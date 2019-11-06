use alloc::{boxed::Box, vec::Vec};
use core::fmt;
use core::ops;
use core::{u32, usize};
use func::{FuncInstance, FuncInstanceInternal, FuncRef};
use host::{AsyncExternals, Externals};
use isa;
use memory::MemoryRef;
use memory_units::Pages;
use module::ModuleRef;
use nan_preserving_float::{F32, F64};
use parity_wasm::elements::Local;
use validation::{DEFAULT_MEMORY_INDEX, DEFAULT_TABLE_INDEX};
use value::{
    ArithmeticOps, ExtendInto, Float, Integer, LittleEndianConvert, RuntimeValue, TransmuteInto,
    TryTruncateInto, WrapInto,
};
use {Signature, Trap, TrapKind, ValueType};
use futures::{Future, future};
use std::cell::RefCell;
use std::rc::Rc;
/// Maximum number of bytes on the value stack.
pub const DEFAULT_VALUE_STACK_LIMIT: usize = 1024 * 1024;

/// Maximum number of levels on the call stack.
pub const DEFAULT_CALL_STACK_LIMIT: usize = 64 * 1024;

/// This is a wrapper around u64 to allow us to treat runtime values as a tag-free `u64`
/// (where if the runtime value is <64 bits the upper bits are 0). This is safe, since
/// all of the possible runtime values are valid to create from 64 defined bits, so if
/// types don't line up we get a logic error (which will ideally be caught by the wasm
/// spec tests) and not undefined behaviour.
///
/// At the boundary between the interpreter and the outside world we convert to the public
/// `RuntimeValue` type, which can then be matched on. We can create a `RuntimeValue` from
/// a `RuntimeValueInternal` only when the type is statically known, which it always is
/// at these boundaries.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
#[repr(transparent)]
struct RuntimeValueInternal(pub u64);

impl RuntimeValueInternal {
    pub fn with_type(self, ty: ValueType) -> RuntimeValue {
        match ty {
            ValueType::I32 => RuntimeValue::I32(<_>::from_runtime_value_internal(self)),
            ValueType::I64 => RuntimeValue::I64(<_>::from_runtime_value_internal(self)),
            ValueType::F32 => RuntimeValue::F32(<_>::from_runtime_value_internal(self)),
            ValueType::F64 => RuntimeValue::F64(<_>::from_runtime_value_internal(self)),
        }
    }
}

trait FromRuntimeValueInternal
where
    Self: Sized,
{
    fn from_runtime_value_internal(val: RuntimeValueInternal) -> Self;
}

macro_rules! impl_from_runtime_value_internal {
	($($t:ty),*) =>	{
		$(
			impl FromRuntimeValueInternal for $t {
				fn from_runtime_value_internal(
					RuntimeValueInternal(val): RuntimeValueInternal,
				) -> Self {
					val	as _
				}
			}

			impl From<$t> for RuntimeValueInternal {
				fn from(other: $t) -> Self {
					RuntimeValueInternal(other as _)
				}
			}
		)*
	};
}

macro_rules! impl_from_runtime_value_internal_float	{
	($($t:ty),*) =>	{
		$(
			impl FromRuntimeValueInternal for $t {
				fn from_runtime_value_internal(
					RuntimeValueInternal(val): RuntimeValueInternal,
				) -> Self {
					<$t>::from_bits(val	as _)
				}
			}

			impl From<$t> for RuntimeValueInternal {
				fn from(other: $t) -> Self {
					RuntimeValueInternal(other.to_bits() as	_)
				}
			}
		)*
	};
}

impl_from_runtime_value_internal!(i8, u8, i16, u16, i32, u32, i64, u64);
impl_from_runtime_value_internal_float!(f32, f64, F32, F64);

impl From<bool> for RuntimeValueInternal {
    fn from(other: bool) -> Self {
        (if other { 1 } else { 0 }).into()
    }
}

impl FromRuntimeValueInternal for bool {
    fn from_runtime_value_internal(RuntimeValueInternal(val): RuntimeValueInternal) -> Self {
        val != 0
    }
}

impl From<RuntimeValue> for RuntimeValueInternal {
    fn from(other: RuntimeValue) -> Self {
        match other {
            RuntimeValue::I32(val) => val.into(),
            RuntimeValue::I64(val) => val.into(),
            RuntimeValue::F32(val) => val.into(),
            RuntimeValue::F64(val) => val.into(),
        }
    }
}

/// Interpreter action to execute after executing instruction.
pub enum InstructionOutcome {
    /// Continue with next instruction.
    RunNextInstruction,
    /// Branch to an instruction at the given position.
    Branch(isa::Target),
    /// Execute function call.
    ExecuteCall(FuncRef),
    /// Return from current function block.
    Return(isa::DropKeep),
}

#[derive(PartialEq, Eq)]
/// Function execution state, related to pause and resume.
pub enum InterpreterState {
    /// The interpreter has been created, but has not been executed.
    Initialized,
    /// The interpreter has started execution, and cannot be called again if it exits normally, or no Host traps happened.
    Started,
    /// The interpreter has been executed, and returned a Host trap. It can resume execution by providing back a return
    /// value.
    Resumable(Option<ValueType>),
}

impl InterpreterState {
    pub fn is_resumable(&self) -> bool {
        match self {
            &InterpreterState::Resumable(_) => true,
            _ => false,
        }
    }
}

/// Function run result.
enum RunResult {
    /// Function has returned.
    Return,
    /// Function is calling other function.
    NestedCall(FuncRef),
}

/// Function interpreter.
pub struct Interpreter {
    value_stack: ValueStack,
    call_stack: CallStack,
    return_type: Option<ValueType>,
    state: InterpreterState,
}

impl Interpreter {
    pub fn new(
        func: &FuncRef,
        args: &[RuntimeValue],
        mut stack_recycler: Option<&mut StackRecycler>,
    ) -> Result<Interpreter, Trap> {
        let mut value_stack = StackRecycler::recreate_value_stack(&mut stack_recycler);
        for &arg in args {
            let arg = arg.into();
            value_stack.push(arg).map_err(
                // There is not enough space for pushing initial arguments.
                // Weird, but bail out anyway.
                |_| Trap::from(TrapKind::StackOverflow),
            )?;
        }

        let mut call_stack = StackRecycler::recreate_call_stack(&mut stack_recycler);
        let initial_frame = FunctionContext::new(func.clone());
        call_stack.push(initial_frame);

        let return_type = func.signature().return_type();

        Ok(Interpreter {
            value_stack,
            call_stack,
            return_type,
            state: InterpreterState::Initialized,
        })
    }

    pub fn state(&self) -> &InterpreterState {
        &self.state
    }

    pub fn start_execution<'a, E: Externals + 'a>(
        &mut self,
        externals: &'a mut E,
    ) -> Result<Option<RuntimeValue>, Trap> {
        // Ensure that the VM has not been executed. This is checked in `FuncInvocation::start_execution`.
        assert!(self.state == InterpreterState::Initialized);

        self.state = InterpreterState::Started;
        self.run_interpreter_loop(externals)?;

        let opt_return_value = self
            .return_type
            .map(|vt| self.value_stack.pop().with_type(vt));

        // Ensure that stack is empty after the execution. This is guaranteed by the validation properties.
        assert!(self.value_stack.len() == 0);

        Ok(opt_return_value)
    }

    pub fn start_execution_async<'a,E:AsyncExternals +'a>(
        mut self,
        externals: Rc<E>,
    ) -> Box<dyn Future<Item = Option<RuntimeValue>, Error = Trap>+'a> 
    {
        // Ensure that the VM has not been executed. This is checked in `FuncInvocation::start_execution`.
        assert!(self.state == InterpreterState::Initialized);

        self.state = InterpreterState::Started;
        Box::new(self.run_interpreter_loop_async(externals))
    }

    pub fn resume_execution<'a, E: Externals + 'a>(
        &mut self,
        return_val: Option<RuntimeValue>,
        externals: &'a mut E,
    ) -> Result<Option<RuntimeValue>, Trap> {
        use core::mem::swap;

        // Ensure that the VM is resumable. This is checked in `FuncInvocation::resume_execution`.
        assert!(self.state.is_resumable());

        let mut resumable_state = InterpreterState::Started;
        swap(&mut self.state, &mut resumable_state);

        if let Some(return_val) = return_val {
            self.value_stack
                .push(return_val.into())
                .map_err(Trap::new)?;
        }

        self.run_interpreter_loop(externals)?;

        let opt_return_value = self
            .return_type
            .map(|vt| self.value_stack.pop().with_type(vt));

        // Ensure that stack is empty after the execution. This is guaranteed by the validation properties.
        assert!(self.value_stack.len() == 0);

        Ok(opt_return_value)
    }

    fn run_interpreter_loop<'a, E: Externals + 'a>(
        &mut self,
        externals: &'a mut E,
    ) -> Result<(), Trap> {
        loop {
            let mut function_context = self.call_stack.pop().expect(
                "on loop entry - not empty; on loop continue - checking for emptiness; qed",
            );
            let function_ref = function_context.function.clone();
            let function_body = function_ref
				.body()
				.expect(
					"Host functions checked in function_return below; Internal functions always have a body; qed"
				);

            if !function_context.is_initialized() {
                // Initialize stack frame for the function call.
                function_context.initialize(&function_body.locals, &mut self.value_stack)?;
            }

            let function_return = self
                .do_run_function(&mut function_context, &function_body.code)
                .map_err(Trap::new)?;

            match function_return {
                RunResult::Return => {
                    if self.call_stack.is_empty() {
                        // This was the last frame in the call stack. This means we
                        // are done executing.
                        return Ok(());
                    }
                }
                RunResult::NestedCall(nested_func) => {
                    if self.call_stack.is_full() {
                        return Err(TrapKind::StackOverflow.into());
                    }

                    match *nested_func.as_internal() {
                        FuncInstanceInternal::Internal { .. } => {
                            let nested_context = FunctionContext::new(nested_func.clone());
                            self.call_stack.push(function_context);
                            self.call_stack.push(nested_context);
                        }
                        FuncInstanceInternal::Host { ref signature, .. } => {
                            let args = prepare_function_args(signature, &mut self.value_stack);
                            // We push the function context first. If the VM is not resumable, it does no harm. If it is, we then save the context here.
                            self.call_stack.push(function_context);

                            let return_val =
                                match FuncInstance::invoke(&nested_func, &args, externals) {
                                    Ok(val) => val,
                                    Err(trap) => {
                                        if trap.kind().is_host() {
                                            self.state = InterpreterState::Resumable(
                                                nested_func.signature().return_type(),
                                            );
                                        }
                                        return Err(trap);
                                    }
                                };

                            // Check if `return_val` matches the signature.
                            let value_ty = return_val.as_ref().map(|val| val.value_type());
                            let expected_ty = nested_func.signature().return_type();
                            if value_ty != expected_ty {
                                return Err(TrapKind::UnexpectedSignature.into());
                            }

                            if let Some(return_val) = return_val {
                                self.value_stack
                                    .push(return_val.into())
                                    .map_err(Trap::new)?;
                            }
                        }
                    }
                }
            }
        }
    }

    fn do_run_function(
        &mut self,
        function_context: &mut FunctionContext,
        instructions: &isa::Instructions,
    ) -> Result<RunResult, TrapKind> {
        let mut iter = instructions.iterate_from(function_context.position);

        loop {
            let instruction = iter.next().expect(
                "Ran out of instructions, this should be impossible \
                 since validation ensures that we either have an explicit \
                 return or an implicit block `end`.",
            );

            match self.run_instruction(function_context, &instruction)? {
                InstructionOutcome::RunNextInstruction => {}
                InstructionOutcome::Branch(target) => {
                    iter = instructions.iterate_from(target.dst_pc);
                    self.value_stack.drop_keep(target.drop_keep);
                }
                InstructionOutcome::ExecuteCall(func_ref) => {
                    function_context.position = iter.position();
                    return Ok(RunResult::NestedCall(func_ref));
                }
                InstructionOutcome::Return(drop_keep) => {
                    self.value_stack.drop_keep(drop_keep);
                    break;
                }
            }
        }

        Ok(RunResult::Return)
    }


    fn run_interpreter_loop_async<'a,E:AsyncExternals +'a>(
        self,
        externals: Rc<E>,
    ) -> Box<dyn Future<Item = Option<RuntimeValue>, Error = Trap> +'a>   
    {
        //return loop?
        Box::new(
            future::loop_fn(Rc::new(RefCell::new(self)),move |rs| {
                let mut function_context = rs.borrow_mut().call_stack.pop().expect(
                    "on loop entry - not empty; on loop continue - checking for emptiness; qed",
                );
                let function_ref = function_context.function.clone();
                let function_body = function_ref
				.body()
				.expect(
					"Host functions checked in function_return below; Internal functions always have a body; qed"
				);

                if !function_context.is_initialized() {
                    // Initialize stack frame for the function call.
                    if let Err(err)= function_context
                        .initialize(&function_body.locals, &mut rs.borrow_mut().value_stack){
                            return future::Either::A(future::err(err.into()))
                        }
                        
                }

                let function_return = match rs.borrow_mut()
                    .do_run_function(&mut function_context, &function_body.code)
                    .map_err(Trap::new){
                        Ok(f)=>f,
                        Err(err)=>{
                            return future::Either::A(future::err(err.into()))
                        }
                    };
                match function_return {
                    RunResult::Return => {
                        if rs.borrow().call_stack.is_empty() {
                            // This was the last frame in the call stack. This means we
                            // are done executing.
                            let mut r=rs.borrow_mut();
                            let opt_return_value = r
                                .return_type
                                .map(move |vt| r.value_stack.pop().with_type(vt));
                            assert!(rs.borrow().value_stack.len() == 0);
                            return future::Either::A(future::ok(future::Loop::Break(opt_return_value)))
                        }
                        return future::Either::A(future::ok(future::Loop::Continue(Rc::clone(&rs))));
                    }
                    RunResult::NestedCall(nested_func) => {
                        if rs.borrow().call_stack.is_full() {
                            return future::Either::A(future::err(TrapKind::StackOverflow.into()));
                        }
                        match *nested_func.as_internal() {
                            FuncInstanceInternal::Internal { .. } => {
                                let nested_context = FunctionContext::new(nested_func.clone());
                                rs.borrow_mut().call_stack.push(function_context);
                                rs.borrow_mut().call_stack.push(nested_context);
                                return future::Either::A(future::ok(future::Loop::Continue(Rc::clone(&rs))));
                            }
                            FuncInstanceInternal::Host { ref signature, .. } => {
                                let args = prepare_function_args(signature, &mut rs.borrow_mut().value_stack);
                                // We push the function context first. If the VM is not resumable, it does no harm. If it is, we then save the context here.
                                rs.borrow_mut().call_stack.push(function_context);
                                let nf=nested_func.clone();
                                let f = FuncInstance::invoke_async(nested_func.clone(), args, Rc::clone(&externals))
                                    .then(move |ret| {
                                        let s=Rc::clone(&rs);
                                        let return_val = match ret {
                                            Ok(val) => val,
                                            Err(trap) => {
                                                if trap.kind().is_host() {
                                                    s.borrow_mut().state = InterpreterState::Resumable(
                                                        nf.signature().return_type(),
                                                    );
                                                }
                                                return future::err(trap);
                                            }
                                        };
                                        // Check if `return_val` matches the signature.
                                        let value_ty =
                                            return_val.as_ref().map(|val| val.value_type());
                                        let expected_ty = nf.signature().return_type();
                                        if value_ty != expected_ty {
                                            return future::err(
                                                TrapKind::UnexpectedSignature.into(),
                                            );
                                        }
                                        if let Some(return_val) = return_val {
                                            if let Err(err)=s.borrow_mut().value_stack
                                                .push(return_val.into())
                                                .map_err(Trap::new){
                                                    return future::err(err)
                                                }

                                        }
                                        future::ok(future::Loop::Continue(Rc::clone(&rs)))
                                    });
                                return future::Either::B(Box::new(f));
                                // return future::Either::B(future::ok(future::Loop::Continue(Rc::clone(&rs))));
                            } //host
                        } //nested as internal
                    } //nested
                } //end of match
                  //have to explicitly continue as we aren't a real loop
                  // return future::ok(future::Loop::Continue(Rc::clone(&rs)));
            }), //End of Loop
        ) //End of Box
    }

    //Async
    #[inline(always)]
    fn run_instruction(
        &mut self,
        context: &mut FunctionContext,
        instruction: &isa::Instruction,
    ) -> Result<InstructionOutcome, TrapKind> {
        match instruction {
            &isa::Instruction::Unreachable => self.run_unreachable(context),

            &isa::Instruction::Br(target) => self.run_br(context, target.clone()),
            &isa::Instruction::BrIfEqz(target) => self.run_br_eqz(target.clone()),
            &isa::Instruction::BrIfNez(target) => self.run_br_nez(target.clone()),
            &isa::Instruction::BrTable(targets) => self.run_br_table(targets),
            &isa::Instruction::Return(drop_keep) => self.run_return(drop_keep),

            &isa::Instruction::Call(index) => self.run_call(context, index),
            &isa::Instruction::CallIndirect(index) => self.run_call_indirect(context, index),

            &isa::Instruction::Drop => self.run_drop(),
            &isa::Instruction::Select => self.run_select(),

            &isa::Instruction::GetLocal(depth) => self.run_get_local(depth),
            &isa::Instruction::SetLocal(depth) => self.run_set_local(depth),
            &isa::Instruction::TeeLocal(depth) => self.run_tee_local(depth),
            &isa::Instruction::GetGlobal(index) => self.run_get_global(context, index),
            &isa::Instruction::SetGlobal(index) => self.run_set_global(context, index),

            &isa::Instruction::I32Load(offset) => self.run_load::<i32>(context, offset),
            &isa::Instruction::I64Load(offset) => self.run_load::<i64>(context, offset),
            &isa::Instruction::F32Load(offset) => self.run_load::<F32>(context, offset),
            &isa::Instruction::F64Load(offset) => self.run_load::<F64>(context, offset),
            &isa::Instruction::I32Load8S(offset) => {
                self.run_load_extend::<i8, i32>(context, offset)
            }
            &isa::Instruction::I32Load8U(offset) => {
                self.run_load_extend::<u8, i32>(context, offset)
            }
            &isa::Instruction::I32Load16S(offset) => {
                self.run_load_extend::<i16, i32>(context, offset)
            }
            &isa::Instruction::I32Load16U(offset) => {
                self.run_load_extend::<u16, i32>(context, offset)
            }
            &isa::Instruction::I64Load8S(offset) => {
                self.run_load_extend::<i8, i64>(context, offset)
            }
            &isa::Instruction::I64Load8U(offset) => {
                self.run_load_extend::<u8, i64>(context, offset)
            }
            &isa::Instruction::I64Load16S(offset) => {
                self.run_load_extend::<i16, i64>(context, offset)
            }
            &isa::Instruction::I64Load16U(offset) => {
                self.run_load_extend::<u16, i64>(context, offset)
            }
            &isa::Instruction::I64Load32S(offset) => {
                self.run_load_extend::<i32, i64>(context, offset)
            }
            &isa::Instruction::I64Load32U(offset) => {
                self.run_load_extend::<u32, i64>(context, offset)
            }

            &isa::Instruction::I32Store(offset) => self.run_store::<i32>(context, offset),
            &isa::Instruction::I64Store(offset) => self.run_store::<i64>(context, offset),
            &isa::Instruction::F32Store(offset) => self.run_store::<F32>(context, offset),
            &isa::Instruction::F64Store(offset) => self.run_store::<F64>(context, offset),
            &isa::Instruction::I32Store8(offset) => self.run_store_wrap::<i32, i8>(context, offset),
            &isa::Instruction::I32Store16(offset) => {
                self.run_store_wrap::<i32, i16>(context, offset)
            }
            &isa::Instruction::I64Store8(offset) => self.run_store_wrap::<i64, i8>(context, offset),
            &isa::Instruction::I64Store16(offset) => {
                self.run_store_wrap::<i64, i16>(context, offset)
            }
            &isa::Instruction::I64Store32(offset) => {
                self.run_store_wrap::<i64, i32>(context, offset)
            }

            &isa::Instruction::CurrentMemory => self.run_current_memory(context),
            &isa::Instruction::GrowMemory => self.run_grow_memory(context),

            &isa::Instruction::I32Const(val) => self.run_const(val.into()),
            &isa::Instruction::I64Const(val) => self.run_const(val.into()),
            &isa::Instruction::F32Const(val) => self.run_const(val.into()),
            &isa::Instruction::F64Const(val) => self.run_const(val.into()),

            &isa::Instruction::I32Eqz => self.run_eqz::<i32>(),
            &isa::Instruction::I32Eq => self.run_eq::<i32>(),
            &isa::Instruction::I32Ne => self.run_ne::<i32>(),
            &isa::Instruction::I32LtS => self.run_lt::<i32>(),
            &isa::Instruction::I32LtU => self.run_lt::<u32>(),
            &isa::Instruction::I32GtS => self.run_gt::<i32>(),
            &isa::Instruction::I32GtU => self.run_gt::<u32>(),
            &isa::Instruction::I32LeS => self.run_lte::<i32>(),
            &isa::Instruction::I32LeU => self.run_lte::<u32>(),
            &isa::Instruction::I32GeS => self.run_gte::<i32>(),
            &isa::Instruction::I32GeU => self.run_gte::<u32>(),

            &isa::Instruction::I64Eqz => self.run_eqz::<i64>(),
            &isa::Instruction::I64Eq => self.run_eq::<i64>(),
            &isa::Instruction::I64Ne => self.run_ne::<i64>(),
            &isa::Instruction::I64LtS => self.run_lt::<i64>(),
            &isa::Instruction::I64LtU => self.run_lt::<u64>(),
            &isa::Instruction::I64GtS => self.run_gt::<i64>(),
            &isa::Instruction::I64GtU => self.run_gt::<u64>(),
            &isa::Instruction::I64LeS => self.run_lte::<i64>(),
            &isa::Instruction::I64LeU => self.run_lte::<u64>(),
            &isa::Instruction::I64GeS => self.run_gte::<i64>(),
            &isa::Instruction::I64GeU => self.run_gte::<u64>(),

            &isa::Instruction::F32Eq => self.run_eq::<F32>(),
            &isa::Instruction::F32Ne => self.run_ne::<F32>(),
            &isa::Instruction::F32Lt => self.run_lt::<F32>(),
            &isa::Instruction::F32Gt => self.run_gt::<F32>(),
            &isa::Instruction::F32Le => self.run_lte::<F32>(),
            &isa::Instruction::F32Ge => self.run_gte::<F32>(),

            &isa::Instruction::F64Eq => self.run_eq::<F64>(),
            &isa::Instruction::F64Ne => self.run_ne::<F64>(),
            &isa::Instruction::F64Lt => self.run_lt::<F64>(),
            &isa::Instruction::F64Gt => self.run_gt::<F64>(),
            &isa::Instruction::F64Le => self.run_lte::<F64>(),
            &isa::Instruction::F64Ge => self.run_gte::<F64>(),

            &isa::Instruction::I32Clz => self.run_clz::<i32>(),
            &isa::Instruction::I32Ctz => self.run_ctz::<i32>(),
            &isa::Instruction::I32Popcnt => self.run_popcnt::<i32>(),
            &isa::Instruction::I32Add => self.run_add::<i32>(),
            &isa::Instruction::I32Sub => self.run_sub::<i32>(),
            &isa::Instruction::I32Mul => self.run_mul::<i32>(),
            &isa::Instruction::I32DivS => self.run_div::<i32, i32>(),
            &isa::Instruction::I32DivU => self.run_div::<i32, u32>(),
            &isa::Instruction::I32RemS => self.run_rem::<i32, i32>(),
            &isa::Instruction::I32RemU => self.run_rem::<i32, u32>(),
            &isa::Instruction::I32And => self.run_and::<i32>(),
            &isa::Instruction::I32Or => self.run_or::<i32>(),
            &isa::Instruction::I32Xor => self.run_xor::<i32>(),
            &isa::Instruction::I32Shl => self.run_shl::<i32>(0x1F),
            &isa::Instruction::I32ShrS => self.run_shr::<i32, i32>(0x1F),
            &isa::Instruction::I32ShrU => self.run_shr::<i32, u32>(0x1F),
            &isa::Instruction::I32Rotl => self.run_rotl::<i32>(),
            &isa::Instruction::I32Rotr => self.run_rotr::<i32>(),

            &isa::Instruction::I64Clz => self.run_clz::<i64>(),
            &isa::Instruction::I64Ctz => self.run_ctz::<i64>(),
            &isa::Instruction::I64Popcnt => self.run_popcnt::<i64>(),
            &isa::Instruction::I64Add => self.run_add::<i64>(),
            &isa::Instruction::I64Sub => self.run_sub::<i64>(),
            &isa::Instruction::I64Mul => self.run_mul::<i64>(),
            &isa::Instruction::I64DivS => self.run_div::<i64, i64>(),
            &isa::Instruction::I64DivU => self.run_div::<i64, u64>(),
            &isa::Instruction::I64RemS => self.run_rem::<i64, i64>(),
            &isa::Instruction::I64RemU => self.run_rem::<i64, u64>(),
            &isa::Instruction::I64And => self.run_and::<i64>(),
            &isa::Instruction::I64Or => self.run_or::<i64>(),
            &isa::Instruction::I64Xor => self.run_xor::<i64>(),
            &isa::Instruction::I64Shl => self.run_shl::<i64>(0x3F),
            &isa::Instruction::I64ShrS => self.run_shr::<i64, i64>(0x3F),
            &isa::Instruction::I64ShrU => self.run_shr::<i64, u64>(0x3F),
            &isa::Instruction::I64Rotl => self.run_rotl::<i64>(),
            &isa::Instruction::I64Rotr => self.run_rotr::<i64>(),

            &isa::Instruction::F32Abs => self.run_abs::<F32>(),
            &isa::Instruction::F32Neg => self.run_neg::<F32>(),
            &isa::Instruction::F32Ceil => self.run_ceil::<F32>(),
            &isa::Instruction::F32Floor => self.run_floor::<F32>(),
            &isa::Instruction::F32Trunc => self.run_trunc::<F32>(),
            &isa::Instruction::F32Nearest => self.run_nearest::<F32>(),
            &isa::Instruction::F32Sqrt => self.run_sqrt::<F32>(),
            &isa::Instruction::F32Add => self.run_add::<F32>(),
            &isa::Instruction::F32Sub => self.run_sub::<F32>(),
            &isa::Instruction::F32Mul => self.run_mul::<F32>(),
            &isa::Instruction::F32Div => self.run_div::<F32, F32>(),
            &isa::Instruction::F32Min => self.run_min::<F32>(),
            &isa::Instruction::F32Max => self.run_max::<F32>(),
            &isa::Instruction::F32Copysign => self.run_copysign::<F32>(),

            &isa::Instruction::F64Abs => self.run_abs::<F64>(),
            &isa::Instruction::F64Neg => self.run_neg::<F64>(),
            &isa::Instruction::F64Ceil => self.run_ceil::<F64>(),
            &isa::Instruction::F64Floor => self.run_floor::<F64>(),
            &isa::Instruction::F64Trunc => self.run_trunc::<F64>(),
            &isa::Instruction::F64Nearest => self.run_nearest::<F64>(),
            &isa::Instruction::F64Sqrt => self.run_sqrt::<F64>(),
            &isa::Instruction::F64Add => self.run_add::<F64>(),
            &isa::Instruction::F64Sub => self.run_sub::<F64>(),
            &isa::Instruction::F64Mul => self.run_mul::<F64>(),
            &isa::Instruction::F64Div => self.run_div::<F64, F64>(),
            &isa::Instruction::F64Min => self.run_min::<F64>(),
            &isa::Instruction::F64Max => self.run_max::<F64>(),
            &isa::Instruction::F64Copysign => self.run_copysign::<F64>(),

            &isa::Instruction::I32WrapI64 => self.run_wrap::<i64, i32>(),
            &isa::Instruction::I32TruncSF32 => self.run_trunc_to_int::<F32, i32, i32>(),
            &isa::Instruction::I32TruncUF32 => self.run_trunc_to_int::<F32, u32, i32>(),
            &isa::Instruction::I32TruncSF64 => self.run_trunc_to_int::<F64, i32, i32>(),
            &isa::Instruction::I32TruncUF64 => self.run_trunc_to_int::<F64, u32, i32>(),
            &isa::Instruction::I64ExtendSI32 => self.run_extend::<i32, i64, i64>(),
            &isa::Instruction::I64ExtendUI32 => self.run_extend::<u32, u64, i64>(),
            &isa::Instruction::I64TruncSF32 => self.run_trunc_to_int::<F32, i64, i64>(),
            &isa::Instruction::I64TruncUF32 => self.run_trunc_to_int::<F32, u64, i64>(),
            &isa::Instruction::I64TruncSF64 => self.run_trunc_to_int::<F64, i64, i64>(),
            &isa::Instruction::I64TruncUF64 => self.run_trunc_to_int::<F64, u64, i64>(),
            &isa::Instruction::F32ConvertSI32 => self.run_extend::<i32, F32, F32>(),
            &isa::Instruction::F32ConvertUI32 => self.run_extend::<u32, F32, F32>(),
            &isa::Instruction::F32ConvertSI64 => self.run_wrap::<i64, F32>(),
            &isa::Instruction::F32ConvertUI64 => self.run_wrap::<u64, F32>(),
            &isa::Instruction::F32DemoteF64 => self.run_wrap::<F64, F32>(),
            &isa::Instruction::F64ConvertSI32 => self.run_extend::<i32, F64, F64>(),
            &isa::Instruction::F64ConvertUI32 => self.run_extend::<u32, F64, F64>(),
            &isa::Instruction::F64ConvertSI64 => self.run_extend::<i64, F64, F64>(),
            &isa::Instruction::F64ConvertUI64 => self.run_extend::<u64, F64, F64>(),
            &isa::Instruction::F64PromoteF32 => self.run_extend::<F32, F64, F64>(),

            &isa::Instruction::I32ReinterpretF32 => self.run_reinterpret::<F32, i32>(),
            &isa::Instruction::I64ReinterpretF64 => self.run_reinterpret::<F64, i64>(),
            &isa::Instruction::F32ReinterpretI32 => self.run_reinterpret::<i32, F32>(),
            &isa::Instruction::F64ReinterpretI64 => self.run_reinterpret::<i64, F64>(),
        }
    }

    fn run_unreachable(
        &mut self,
        _context: &mut FunctionContext,
    ) -> Result<InstructionOutcome, TrapKind> {
        Err(TrapKind::Unreachable)
    }

    fn run_br(
        &mut self,
        _context: &mut FunctionContext,
        target: isa::Target,
    ) -> Result<InstructionOutcome, TrapKind> {
        Ok(InstructionOutcome::Branch(target))
    }

    fn run_br_nez(&mut self, target: isa::Target) -> Result<InstructionOutcome, TrapKind> {
        let condition = self.value_stack.pop_as();
        if condition {
            Ok(InstructionOutcome::Branch(target))
        } else {
            Ok(InstructionOutcome::RunNextInstruction)
        }
    }

    fn run_br_eqz(&mut self, target: isa::Target) -> Result<InstructionOutcome, TrapKind> {
        let condition = self.value_stack.pop_as();
        if condition {
            Ok(InstructionOutcome::RunNextInstruction)
        } else {
            Ok(InstructionOutcome::Branch(target))
        }
    }

    fn run_br_table(&mut self, targets: isa::BrTargets) -> Result<InstructionOutcome, TrapKind> {
        let index: u32 = self.value_stack.pop_as();

        let dst = targets.get(index);

        Ok(InstructionOutcome::Branch(dst))
    }

    fn run_return(&mut self, drop_keep: isa::DropKeep) -> Result<InstructionOutcome, TrapKind> {
        Ok(InstructionOutcome::Return(drop_keep))
    }

    fn run_call(
        &mut self,
        context: &mut FunctionContext,
        func_idx: u32,
    ) -> Result<InstructionOutcome, TrapKind> {
        let func = context
            .module()
            .func_by_index(func_idx)
            .expect("Due to validation func should exists");
        Ok(InstructionOutcome::ExecuteCall(func))
    }

    fn run_call_indirect(
        &mut self,
        context: &mut FunctionContext,
        signature_idx: u32,
    ) -> Result<InstructionOutcome, TrapKind> {
        let table_func_idx: u32 = self.value_stack.pop_as();
        let table = context
            .module()
            .table_by_index(DEFAULT_TABLE_INDEX)
            .expect("Due to validation table should exists");
        let func_ref = table
            .get(table_func_idx)
            .map_err(|_| TrapKind::TableAccessOutOfBounds)?
            .ok_or_else(|| TrapKind::ElemUninitialized)?;

        {
            let actual_function_type = func_ref.signature();
            let required_function_type = context
                .module()
                .signature_by_index(signature_idx)
                .expect("Due to validation type should exists");

            if &*required_function_type != actual_function_type {
                return Err(TrapKind::UnexpectedSignature);
            }
        }

        Ok(InstructionOutcome::ExecuteCall(func_ref))
    }

    fn run_drop(&mut self) -> Result<InstructionOutcome, TrapKind> {
        let _ = self.value_stack.pop();
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_select(&mut self) -> Result<InstructionOutcome, TrapKind> {
        let (left, mid, right) = self.value_stack.pop_triple();

        let condition = <_>::from_runtime_value_internal(right);
        let val = if condition { left } else { mid };
        self.value_stack.push(val)?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_get_local(&mut self, index: u32) -> Result<InstructionOutcome, TrapKind> {
        let val = *self.value_stack.pick_mut(index as usize);
        self.value_stack.push(val)?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_set_local(&mut self, index: u32) -> Result<InstructionOutcome, TrapKind> {
        let val = self.value_stack.pop();
        *self.value_stack.pick_mut(index as usize) = val;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_tee_local(&mut self, index: u32) -> Result<InstructionOutcome, TrapKind> {
        let val = self.value_stack.top().clone();
        *self.value_stack.pick_mut(index as usize) = val;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_get_global(
        &mut self,
        context: &mut FunctionContext,
        index: u32,
    ) -> Result<InstructionOutcome, TrapKind> {
        let global = context
            .module()
            .global_by_index(index)
            .expect("Due to validation global should exists");
        let val = global.get();
        self.value_stack.push(val.into())?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_set_global(
        &mut self,
        context: &mut FunctionContext,
        index: u32,
    ) -> Result<InstructionOutcome, TrapKind> {
        let val = self.value_stack.pop();
        let global = context
            .module()
            .global_by_index(index)
            .expect("Due to validation global should exists");
        global
            .set(val.with_type(global.value_type()))
            .expect("Due to validation set to a global should succeed");
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_load<T>(
        &mut self,
        context: &mut FunctionContext,
        offset: u32,
    ) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: LittleEndianConvert,
    {
        let raw_address = self.value_stack.pop_as();
        let address = effective_address(offset, raw_address)?;
        let m = context
            .memory()
            .expect("Due to validation memory should exists");
        let n: T = m
            .get_value(address)
            .map_err(|_| TrapKind::MemoryAccessOutOfBounds)?;
        self.value_stack.push(n.into())?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_load_extend<T, U>(
        &mut self,
        context: &mut FunctionContext,
        offset: u32,
    ) -> Result<InstructionOutcome, TrapKind>
    where
        T: ExtendInto<U>,
        RuntimeValueInternal: From<U>,
        T: LittleEndianConvert,
    {
        let raw_address = self.value_stack.pop_as();
        let address = effective_address(offset, raw_address)?;
        let m = context
            .memory()
            .expect("Due to validation memory should exists");
        let v: T = m
            .get_value(address)
            .map_err(|_| TrapKind::MemoryAccessOutOfBounds)?;
        let stack_value: U = v.extend_into();
        self.value_stack
            .push(stack_value.into())
            .map_err(Into::into)
            .map(|_| InstructionOutcome::RunNextInstruction)
    }

    fn run_store<T>(
        &mut self,
        context: &mut FunctionContext,
        offset: u32,
    ) -> Result<InstructionOutcome, TrapKind>
    where
        T: FromRuntimeValueInternal,
        T: LittleEndianConvert,
    {
        let stack_value = self.value_stack.pop_as::<T>();
        let raw_address = self.value_stack.pop_as::<u32>();
        let address = effective_address(offset, raw_address)?;

        let m = context
            .memory()
            .expect("Due to validation memory should exists");
        m.set_value(address, stack_value)
            .map_err(|_| TrapKind::MemoryAccessOutOfBounds)?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_store_wrap<T, U>(
        &mut self,
        context: &mut FunctionContext,
        offset: u32,
    ) -> Result<InstructionOutcome, TrapKind>
    where
        T: FromRuntimeValueInternal,
        T: WrapInto<U>,
        U: LittleEndianConvert,
    {
        let stack_value: T = <_>::from_runtime_value_internal(self.value_stack.pop());
        let stack_value = stack_value.wrap_into();
        let raw_address = self.value_stack.pop_as::<u32>();
        let address = effective_address(offset, raw_address)?;
        let m = context
            .memory()
            .expect("Due to validation memory should exists");
        m.set_value(address, stack_value)
            .map_err(|_| TrapKind::MemoryAccessOutOfBounds)?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_current_memory(
        &mut self,
        context: &mut FunctionContext,
    ) -> Result<InstructionOutcome, TrapKind> {
        let m = context
            .memory()
            .expect("Due to validation memory should exists");
        let s = m.current_size().0;
        self.value_stack.push(RuntimeValueInternal(s as _))?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_grow_memory(
        &mut self,
        context: &mut FunctionContext,
    ) -> Result<InstructionOutcome, TrapKind> {
        let pages: u32 = self.value_stack.pop_as();
        let m = context
            .memory()
            .expect("Due to validation memory should exists");
        let m = match m.grow(Pages(pages as usize)) {
            Ok(Pages(new_size)) => new_size as u32,
            Err(_) => u32::MAX, // Returns -1 (or 0xFFFFFFFF) in case of error.
        };
        self.value_stack.push(RuntimeValueInternal(m as _))?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_const(&mut self, val: RuntimeValue) -> Result<InstructionOutcome, TrapKind> {
        self.value_stack
            .push(val.into())
            .map_err(Into::into)
            .map(|_| InstructionOutcome::RunNextInstruction)
    }

    fn run_relop<T, F>(&mut self, f: F) -> Result<InstructionOutcome, TrapKind>
    where
        T: FromRuntimeValueInternal,
        F: FnOnce(T, T) -> bool,
    {
        let (left, right) = self.value_stack.pop_pair_as::<T>();
        let v = if f(left, right) {
            RuntimeValueInternal(1)
        } else {
            RuntimeValueInternal(0)
        };
        self.value_stack.push(v)?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_eqz<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        T: FromRuntimeValueInternal,
        T: PartialEq<T> + Default,
    {
        let v = self.value_stack.pop_as::<T>();
        let v = RuntimeValueInternal(if v == Default::default() { 1 } else { 0 });
        self.value_stack.push(v)?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_eq<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        T: FromRuntimeValueInternal + PartialEq<T>,
    {
        self.run_relop(|left: T, right: T| left == right)
    }

    fn run_ne<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        T: FromRuntimeValueInternal + PartialEq<T>,
    {
        self.run_relop(|left: T, right: T| left != right)
    }

    fn run_lt<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        T: FromRuntimeValueInternal + PartialOrd<T>,
    {
        self.run_relop(|left: T, right: T| left < right)
    }

    fn run_gt<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        T: FromRuntimeValueInternal + PartialOrd<T>,
    {
        self.run_relop(|left: T, right: T| left > right)
    }

    fn run_lte<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        T: FromRuntimeValueInternal + PartialOrd<T>,
    {
        self.run_relop(|left: T, right: T| left <= right)
    }

    fn run_gte<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        T: FromRuntimeValueInternal + PartialOrd<T>,
    {
        self.run_relop(|left: T, right: T| left >= right)
    }

    fn run_unop<T, U, F>(&mut self, f: F) -> Result<InstructionOutcome, TrapKind>
    where
        F: FnOnce(T) -> U,
        T: FromRuntimeValueInternal,
        RuntimeValueInternal: From<U>,
    {
        let v = self.value_stack.pop_as::<T>();
        let v = f(v);
        self.value_stack.push(v.into())?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_clz<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: Integer<T> + FromRuntimeValueInternal,
    {
        self.run_unop(|v: T| v.leading_zeros())
    }

    fn run_ctz<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: Integer<T> + FromRuntimeValueInternal,
    {
        self.run_unop(|v: T| v.trailing_zeros())
    }

    fn run_popcnt<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: Integer<T> + FromRuntimeValueInternal,
    {
        self.run_unop(|v: T| v.count_ones())
    }

    fn run_add<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: ArithmeticOps<T> + FromRuntimeValueInternal,
    {
        let (left, right) = self.value_stack.pop_pair_as::<T>();
        let v = left.add(right);
        self.value_stack.push(v.into())?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_sub<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: ArithmeticOps<T> + FromRuntimeValueInternal,
    {
        let (left, right) = self.value_stack.pop_pair_as::<T>();
        let v = left.sub(right);
        self.value_stack.push(v.into())?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_mul<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: ArithmeticOps<T> + FromRuntimeValueInternal,
    {
        let (left, right) = self.value_stack.pop_pair_as::<T>();
        let v = left.mul(right);
        self.value_stack.push(v.into())?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_div<T, U>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: TransmuteInto<U> + FromRuntimeValueInternal,
        U: ArithmeticOps<U> + TransmuteInto<T>,
    {
        let (left, right) = self.value_stack.pop_pair_as::<T>();
        let (left, right) = (left.transmute_into(), right.transmute_into());
        let v = left.div(right)?;
        let v = v.transmute_into();
        self.value_stack.push(v.into())?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_rem<T, U>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: TransmuteInto<U> + FromRuntimeValueInternal,
        U: Integer<U> + TransmuteInto<T>,
    {
        let (left, right) = self.value_stack.pop_pair_as::<T>();
        let (left, right) = (left.transmute_into(), right.transmute_into());
        let v = left.rem(right)?;
        let v = v.transmute_into();
        self.value_stack.push(v.into())?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_and<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<<T as ops::BitAnd>::Output>,
        T: ops::BitAnd<T> + FromRuntimeValueInternal,
    {
        let (left, right) = self.value_stack.pop_pair_as::<T>();
        let v = left.bitand(right);
        self.value_stack.push(v.into())?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_or<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<<T as ops::BitOr>::Output>,
        T: ops::BitOr<T> + FromRuntimeValueInternal,
    {
        let (left, right) = self.value_stack.pop_pair_as::<T>();
        let v = left.bitor(right);
        self.value_stack.push(v.into())?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_xor<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<<T as ops::BitXor>::Output>,
        T: ops::BitXor<T> + FromRuntimeValueInternal,
    {
        let (left, right) = self.value_stack.pop_pair_as::<T>();
        let v = left.bitxor(right);
        self.value_stack.push(v.into())?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_shl<T>(&mut self, mask: T) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<<T as ops::Shl<T>>::Output>,
        T: ops::Shl<T> + ops::BitAnd<T, Output = T> + FromRuntimeValueInternal,
    {
        let (left, right) = self.value_stack.pop_pair_as::<T>();
        let v = left.shl(right & mask);
        self.value_stack.push(v.into())?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_shr<T, U>(&mut self, mask: U) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: TransmuteInto<U> + FromRuntimeValueInternal,
        U: ops::Shr<U> + ops::BitAnd<U, Output = U>,
        <U as ops::Shr<U>>::Output: TransmuteInto<T>,
    {
        let (left, right) = self.value_stack.pop_pair_as::<T>();
        let (left, right) = (left.transmute_into(), right.transmute_into());
        let v = left.shr(right & mask);
        let v = v.transmute_into();
        self.value_stack.push(v.into())?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_rotl<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: Integer<T> + FromRuntimeValueInternal,
    {
        let (left, right) = self.value_stack.pop_pair_as::<T>();
        let v = left.rotl(right);
        self.value_stack.push(v.into())?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_rotr<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: Integer<T> + FromRuntimeValueInternal,
    {
        let (left, right) = self.value_stack.pop_pair_as::<T>();
        let v = left.rotr(right);
        self.value_stack.push(v.into())?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_abs<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: Float<T> + FromRuntimeValueInternal,
    {
        self.run_unop(|v: T| v.abs())
    }

    fn run_neg<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<<T as ops::Neg>::Output>,
        T: ops::Neg + FromRuntimeValueInternal,
    {
        self.run_unop(|v: T| v.neg())
    }

    fn run_ceil<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: Float<T> + FromRuntimeValueInternal,
    {
        self.run_unop(|v: T| v.ceil())
    }

    fn run_floor<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: Float<T> + FromRuntimeValueInternal,
    {
        self.run_unop(|v: T| v.floor())
    }

    fn run_trunc<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: Float<T> + FromRuntimeValueInternal,
    {
        self.run_unop(|v: T| v.trunc())
    }

    fn run_nearest<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: Float<T> + FromRuntimeValueInternal,
    {
        self.run_unop(|v: T| v.nearest())
    }

    fn run_sqrt<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: Float<T> + FromRuntimeValueInternal,
    {
        self.run_unop(|v: T| v.sqrt())
    }

    fn run_min<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: Float<T> + FromRuntimeValueInternal,
    {
        let (left, right) = self.value_stack.pop_pair_as::<T>();
        let v = left.min(right);
        self.value_stack.push(v.into())?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_max<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: Float<T> + FromRuntimeValueInternal,
    {
        let (left, right) = self.value_stack.pop_pair_as::<T>();
        let v = left.max(right);
        self.value_stack.push(v.into())?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_copysign<T>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<T>,
        T: Float<T> + FromRuntimeValueInternal,
    {
        let (left, right) = self.value_stack.pop_pair_as::<T>();
        let v = left.copysign(right);
        self.value_stack.push(v.into())?;
        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_wrap<T, U>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<U>,
        T: WrapInto<U> + FromRuntimeValueInternal,
    {
        self.run_unop(|v: T| v.wrap_into())
    }

    fn run_trunc_to_int<T, U, V>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<V>,
        T: TryTruncateInto<U, TrapKind> + FromRuntimeValueInternal,
        U: TransmuteInto<V>,
    {
        let v = self.value_stack.pop_as::<T>();

        v.try_truncate_into()
            .map(|v| v.transmute_into())
            .map(|v| self.value_stack.push(v.into()))
            .map(|_| InstructionOutcome::RunNextInstruction)
    }

    fn run_extend<T, U, V>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<V>,
        T: ExtendInto<U> + FromRuntimeValueInternal,
        U: TransmuteInto<V>,
    {
        let v = self.value_stack.pop_as::<T>();

        let v = v.extend_into().transmute_into();
        self.value_stack.push(v.into())?;

        Ok(InstructionOutcome::RunNextInstruction)
    }

    fn run_reinterpret<T, U>(&mut self) -> Result<InstructionOutcome, TrapKind>
    where
        RuntimeValueInternal: From<U>,
        T: FromRuntimeValueInternal,
        T: TransmuteInto<U>,
    {
        let v = self.value_stack.pop_as::<T>();

        let v = v.transmute_into();
        self.value_stack.push(v.into())?;

        Ok(InstructionOutcome::RunNextInstruction)
    }
}

/// Function execution context.
struct FunctionContext {
    /// Is context initialized.
    pub is_initialized: bool,
    /// Internal function reference.
    pub function: FuncRef,
    pub module: ModuleRef,
    pub memory: Option<MemoryRef>,
    /// Current instruction position.
    pub position: u32,
}

impl FunctionContext {
    pub fn new(function: FuncRef) -> Self {
        let module = match function.as_internal() {
			FuncInstanceInternal::Internal { module, .. } => module.upgrade().expect("module deallocated"),
			FuncInstanceInternal::Host { .. } => panic!("Host functions can't be called as internally defined functions; Thus FunctionContext can be created only with internally defined functions; qed"),
		};
        let memory = module.memory_by_index(DEFAULT_MEMORY_INDEX);
        FunctionContext {
            is_initialized: false,
            function: function,
            module: ModuleRef(module),
            memory: memory,
            position: 0,
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    pub fn initialize(
        &mut self,
        locals: &[Local],
        value_stack: &mut ValueStack,
    ) -> Result<(), TrapKind> {
        debug_assert!(!self.is_initialized);

        let num_locals = locals.iter().map(|l| l.count() as usize).sum();

        value_stack.extend(num_locals)?;

        self.is_initialized = true;
        Ok(())
    }

    pub fn module(&self) -> ModuleRef {
        self.module.clone()
    }

    pub fn memory(&self) -> Option<&MemoryRef> {
        self.memory.as_ref()
    }
}

impl fmt::Debug for FunctionContext {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FunctionContext")
    }
}

fn effective_address(address: u32, offset: u32) -> Result<u32, TrapKind> {
    match offset.checked_add(address) {
        None => Err(TrapKind::MemoryAccessOutOfBounds),
        Some(address) => Ok(address),
    }
}

fn prepare_function_args(
    signature: &Signature,
    caller_stack: &mut ValueStack,
) -> Vec<RuntimeValue> {
    let mut out = signature
        .params()
        .iter()
        .rev()
        .map(|&param_ty| caller_stack.pop().with_type(param_ty))
        .collect::<Vec<RuntimeValue>>();
    out.reverse();
    out
}

pub fn check_function_args(signature: &Signature, args: &[RuntimeValue]) -> Result<(), Trap> {
    if signature.params().len() != args.len() {
        return Err(TrapKind::UnexpectedSignature.into());
    }

    if signature
        .params()
        .iter()
        .zip(args)
        .any(|(expected_type, param_value)| {
            let actual_type = param_value.value_type();
            &actual_type != expected_type
        })
    {
        return Err(TrapKind::UnexpectedSignature.into());
    }

    Ok(())
}

#[derive(Debug)]
struct ValueStack {
    buf: Box<[RuntimeValueInternal]>,
    /// Index of the first free place in the stack.
    sp: usize,
}

impl ValueStack {
    #[inline]
    fn drop_keep(&mut self, drop_keep: isa::DropKeep) {
        if drop_keep.keep == isa::Keep::Single {
            let top = *self.top();
            *self.pick_mut(drop_keep.drop as usize + 1) = top;
        }

        let cur_stack_len = self.len();
        self.sp = cur_stack_len - drop_keep.drop as usize;
    }

    #[inline]
    fn pop_as<T>(&mut self) -> T
    where
        T: FromRuntimeValueInternal,
    {
        let value = self.pop();

        T::from_runtime_value_internal(value)
    }

    #[inline]
    fn pop_pair_as<T>(&mut self) -> (T, T)
    where
        T: FromRuntimeValueInternal,
    {
        let right = self.pop_as();
        let left = self.pop_as();
        (left, right)
    }

    #[inline]
    fn pop_triple(
        &mut self,
    ) -> (
        RuntimeValueInternal,
        RuntimeValueInternal,
        RuntimeValueInternal,
    ) {
        let right = self.pop();
        let mid = self.pop();
        let left = self.pop();
        (left, mid, right)
    }

    #[inline]
    fn top(&self) -> &RuntimeValueInternal {
        self.pick(1)
    }

    fn pick(&self, depth: usize) -> &RuntimeValueInternal {
        &self.buf[self.sp - depth]
    }

    #[inline]
    fn pick_mut(&mut self, depth: usize) -> &mut RuntimeValueInternal {
        &mut self.buf[self.sp - depth]
    }

    #[inline]
    fn pop(&mut self) -> RuntimeValueInternal {
        self.sp -= 1;
        self.buf[self.sp]
    }

    #[inline]
    fn push(&mut self, value: RuntimeValueInternal) -> Result<(), TrapKind> {
        let cell = self
            .buf
            .get_mut(self.sp)
            .ok_or_else(|| TrapKind::StackOverflow)?;
        *cell = value;
        self.sp += 1;
        Ok(())
    }

    fn extend(&mut self, len: usize) -> Result<(), TrapKind> {
        let cells = self
            .buf
            .get_mut(self.sp..self.sp + len)
            .ok_or_else(|| TrapKind::StackOverflow)?;
        for cell in cells {
            *cell = Default::default();
        }
        self.sp += len;
        Ok(())
    }

    #[inline]
    fn len(&self) -> usize {
        self.sp
    }
}

struct CallStack {
    buf: Vec<FunctionContext>,
    limit: usize,
}

impl CallStack {
    fn push(&mut self, ctx: FunctionContext) {
        self.buf.push(ctx);
    }

    fn pop(&mut self) -> Option<FunctionContext> {
        self.buf.pop()
    }

    fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    fn is_full(&self) -> bool {
        self.buf.len() + 1 >= self.limit
    }
}

/// Used to recycle stacks instead of allocating them repeatedly.
pub struct StackRecycler {
    value_stack_buf: Option<Box<[RuntimeValueInternal]>>,
    value_stack_limit: usize,
    call_stack_buf: Option<Vec<FunctionContext>>,
    call_stack_limit: usize,
}

impl StackRecycler {
    /// Limit stacks created by this recycler to
    /// - `value_stack_limit` bytes for values and
    /// - `call_stack_limit` levels for calls.
    pub fn with_limits(value_stack_limit: usize, call_stack_limit: usize) -> Self {
        Self {
            value_stack_buf: None,
            value_stack_limit,
            call_stack_buf: None,
            call_stack_limit,
        }
    }

    /// Clears any values left on the stack to avoid
    /// leaking them to future export invocations.
    ///
    /// This is a secondary defense to prevent modules from
    /// exploiting faulty stack handling in the interpreter.
    ///
    /// Do note that there are additional channels that
    /// can leak information into an untrusted module.
    pub fn clear(&mut self) {
        if let Some(buf) = &mut self.value_stack_buf {
            for cell in buf.iter_mut() {
                *cell = RuntimeValueInternal(0);
            }
        }
    }

    fn recreate_value_stack(this: &mut Option<&mut Self>) -> ValueStack {
        let limit = this
            .as_ref()
            .map_or(DEFAULT_VALUE_STACK_LIMIT, |this| this.value_stack_limit)
            / ::core::mem::size_of::<RuntimeValueInternal>();

        let buf = this
            .as_mut()
            .and_then(|this| this.value_stack_buf.take())
            .unwrap_or_else(|| {
                let mut buf = Vec::new();
                buf.reserve_exact(limit);
                buf.resize(limit, RuntimeValueInternal(0));
                buf.into_boxed_slice()
            });

        ValueStack { buf, sp: 0 }
    }

    fn recreate_call_stack(this: &mut Option<&mut Self>) -> CallStack {
        let limit = this
            .as_ref()
            .map_or(DEFAULT_CALL_STACK_LIMIT, |this| this.call_stack_limit);

        let buf = this
            .as_mut()
            .and_then(|this| this.call_stack_buf.take())
            .unwrap_or_default();

        CallStack { buf, limit }
    }

    pub(crate) fn recycle(&mut self, mut interpreter: Interpreter) {
        interpreter.call_stack.buf.clear();

        self.value_stack_buf = Some(interpreter.value_stack.buf);
        self.call_stack_buf = Some(interpreter.call_stack.buf);
    }
}

impl Default for StackRecycler {
    fn default() -> Self {
        Self::with_limits(DEFAULT_VALUE_STACK_LIMIT, DEFAULT_CALL_STACK_LIMIT)
    }
}
