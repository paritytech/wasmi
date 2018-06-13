use std::mem;
use std::ops;
use std::{u32, usize};
use std::fmt;
use std::iter::repeat;
use std::collections::VecDeque;
use parity_wasm::elements::{BlockType, Local};
use {Error, Trap, TrapKind, Signature};
use module::ModuleRef;
use func::{FuncRef, FuncInstance, FuncInstanceInternal};
use value::{
	RuntimeValue, FromRuntimeValue, WrapInto, TryTruncateInto, ExtendInto,
	ArithmeticOps, Integer, Float, LittleEndianConvert, TransmuteInto,
};
use host::Externals;
use common::{DEFAULT_MEMORY_INDEX, DEFAULT_TABLE_INDEX};
use common::stack::StackWithLimit;
use memory_units::Pages;
use nan_preserving_float::{F32, F64};
use isa;

/// Maximum number of entries in value stack.
pub const DEFAULT_VALUE_STACK_LIMIT: usize = 16384;

/// Function interpreter.
pub struct Interpreter<'a, E: Externals + 'a> {
	externals: &'a mut E,
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
	Return,
}

/// Function run result.
enum RunResult {
	/// Function has returned (optional) value.
	Return(Option<RuntimeValue>),
	/// Function is calling other function.
	NestedCall(FuncRef),
}

impl<'a, E: Externals> Interpreter<'a, E> {
	pub fn new(externals: &'a mut E) -> Interpreter<'a, E> {
		Interpreter {
			externals,
		}
	}

	pub fn start_execution(&mut self, func: &FuncRef, args: &[RuntimeValue]) -> Result<Option<RuntimeValue>, Trap> {
		let context = FunctionContext::new(
			func.clone(),
			DEFAULT_VALUE_STACK_LIMIT,
			func.signature(),
			args.into_iter().cloned().collect(),
		);

		let mut function_stack = VecDeque::new();
		function_stack.push_back(context);

		self.run_interpreter_loop(&mut function_stack)
	}

	fn run_interpreter_loop(&mut self, function_stack: &mut VecDeque<FunctionContext>) -> Result<Option<RuntimeValue>, Trap> {
		loop {
			let mut function_context = function_stack.pop_back().expect("on loop entry - not empty; on loop continue - checking for emptiness; qed");
			let function_ref = function_context.function.clone();
			let function_body = function_ref
				.body()
				.expect(
					"Host functions checked in function_return below; Internal functions always have a body; qed"
				);
			if !function_context.is_initialized() {
				let return_type = function_context.return_type;
				function_context.initialize(&function_body.locals);
			}

			let function_return = self.do_run_function(
				&mut function_context,
				&function_body.code.code,
			).map_err(Trap::new)?;

			match function_return {
				RunResult::Return(return_value) => {
					match function_stack.back_mut() {
						Some(caller_context) => if let Some(return_value) = return_value {
							caller_context.value_stack_mut().push(return_value).map_err(Trap::new)?;
						},
						None => return Ok(return_value),
					}
				},
				RunResult::NestedCall(nested_func) => {
					match *nested_func.as_internal() {
						FuncInstanceInternal::Internal { .. } => {
							let nested_context = function_context.nested(nested_func.clone()).map_err(Trap::new)?;
							function_stack.push_back(function_context);
							function_stack.push_back(nested_context);
						},
						FuncInstanceInternal::Host { ref signature, .. } => {
							let args = prepare_function_args(signature, &mut function_context.value_stack);
							let return_val = FuncInstance::invoke(&nested_func, &args, self.externals)?;

							// Check if `return_val` matches the signature.
							let value_ty = return_val.clone().map(|val| val.value_type());
							let expected_ty = nested_func.signature().return_type();
							if value_ty != expected_ty {
								return Err(TrapKind::UnexpectedSignature.into());
							}

							if let Some(return_val) = return_val {
								function_context.value_stack_mut().push(return_val).map_err(Trap::new)?;
							}
							function_stack.push_back(function_context);
						}
					}
				},
			}
		}
	}

	fn do_run_function(&mut self, function_context: &mut FunctionContext, instructions: &[isa::Instruction]) -> Result<RunResult, TrapKind> {
		loop {
			let instruction = &instructions[function_context.position];

			match self.run_instruction(function_context, instruction)? {
				InstructionOutcome::RunNextInstruction => function_context.position += 1,
				InstructionOutcome::Branch(target) => {
					function_context.position = target.dst_pc as usize;

					assert!(target.keep <= 1);
					let keep = if target.keep == 1 {
						Some(function_context.value_stack_mut().pop())
					} else {
						None
					};

					let cur_stack_len = function_context.value_stack.len();
					function_context.value_stack_mut().resize(cur_stack_len - target.drop as usize);
					if let Some(keep) = keep {
						function_context.value_stack_mut().push(keep)?;
					}
				},
				InstructionOutcome::ExecuteCall(func_ref) => {
					function_context.position += 1;
					return Ok(RunResult::NestedCall(func_ref));
				},
				InstructionOutcome::Return => break,
			}
		}

		Ok(RunResult::Return(match function_context.return_type {
			BlockType::Value(_) => {
				let result = function_context
					.value_stack_mut()
					.pop();
				Some(result)
			},
			BlockType::NoResult => None,
		}))
	}

	fn run_instruction(&mut self, context: &mut FunctionContext, instruction: &isa::Instruction) -> Result<InstructionOutcome, TrapKind> {
		match instruction {
			&isa::Instruction::Unreachable => self.run_unreachable(context),

			&isa::Instruction::Br(ref target) => self.run_br(context, target.clone()),
			&isa::Instruction::BrIfEqz(ref target) => self.run_br_eqz(context, target.clone()),
			&isa::Instruction::BrIfNez(ref target) => self.run_br_nez(context, target.clone()),
			&isa::Instruction::BrTable(ref targets) => self.run_br_table(context, targets),
			&isa::Instruction::Return { drop, keep } => self.run_return(context),

			&isa::Instruction::Call(index) => self.run_call(context, index),
			&isa::Instruction::CallIndirect(index) => self.run_call_indirect(context, index),

			&isa::Instruction::Drop => self.run_drop(context),
			&isa::Instruction::Select => self.run_select(context),

			&isa::Instruction::GetLocal(depth) => self.run_get_local(context, depth),
			&isa::Instruction::SetLocal(depth) => self.run_set_local(context, depth),
			&isa::Instruction::TeeLocal(depth) => self.run_tee_local(context, depth),
			&isa::Instruction::GetGlobal(index) => self.run_get_global(context, index),
			&isa::Instruction::SetGlobal(index) => self.run_set_global(context, index),

			&isa::Instruction::I32Load(offset) => self.run_load::<i32>(context, offset),
			&isa::Instruction::I64Load(offset) => self.run_load::<i64>(context, offset),
			&isa::Instruction::F32Load(offset) => self.run_load::<F32>(context, offset),
			&isa::Instruction::F64Load(offset) => self.run_load::<F64>(context, offset),
			&isa::Instruction::I32Load8S(offset) => self.run_load_extend::<i8, i32>(context, offset),
			&isa::Instruction::I32Load8U(offset) => self.run_load_extend::<u8, i32>(context, offset),
			&isa::Instruction::I32Load16S(offset) => self.run_load_extend::<i16, i32>(context, offset),
			&isa::Instruction::I32Load16U(offset) => self.run_load_extend::<u16, i32>(context, offset),
			&isa::Instruction::I64Load8S(offset) => self.run_load_extend::<i8, i64>(context, offset),
			&isa::Instruction::I64Load8U(offset) => self.run_load_extend::<u8, i64>(context, offset),
			&isa::Instruction::I64Load16S(offset) => self.run_load_extend::<i16, i64>(context, offset),
			&isa::Instruction::I64Load16U(offset) => self.run_load_extend::<u16, i64>(context, offset),
			&isa::Instruction::I64Load32S(offset) => self.run_load_extend::<i32, i64>(context, offset),
			&isa::Instruction::I64Load32U(offset) => self.run_load_extend::<u32, i64>(context, offset),

			&isa::Instruction::I32Store(offset) => self.run_store::<i32>(context, offset),
			&isa::Instruction::I64Store(offset) => self.run_store::<i64>(context, offset),
			&isa::Instruction::F32Store(offset) => self.run_store::<F32>(context, offset),
			&isa::Instruction::F64Store(offset) => self.run_store::<F64>(context, offset),
			&isa::Instruction::I32Store8(offset) => self.run_store_wrap::<i32, i8>(context, offset),
			&isa::Instruction::I32Store16(offset) => self.run_store_wrap::<i32, i16>(context, offset),
			&isa::Instruction::I64Store8(offset) => self.run_store_wrap::<i64, i8>(context, offset),
			&isa::Instruction::I64Store16(offset) => self.run_store_wrap::<i64, i16>(context, offset),
			&isa::Instruction::I64Store32(offset) => self.run_store_wrap::<i64, i32>(context, offset),

			&isa::Instruction::CurrentMemory => self.run_current_memory(context),
			&isa::Instruction::GrowMemory => self.run_grow_memory(context),

			&isa::Instruction::I32Const(val) => self.run_const(context, val.into()),
			&isa::Instruction::I64Const(val) => self.run_const(context, val.into()),
			&isa::Instruction::F32Const(val) => self.run_const(context, RuntimeValue::decode_f32(val)),
			&isa::Instruction::F64Const(val) => self.run_const(context, RuntimeValue::decode_f64(val)),

			&isa::Instruction::I32Eqz => self.run_eqz::<i32>(context),
			&isa::Instruction::I32Eq => self.run_eq::<i32>(context),
			&isa::Instruction::I32Ne => self.run_ne::<i32>(context),
			&isa::Instruction::I32LtS => self.run_lt::<i32>(context),
			&isa::Instruction::I32LtU => self.run_lt::<u32>(context),
			&isa::Instruction::I32GtS => self.run_gt::<i32>(context),
			&isa::Instruction::I32GtU => self.run_gt::<u32>(context),
			&isa::Instruction::I32LeS => self.run_lte::<i32>(context),
			&isa::Instruction::I32LeU => self.run_lte::<u32>(context),
			&isa::Instruction::I32GeS => self.run_gte::<i32>(context),
			&isa::Instruction::I32GeU => self.run_gte::<u32>(context),

			&isa::Instruction::I64Eqz => self.run_eqz::<i64>(context),
			&isa::Instruction::I64Eq => self.run_eq::<i64>(context),
			&isa::Instruction::I64Ne => self.run_ne::<i64>(context),
			&isa::Instruction::I64LtS => self.run_lt::<i64>(context),
			&isa::Instruction::I64LtU => self.run_lt::<u64>(context),
			&isa::Instruction::I64GtS => self.run_gt::<i64>(context),
			&isa::Instruction::I64GtU => self.run_gt::<u64>(context),
			&isa::Instruction::I64LeS => self.run_lte::<i64>(context),
			&isa::Instruction::I64LeU => self.run_lte::<u64>(context),
			&isa::Instruction::I64GeS => self.run_gte::<i64>(context),
			&isa::Instruction::I64GeU => self.run_gte::<u64>(context),

			&isa::Instruction::F32Eq => self.run_eq::<F32>(context),
			&isa::Instruction::F32Ne => self.run_ne::<F32>(context),
			&isa::Instruction::F32Lt => self.run_lt::<F32>(context),
			&isa::Instruction::F32Gt => self.run_gt::<F32>(context),
			&isa::Instruction::F32Le => self.run_lte::<F32>(context),
			&isa::Instruction::F32Ge => self.run_gte::<F32>(context),

			&isa::Instruction::F64Eq => self.run_eq::<F64>(context),
			&isa::Instruction::F64Ne => self.run_ne::<F64>(context),
			&isa::Instruction::F64Lt => self.run_lt::<F64>(context),
			&isa::Instruction::F64Gt => self.run_gt::<F64>(context),
			&isa::Instruction::F64Le => self.run_lte::<F64>(context),
			&isa::Instruction::F64Ge => self.run_gte::<F64>(context),

			&isa::Instruction::I32Clz => self.run_clz::<i32>(context),
			&isa::Instruction::I32Ctz => self.run_ctz::<i32>(context),
			&isa::Instruction::I32Popcnt => self.run_popcnt::<i32>(context),
			&isa::Instruction::I32Add => self.run_add::<i32>(context),
			&isa::Instruction::I32Sub => self.run_sub::<i32>(context),
			&isa::Instruction::I32Mul => self.run_mul::<i32>(context),
			&isa::Instruction::I32DivS => self.run_div::<i32, i32>(context),
			&isa::Instruction::I32DivU => self.run_div::<i32, u32>(context),
			&isa::Instruction::I32RemS => self.run_rem::<i32, i32>(context),
			&isa::Instruction::I32RemU => self.run_rem::<i32, u32>(context),
			&isa::Instruction::I32And => self.run_and::<i32>(context),
			&isa::Instruction::I32Or => self.run_or::<i32>(context),
			&isa::Instruction::I32Xor => self.run_xor::<i32>(context),
			&isa::Instruction::I32Shl => self.run_shl::<i32>(context, 0x1F),
			&isa::Instruction::I32ShrS => self.run_shr::<i32, i32>(context, 0x1F),
			&isa::Instruction::I32ShrU => self.run_shr::<i32, u32>(context, 0x1F),
			&isa::Instruction::I32Rotl => self.run_rotl::<i32>(context),
			&isa::Instruction::I32Rotr => self.run_rotr::<i32>(context),

			&isa::Instruction::I64Clz => self.run_clz::<i64>(context),
			&isa::Instruction::I64Ctz => self.run_ctz::<i64>(context),
			&isa::Instruction::I64Popcnt => self.run_popcnt::<i64>(context),
			&isa::Instruction::I64Add => self.run_add::<i64>(context),
			&isa::Instruction::I64Sub => self.run_sub::<i64>(context),
			&isa::Instruction::I64Mul => self.run_mul::<i64>(context),
			&isa::Instruction::I64DivS => self.run_div::<i64, i64>(context),
			&isa::Instruction::I64DivU => self.run_div::<i64, u64>(context),
			&isa::Instruction::I64RemS => self.run_rem::<i64, i64>(context),
			&isa::Instruction::I64RemU => self.run_rem::<i64, u64>(context),
			&isa::Instruction::I64And => self.run_and::<i64>(context),
			&isa::Instruction::I64Or => self.run_or::<i64>(context),
			&isa::Instruction::I64Xor => self.run_xor::<i64>(context),
			&isa::Instruction::I64Shl => self.run_shl::<i64>(context, 0x3F),
			&isa::Instruction::I64ShrS => self.run_shr::<i64, i64>(context, 0x3F),
			&isa::Instruction::I64ShrU => self.run_shr::<i64, u64>(context, 0x3F),
			&isa::Instruction::I64Rotl => self.run_rotl::<i64>(context),
			&isa::Instruction::I64Rotr => self.run_rotr::<i64>(context),

			&isa::Instruction::F32Abs => self.run_abs::<F32>(context),
			&isa::Instruction::F32Neg => self.run_neg::<F32>(context),
			&isa::Instruction::F32Ceil => self.run_ceil::<F32>(context),
			&isa::Instruction::F32Floor => self.run_floor::<F32>(context),
			&isa::Instruction::F32Trunc => self.run_trunc::<F32>(context),
			&isa::Instruction::F32Nearest => self.run_nearest::<F32>(context),
			&isa::Instruction::F32Sqrt => self.run_sqrt::<F32>(context),
			&isa::Instruction::F32Add => self.run_add::<F32>(context),
			&isa::Instruction::F32Sub => self.run_sub::<F32>(context),
			&isa::Instruction::F32Mul => self.run_mul::<F32>(context),
			&isa::Instruction::F32Div => self.run_div::<F32, F32>(context),
			&isa::Instruction::F32Min => self.run_min::<F32>(context),
			&isa::Instruction::F32Max => self.run_max::<F32>(context),
			&isa::Instruction::F32Copysign => self.run_copysign::<F32>(context),

			&isa::Instruction::F64Abs => self.run_abs::<F64>(context),
			&isa::Instruction::F64Neg => self.run_neg::<F64>(context),
			&isa::Instruction::F64Ceil => self.run_ceil::<F64>(context),
			&isa::Instruction::F64Floor => self.run_floor::<F64>(context),
			&isa::Instruction::F64Trunc => self.run_trunc::<F64>(context),
			&isa::Instruction::F64Nearest => self.run_nearest::<F64>(context),
			&isa::Instruction::F64Sqrt => self.run_sqrt::<F64>(context),
			&isa::Instruction::F64Add => self.run_add::<F64>(context),
			&isa::Instruction::F64Sub => self.run_sub::<F64>(context),
			&isa::Instruction::F64Mul => self.run_mul::<F64>(context),
			&isa::Instruction::F64Div => self.run_div::<F64, F64>(context),
			&isa::Instruction::F64Min => self.run_min::<F64>(context),
			&isa::Instruction::F64Max => self.run_max::<F64>(context),
			&isa::Instruction::F64Copysign => self.run_copysign::<F64>(context),

			&isa::Instruction::I32WrapI64 => self.run_wrap::<i64, i32>(context),
			&isa::Instruction::I32TruncSF32 => self.run_trunc_to_int::<F32, i32, i32>(context),
			&isa::Instruction::I32TruncUF32 => self.run_trunc_to_int::<F32, u32, i32>(context),
			&isa::Instruction::I32TruncSF64 => self.run_trunc_to_int::<F64, i32, i32>(context),
			&isa::Instruction::I32TruncUF64 => self.run_trunc_to_int::<F64, u32, i32>(context),
			&isa::Instruction::I64ExtendSI32 => self.run_extend::<i32, i64, i64>(context),
			&isa::Instruction::I64ExtendUI32 => self.run_extend::<u32, u64, i64>(context),
			&isa::Instruction::I64TruncSF32 => self.run_trunc_to_int::<F32, i64, i64>(context),
			&isa::Instruction::I64TruncUF32 => self.run_trunc_to_int::<F32, u64, i64>(context),
			&isa::Instruction::I64TruncSF64 => self.run_trunc_to_int::<F64, i64, i64>(context),
			&isa::Instruction::I64TruncUF64 => self.run_trunc_to_int::<F64, u64, i64>(context),
			&isa::Instruction::F32ConvertSI32 => self.run_extend::<i32, F32, F32>(context),
			&isa::Instruction::F32ConvertUI32 => self.run_extend::<u32, F32, F32>(context),
			&isa::Instruction::F32ConvertSI64 => self.run_wrap::<i64, F32>(context),
			&isa::Instruction::F32ConvertUI64 => self.run_wrap::<u64, F32>(context),
			&isa::Instruction::F32DemoteF64 => self.run_wrap::<F64, F32>(context),
			&isa::Instruction::F64ConvertSI32 => self.run_extend::<i32, F64, F64>(context),
			&isa::Instruction::F64ConvertUI32 => self.run_extend::<u32, F64, F64>(context),
			&isa::Instruction::F64ConvertSI64 => self.run_extend::<i64, F64, F64>(context),
			&isa::Instruction::F64ConvertUI64 => self.run_extend::<u64, F64, F64>(context),
			&isa::Instruction::F64PromoteF32 => self.run_extend::<F32, F64, F64>(context),

			&isa::Instruction::I32ReinterpretF32 => self.run_reinterpret::<F32, i32>(context),
			&isa::Instruction::I64ReinterpretF64 => self.run_reinterpret::<F64, i64>(context),
			&isa::Instruction::F32ReinterpretI32 => self.run_reinterpret::<i32, F32>(context),
			&isa::Instruction::F64ReinterpretI64 => self.run_reinterpret::<i64, F64>(context),
		}
	}

	fn run_unreachable(&mut self, _context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind> {
		Err(TrapKind::Unreachable)
	}

	fn run_br(&mut self, _context: &mut FunctionContext, target: isa::Target) -> Result<InstructionOutcome, TrapKind> {
		Ok(InstructionOutcome::Branch(target))
	}

	fn run_br_nez(&mut self, context: &mut FunctionContext, target: isa::Target) -> Result<InstructionOutcome, TrapKind> {
		let condition = context.value_stack_mut().pop_as();
		if condition {
			Ok(InstructionOutcome::Branch(target))
		} else {
			Ok(InstructionOutcome::RunNextInstruction)
		}
	}

	fn run_br_eqz(&mut self, context: &mut FunctionContext, target: isa::Target) -> Result<InstructionOutcome, TrapKind> {
		let condition = context.value_stack_mut().pop_as();
		if condition {
			Ok(InstructionOutcome::RunNextInstruction)
		} else {

			Ok(InstructionOutcome::Branch(target))
		}
	}

	fn run_br_table(&mut self, context: &mut FunctionContext, table: &[isa::Target]) -> Result<InstructionOutcome, TrapKind> {
		let index: u32 = context.value_stack_mut()
			.pop_as();

		let dst =
		if (index as usize) < table.len() - 1 {
			table[index as usize].clone()
		} else {
			let len = table.len();
			table[len - 1].clone()
		};
		Ok(InstructionOutcome::Branch(dst))
	}

	fn run_return(&mut self, _context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind> {
		Ok(InstructionOutcome::Return)
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
		let table_func_idx: u32 = context
			.value_stack_mut()
			.pop_as();
		let table = context
			.module()
			.table_by_index(DEFAULT_TABLE_INDEX)
			.expect("Due to validation table should exists");
		let func_ref = table.get(table_func_idx)
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

	fn run_drop(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind> {
		let _ = context
			.value_stack_mut()
			.pop();
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_select(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind> {
		let (left, mid, right) = context
			.value_stack_mut()
			.pop_triple();

		let condition = right
			.try_into()
			.expect("Due to validation stack top should be I32");
		let val = if condition { left } else { mid };
		context.value_stack_mut().push(val)?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_get_local(&mut self, context: &mut FunctionContext, index: u32) -> Result<InstructionOutcome, TrapKind> {
		let val = context.get_local(index as usize);
		context.value_stack_mut().push(val)?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_set_local(&mut self, context: &mut FunctionContext, index: u32) -> Result<InstructionOutcome, TrapKind> {
		let arg = context
			.value_stack_mut()
			.pop();
		context.set_local(index as usize, arg);
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_tee_local(&mut self, context: &mut FunctionContext, index: u32) -> Result<InstructionOutcome, TrapKind> {
		let arg = context
			.value_stack()
			.top()
			.clone();
		context.set_local(index as usize, arg);
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
		context.value_stack_mut().push(val)?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_set_global(
		&mut self,
		context: &mut FunctionContext,
		index: u32,
	) -> Result<InstructionOutcome, TrapKind> {
		let val = context
			.value_stack_mut()
			.pop();
		let global = context
			.module()
			.global_by_index(index)
			.expect("Due to validation global should exists");
		global.set(val).expect("Due to validation set to a global should succeed");
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_load<T>(&mut self, context: &mut FunctionContext, offset: u32) -> Result<InstructionOutcome, TrapKind>
		where RuntimeValue: From<T>, T: LittleEndianConvert {
		let raw_address = context
			.value_stack_mut()
			.pop_as();
		let address =
			effective_address(
				offset,
				raw_address,
			)?;
		let m = context.module()
			.memory_by_index(DEFAULT_MEMORY_INDEX)
			.expect("Due to validation memory should exists");
		let b = m.get(address, mem::size_of::<T>())
			.map_err(|_| TrapKind::MemoryAccessOutOfBounds)?;
		let n = T::from_little_endian(&b)
			.expect("Can't fail since buffer length should be size_of::<T>");
		context.value_stack_mut().push(n.into())?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_load_extend<T, U>(&mut self, context: &mut FunctionContext, offset: u32) -> Result<InstructionOutcome, TrapKind>
		where T: ExtendInto<U>, RuntimeValue: From<U>, T: LittleEndianConvert {
		let raw_address = context
			.value_stack_mut()
			.pop_as();
		let address =
			effective_address(
				offset,
				raw_address,
			)?;
		let m = context.module()
			.memory_by_index(DEFAULT_MEMORY_INDEX)
			.expect("Due to validation memory should exists");
		let b = m.get(address, mem::size_of::<T>())
			.map_err(|_| TrapKind::MemoryAccessOutOfBounds)?;
		let v = T::from_little_endian(&b)
			.expect("Can't fail since buffer length should be size_of::<T>");
		let stack_value: U = v.extend_into();
		context
			.value_stack_mut()
			.push(stack_value.into())
			.map_err(Into::into)
			.map(|_| InstructionOutcome::RunNextInstruction)
	}

	fn run_store<T>(&mut self, context: &mut FunctionContext, offset: u32) -> Result<InstructionOutcome, TrapKind>
		where T: FromRuntimeValue, T: LittleEndianConvert {
		let stack_value = context
			.value_stack_mut()
			.pop_as::<T>()
			.into_little_endian();
		let raw_address = context
			.value_stack_mut()
			.pop_as::<u32>();
		let address =
			effective_address(
				offset,
				raw_address,
			)?;

		let m = context.module()
			.memory_by_index(DEFAULT_MEMORY_INDEX)
			.expect("Due to validation memory should exists");
		m.set(address, &stack_value)
			.map_err(|_| TrapKind::MemoryAccessOutOfBounds)?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_store_wrap<T, U>(
		&mut self,
		context: &mut FunctionContext,
		offset: u32,
	) -> Result<InstructionOutcome, TrapKind>
	where
		T: FromRuntimeValue,
		T: WrapInto<U>,
		U: LittleEndianConvert,
	{
		let stack_value: T = context
			.value_stack_mut()
			.pop()
			.try_into()
			.expect("Due to validation value should be of proper type");
		let stack_value = stack_value.wrap_into().into_little_endian();
		let raw_address = context
			.value_stack_mut()
			.pop_as::<u32>();
		let address =
			effective_address(
				offset,
				raw_address,
			)?;
		let m = context.module()
			.memory_by_index(DEFAULT_MEMORY_INDEX)
			.expect("Due to validation memory should exists");
		m.set(address, &stack_value)
			.map_err(|_| TrapKind::MemoryAccessOutOfBounds)?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_current_memory(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind> {
		let m = context.module()
			.memory_by_index(DEFAULT_MEMORY_INDEX)
			.expect("Due to validation memory should exists");
		let s = m.current_size().0;
		context
			.value_stack_mut()
			.push(RuntimeValue::I32(s as i32))?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_grow_memory(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind> {
		let pages: u32 = context
			.value_stack_mut()
			.pop_as();
		let m = context.module()
			.memory_by_index(DEFAULT_MEMORY_INDEX)
			.expect("Due to validation memory should exists");
		let m = match m.grow(Pages(pages as usize)) {
			Ok(Pages(new_size)) => new_size as u32,
			Err(_) => u32::MAX, // Returns -1 (or 0xFFFFFFFF) in case of error.
		};
		context
			.value_stack_mut()
			.push(RuntimeValue::I32(m as i32))?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_const(&mut self, context: &mut FunctionContext, val: RuntimeValue) -> Result<InstructionOutcome, TrapKind> {
		context
			.value_stack_mut()
			.push(val)
			.map_err(Into::into)
			.map(|_| InstructionOutcome::RunNextInstruction)
	}

	fn run_relop<T, F>(&mut self, context: &mut FunctionContext, f: F) -> Result<InstructionOutcome, TrapKind>
	where
		T: FromRuntimeValue,
		F: FnOnce(T, T) -> bool,
	{
		let (left, right) = context
			.value_stack_mut()
			.pop_pair_as::<T>()
			.expect("Due to validation stack should contain pair of values");
		let v = if f(left, right) {
			RuntimeValue::I32(1)
		} else {
			RuntimeValue::I32(0)
		};
		context.value_stack_mut().push(v)?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_eqz<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
		where T: FromRuntimeValue, T: PartialEq<T> + Default {
		let v = context
			.value_stack_mut()
			.pop_as::<T>();
		let v = RuntimeValue::I32(if v == Default::default() { 1 } else { 0 });
		context.value_stack_mut().push(v)?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_eq<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where T: FromRuntimeValue + PartialEq<T>
	{
		self.run_relop(context, |left: T, right: T| left == right)
	}

	fn run_ne<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
		where T: FromRuntimeValue + PartialEq<T> {
		self.run_relop(context, |left: T, right: T| left != right)
	}

	fn run_lt<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
		where T: FromRuntimeValue + PartialOrd<T> {
		self.run_relop(context, |left: T, right: T| left < right)
	}

	fn run_gt<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
		where T: FromRuntimeValue + PartialOrd<T> {
		self.run_relop(context, |left: T, right: T| left > right)
	}

	fn run_lte<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
		where T: FromRuntimeValue + PartialOrd<T> {
		self.run_relop(context, |left: T, right: T| left <= right)
	}

	fn run_gte<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
		where T: FromRuntimeValue + PartialOrd<T> {
		self.run_relop(context, |left: T, right: T| left >= right)
	}

	fn run_unop<T, U, F>(&mut self, context: &mut FunctionContext, f: F) -> Result<InstructionOutcome, TrapKind>
	where
		F: FnOnce(T) -> U,
		T: FromRuntimeValue,
		RuntimeValue: From<U>
	{
		let v = context
			.value_stack_mut()
			.pop_as::<T>();
		let v = f(v);
		context.value_stack_mut().push(v.into())?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_clz<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<T>, T: Integer<T> + FromRuntimeValue {
		self.run_unop(context, |v: T| v.leading_zeros())
	}

	fn run_ctz<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<T>, T: Integer<T> + FromRuntimeValue {
		self.run_unop(context, |v: T| v.trailing_zeros())
	}

	fn run_popcnt<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<T>, T: Integer<T> + FromRuntimeValue {
		self.run_unop(context, |v: T| v.count_ones())
	}

	fn run_add<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
		where RuntimeValue: From<T>, T: ArithmeticOps<T> + FromRuntimeValue {
		let (left, right) = context
			.value_stack_mut()
			.pop_pair_as::<T>()
			.expect("Due to validation stack should contain pair of values");
		let v = left.add(right);
		context.value_stack_mut().push(v.into())?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_sub<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
		where RuntimeValue: From<T>, T: ArithmeticOps<T> + FromRuntimeValue {
		let (left, right) = context
			.value_stack_mut()
			.pop_pair_as::<T>()
			.expect("Due to validation stack should contain pair of values");
		let v = left.sub(right);
		context.value_stack_mut().push(v.into())?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_mul<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<T>, T: ArithmeticOps<T> + FromRuntimeValue {
		let (left, right) = context
			.value_stack_mut()
			.pop_pair_as::<T>()
			.expect("Due to validation stack should contain pair of values");
		let v = left.mul(right);
		context.value_stack_mut().push(v.into())?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_div<T, U>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<T>, T: TransmuteInto<U> + FromRuntimeValue, U: ArithmeticOps<U> + TransmuteInto<T> {
		let (left, right) = context
			.value_stack_mut()
			.pop_pair_as::<T>()
			.expect("Due to validation stack should contain pair of values");
		let (left, right) = (left.transmute_into(), right.transmute_into());
		let v = left.div(right)?;
		let v = v.transmute_into();
		context.value_stack_mut().push(v.into())?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_rem<T, U>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<T>, T: TransmuteInto<U> + FromRuntimeValue, U: Integer<U> + TransmuteInto<T> {
		let (left, right) = context
			.value_stack_mut()
			.pop_pair_as::<T>()
			.expect("Due to validation stack should contain pair of values");
		let (left, right) = (left.transmute_into(), right.transmute_into());
		let v = left.rem(right)?;
		let v = v.transmute_into();
		context.value_stack_mut().push(v.into())?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_and<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<<T as ops::BitAnd>::Output>, T: ops::BitAnd<T> + FromRuntimeValue {
		let (left, right) = context
			.value_stack_mut()
			.pop_pair_as::<T>()
			.expect("Due to validation stack should contain pair of values");
		let v = left.bitand(right);
		context.value_stack_mut().push(v.into())?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_or<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<<T as ops::BitOr>::Output>, T: ops::BitOr<T> + FromRuntimeValue {
		let (left, right) = context
			.value_stack_mut()
			.pop_pair_as::<T>()
			.expect("Due to validation stack should contain pair of values");
		let v = left.bitor(right);
		context.value_stack_mut().push(v.into())?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_xor<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<<T as ops::BitXor>::Output>, T: ops::BitXor<T> + FromRuntimeValue {
		let (left, right) = context
			.value_stack_mut()
			.pop_pair_as::<T>()
			.expect("Due to validation stack should contain pair of values");
		let v = left.bitxor(right);
		context.value_stack_mut().push(v.into())?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_shl<T>(&mut self, context: &mut FunctionContext, mask: T) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<<T as ops::Shl<T>>::Output>, T: ops::Shl<T> + ops::BitAnd<T, Output=T> + FromRuntimeValue {
		let (left, right) = context
			.value_stack_mut()
			.pop_pair_as::<T>()
			.expect("Due to validation stack should contain pair of values");
		let v = left.shl(right & mask);
		context.value_stack_mut().push(v.into())?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_shr<T, U>(&mut self, context: &mut FunctionContext, mask: U) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<T>, T: TransmuteInto<U> + FromRuntimeValue, U: ops::Shr<U> + ops::BitAnd<U, Output=U>, <U as ops::Shr<U>>::Output: TransmuteInto<T> {
		let (left, right) = context
			.value_stack_mut()
			.pop_pair_as::<T>()
			.expect("Due to validation stack should contain pair of values");
		let (left, right) = (left.transmute_into(), right.transmute_into());
		let v = left.shr(right & mask);
		let v = v.transmute_into();
		context.value_stack_mut().push(v.into())?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_rotl<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<T>, T: Integer<T> + FromRuntimeValue {
		let (left, right) = context
			.value_stack_mut()
			.pop_pair_as::<T>()
			.expect("Due to validation stack should contain pair of values");
		let v = left.rotl(right);
		context.value_stack_mut().push(v.into())?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_rotr<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<T>, T: Integer<T> + FromRuntimeValue
	{
		let (left, right) = context
			.value_stack_mut()
			.pop_pair_as::<T>()
			.expect("Due to validation stack should contain pair of values");
		let v = left.rotr(right);
		context.value_stack_mut().push(v.into())?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_abs<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<T>, T: Float<T> + FromRuntimeValue
	{
		self.run_unop(context, |v: T| v.abs())
	}

	fn run_neg<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where
		RuntimeValue: From<<T as ops::Neg>::Output>,
		T: ops::Neg + FromRuntimeValue
	{
		self.run_unop(context, |v: T| v.neg())
	}

	fn run_ceil<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<T>, T: Float<T> + FromRuntimeValue
	{
		self.run_unop(context, |v: T| v.ceil())
	}

	fn run_floor<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<T>, T: Float<T> + FromRuntimeValue
	{
		self.run_unop(context, |v: T| v.floor())
	}

	fn run_trunc<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<T>, T: Float<T> + FromRuntimeValue
	{
		self.run_unop(context, |v: T| v.trunc())
	}

	fn run_nearest<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<T>, T: Float<T> + FromRuntimeValue
	{
		self.run_unop(context, |v: T| v.nearest())
	}

	fn run_sqrt<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<T>, T: Float<T> + FromRuntimeValue
	{
		self.run_unop(context, |v: T| v.sqrt())
	}

	fn run_min<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<T>, T: Float<T> + FromRuntimeValue
	{
		let (left, right) = context
			.value_stack_mut()
			.pop_pair_as::<T>()
			.expect("Due to validation stack should contain pair of values");
		let v = left.min(right);
		context.value_stack_mut().push(v.into())?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_max<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<T>, T: Float<T> + FromRuntimeValue {
		let (left, right) = context
			.value_stack_mut()
			.pop_pair_as::<T>()
			.expect("Due to validation stack should contain pair of values");
		let v = left.max(right);
		context.value_stack_mut().push(v.into())?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_copysign<T>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<T>, T: Float<T> + FromRuntimeValue {
		let (left, right) = context
			.value_stack_mut()
			.pop_pair_as::<T>()
			.expect("Due to validation stack should contain pair of values");
		let v = left.copysign(right);
		context.value_stack_mut().push(v.into())?;
		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_wrap<T, U>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where RuntimeValue: From<U>, T: WrapInto<U> + FromRuntimeValue {
		self.run_unop(context, |v: T| v.wrap_into())
	}

	fn run_trunc_to_int<T, U, V>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
		where RuntimeValue: From<V>, T: TryTruncateInto<U, TrapKind> + FromRuntimeValue, U: TransmuteInto<V>,  {
		let v = context
			.value_stack_mut()
			.pop_as::<T>();

		v.try_truncate_into()
			.map(|v| v.transmute_into())
			.map(|v| context.value_stack_mut().push(v.into()))
			.map(|_| InstructionOutcome::RunNextInstruction)
	}

	fn run_extend<T, U, V>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where
		RuntimeValue: From<V>, T: ExtendInto<U> + FromRuntimeValue, U: TransmuteInto<V>
	{
		let v = context
			.value_stack_mut()
			.pop_as::<T>();

		let v = v.extend_into().transmute_into();
		context.value_stack_mut().push(v.into())?;

		Ok(InstructionOutcome::RunNextInstruction)
	}

	fn run_reinterpret<T, U>(&mut self, context: &mut FunctionContext) -> Result<InstructionOutcome, TrapKind>
	where
		RuntimeValue: From<U>, T: FromRuntimeValue, T: TransmuteInto<U>
	{
		let v = context
			.value_stack_mut()
			.pop_as::<T>();

		let v = v.transmute_into();
		context.value_stack_mut().push(v.into())?;

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
	/// Function return type.
	pub return_type: BlockType,
	/// Local variables.
	pub locals: Vec<RuntimeValue>,
	/// Values stack.
	pub value_stack: ValueStack,
	/// Current instruction position.
	pub position: usize,
}

impl FunctionContext {
	pub fn new(function: FuncRef, value_stack_limit: usize, signature: &Signature, args: Vec<RuntimeValue>) -> Self {
		let module = match *function.as_internal() {
			FuncInstanceInternal::Internal { ref module, .. } => module.upgrade().expect("module deallocated"),
			FuncInstanceInternal::Host { .. } => panic!("Host functions can't be called as internally defined functions; Thus FunctionContext can be created only with internally defined functions; qed"),
		};
		FunctionContext {
			is_initialized: false,
			function: function,
			module: ModuleRef(module),
			return_type: signature.return_type().map(|vt| BlockType::Value(vt.into_elements())).unwrap_or(BlockType::NoResult),
			value_stack: ValueStack::with_limit(value_stack_limit),
			locals: args,
			position: 0,
		}
	}

	pub fn nested(&mut self, function: FuncRef) -> Result<Self, TrapKind> {
		let (function_locals, module, function_return_type) = {
			let module = match *function.as_internal() {
				FuncInstanceInternal::Internal { ref module, .. } => module.upgrade().expect("module deallocated"),
				FuncInstanceInternal::Host { .. } => panic!("Host functions can't be called as internally defined functions; Thus FunctionContext can be created only with internally defined functions; qed"),
			};
			let function_type = function.signature();
			let function_return_type = function_type.return_type().map(|vt| BlockType::Value(vt.into_elements())).unwrap_or(BlockType::NoResult);
			let function_locals = prepare_function_args(function_type, &mut self.value_stack);
			(function_locals, module, function_return_type)
		};

		Ok(FunctionContext {
			is_initialized: false,
			function: function,
			module: ModuleRef(module),
			return_type: function_return_type,
			value_stack: ValueStack::with_limit(self.value_stack.limit() - self.value_stack.len()),
			locals: function_locals,
			position: 0,
		})
	}

	pub fn is_initialized(&self) -> bool {
		self.is_initialized
	}

	pub fn initialize(&mut self, locals: &[Local]) {
		debug_assert!(!self.is_initialized);
		self.is_initialized = true;

		let locals = locals.iter()
			.flat_map(|l| repeat(l.value_type()).take(l.count() as usize))
			.map(::types::ValueType::from_elements)
			.map(RuntimeValue::default)
			.collect::<Vec<_>>();
		self.locals.extend(locals);
	}

	pub fn module(&self) -> ModuleRef {
		self.module.clone()
	}

	pub fn set_local(&mut self, index: usize, value: RuntimeValue) {
		let l = self.locals.get_mut(index).expect("Due to validation local should exists");
		*l = value;
	}

	pub fn get_local(&mut self, index: usize) -> RuntimeValue {
		self.locals.get(index)
			.cloned()
			.expect("Due to validation local should exists")
	}

	pub fn value_stack(&self) -> &ValueStack {
		&self.value_stack
	}

	pub fn value_stack_mut(&mut self) -> &mut ValueStack {
		&mut self.value_stack
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
	let mut args = signature
		.params()
		.iter()
		.map(|_| caller_stack.pop())
		.collect::<Vec<RuntimeValue>>();
	args.reverse();
	check_function_args(signature, &args).expect("Due to validation arguments should match");
	args
}

pub fn check_function_args(signature: &Signature, args: &[RuntimeValue]) -> Result<(), Error> {
	if signature.params().len() != args.len() {
		return Err(
			Error::Function(
				format!(
					"not enough arguments, given {} but expected: {}",
					args.len(),
					signature.params().len(),
				)
			)
		);
	}

	signature.params().iter().cloned().zip(args).map(|(expected_type, param_value)| {
		let actual_type = param_value.value_type();
		if actual_type != expected_type {
			return Err(Error::Function(format!("invalid parameter type {:?} when expected {:?}", actual_type, expected_type)));
		}
		Ok(())
	}).collect::<Result<Vec<_>, _>>()?;

	Ok(())
}

struct ValueStack {
	stack_with_limit: StackWithLimit<RuntimeValue>,
}

impl ValueStack {
	fn with_limit(limit: usize) -> ValueStack {
		ValueStack {
			stack_with_limit: StackWithLimit::with_limit(limit),
		}
	}

	fn pop_as<T>(&mut self) -> T
	where
		T: FromRuntimeValue,
	{
		let value = self.stack_with_limit
			.pop()
			.expect("Due to validation stack shouldn't be empty");
		value.try_into().expect("Due to validation stack top's type should match")
	}

	fn pop_pair_as<T>(&mut self) -> Result<(T, T), Error>
	where
		T: FromRuntimeValue,
	{
		let right = self.pop_as();
		let left = self.pop_as();
		Ok((left, right))
	}

	fn pop_triple(&mut self) -> (RuntimeValue, RuntimeValue, RuntimeValue) {
		let right = self.stack_with_limit.pop().expect("Due to validation stack shouldn't be empty");
		let mid = self.stack_with_limit.pop().expect("Due to validation stack shouldn't be empty");
		let left = self.stack_with_limit.pop().expect("Due to validation stack shouldn't be empty");
		(left, mid, right)
	}

	fn pop(&mut self) -> RuntimeValue {
		self.stack_with_limit.pop().expect("Due to validation stack shouldn't be empty")
	}

	fn push(&mut self, value: RuntimeValue) -> Result<(), TrapKind> {
		self.stack_with_limit.push(value)
			.map_err(|_| TrapKind::StackOverflow)
	}

	fn resize(&mut self, new_len: usize) {
		self.stack_with_limit.resize(new_len, RuntimeValue::I32(0));
	}

	fn len(&self) -> usize {
		self.stack_with_limit.len()
	}

	fn limit(&self) -> usize {
		self.stack_with_limit.limit()
	}

	fn top(&self) -> &RuntimeValue {
		self.stack_with_limit.top().expect("Due to validation stack shouldn't be empty")
	}
}
