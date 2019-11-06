use alloc::{
    borrow::Cow,
    rc::{Rc, Weak},
    vec::Vec,
};
use core::fmt;
use host::{Externals,AsyncExternals};
use isa;
use module::ModuleInstance;
use parity_wasm::elements::Local;
use runner::{check_function_args, Interpreter, InterpreterState, StackRecycler};
use types::ValueType;
use value::RuntimeValue;
use {Signature, Trap};
use futures::{Future, future};

/// Reference to a function (See [`FuncInstance`] for details).
///
/// This reference has a reference-counting semantics.
///
/// [`FuncInstance`]: struct.FuncInstance.html
#[derive(Clone, Debug)]
pub struct FuncRef(Rc<FuncInstance>);

impl ::core::ops::Deref for FuncRef {
    type Target = FuncInstance;
    fn deref(&self) -> &FuncInstance {
        &self.0
    }
}

/// Runtime representation of a function.
///
/// Functions are the unit of organization of code in WebAssembly. Each function takes a sequence of values
/// as parameters and either optionally return a value or trap.
/// Functions can call other function including itself (i.e recursive calls are allowed) and imported functions
/// (i.e functions defined in another module or by the host environment).
///
/// Functions can be defined either:
///
/// - by a wasm module,
/// - by the host environment and passed to a wasm module as an import.
///   See more in [`Externals`].
///
/// [`Externals`]: trait.Externals.html
pub struct FuncInstance(FuncInstanceInternal);

#[derive(Clone)]
pub(crate) enum FuncInstanceInternal {
    Internal {
        signature: Rc<Signature>,
        module: Weak<ModuleInstance>,
        body: Rc<FuncBody>,
    },
    Host {
        signature: Signature,
        host_func_index: usize,
    },
}

impl fmt::Debug for FuncInstance {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.as_internal() {
            &FuncInstanceInternal::Internal { ref signature, .. } => {
                // We can't write description of self.module here, because it generate
                // debug string for function instances and this will lead to infinite loop.
                write!(f, "Internal {{ signature={:?} }}", signature,)
            }
            &FuncInstanceInternal::Host { ref signature, .. } => {
                write!(f, "Host {{ signature={:?} }}", signature)
            }
        }
    }
}

impl FuncInstance {
    /// Allocate a function instance for a host function.
    ///
    /// When this function instance will be called by the wasm code,
    /// the instance of [`Externals`] will be invoked by calling `invoke_index`
    /// with specified `host_func_index` here.
    /// This call will be made with the `signature` provided here.
    ///
    /// [`Externals`]: trait.Externals.html
    pub fn alloc_host(signature: Signature, host_func_index: usize) -> FuncRef {
        let func = FuncInstanceInternal::Host {
            signature,
            host_func_index,
        };
        FuncRef(Rc::new(FuncInstance(func)))
    }

    /// Returns [signature] of this function instance.
    ///
    /// This function instance can only be called with matching signatures.
    ///
    /// [signature]: struct.Signature.html
    pub fn signature(&self) -> &Signature {
        match *self.as_internal() {
            FuncInstanceInternal::Internal { ref signature, .. } => signature,
            FuncInstanceInternal::Host { ref signature, .. } => signature,
        }
    }

    pub(crate) fn as_internal(&self) -> &FuncInstanceInternal {
        &self.0
    }

    pub(crate) fn alloc_internal(
        module: Weak<ModuleInstance>,
        signature: Rc<Signature>,
        body: FuncBody,
    ) -> FuncRef {
        let func = FuncInstanceInternal::Internal {
            signature,
            module: module,
            body: Rc::new(body),
        };
        FuncRef(Rc::new(FuncInstance(func)))
    }

    pub(crate) fn body(&self) -> Option<Rc<FuncBody>> {
        match *self.as_internal() {
            FuncInstanceInternal::Internal { ref body, .. } => Some(Rc::clone(body)),
            FuncInstanceInternal::Host { .. } => None,
        }
    }

    /// Invoke this function.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `args` types is not match function [`signature`] or
    /// if [`Trap`] at execution time occured.
    ///
    /// [`signature`]: #method.signature
    /// [`Trap`]: #enum.Trap.html
    pub fn invoke<E: Externals>(
        func: &FuncRef,
        args: &[RuntimeValue],
        externals: &mut E,
    ) -> Result<Option<RuntimeValue>, Trap> {
        check_function_args(func.signature(), &args)?;
        match *func.as_internal() {
            FuncInstanceInternal::Internal { .. } => {
                let mut interpreter = Interpreter::new(func, args, None)?;
                interpreter.start_execution(externals)
            }
            FuncInstanceInternal::Host {
                ref host_func_index,
                ..
            } => externals.invoke_index(*host_func_index, args.into()),
        }
    }

    /// Invoke this function asynchronously.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `args` types is not match function [`signature`] or
    /// if [`Trap`] at execution time occured.
    ///
    /// [`signature`]: #method.signature
    /// [`Trap`]: #enum.Trap.html
    pub fn invoke_async<'a,E: AsyncExternals +'a>(
        func: FuncRef,
        args: Vec<RuntimeValue>,
        externals: Rc<E>,
    ) -> Box<(dyn Future<Item = Option<RuntimeValue>,Error= Trap>+'a)> 
    {

            if let Err(err)=check_function_args(func.signature(), &args){
                return Box::new(future::err(err))
            }
            match *func.as_internal() {
                FuncInstanceInternal::Internal { .. } => {
                    let interpreter = Interpreter::new(&func, &args, None);
                    match interpreter{
                        Ok(i)=>{
                            i.start_execution_async(externals)
                        },
                        Err(err)=>{
                            Box::new(future::err(err))
                        }
                    }
                    
                }
                FuncInstanceInternal::Host {
                    ref host_func_index,
                    ..
                } => {
                    externals
                        .invoke_index_async(*host_func_index, args.into())
                        
                }
            }
    }

    /// Invoke this function using recycled stacks.
    ///
    /// # Errors
    ///
    /// Same as [`invoke`].
    ///
    /// [`invoke`]: #method.invoke
    pub fn invoke_with_stack<E: Externals>(
        func: &FuncRef,
        args: &[RuntimeValue],
        externals: &mut E,
        stack_recycler: &mut StackRecycler,
    ) -> Result<Option<RuntimeValue>, Trap> {
        check_function_args(func.signature(), &args)?;
        match *func.as_internal() {
            FuncInstanceInternal::Internal { .. } => {
                let mut interpreter = Interpreter::new(func, args, Some(stack_recycler))?;
                let return_value = interpreter.start_execution(externals);
                stack_recycler.recycle(interpreter);
                return_value
            }
            FuncInstanceInternal::Host {
                ref host_func_index,
                ..
            } => externals.invoke_index(*host_func_index, args.into()),
        }
    }

    /// Invoke the function, get a resumable handle. This handle can then be used to [`start_execution`]. If a
    /// Host trap happens, caller can use [`resume_execution`] to feed the expected return value back in, and then
    /// continue the execution.
    ///
    /// This is an experimental API, and this functionality may not be available in other WebAssembly engines.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `args` types is not match function [`signature`].
    ///
    /// [`signature`]: #method.signature
    /// [`Trap`]: #enum.Trap.html
    /// [`start_execution`]: struct.FuncInvocation.html#method.start_execution
    /// [`resume_execution`]: struct.FuncInvocation.html#method.resume_execution
    pub fn invoke_resumable<'args>(
        func: &FuncRef,
        args: impl Into<Cow<'args, [RuntimeValue]>>,
    ) -> Result<FuncInvocation<'args>, Trap> {
        let args = args.into();
        check_function_args(func.signature(), &args)?;
        match *func.as_internal() {
            FuncInstanceInternal::Internal { .. } => {
                let interpreter = Interpreter::new(func, &*args, None)?;
                Ok(FuncInvocation {
                    kind: FuncInvocationKind::Internal(interpreter),
                })
            }
            FuncInstanceInternal::Host {
                ref host_func_index,
                ..
            } => Ok(FuncInvocation {
                kind: FuncInvocationKind::Host {
                    args,
                    host_func_index: *host_func_index,
                    finished: false,
                },
            }),
        }
    }
}

/// A resumable invocation error.
#[derive(Debug)]
pub enum ResumableError {
    /// Trap happened.
    Trap(Trap),
    /// The invocation is not resumable.
    ///
    /// Invocations are only resumable if a host function is called, and the host function returns a trap of `Host` kind. For other cases, this error will be returned. This includes:
    /// - The invocation is directly a host function.
    /// - The invocation has not been started.
    /// - The invocation returns normally or returns any trap other than `Host` kind.
    ///
    /// This error is returned by [`resume_execution`].
    ///
    /// [`resume_execution`]: struct.FuncInvocation.html#method.resume_execution
    NotResumable,
    /// The invocation has already been started.
    ///
    /// This error is returned by [`start_execution`].
    ///
    /// [`start_execution`]: struct.FuncInvocation.html#method.start_execution
    AlreadyStarted,
}

impl From<Trap> for ResumableError {
    fn from(trap: Trap) -> Self {
        ResumableError::Trap(trap)
    }
}

/// A resumable invocation handle. This struct is returned by `FuncInstance::invoke_resumable`.
pub struct FuncInvocation<'args> {
    kind: FuncInvocationKind<'args>,
}

enum FuncInvocationKind<'args> {
    Internal(Interpreter),
    Host {
        args: Cow<'args, [RuntimeValue]>,
        host_func_index: usize,
        finished: bool,
    },
}

impl<'args> FuncInvocation<'args> {
    /// Whether this invocation is currently resumable.
    pub fn is_resumable(&self) -> bool {
        match &self.kind {
            &FuncInvocationKind::Internal(ref interpreter) => interpreter.state().is_resumable(),
            &FuncInvocationKind::Host { .. } => false,
        }
    }

    /// If the invocation is resumable, the expected return value type to be feed back in.
    pub fn resumable_value_type(&self) -> Option<ValueType> {
        match &self.kind {
            &FuncInvocationKind::Internal(ref interpreter) => match interpreter.state() {
                &InterpreterState::Resumable(ref value_type) => value_type.clone(),
                _ => None,
            },
            &FuncInvocationKind::Host { .. } => None,
        }
    }

    /// Start the invocation execution.
    pub fn start_execution<'externals, E: Externals + 'externals>(
        &mut self,
        externals: &'externals mut E,
    ) -> Result<Option<RuntimeValue>, ResumableError> {
        match self.kind {
            FuncInvocationKind::Internal(ref mut interpreter) => {
                if interpreter.state() != &InterpreterState::Initialized {
                    return Err(ResumableError::AlreadyStarted);
                }
                Ok(interpreter.start_execution(externals)?)
            }
            FuncInvocationKind::Host {
                ref args,
                ref mut finished,
                ref host_func_index,
            } => {
                if *finished {
                    return Err(ResumableError::AlreadyStarted);
                }
                *finished = true;
                Ok(externals.invoke_index(*host_func_index, args.as_ref().into())?)
            }
        }
    }

    /// Resume an execution if a previous trap of Host kind happened.
    ///
    /// `return_val` must be of the value type [`resumable_value_type`], defined by the host function import. Otherwise,
    /// `UnexpectedSignature` trap will be returned. The current invocation must also be resumable
    /// [`is_resumable`]. Otherwise, a `NotResumable` error will be returned.
    ///
    /// [`resumable_value_type`]: #method.resumable_value_type
    /// [`is_resumable`]: #method.is_resumable
    pub fn resume_execution<'externals, E: Externals + 'externals>(
        &mut self,
        return_val: Option<RuntimeValue>,
        externals: &'externals mut E,
    ) -> Result<Option<RuntimeValue>, ResumableError> {
        use crate::TrapKind;

        if return_val.map(|v| v.value_type()) != self.resumable_value_type() {
            return Err(ResumableError::Trap(Trap::new(
                TrapKind::UnexpectedSignature,
            )));
        }

        match &mut self.kind {
            FuncInvocationKind::Internal(interpreter) => {
                if interpreter.state().is_resumable() {
                    Ok(interpreter.resume_execution(return_val, externals)?)
                } else {
                    Err(ResumableError::AlreadyStarted)
                }
            }
            FuncInvocationKind::Host { .. } => Err(ResumableError::NotResumable),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FuncBody {
    pub locals: Vec<Local>,
    pub code: isa::Instructions,
}
