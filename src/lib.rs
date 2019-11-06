//! # wasmi
//!
//! This library allows WebAssembly modules to be loaded in binary format and their functions invoked.
//!
//! # Introduction
//!
//! WebAssembly (wasm) is a safe, portable and compact format that is designed for efficient execution.
//!
//! Wasm code is distributed in the form of modules that contains definitions of:
//!
//! - functions,
//! - global variables,
//! - linear memory instances and
//! - tables.
//!
//! Each of these definitions can be imported and exported.
//!
//! In addition to these definitions, modules can define initialization data for their memory or tables. This initialization data can take the
//! form of segments, copied to given offsets. They can also define a `start` function that is automatically executed when the module is loaded.
//!
//! ## Loading and Validation
//!
//! Before execution, a module must be validated. This process checks that the module is well-formed
//! and makes only allowed operations.
//!
//! A valid module can't access memory outside its sandbox, can't cause stack underflows
//! and can only call functions with correct signatures.
//!
//! ## Instantiation
//!
//! In order to execute code from a wasm module, it must be instantiated.
//! Instantiation includes the following steps:
//!
//! 1. Creating an empty module instance.
//! 2. Resolving the definition instances for each declared import in the module.
//! 3. Instantiating definitions declared in the module (e.g. allocate global variables, allocate linear memory, etc.).
//! 4. Initializing memory and table contents by copying segments into them.
//! 5. Executing the `start` function, if any.
//!
//! After these steps, the module instance is ready to execute functions.
//!
//! ## Execution
//!
//! It only is allowed to call functions which are exported by the module.
//! Functions can either return a result or trap (e.g. there can't be linking error in the middle of the function execution).
//! This property is ensured by the validation process.
//!
//! # Examples
//!
//! ```rust
//! extern crate wasmi;
//! extern crate wabt;
//!
//! use wasmi::{ModuleInstance, ImportsBuilder, NopExternals, RuntimeValue};
//!
//! fn main() {
//!     // Parse WAT (WebAssembly Text format) into wasm bytecode.
//!     let wasm_binary: Vec<u8> =
//!         wabt::wat2wasm(
//!             r#"
//!             (module
//!                 (func (export "test") (result i32)
//!                     i32.const 1337
//!                 )
//!             )
//!             "#,
//!         )
//!         .expect("failed to parse wat");
//!
//!     // Load wasm binary and prepare it for instantiation.
//!     let module = wasmi::Module::from_buffer(&wasm_binary)
//!         .expect("failed to load wasm");
//!
//!     // Instantiate a module with empty imports and
//!     // assert that there is no `start` function.
//!     let instance =
//!         ModuleInstance::new(
//!             &module,
//!             &ImportsBuilder::default()
//!         )
//!         .expect("failed to instantiate wasm module")
//!         .assert_no_start();
//!
//!     // Finally, invoke the exported function "test" with no parameters
//!     // and empty external function executor.
//!     assert_eq!(
//!         instance.invoke_export(
//!             "test",
//!             &[],
//!             &mut NopExternals,
//!         ).expect("failed to execute export"),
//!         Some(RuntimeValue::I32(1337)),
//!     );
//! }
//! ```

#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
#[macro_use]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std as alloc;

#[cfg(feature = "std")]
#[macro_use]
extern crate core;

#[cfg(test)]
extern crate assert_matches;
#[cfg(test)]
extern crate wabt;

extern crate memory_units as memory_units_crate;
extern crate parity_wasm;

extern crate wasmi_validation as validation;

extern crate futures;

use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt;
#[cfg(feature = "std")]
use std::error;

#[cfg(not(feature = "std"))]
extern crate libm;

extern crate num_rational;
extern crate num_traits;

/// Error type which can be thrown by wasm code or by host environment.
///
/// Under some conditions, wasm execution may produce a `Trap`, which immediately aborts execution.
/// Traps can't be handled by WebAssembly code, but are reported to the embedder.
#[derive(Debug)]
pub struct Trap {
    kind: TrapKind,
}

impl Trap {
    /// Create new trap.
    pub fn new(kind: TrapKind) -> Trap {
        Trap { kind }
    }

    /// Returns kind of this trap.
    pub fn kind(&self) -> &TrapKind {
        &self.kind
    }
}

impl fmt::Display for Trap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Trap: {:?}", self.kind)
    }
}

#[cfg(feature = "std")]
impl error::Error for Trap {
    fn description(&self) -> &str {
        "runtime trap"
    }
}

/// Error type which can be thrown by wasm code or by host environment.
///
/// See [`Trap`] for details.
///
/// [`Trap`]: struct.Trap.html
#[derive(Debug)]
pub enum TrapKind {
    /// Wasm code executed `unreachable` opcode.
    ///
    /// `unreachable` is a special opcode which always traps upon execution.
    /// This opcode have a similar purpose as `ud2` in x86.
    Unreachable,

    /// Attempt to load or store at the address which
    /// lies outside of bounds of the memory.
    ///
    /// Since addresses are interpreted as unsigned integers, out of bounds access
    /// can't happen with negative addresses (i.e. they will always wrap).
    MemoryAccessOutOfBounds,

    /// Attempt to access table element at index which
    /// lies outside of bounds.
    ///
    /// This typically can happen when `call_indirect` is executed
    /// with index that lies out of bounds.
    ///
    /// Since indexes are interpreted as unsinged integers, out of bounds access
    /// can't happen with negative indexes (i.e. they will always wrap).
    TableAccessOutOfBounds,

    /// Attempt to access table element which is uninitialized (i.e. `None`).
    ///
    /// This typically can happen when `call_indirect` is executed.
    ElemUninitialized,

    /// Attempt to divide by zero.
    ///
    /// This trap typically can happen if `div` or `rem` is executed with
    /// zero as divider.
    DivisionByZero,

    /// Attempt to make a conversion to an int failed.
    ///
    /// This can happen when:
    ///
    /// - trying to do signed division (or get the remainder) -2<sup>N-1</sup> over -1. This is
    ///   because the result +2<sup>N-1</sup> isn't representable as a N-bit signed integer.
    /// - trying to truncate NaNs, infinity, or value for which the result is out of range into an integer.
    InvalidConversionToInt,

    /// Stack overflow.
    ///
    /// This is likely caused by some infinite or very deep recursion.
    /// Extensive inlining might also be the cause of stack overflow.
    StackOverflow,

    /// Attempt to invoke a function with mismatching signature.
    ///
    /// This can happen if [`FuncInstance`] was invoked
    /// with mismatching [signature][`Signature`].
    ///
    /// This can always happen with indirect calls. `call_indirect` instruction always
    /// specifies the expected signature of function. If `call_indirect` is executed
    /// with index that points on function with signature different that is
    /// expected by this `call_indirect`, this trap is raised.
    ///
    /// [`Signature`]: struct.Signature.html
    UnexpectedSignature,

    /// Error specified by the host.
    ///
    /// Typically returned from an implementation of [`Externals`].
    ///
    /// [`Externals`]: trait.Externals.html
    Host(Box<dyn host::HostError>),
}

impl TrapKind {
    /// Whether this trap is specified by the host.
    pub fn is_host(&self) -> bool {
        match self {
            &TrapKind::Host(_) => true,
            _ => false,
        }
    }
}

/// Internal interpreter error.
#[derive(Debug)]
pub enum Error {
    /// Module validation error. Might occur only at load time.
    Validation(String),
    /// Error while instantiating a module. Might occur when provided
    /// with incorrect exports (i.e. linkage failure).
    Instantiation(String),
    /// Function-level error.
    Function(String),
    /// Table-level error.
    Table(String),
    /// Memory-level error.
    Memory(String),
    /// Global-level error.
    Global(String),
    /// Value-level error.
    Value(String),
    /// Trap.
    Trap(Trap),
    /// Custom embedder error.
    Host(Box<dyn host::HostError>),
}

impl Error {
    /// Returns [`HostError`] if this `Error` represents some host error.
    ///
    /// I.e. if this error have variant [`Host`] or [`Trap`][`Trap`] with [host][`TrapKind::Host`] error.
    ///
    /// [`HostError`]: trait.HostError.html
    /// [`Host`]: enum.Error.html#variant.Host
    /// [`Trap`]: enum.Error.html#variant.Trap
    /// [`TrapKind::Host`]: enum.TrapKind.html#variant.Host
    pub fn as_host_error(&self) -> Option<&dyn host::HostError> {
        match *self {
            Error::Host(ref host_err) => Some(&**host_err),
            Error::Trap(ref trap) => match *trap.kind() {
                TrapKind::Host(ref host_err) => Some(&**host_err),
                _ => None,
            },
            _ => None,
        }
    }
}

impl Into<String> for Error {
    fn into(self) -> String {
        match self {
            Error::Validation(s) => s,
            Error::Instantiation(s) => s,
            Error::Function(s) => s,
            Error::Table(s) => s,
            Error::Memory(s) => s,
            Error::Global(s) => s,
            Error::Value(s) => s,
            Error::Trap(s) => format!("trap: {:?}", s),
            Error::Host(e) => format!("user: {}", e),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::Validation(ref s) => write!(f, "Validation: {}", s),
            Error::Instantiation(ref s) => write!(f, "Instantiation: {}", s),
            Error::Function(ref s) => write!(f, "Function: {}", s),
            Error::Table(ref s) => write!(f, "Table: {}", s),
            Error::Memory(ref s) => write!(f, "Memory: {}", s),
            Error::Global(ref s) => write!(f, "Global: {}", s),
            Error::Value(ref s) => write!(f, "Value: {}", s),
            Error::Trap(ref s) => write!(f, "Trap: {:?}", s),
            Error::Host(ref e) => write!(f, "User: {}", e),
        }
    }
}

#[cfg(feature = "std")]
impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::Validation(ref s) => s,
            Error::Instantiation(ref s) => s,
            Error::Function(ref s) => s,
            Error::Table(ref s) => s,
            Error::Memory(ref s) => s,
            Error::Global(ref s) => s,
            Error::Value(ref s) => s,
            Error::Trap(_) => "Trap",
            Error::Host(_) => "Host error",
        }
    }
}

impl<U> From<U> for Error
where
    U: host::HostError + Sized,
{
    fn from(e: U) -> Self {
        Error::Host(Box::new(e))
    }
}

impl<U> From<U> for Trap
where
    U: host::HostError + Sized,
{
    fn from(e: U) -> Self {
        Trap::new(TrapKind::Host(Box::new(e)))
    }
}

impl From<Trap> for Error {
    fn from(e: Trap) -> Error {
        Error::Trap(e)
    }
}

impl From<TrapKind> for Trap {
    fn from(e: TrapKind) -> Trap {
        Trap::new(e)
    }
}

impl From<validation::Error> for Error {
    fn from(e: validation::Error) -> Error {
        Error::Validation(e.to_string())
    }
}

mod func;
mod global;
mod host;
mod imports;
mod isa;
mod memory;
mod module;
pub mod nan_preserving_float;
mod prepare;
mod runner;
mod table;
mod types;
mod value;

#[cfg(test)]
mod tests;

pub use self::func::{FuncInstance, FuncInvocation, FuncRef, ResumableError};
pub use self::global::{GlobalInstance, GlobalRef};
pub use self::host::{Externals, AsyncExternals, HostError, NopExternals, RuntimeArgs};
pub use self::imports::{ImportResolver, ImportsBuilder, ModuleImportResolver};
pub use self::memory::{MemoryInstance, MemoryRef, LINEAR_MEMORY_PAGE_SIZE};
pub use self::module::{ExternVal, ModuleInstance, ModuleRef, NotStartedModuleRef};
pub use self::runner::{StackRecycler, DEFAULT_CALL_STACK_LIMIT, DEFAULT_VALUE_STACK_LIMIT};
pub use self::table::{TableInstance, TableRef};
pub use self::types::{GlobalDescriptor, MemoryDescriptor, Signature, TableDescriptor, ValueType};
pub use self::value::{Error as ValueError, FromRuntimeValue, LittleEndianConvert, RuntimeValue};

/// WebAssembly-specific sizes and units.
pub mod memory_units {
    pub use memory_units_crate::wasm32::*;
    pub use memory_units_crate::{size_of, ByteSize, Bytes, RoundUpTo};
}

/// Deserialized module prepared for instantiation.
pub struct Module {
    code_map: Vec<isa::Instructions>,
    module: parity_wasm::elements::Module,
}

impl Module {
    /// Create `Module` from `parity_wasm::elements::Module`.
    ///
    /// This function will load, validate and prepare a `parity_wasm`'s `Module`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if provided `Module` is not valid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// extern crate parity_wasm;
    /// extern crate wasmi;
    ///
    /// use parity_wasm::builder;
    /// use parity_wasm::elements;
    ///
    /// fn main() {
    ///     let parity_module =
    ///         builder::module()
    ///             .function()
    ///                 .signature().with_param(elements::ValueType::I32).build()
    ///                 .body().build()
    ///             .build()
    ///         .build();
    ///
    ///     let module = wasmi::Module::from_parity_wasm_module(parity_module)
    ///         .expect("parity-wasm builder generated invalid module!");
    ///
    ///     // Instantiate `module`, etc...
    /// }
    /// ```
    pub fn from_parity_wasm_module(module: parity_wasm::elements::Module) -> Result<Module, Error> {
        let prepare::CompiledModule { code_map, module } = prepare::compile_module(module)?;

        Ok(Module { code_map, module })
    }

    /// Fail if the module contains any floating-point operations
    ///
    /// # Errors
    ///
    /// Returns `Err` if provided `Module` is not valid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate wasmi;
    /// # extern crate wabt;
    ///
    /// let wasm_binary: Vec<u8> =
    ///     wabt::wat2wasm(
    ///         r#"
    ///         (module
    ///          (func $add (param $lhs i32) (param $rhs i32) (result i32)
    ///                get_local $lhs
    ///                get_local $rhs
    ///                i32.add))
    ///         "#,
    ///     )
    ///     .expect("failed to parse wat");
    ///
    /// // Load wasm binary and prepare it for instantiation.
    /// let module = wasmi::Module::from_buffer(&wasm_binary).expect("Parsing failed");
    /// assert!(module.deny_floating_point().is_ok());
    ///
    /// let wasm_binary: Vec<u8> =
    ///     wabt::wat2wasm(
    ///         r#"
    ///         (module
    ///          (func $add (param $lhs f32) (param $rhs f32) (result f32)
    ///                get_local $lhs
    ///                get_local $rhs
    ///                f32.add))
    ///         "#,
    ///     )
    ///     .expect("failed to parse wat");
    ///
    /// let module = wasmi::Module::from_buffer(&wasm_binary).expect("Parsing failed");
    /// assert!(module.deny_floating_point().is_err());
    ///
    /// let wasm_binary: Vec<u8> =
    ///     wabt::wat2wasm(
    ///         r#"
    ///         (module
    ///          (func $add (param $lhs f32) (param $rhs f32) (result f32)
    ///                get_local $lhs))
    ///         "#,
    ///     )
    ///     .expect("failed to parse wat");
    ///
    /// let module = wasmi::Module::from_buffer(&wasm_binary).expect("Parsing failed");
    /// assert!(module.deny_floating_point().is_err());
    /// ```
    pub fn deny_floating_point(&self) -> Result<(), Error> {
        prepare::deny_floating_point(&self.module).map_err(Into::into)
    }

    /// Create `Module` from a given buffer.
    ///
    /// This function will deserialize wasm module from a given module,
    /// validate and prepare it for instantiation.
    ///
    /// # Errors
    ///
    /// Returns `Err` if wasm binary in provided `buffer` is not valid wasm binary.
    ///
    /// # Examples
    ///
    /// ```rust
    /// extern crate wasmi;
    ///
    /// fn main() {
    ///     let module =
    ///         wasmi::Module::from_buffer(
    ///             // Minimal module:
    ///             //   \0asm - magic
    ///             //    0x01 - version (in little-endian)
    ///             &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]
    ///         ).expect("Failed to load minimal module");
    ///
    ///     // Instantiate `module`, etc...
    /// }
    /// ```
    pub fn from_buffer<B: AsRef<[u8]>>(buffer: B) -> Result<Module, Error> {
        let module = parity_wasm::elements::deserialize_buffer(buffer.as_ref())
            .map_err(|e: parity_wasm::elements::Error| Error::Validation(e.to_string()))?;
        Module::from_parity_wasm_module(module)
    }

    pub(crate) fn module(&self) -> &parity_wasm::elements::Module {
        &self.module
    }

    pub(crate) fn code(&self) -> &Vec<isa::Instructions> {
        &self.code_map
    }
}
