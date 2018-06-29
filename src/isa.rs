//! An instruction set used by wasmi.
//!
//! The instruction set is mostly derived from Wasm. However,
//! there is a substantial difference.
//!
//! # Structured Stack Machine vs Traditional One
//!
//! Wasm is a structured stack machine. Wasm encodes control flow in structures
//! similar to that commonly found in a programming languages
//! such as if, while. That contrasts to a traditional stack machine which
//!  encodes all control flow with goto-like instructions.
//!
//! Structured stack machine code aligns well with goals of Wasm,
//! namely providing fast validation of Wasm code and compilation to native code.
//!
//! Unfortunately, the downside of structured stack machine code is
//! that it is less convenient to interpret. For example, let's look at
//! the following example in hypothetical structured stack machine:
//!
//! ```plain
//! loop
//!   ...
//!   if_true_jump_to_end
//!   ...
//! end
//! ```
//!
//! To execute `if_true_jump_to_end` , the interpreter needs to skip all instructions
//! until it reaches the *matching* `end`. That's quite inefficient compared
//! to a plain goto to the specific position.
//!
//! # Differences from Wasm
//!
//! - There is no `nop` instruction.
//! - All control flow strucutres are flattened to plain gotos.
//! - Implicit returns via reaching function scope `End` are replaced with an explicit `return` instruction.
//! - Locals live on the value stack now.
//! - Load/store instructions doesn't take `align` parameter.
//! - *.const store value in straight encoding.
//! - Reserved immediates are ignored for `call_indirect`, `current_memory`, `grow_memory`.
//!

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Keep {
	None,
	Single,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DropKeep {
	pub drop: u32,
	pub keep: Keep,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
	pub dst_pc: u32,
	pub drop_keep: DropKeep,
}

#[allow(unused)] // TODO: Remove
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
	/// Push a local variable or an argument from the specified depth.
	GetLocal(u32),

	/// Pop a value and put it in at the specified depth.
	SetLocal(u32),

	/// Copy a value to the specified depth.
	TeeLocal(u32),

	/// Similar to the Wasm ones, but instead of a label depth
	/// they specify direct PC.
	Br(Target),
	BrIfEqz(Target),
	BrIfNez(Target),

	/// Last one is the default.
	///
	/// Can be less than zero.
	BrTable(Box<[Target]>),

	Unreachable,
	Return(DropKeep),

	Call(u32),
	CallIndirect(u32),

	Drop,
	Select,

	GetGlobal(u32),
	SetGlobal(u32),

	I32Load(u32),
	I64Load(u32),
	F32Load(u32),
	F64Load(u32),
	I32Load8S(u32),
	I32Load8U(u32),
	I32Load16S(u32),
	I32Load16U(u32),
	I64Load8S(u32),
	I64Load8U(u32),
	I64Load16S(u32),
	I64Load16U(u32),
	I64Load32S(u32),
	I64Load32U(u32),
	I32Store(u32),
	I64Store(u32),
	F32Store(u32),
	F64Store(u32),
	I32Store8(u32),
	I32Store16(u32),
	I64Store8(u32),
	I64Store16(u32),
	I64Store32(u32),

	CurrentMemory,
	GrowMemory,

	I32Const(i32),
	I64Const(i64),
	F32Const(u32),
	F64Const(u64),

	I32Eqz,
	I32Eq,
	I32Ne,
	I32LtS,
	I32LtU,
	I32GtS,
	I32GtU,
	I32LeS,
	I32LeU,
	I32GeS,
	I32GeU,

	I64Eqz,
	I64Eq,
	I64Ne,
	I64LtS,
	I64LtU,
	I64GtS,
	I64GtU,
	I64LeS,
	I64LeU,
	I64GeS,
	I64GeU,

	F32Eq,
	F32Ne,
	F32Lt,
	F32Gt,
	F32Le,
	F32Ge,

	F64Eq,
	F64Ne,
	F64Lt,
	F64Gt,
	F64Le,
	F64Ge,

	I32Clz,
	I32Ctz,
	I32Popcnt,
	I32Add,
	I32Sub,
	I32Mul,
	I32DivS,
	I32DivU,
	I32RemS,
	I32RemU,
	I32And,
	I32Or,
	I32Xor,
	I32Shl,
	I32ShrS,
	I32ShrU,
	I32Rotl,
	I32Rotr,

	I64Clz,
	I64Ctz,
	I64Popcnt,
	I64Add,
	I64Sub,
	I64Mul,
	I64DivS,
	I64DivU,
	I64RemS,
	I64RemU,
	I64And,
	I64Or,
	I64Xor,
	I64Shl,
	I64ShrS,
	I64ShrU,
	I64Rotl,
	I64Rotr,
	F32Abs,
	F32Neg,
	F32Ceil,
	F32Floor,
	F32Trunc,
	F32Nearest,
	F32Sqrt,
	F32Add,
	F32Sub,
	F32Mul,
	F32Div,
	F32Min,
	F32Max,
	F32Copysign,
	F64Abs,
	F64Neg,
	F64Ceil,
	F64Floor,
	F64Trunc,
	F64Nearest,
	F64Sqrt,
	F64Add,
	F64Sub,
	F64Mul,
	F64Div,
	F64Min,
	F64Max,
	F64Copysign,

	I32WrapI64,
	I32TruncSF32,
	I32TruncUF32,
	I32TruncSF64,
	I32TruncUF64,
	I64ExtendSI32,
	I64ExtendUI32,
	I64TruncSF32,
	I64TruncUF32,
	I64TruncSF64,
	I64TruncUF64,
	F32ConvertSI32,
	F32ConvertUI32,
	F32ConvertSI64,
	F32ConvertUI64,
	F32DemoteF64,
	F64ConvertSI32,
	F64ConvertUI32,
	F64ConvertSI64,
	F64ConvertUI64,
	F64PromoteF32,

	I32ReinterpretF32,
	I64ReinterpretF64,
	F32ReinterpretI32,
	F64ReinterpretI64,
}

#[derive(Debug, Clone)]
pub struct Instructions {
	pub code: Vec<Instruction>,
}
