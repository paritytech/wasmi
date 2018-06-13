use super::{validate_module, ValidatedModule};
use parity_wasm::builder::module;
use parity_wasm::elements::{
	External, GlobalEntry, GlobalType, ImportEntry, InitExpr, MemoryType,
	Instruction, Instructions, TableType, ValueType, BlockType, deserialize_buffer,
	Module,
};
use isa;
use wabt;

#[test]
fn empty_is_valid() {
	let module = module().build();
	assert!(validate_module(module).is_ok());
}

#[test]
fn limits() {
	let test_cases = vec![
		// min > max
		(10, Some(9), false),
		// min = max
		(10, Some(10), true),
		// table/memory is always valid without max
		(10, None, true),
	];

	for (min, max, is_valid) in test_cases {
		// defined table
		let m = module()
			.table()
				.with_min(min)
				.with_max(max)
				.build()
			.build();
		assert_eq!(validate_module(m).is_ok(), is_valid);

		// imported table
		let m = module()
			.with_import(
				ImportEntry::new(
					"core".into(),
					"table".into(),
					External::Table(TableType::new(min, max))
				)
			)
			.build();
		assert_eq!(validate_module(m).is_ok(), is_valid);

		// defined memory
		let m = module()
			.memory()
				.with_min(min)
				.with_max(max)
				.build()
			.build();
		assert_eq!(validate_module(m).is_ok(), is_valid);

		// imported table
		let m = module()
			.with_import(
				ImportEntry::new(
					"core".into(),
					"memory".into(),
					External::Memory(MemoryType::new(min, max))
				)
			)
			.build();
		assert_eq!(validate_module(m).is_ok(), is_valid);
	}
}

#[test]
fn global_init_const() {
	let m = module()
		.with_global(
			GlobalEntry::new(
				GlobalType::new(ValueType::I32, true),
				InitExpr::new(
					vec![Instruction::I32Const(42), Instruction::End]
				)
			)
		)
		.build();
	assert!(validate_module(m).is_ok());

	// init expr type differs from declared global type
	let m = module()
		.with_global(
			GlobalEntry::new(
				GlobalType::new(ValueType::I64, true),
				InitExpr::new(vec![Instruction::I32Const(42), Instruction::End])
			)
		)
		.build();
	assert!(validate_module(m).is_err());
}

#[test]
fn global_init_global() {
	let m = module()
		.with_import(
			ImportEntry::new(
				"env".into(),
				"ext_global".into(),
				External::Global(GlobalType::new(ValueType::I32, false))
			)
		)
		.with_global(
			GlobalEntry::new(
				GlobalType::new(ValueType::I32, true),
				InitExpr::new(vec![Instruction::GetGlobal(0), Instruction::End])
			)
		)
		.build();
	assert!(validate_module(m).is_ok());

	// get_global can reference only previously defined globals
	let m = module()
		.with_global(
			GlobalEntry::new(
				GlobalType::new(ValueType::I32, true),
				InitExpr::new(vec![Instruction::GetGlobal(0), Instruction::End])
			)
		)
		.build();
	assert!(validate_module(m).is_err());

	// get_global can reference only const globals
	let m = module()
		.with_import(
			ImportEntry::new(
				"env".into(),
				"ext_global".into(),
				External::Global(GlobalType::new(ValueType::I32, true))
			)
		)
		.with_global(
			GlobalEntry::new(
				GlobalType::new(ValueType::I32, true),
				InitExpr::new(vec![Instruction::GetGlobal(0), Instruction::End])
			)
		)
		.build();
	assert!(validate_module(m).is_err());

	// get_global in init_expr can only refer to imported globals.
	let m = module()
		.with_global(
			GlobalEntry::new(
				GlobalType::new(ValueType::I32, false),
				InitExpr::new(vec![Instruction::I32Const(0), Instruction::End])
			)
		)
		.with_global(
			GlobalEntry::new(
				GlobalType::new(ValueType::I32, true),
				InitExpr::new(vec![Instruction::GetGlobal(0), Instruction::End])
			)
		)
		.build();
	assert!(validate_module(m).is_err());
}

#[test]
fn global_init_misc() {
	// without delimiting End opcode
	let m = module()
		.with_global(
			GlobalEntry::new(
				GlobalType::new(ValueType::I32, true),
				InitExpr::new(vec![Instruction::I32Const(42)])
			)
		)
		.build();
	assert!(validate_module(m).is_err());

	// empty init expr
	let m = module()
		.with_global(
			GlobalEntry::new(
				GlobalType::new(ValueType::I32, true),
				InitExpr::new(vec![Instruction::End])
			)
		)
		.build();
	assert!(validate_module(m).is_err());

	// not an constant opcode used
	let m = module()
		.with_global(
			GlobalEntry::new(
				GlobalType::new(ValueType::I32, true),
				InitExpr::new(vec![Instruction::Unreachable, Instruction::End])
			)
		)
		.build();
	assert!(validate_module(m).is_err());
}

#[test]
fn module_limits_validity() {
	// module cannot contain more than 1 memory atm.
	let m = module()
		.with_import(
			ImportEntry::new(
				"core".into(),
				"memory".into(),
				External::Memory(MemoryType::new(10, None))
			)
		)
		.memory()
			.with_min(10)
			.build()
		.build();
	assert!(validate_module(m).is_err());

	// module cannot contain more than 1 table atm.
	let m = module()
		.with_import(
			ImportEntry::new(
				"core".into(),
				"table".into(),
				External::Table(TableType::new(10, None))
			)
		)
		.table()
			.with_min(10)
			.build()
		.build();
	assert!(validate_module(m).is_err());
}

#[test]
fn funcs() {
	// recursive function calls is legal.
	let m = module()
		.function()
			.signature().return_type().i32().build()
			.body().with_instructions(Instructions::new(vec![
				Instruction::Call(1),
				Instruction::End,
			])).build()
			.build()
		.function()
			.signature().return_type().i32().build()
			.body().with_instructions(Instructions::new(vec![
				Instruction::Call(0),
				Instruction::End,
			])).build()
			.build()
		.build();
	assert!(validate_module(m).is_ok());
}

#[test]
fn globals() {
	// import immutable global is legal.
	let m = module()
		.with_import(
			ImportEntry::new(
				"env".into(),
				"ext_global".into(),
				External::Global(GlobalType::new(ValueType::I32, false))
			)
		)
		.build();
	assert!(validate_module(m).is_ok());

	// import mutable global is invalid.
	let m = module()
		.with_import(
			ImportEntry::new(
				"env".into(),
				"ext_global".into(),
				External::Global(GlobalType::new(ValueType::I32, true))
			)
		)
		.build();
	assert!(validate_module(m).is_err());
}

#[test]
fn if_else_with_return_type_validation() {
	let m = module()
		.function()
			.signature().build()
			.body().with_instructions(Instructions::new(vec![
				Instruction::I32Const(1),
				Instruction::If(BlockType::NoResult),
					Instruction::I32Const(1),
					Instruction::If(BlockType::Value(ValueType::I32)),
						Instruction::I32Const(1),
					Instruction::Else,
						Instruction::I32Const(2),
					Instruction::End,
					Instruction::Drop,
				Instruction::End,
				Instruction::End,
			])).build()
			.build()
		.build();
	validate_module(m).unwrap();
}

fn validate(wat: &str) -> ValidatedModule {
	let wasm = wabt::wat2wasm(wat).unwrap();
	let module = deserialize_buffer::<Module>(&wasm).unwrap();
	let validated_module = validate_module(module).unwrap();
	validated_module
}

fn compile(wat: &str) -> Vec<isa::Instruction> {
	let validated_module = validate(wat);
	let code = &validated_module.code_map[0];
	code.code.clone()
}

#[test]
fn implicit_return_no_value() {
	let code = compile(r#"
		(module
			(func (export "call")
			)
		)
	"#);
	assert_eq!(
		code,
		vec![
			isa::Instruction::Return {
				drop: 0,
				keep: 0,
			}
		]
	)
}

#[test]
fn implicit_return_with_value() {
	let code = compile(r#"
		(module
			(func (export "call") (result i32)
				i32.const 0
			)
		)
	"#);
	assert_eq!(
		code,
		vec![
			isa::Instruction::I32Const(0),
			isa::Instruction::Return {
				drop: 0,
				keep: 1,
			}
		]
	)
}

#[test]
fn implicit_return_param() {
	let code = compile(r#"
		(module
			(func (export "call") (param i32)
			)
		)
	"#);
	assert_eq!(
		code,
		vec![
			isa::Instruction::Return {
				drop: 1,
				keep: 0,
			}
		]
	)
}

#[test]
fn get_local() {
	let code = compile(r#"
		(module
			(func (export "call") (param i32) (result i32)
				get_local 0
			)
		)
	"#);
	assert_eq!(
		code,
		vec![
			isa::Instruction::GetLocal(1),
			isa::Instruction::Return {
				drop: 1,
				keep: 1,
			}
		]
	)
}

#[test]
fn explicit_return() {
	let code = compile(r#"
		(module
			(func (export "call") (param i32) (result i32)
				get_local 0
				return
			)
		)
	"#);
	assert_eq!(
		code,
		vec![
			isa::Instruction::GetLocal(1),
			isa::Instruction::Return {
				drop: 1,
				keep: 1,
			},
			isa::Instruction::Return {
				drop: 1,
				keep: 1,
			}
		]
	)
}

#[test]
fn add_params() {
	let code = compile(r#"
		(module
			(func (export "call") (param i32) (param i32) (result i32)
				get_local 0
				get_local 1
				i32.add
			)
		)
	"#);
	assert_eq!(
		code,
		vec![
			// This is tricky. Locals are now loaded from the stack. The load
			// happens from address relative of the current stack pointer. The first load
			// takes the value below the previous one (i.e the second argument) and then, it increments
			// the stack pointer. And then the same thing hapens with the value below the previous one
			// (which happens to be the value loaded by the first get_local).
			isa::Instruction::GetLocal(2),
			isa::Instruction::GetLocal(2),
			isa::Instruction::I32Add,
			isa::Instruction::Return {
				drop: 2,
				keep: 1,
			}
		]
	)
}

#[test]
fn drop_locals() {
	let code = compile(r#"
		(module
			(func (export "call") (param i32)
				(local i32)
				get_local 0
				set_local 1
			)
		)
	"#);
	assert_eq!(
		code,
		vec![
			isa::Instruction::GetLocal(2),
			isa::Instruction::SetLocal(1),
			isa::Instruction::Return {
				drop: 2,
				keep: 0,
			}
		]
	)
}

#[test]
fn if_without_else() {
	let code = compile(r#"
		(module
			(func (export "call") (param i32) (result i32)
				i32.const 1
				if
					i32.const 2
					return
				end
				i32.const 3
			)
		)
	"#);
	assert_eq!(
		code,
		vec![
			isa::Instruction::I32Const(1),
			isa::Instruction::BrIfEqz(isa::Target {
				dst_pc: 4,
				drop: 0,
				keep: 0,
			}),
			isa::Instruction::I32Const(2),
			isa::Instruction::Return {
				drop: 1, // 1 param
				keep: 1, // 1 result
			},
			isa::Instruction::I32Const(3),
			isa::Instruction::Return {
				drop: 1,
				keep: 1,
			},
		]
	)
}

#[test]
fn if_else() {
	let code = compile(r#"
		(module
			(func (export "call")
				(local i32)
				i32.const 1
				if
					i32.const 2
					set_local 0
				else
					i32.const 3
					set_local 0
				end
			)
		)
	"#);
	assert_eq!(
		code,
		vec![
			isa::Instruction::I32Const(1),
			isa::Instruction::BrIfEqz(isa::Target {
				dst_pc: 5,
				drop: 0,
				keep: 0,
			}),
			isa::Instruction::I32Const(2),
			isa::Instruction::SetLocal(1),
			isa::Instruction::Br(isa::Target {
				dst_pc: 7,
				drop: 0,
				keep: 0,
			}),
			isa::Instruction::I32Const(3),
			isa::Instruction::SetLocal(1),
			isa::Instruction::Return {
				drop: 1,
				keep: 0,
			},
		]
	)
}

#[test]
fn if_else_returns_result() {
	let code = compile(r#"
		(module
			(func (export "call")
				i32.const 1
				if (result i32)
					i32.const 2
				else
					i32.const 3
				end
				drop
			)
		)
	"#);
	assert_eq!(
		code,
		vec![
			isa::Instruction::I32Const(1),
			isa::Instruction::BrIfEqz(isa::Target {
				dst_pc: 4,
				drop: 0,
				keep: 0,
			}),
			isa::Instruction::I32Const(2),
			isa::Instruction::Br(isa::Target {
				dst_pc: 5,
				drop: 0,
				keep: 0,
			}),
			isa::Instruction::I32Const(3),
			isa::Instruction::Drop,
			isa::Instruction::Return {
				drop: 0,
				keep: 0,
			},
		]
	)
}

#[test]
fn if_else_branch_from_true_branch() {
	let code = compile(r#"
		(module
			(func (export "call")
				i32.const 1
				if (result i32)
					i32.const 1
					i32.const 1
					br_if 0
					drop
					i32.const 2
				else
					i32.const 3
				end
				drop
			)
		)
	"#);
	assert_eq!(
		code,
		vec![
			isa::Instruction::I32Const(1),
			isa::Instruction::BrIfEqz(isa::Target {
				dst_pc: 8,
				drop: 0,
				keep: 0,
			}),
			isa::Instruction::I32Const(1),
			isa::Instruction::I32Const(1),
			isa::Instruction::BrIfNez(isa::Target {
				dst_pc: 9,
				drop: 1, // TODO: Is this correct?
				keep: 1, // TODO: Is this correct?
			}),
			isa::Instruction::Drop,
			isa::Instruction::I32Const(2),
			isa::Instruction::Br(isa::Target {
				dst_pc: 9,
				drop: 0,
				keep: 0,
			}),
			isa::Instruction::I32Const(3),
			isa::Instruction::Drop,
			isa::Instruction::Return {
				drop: 0,
				keep: 0,
			},
		]
	)
}

#[test]
fn if_else_branch_from_false_branch() {
	let code = compile(r#"
		(module
			(func (export "call")
				i32.const 1
				if (result i32)
					i32.const 1
				else
					i32.const 2
					i32.const 1
					br_if 0
					drop
					i32.const 3
				end
				drop
			)
		)
	"#);
	assert_eq!(
		code,
		vec![
			isa::Instruction::I32Const(1),
			isa::Instruction::BrIfEqz(isa::Target {
				dst_pc: 4,
				drop: 0,
				keep: 0,
			}),
			isa::Instruction::I32Const(1),
			isa::Instruction::Br(isa::Target {
				dst_pc: 9,
				drop: 0,
				keep: 0,
			}),
			isa::Instruction::I32Const(2),
			isa::Instruction::I32Const(1),
			isa::Instruction::BrIfNez(isa::Target {
				dst_pc: 9,
				drop: 1, // TODO: Is this correct?
				keep: 1,
			}),
			isa::Instruction::Drop,
			isa::Instruction::I32Const(3),
			isa::Instruction::Drop,
			isa::Instruction::Return {
				drop: 0,
				keep: 0,
			},
		]
	)
}

#[test]
fn empty_loop() {
	let code = compile(r#"
		(module
			(func (export "call")
				loop (result i32)
					i32.const 1
					br_if 0
					i32.const 2
				end
				drop
			)
		)
	"#);
	assert_eq!(
		code,
		vec![
			isa::Instruction::I32Const(1),
			isa::Instruction::BrIfNez(isa::Target {
				dst_pc: 0,
				drop: 1,
				keep: 0,
			}),
			isa::Instruction::I32Const(2),
			isa::Instruction::Drop,
			isa::Instruction::Return {
				drop: 0,
				keep: 0,
			},
		]
	)
}

// TODO: Loop
// TODO: Empty loop?
// TODO: brtable

#[test]
fn wabt_example() {
	let code = compile(r#"
		(module
			(func (export "call") (param i32) (result i32)
				block $exit
					get_local 0
					br_if $exit
					i32.const 1
					return
				end
				i32.const 2
				return
			)
		)
	"#);
	assert_eq!(
		code,
		vec![
			isa::Instruction::GetLocal(1),
			isa::Instruction::BrIfNez(isa::Target {
				dst_pc: 4,
				keep: 0,
				drop: 1,
			}),
			isa::Instruction::I32Const(1),
			isa::Instruction::Return {
				drop: 1, // 1 parameter
				keep: 1, // return value
			},
			isa::Instruction::I32Const(2),
			isa::Instruction::Return {
				drop: 1,
				keep: 1,
			},
			isa::Instruction::Return {
				drop: 1,
				keep: 1,
			},
		]
	)
}
