use super::validate_module;
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

fn compile(wat: &str) -> Vec<isa::Instruction> {
	let wasm = wabt::wat2wasm(wat).unwrap();
	let module = deserialize_buffer::<Module>(&wasm).unwrap();
	let validated_module = validate_module(module).unwrap();
	let code = &validated_module.code_map[0];
	code.code.clone()
}

#[test]
fn explicit_return_no_value() {
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
fn explicit_return_with_value() {
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
fn explicit_return_param() {
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
