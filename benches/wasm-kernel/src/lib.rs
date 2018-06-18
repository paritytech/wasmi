#![no_std]
#![feature(lang_items)]
#![feature(core_intrinsics)]
#![feature(panic_implementation)]

extern crate rlibc;
extern crate tiny_keccak;

use tiny_keccak::Keccak;

mod rev_complement;

pub struct TinyKeccakTestData {
	data: &'static [u8],
	result: &'static mut [u8],
}

#[no_mangle]
pub extern "C" fn prepare_tiny_keccak() -> *const TinyKeccakTestData {
	static DATA: [u8; 4096] = [254u8; 4096];
	static mut RESULT: [u8; 32] = [0u8; 32];

	static mut TEST_DATA: Option<TinyKeccakTestData> = None;

	unsafe {
		if let None = TEST_DATA {
			TEST_DATA = Some(TinyKeccakTestData {
				data: &DATA,
				result: &mut RESULT,
			});
		}
		TEST_DATA.as_ref().unwrap() as *const TinyKeccakTestData
	}
}

#[no_mangle]
pub extern "C" fn bench_tiny_keccak(test_data: *const TinyKeccakTestData) {
	unsafe {
		let mut keccak = Keccak::new_keccak256();
		keccak.update((*test_data).data);
		keccak.finalize((*test_data).result);
	}
}

pub struct RevComplementTestData {
	input: Box<[u8]>,
	output: Box<[u8]>,
}

#[no_mangle]
pub extern "C" fn prepare_rev_complement(size: usize) -> *mut RevComplementTestData {
	let input = vec![0; size];
	let output = vec![0; size];

	let test_data = Box::new(
		RevComplementTestData {
			input: input.into_boxed_slice(),
			output: output.into_boxed_slice(),
		}
	);

	// Basically leak the pointer to the test data. This shouldn't be harmful since `prepare` is called
	// only once per bench run (not for the iteration), and afterwards whole memory instance is discarded.
	Box::into_raw(test_data)
}

#[no_mangle]
pub extern "C" fn rev_complement_input_ptr(test_data: *mut RevComplementTestData) -> *mut u8 {
	unsafe {
		(*test_data).input.as_mut_ptr()
	}
}

#[no_mangle]
pub extern "C" fn rev_complement_output_ptr(test_data: *mut RevComplementTestData) -> *const u8 {
	unsafe {
		(*test_data).output.as_mut_ptr()
	}
}

#[no_mangle]
pub extern "C" fn bench_rev_complement(test_data: *mut RevComplementTestData) {
	unsafe {
		let result = rev_complement::run(&*(*test_data).input);
		(*test_data).output.copy_from_slice(&result);
	}
}
