//! An implementation of a `ByteBuf` based on virtual memory.
//!
//! This implementation uses `VirtualAlloc` and `VirtualFree` on Windows systems.
//!
//! It reserves virtual memory up to `WASM_MAX_PAGES` for efficiency.
//!
//! This implementation is several orders of magnitudes faster than `vec_memory` implementation.
//!
//! NOTE: Pages in this source file refer to wasm32 pages which are defined by the spec as 64KiB in size.

use winapi::shared::basetsd::SIZE_T;
use winapi::shared::minwindef::LPVOID;
use winapi::shared::ntdef::NULL;
use winapi::um::memoryapi::{VirtualAlloc, VirtualFree};
use winapi::um::winnt::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};

use std::ptr::NonNull;
use std::slice;

struct VAlloc {
    /// The pointer that points to the start of the mapping.
    ///
    /// This value doesn't change after creation.
    ptr: NonNull<u8>,
    /// The length of this mapping.
    ///
    /// Cannot be more than `isize::max_value()`. This value doesn't change after creation.
    len: usize,
}

const WASM_PAGE_SIZE: SIZE_T = 65536; // Size of a wasm32 page in bytes
const WASM_MAX_PAGES: SIZE_T = 65536 * WASM_PAGE_SIZE; // The maximum number of pages of a wasm32 program in bytes.

impl VAlloc {
    /// Reserves new pages up to `65536 * 64KiB` and commits initial pages in the range [0, initial).
    ///
    /// Returns `Err` if:
    /// - `len` should not exceed `isize::max_value()`
    /// - `len` should be greater than 0.
    /// - `VirtualAlloc` returns an error (almost certainly means out of memory).
    /// - `VirtualAlloc` returns an error when committing pages.
    /// - `initial pages` cannot be committed.
    fn new(len: usize) -> Result<Self, &'static str> {
        if len > isize::max_value() as usize {
            return Err("`len` should not exceed `isize::max_value()`");
        }

        if len == 0 {
            return Err("`len` should be greater than 0");
        }

        let ptr = unsafe {
            // Safety Proof:
            // There are not specific safety proofs are required for this call, since the call
            // by itself can't invoke any safety problems (however, misusing its result can).
            //
            // VirtualAlloc zeroes out allocated pages.
            VirtualAlloc(
                // `lpAddress` - let the system to choose the address at which to create the mapping.
                NULL,
                // `dwSize` - allocate the maximum number of wasm32 pages to bypass the overhead of rezising.
                WASM_MAX_PAGES,
                // `flAllocationType` - reserve pages so that they are not committed to memory immediately.
                MEM_RESERVE,
                // `flProtect` - apply READ WRITE !EXECUTE protection.
                PAGE_READWRITE,
            )
        };

        // Checking if there is an error with allocating memory pages.
        let base_ptr = match ptr {
            NULL => return Err("VirtualAlloc returned an error"),
            _ => ptr as *mut u8,
        };

        // Commit initial pages.
        let ptr = unsafe {
            // Even though we are committing, actual physical pages are not allocated until
            // they are accessed.
            //
            // Safety proof:
            // This should work once pages are successfully reserved.
            // Issue arise only with passing the wrong arguments or resource exhaustion.
            VirtualAlloc(
                // `lpAddress` - set the base address of pages to commit.
                base_ptr as LPVOID,
                // `dwSize` - set the length of pages to commit. This is the same as the initial or minimum.
                len,
                // `flAllocationType` - commit pages so that it can be read/written to.
                MEM_COMMIT,
                // `flProtect` - apply READ WRITE !EXECUTE protection.
                PAGE_READWRITE,
            )
        };

        // Checking if there is an error with committing memory pages.
        if ptr == NULL {
            return Err("VirtualAlloc couldn't commit initial pages");
        }

        let base_ptr = NonNull::new(base_ptr).ok_or("VirtualAlloc returned an error")?;

        Ok(Self { ptr: base_ptr, len })
    }

    /// Commits more pages
    fn grow(&mut self, new_len: usize) -> Result<(), &'static str> {
        // Pointer to memory base
        let base_ptr = self.ptr.as_ptr() as LPVOID;

        // Commit pages.
        let ptr = unsafe {
            // Even though we are committing, actual physical pages are not allocated until
            // they are accessed.
            //
            // Safety proof:
            // This should work once pages are successfully reserved.
            // Issue arise only with passing the wrong arguments or resource exhaustion.
            VirtualAlloc(
                // `lpAddress` - set the base address of pages to commit.
                // Overlapping committed pages are ignored.
                base_ptr,
                // `dwSize` - set the length of pages to commit. This is the same as the initial or minimum.
                new_len,
                // `flAllocationType` - commit pages so that it can be read/written to.
                MEM_COMMIT,
                // `flProtect` - apply READ WRITE !EXECUTE protection.
                PAGE_READWRITE,
            )
        };

        // Checking if there is an error with committing memory pages.
        if ptr == NULL {
            return Err("VirtualAlloc couldn't commit pages on grow");
        }

        // Update length.
        self.len = new_len;

        Ok(())
    }

    fn as_slice(&self) -> &[u8] {
        unsafe {
            // Safety Proof:
            // - Aliasing guarantees of `self.ptr` are not violated since `self` is the only owner.
            // - This pointer was allocated for `self.len` bytes and thus is a valid slice.
            // - `self.len` doesn't change throughout the lifetime of `self`.
            // - The value is returned valid for the duration of lifetime of `self`.
            //   `self` cannot be destroyed while the returned slice is alive.
            // - `self.ptr` is of `NonNull` type and thus `.as_ptr()` can never return NULL.
            // - `self.len` cannot be larger than `isize::max_value()`.
            slice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }

    fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe {
            // Safety Proof:
            // - See the proof for `Self::as_slice`
            // - Additionally, it is not possible to obtain two mutable references for `self.ptr`
            slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len)
        }
    }
}

impl Drop for VAlloc {
    fn drop(&mut self) {
        let ret_val = unsafe {
            // Safety proof:
            // - `self.ptr` was allocated by a call to `VirtualAlloc`.
            VirtualFree(
                // lpAddress - base address of pages to deallocate.
                self.ptr.as_ptr() as LPVOID,
                // dwSize - since address was provided by VirtualAlloc, we can provide 0 to deallocate pages allocated.
                0,
                // dwFreeType - frees pages allocated by VirtualAlloc.
                MEM_RELEASE,
            )
        };

        // `VirtualFree` can fail if the wrong arguments are passed, for example using MEM_RELEASE `dwFreeType`
        // with a non-zero `dwSize`.
        //
        // Asserting here to make sure we do not fail silently and leak memory when we can't free pages.
        //
        // VirtualFree is successful if it returns a non-zero value.
        assert_ne!(ret_val, 0, "VirtualFree failed");
    }
}

pub struct ByteBuf {
    region: Option<VAlloc>,
}

impl ByteBuf {
    pub fn new(len: usize) -> Result<Self, &'static str> {
        let region = if len == 0 {
            None
        } else {
            Some(VAlloc::new(len)?)
        };

        Ok(Self { region })
    }

    /// WebAssembly memory only grows and there is currently no shrink operator.
    /// This implementation, although named `realloc` (to remain compatible with existing code)
    /// really only grows based to the WebAssembly spec.
    ///
    /// This is intentional because `VAlloc` implementation reserves 2^32 bytes ahead of time for efficiency.
    /// And dropping and reallocating those pages defeats the purpose of reserving them ahead of time.
    ///
    /// With the above, any `new_len` lesser than the old `len` returns an error.
    pub fn realloc(&mut self, new_len: usize) -> Result<(), &'static str> {
        if let Some(ref mut region) = self.region {
            // When `new_len` is lesser or equal to current `region.len`, do nothing.
            // Even though that is currently not likely to happen because of prior validations
            // notably in `MemoryInstnce::grow`.
            //
            // Also `MemoryInstnce::grow` already makes sure `new_len` is not greater than
            // specified maximum or `WASM_MAX_PAGES`.
            if new_len > region.len {
                region.grow(new_len)?;
            }
        } else {
            // On the other hand, if region has not been allocated and new length
            // is greater than zero, then a new memory needs to be allocated.
            if new_len > 0 {
                self.region = Some(VAlloc::new(new_len)?);
            }
        }

        Ok(())
    }

    pub fn len(&self) -> usize {
        self.region.as_ref().map(|m| m.len).unwrap_or(0)
    }

    pub fn as_slice(&self) -> &[u8] {
        self.region.as_ref().map(|m| m.as_slice()).unwrap_or(&[])
    }

    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        self.region
            .as_mut()
            .map(|m| m.as_slice_mut())
            .unwrap_or(&mut [])
    }

    pub fn erase(&mut self) -> Result<(), &'static str> {
        let len = self.len();
        if len > 0 {
            // The order is important.
            //
            // 1. First we clear, and thus drop, the current region if any.
            // 2. And then we create a new one.
            //
            // Otherwise we double the peak memory consumption.
            self.region = None;
            self.region = Some(VAlloc::new(len)?);
        }
        Ok(())
    }
}
