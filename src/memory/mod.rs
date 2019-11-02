use alloc::{rc::Rc, string::ToString, vec::Vec};
use core::{
    cell::{Cell, RefCell},
    cmp, fmt,
    ops::Range,
    u32,
};
use memory_units::{Bytes, Pages, RoundUpTo};
use parity_wasm::elements::ResizableLimits;
use value::LittleEndianConvert;
use Error;

#[cfg(all(unix, not(feature = "vec_memory")))]
#[path = "mmap_bytebuf.rs"]
mod bytebuf;

#[cfg(all(windows, not(feature = "vec_memory")))]
#[path = "virtualalloc_bytebuf.rs"]
mod bytebuf;

#[cfg(any(all(not(windows), not(unix)), feature = "vec_memory"))]
#[path = "vec_bytebuf.rs"]
mod bytebuf;

use self::bytebuf::ByteBuf;

/// Size of a page of [linear memory][`MemoryInstance`] - 64KiB.
///
/// The size of a memory is always a integer multiple of a page size.
///
/// [`MemoryInstance`]: struct.MemoryInstance.html
pub const LINEAR_MEMORY_PAGE_SIZE: Bytes = Bytes(65536);

/// Reference to a memory (See [`MemoryInstance`] for details).
///
/// This reference has a reference-counting semantics.
///
/// [`MemoryInstance`]: struct.MemoryInstance.html
///
#[derive(Clone, Debug)]
pub struct MemoryRef(Rc<MemoryInstance>);

impl ::core::ops::Deref for MemoryRef {
    type Target = MemoryInstance;
    fn deref(&self) -> &MemoryInstance {
        &self.0
    }
}

/// Runtime representation of a linear memory (or `memory` for short).
///
/// A memory is a contiguous, mutable array of raw bytes. Wasm code can load and store values
/// from/to a linear memory at any byte address.
/// A trap occurs if an access is not within the bounds of the current memory size.
///
/// A memory is created with an initial size but can be grown dynamically.
/// The growth can be limited by specifying maximum size.
/// The size of a memory is always a integer multiple of a [page size][`LINEAR_MEMORY_PAGE_SIZE`] - 64KiB.
///
/// At the moment, wasm doesn't provide any way to shrink the memory.
///
/// [`LINEAR_MEMORY_PAGE_SIZE`]: constant.LINEAR_MEMORY_PAGE_SIZE.html
pub struct MemoryInstance {
    /// Memory limits.
    limits: ResizableLimits,
    /// Linear memory buffer with lazy allocation.
    buffer: RefCell<ByteBuf>,
    initial: Pages,
    current_size: Cell<usize>,
    maximum: Option<Pages>,
}

impl fmt::Debug for MemoryInstance {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("MemoryInstance")
            .field("limits", &self.limits)
            .field("buffer.len", &self.buffer.borrow().len())
            .field("maximum", &self.maximum)
            .field("initial", &self.initial)
            .finish()
    }
}

struct CheckedRegion {
    offset: usize,
    size: usize,
}

impl CheckedRegion {
    fn range(&self) -> Range<usize> {
        self.offset..self.offset + self.size
    }

    fn intersects(&self, other: &Self) -> bool {
        let low = cmp::max(self.offset, other.offset);
        let high = cmp::min(self.offset + self.size, other.offset + other.size);

        low < high
    }
}

impl MemoryInstance {
    /// Allocate a memory instance.
    ///
    /// The memory allocated with initial number of pages specified by `initial`.
    /// Minimal possible value for `initial` is 0 and maximum possible is `65536`.
    /// (Since maximum addressible memory is 2<sup>32</sup> = 4GiB = 65536 * [64KiB][`LINEAR_MEMORY_PAGE_SIZE`]).
    ///
    /// It is possible to limit maximum number of pages this memory instance can have by specifying
    /// `maximum`. If not specified, this memory instance would be able to allocate up to 4GiB.
    ///
    /// Allocated memory is always zeroed.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    ///
    /// - `initial` is greater than `maximum`
    /// - either `initial` or `maximum` is greater than `65536`.
    ///
    /// [`LINEAR_MEMORY_PAGE_SIZE`]: constant.LINEAR_MEMORY_PAGE_SIZE.html
    pub fn alloc(initial: Pages, maximum: Option<Pages>) -> Result<MemoryRef, Error> {
        {
            use core::convert::TryInto;
            let initial_u32: u32 = initial.0.try_into().map_err(|_| {
                Error::Memory(format!("initial ({}) can't be coerced to u32", initial.0))
            })?;
            let maximum_u32: Option<u32> = match maximum {
                Some(maximum_pages) => Some(maximum_pages.0.try_into().map_err(|_| {
                    Error::Memory(format!(
                        "maximum ({}) can't be coerced to u32",
                        maximum_pages.0
                    ))
                })?),
                None => None,
            };
            validation::validate_memory(initial_u32, maximum_u32).map_err(Error::Memory)?;
        }

        let memory = MemoryInstance::new(initial, maximum)?;
        Ok(MemoryRef(Rc::new(memory)))
    }

    /// Create new linear memory instance.
    fn new(initial: Pages, maximum: Option<Pages>) -> Result<Self, Error> {
        let limits = ResizableLimits::new(initial.0 as u32, maximum.map(|p| p.0 as u32));

        let initial_size: Bytes = initial.into();
        Ok(MemoryInstance {
            limits: limits,
            buffer: RefCell::new(
                ByteBuf::new(initial_size.0).map_err(|err| Error::Memory(err.to_string()))?,
            ),
            initial: initial,
            current_size: Cell::new(initial_size.0),
            maximum: maximum,
        })
    }

    /// Return linear memory limits.
    pub(crate) fn limits(&self) -> &ResizableLimits {
        &self.limits
    }

    /// Returns number of pages this `MemoryInstance` was created with.
    pub fn initial(&self) -> Pages {
        self.initial
    }

    /// Returns maximum amount of pages this `MemoryInstance` can grow to.
    ///
    /// Returns `None` if there is no limit set.
    /// Maximum memory size cannot exceed `65536` pages or 4GiB.
    pub fn maximum(&self) -> Option<Pages> {
        self.maximum
    }

    /// Returns current linear memory size.
    ///
    /// Maximum memory size cannot exceed `65536` pages or 4GiB.
    ///
    /// # Example
    ///
    /// To convert number of pages to number of bytes you can use the following code:
    ///
    /// ```rust
    /// use wasmi::MemoryInstance;
    /// use wasmi::memory_units::*;
    ///
    /// let memory = MemoryInstance::alloc(Pages(1), None).unwrap();
    /// let byte_size: Bytes = memory.current_size().into();
    /// assert_eq!(
    ///     byte_size,
    ///     Bytes(65536),
    /// );
    /// ```
    pub fn current_size(&self) -> Pages {
        Bytes(self.buffer.borrow().len()).round_up_to()
    }

    /// Get value from memory at given offset.
    pub fn get_value<T: LittleEndianConvert>(&self, offset: u32) -> Result<T, Error> {
        let mut buffer = self.buffer.borrow_mut();
        let region =
            self.checked_region(&mut buffer, offset as usize, ::core::mem::size_of::<T>())?;
        Ok(
            T::from_little_endian(&buffer.as_slice_mut()[region.range()])
                .expect("Slice size is checked"),
        )
    }

    /// Copy data from memory at given offset.
    ///
    /// This will allocate vector for you.
    /// If you can provide a mutable slice you can use [`get_into`].
    ///
    /// [`get_into`]: #method.get_into
    pub fn get(&self, offset: u32, size: usize) -> Result<Vec<u8>, Error> {
        let mut buffer = self.buffer.borrow_mut();
        let region = self.checked_region(&mut buffer, offset as usize, size)?;

        Ok(buffer.as_slice_mut()[region.range()].to_vec())
    }

    /// Copy data from given offset in the memory into `target` slice.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the specified region is out of bounds.
    pub fn get_into(&self, offset: u32, target: &mut [u8]) -> Result<(), Error> {
        let mut buffer = self.buffer.borrow_mut();
        let region = self.checked_region(&mut buffer, offset as usize, target.len())?;

        target.copy_from_slice(&buffer.as_slice_mut()[region.range()]);

        Ok(())
    }

    /// Copy data in the memory at given offset.
    pub fn set(&self, offset: u32, value: &[u8]) -> Result<(), Error> {
        let mut buffer = self.buffer.borrow_mut();
        let range = self
            .checked_region(&mut buffer, offset as usize, value.len())?
            .range();

        buffer.as_slice_mut()[range].copy_from_slice(value);

        Ok(())
    }

    /// Copy value in the memory at given offset.
    pub fn set_value<T: LittleEndianConvert>(&self, offset: u32, value: T) -> Result<(), Error> {
        let mut buffer = self.buffer.borrow_mut();
        let range = self
            .checked_region(&mut buffer, offset as usize, ::core::mem::size_of::<T>())?
            .range();
        value.into_little_endian(&mut buffer.as_slice_mut()[range]);
        Ok(())
    }

    /// Increases the size of the linear memory by given number of pages.
    /// Returns previous memory size if succeeds.
    ///
    /// # Errors
    ///
    /// Returns `Err` if attempted to allocate more memory than permited by the limit.
    pub fn grow(&self, additional: Pages) -> Result<Pages, Error> {
        let size_before_grow: Pages = self.current_size();

        if additional == Pages(0) {
            return Ok(size_before_grow);
        }
        if additional > Pages(65536) {
            return Err(Error::Memory(format!(
                "Trying to grow memory by more than 65536 pages"
            )));
        }

        let new_size: Pages = size_before_grow + additional;
        let maximum = self
            .maximum
            .unwrap_or(Pages(validation::LINEAR_MEMORY_MAX_PAGES as usize));
        if new_size > maximum {
            return Err(Error::Memory(format!(
                "Trying to grow memory by {} pages when already have {}",
                additional.0, size_before_grow.0,
            )));
        }

        let new_buffer_length: Bytes = new_size.into();
        self.buffer
            .borrow_mut()
            .realloc(new_buffer_length.0)
            .map_err(|err| Error::Memory(err.to_string()))?;

        self.current_size.set(new_buffer_length.0);

        Ok(size_before_grow)
    }

    fn checked_region(
        &self,
        buffer: &mut ByteBuf,
        offset: usize,
        size: usize,
    ) -> Result<CheckedRegion, Error> {
        let end = offset.checked_add(size).ok_or_else(|| {
            Error::Memory(format!(
                "trying to access memory block of size {} from offset {}",
                size, offset
            ))
        })?;

        if end > buffer.len() {
            return Err(Error::Memory(format!(
                "trying to access region [{}..{}] in memory [0..{}]",
                offset,
                end,
                buffer.len()
            )));
        }

        Ok(CheckedRegion {
            offset: offset,
            size: size,
        })
    }

    fn checked_region_pair(
        &self,
        buffer: &mut ByteBuf,
        offset1: usize,
        size1: usize,
        offset2: usize,
        size2: usize,
    ) -> Result<(CheckedRegion, CheckedRegion), Error> {
        let end1 = offset1.checked_add(size1).ok_or_else(|| {
            Error::Memory(format!(
                "trying to access memory block of size {} from offset {}",
                size1, offset1
            ))
        })?;

        let end2 = offset2.checked_add(size2).ok_or_else(|| {
            Error::Memory(format!(
                "trying to access memory block of size {} from offset {}",
                size2, offset2
            ))
        })?;

        if end1 > buffer.len() {
            return Err(Error::Memory(format!(
                "trying to access region [{}..{}] in memory [0..{}]",
                offset1,
                end1,
                buffer.len()
            )));
        }

        if end2 > buffer.len() {
            return Err(Error::Memory(format!(
                "trying to access region [{}..{}] in memory [0..{}]",
                offset2,
                end2,
                buffer.len()
            )));
        }

        Ok((
            CheckedRegion {
                offset: offset1,
                size: size1,
            },
            CheckedRegion {
                offset: offset2,
                size: size2,
            },
        ))
    }

    /// Copy contents of one memory region to another.
    ///
    /// Semantically equivalent to `memmove`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if either of specified regions is out of bounds.
    pub fn copy(&self, src_offset: usize, dst_offset: usize, len: usize) -> Result<(), Error> {
        let mut buffer = self.buffer.borrow_mut();

        let (read_region, write_region) =
            self.checked_region_pair(&mut buffer, src_offset, len, dst_offset, len)?;

        unsafe {
            ::core::ptr::copy(
                buffer.as_slice()[read_region.range()].as_ptr(),
                buffer.as_slice_mut()[write_region.range()].as_mut_ptr(),
                len,
            )
        }

        Ok(())
    }

    /// Copy contents of one memory region to another (non-overlapping version).
    ///
    /// Semantically equivalent to `memcpy`.
    /// but returns Error if source overlaping with destination.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    ///
    /// - either of specified regions is out of bounds,
    /// - these regions overlaps.
    pub fn copy_nonoverlapping(
        &self,
        src_offset: usize,
        dst_offset: usize,
        len: usize,
    ) -> Result<(), Error> {
        let mut buffer = self.buffer.borrow_mut();

        let (read_region, write_region) =
            self.checked_region_pair(&mut buffer, src_offset, len, dst_offset, len)?;

        if read_region.intersects(&write_region) {
            return Err(Error::Memory(format!(
                "non-overlapping copy is used for overlapping regions"
            )));
        }

        unsafe {
            ::core::ptr::copy_nonoverlapping(
                buffer.as_slice()[read_region.range()].as_ptr(),
                buffer.as_slice_mut()[write_region.range()].as_mut_ptr(),
                len,
            )
        }

        Ok(())
    }

    /// Copy memory between two (possibly distinct) memory instances.
    ///
    /// If the same memory instance passed as `src` and `dst` then usual `copy` will be used.
    pub fn transfer(
        src: &MemoryRef,
        src_offset: usize,
        dst: &MemoryRef,
        dst_offset: usize,
        len: usize,
    ) -> Result<(), Error> {
        if Rc::ptr_eq(&src.0, &dst.0) {
            // `transfer` is invoked with with same source and destination. Let's assume that regions may
            // overlap and use `copy`.
            return src.copy(src_offset, dst_offset, len);
        }

        // Because memory references point to different memory instances, it is safe to `borrow_mut`
        // both buffers at once (modulo `with_direct_access_mut`).
        let mut src_buffer = src.buffer.borrow_mut();
        let mut dst_buffer = dst.buffer.borrow_mut();

        let src_range = src
            .checked_region(&mut src_buffer, src_offset, len)?
            .range();
        let dst_range = dst
            .checked_region(&mut dst_buffer, dst_offset, len)?
            .range();

        dst_buffer.as_slice_mut()[dst_range].copy_from_slice(&src_buffer.as_slice()[src_range]);

        Ok(())
    }

    /// Fill the memory region with the specified value.
    ///
    /// Semantically equivalent to `memset`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the specified region is out of bounds.
    pub fn clear(&self, offset: usize, new_val: u8, len: usize) -> Result<(), Error> {
        let mut buffer = self.buffer.borrow_mut();

        let range = self.checked_region(&mut buffer, offset, len)?.range();

        for val in &mut buffer.as_slice_mut()[range] {
            *val = new_val
        }
        Ok(())
    }

    /// Fill the specified memory region with zeroes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the specified region is out of bounds.
    pub fn zero(&self, offset: usize, len: usize) -> Result<(), Error> {
        self.clear(offset, 0, len)
    }

    /// Set every byte in the entire linear memory to 0, preserving its size.
    ///
    /// Might be useful for some optimization shenanigans.
    pub fn erase(&self) -> Result<(), Error> {
        self.buffer
            .borrow_mut()
            .erase()
            .map_err(|err| Error::Memory(err.to_string()))
    }

    /// Provides direct access to the underlying memory buffer.
    ///
    /// # Panics
    ///
    /// Any call that requires write access to memory (such as [`set`], [`clear`], etc) made within
    /// the closure will panic.
    ///
    /// [`set`]: #method.get
    /// [`clear`]: #method.set
    pub fn with_direct_access<R, F: FnOnce(&[u8]) -> R>(&self, f: F) -> R {
        let buf = self.buffer.borrow();
        f(buf.as_slice())
    }

    /// Provides direct mutable access to the underlying memory buffer.
    ///
    /// # Panics
    ///
    /// Any calls that requires either read or write access to memory (such as [`get`], [`set`], [`copy`], etc) made
    /// within the closure will panic. Proceed with caution.
    ///
    /// [`get`]: #method.get
    /// [`set`]: #method.set
    pub fn with_direct_access_mut<R, F: FnOnce(&mut [u8]) -> R>(&self, f: F) -> R {
        let mut buf = self.buffer.borrow_mut();
        f(buf.as_slice_mut())
    }
}

#[cfg(test)]
mod tests {

    use super::{MemoryInstance, MemoryRef, LINEAR_MEMORY_PAGE_SIZE};
    use memory_units::Pages;
    use std::rc::Rc;
    use Error;

    #[test]
    fn alloc() {
        let mut fixtures = vec![
            (0, None, true),
            (0, Some(0), true),
            (1, None, true),
            (1, Some(1), true),
            (0, Some(1), true),
            (1, Some(0), false),
        ];

        #[cfg(target_pointer_width = "64")]
        fixtures.extend(&[
            (65536, Some(65536), true),
            (65536, Some(0), false),
            (65536, None, true),
        ]);

        for (index, &(initial, maybe_max, expected_ok)) in fixtures.iter().enumerate() {
            let initial: Pages = Pages(initial);
            let maximum: Option<Pages> = maybe_max.map(|m| Pages(m));
            let result = MemoryInstance::alloc(initial, maximum);
            if result.is_ok() != expected_ok {
                panic!(
                    "unexpected error at {}, initial={:?}, max={:?}, expected={}, result={:?}",
                    index, initial, maybe_max, expected_ok, result,
                );
            }
        }
    }

    #[test]
    fn ensure_page_size() {
        use memory_units::ByteSize;
        assert_eq!(LINEAR_MEMORY_PAGE_SIZE, Pages::byte_size());
    }

    fn create_memory(initial_content: &[u8]) -> MemoryInstance {
        let mem = MemoryInstance::new(Pages(1), Some(Pages(1))).unwrap();
        mem.set(0, initial_content)
            .expect("Successful initialize the memory");
        mem
    }

    #[test]
    fn copy_overlaps_1() {
        let mem = create_memory(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        mem.copy(0, 4, 6).expect("Successfully copy the elements");
        let result = mem.get(0, 10).expect("Successfully retrieve the result");
        assert_eq!(result, &[0, 1, 2, 3, 0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn copy_overlaps_2() {
        let mem = create_memory(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        mem.copy(4, 0, 6).expect("Successfully copy the elements");
        let result = mem.get(0, 10).expect("Successfully retrieve the result");
        assert_eq!(result, &[4, 5, 6, 7, 8, 9, 6, 7, 8, 9]);
    }

    #[test]
    fn copy_nonoverlapping() {
        let mem = create_memory(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        mem.copy_nonoverlapping(0, 10, 10)
            .expect("Successfully copy the elements");
        let result = mem.get(10, 10).expect("Successfully retrieve the result");
        assert_eq!(result, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn copy_nonoverlapping_overlaps_1() {
        let mem = create_memory(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let result = mem.copy_nonoverlapping(0, 4, 6);
        match result {
            Err(Error::Memory(_)) => {}
            _ => panic!("Expected Error::Memory(_) result, but got {:?}", result),
        }
    }

    #[test]
    fn copy_nonoverlapping_overlaps_2() {
        let mem = create_memory(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let result = mem.copy_nonoverlapping(4, 0, 6);
        match result {
            Err(Error::Memory(_)) => {}
            _ => panic!("Expected Error::Memory(_), but got {:?}", result),
        }
    }

    #[test]
    fn transfer_works() {
        let src = MemoryRef(Rc::new(create_memory(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])));
        let dst = MemoryRef(Rc::new(create_memory(&[
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        ])));

        MemoryInstance::transfer(&src, 4, &dst, 0, 3).unwrap();

        assert_eq!(src.get(0, 10).unwrap(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            dst.get(0, 10).unwrap(),
            &[4, 5, 6, 13, 14, 15, 16, 17, 18, 19]
        );
    }

    #[test]
    fn transfer_still_works_with_same_memory() {
        let src = MemoryRef(Rc::new(create_memory(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])));

        MemoryInstance::transfer(&src, 4, &src, 0, 3).unwrap();

        assert_eq!(src.get(0, 10).unwrap(), &[4, 5, 6, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn transfer_oob_with_same_memory_errors() {
        let src = MemoryRef(Rc::new(create_memory(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])));
        assert!(MemoryInstance::transfer(&src, 65535, &src, 0, 3).is_err());

        // Check that memories content left untouched
        assert_eq!(src.get(0, 10).unwrap(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn transfer_oob_errors() {
        let src = MemoryRef(Rc::new(create_memory(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])));
        let dst = MemoryRef(Rc::new(create_memory(&[
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        ])));

        assert!(MemoryInstance::transfer(&src, 65535, &dst, 0, 3).is_err());

        // Check that memories content left untouched
        assert_eq!(src.get(0, 10).unwrap(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            dst.get(0, 10).unwrap(),
            &[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        );
    }

    #[test]
    fn clear() {
        let mem = create_memory(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        mem.clear(0, 0x4A, 10)
            .expect("To successfully clear the memory");
        let result = mem.get(0, 10).expect("To successfully retrieve the result");
        assert_eq!(result, &[0x4A; 10]);
    }

    #[test]
    fn get_into() {
        let mem = MemoryInstance::new(Pages(1), None).unwrap();
        mem.set(6, &[13, 17, 129])
            .expect("memory set should not fail");

        let mut data = [0u8; 2];
        mem.get_into(7, &mut data[..])
            .expect("get_into should not fail");

        assert_eq!(data, [17, 129]);
    }

    #[test]
    fn zero_copy() {
        let mem = MemoryInstance::alloc(Pages(1), None).unwrap();
        mem.set(100, &[0]).expect("memory set should not fail");
        mem.with_direct_access_mut(|buf| {
            assert_eq!(
                buf.len(),
                65536,
                "the buffer length is expected to be 1 page long"
            );
            buf[..10].copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        });
        mem.with_direct_access(|buf| {
            assert_eq!(
                buf.len(),
                65536,
                "the buffer length is expected to be 1 page long"
            );
            assert_eq!(&buf[..10], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        });
    }

    #[should_panic]
    #[test]
    fn zero_copy_panics_on_nested_access() {
        let mem = MemoryInstance::alloc(Pages(1), None).unwrap();
        let mem_inner = mem.clone();
        mem.with_direct_access(move |_| {
            let _ = mem_inner.set(0, &[11, 12, 13]);
        });
    }
}
