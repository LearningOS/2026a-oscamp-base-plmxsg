//! # Free-List Allocator
//!
//! Building on the bump allocator, implement a Free-List Allocator that supports memory reclamation.
//!
//! ## How It Works
//!
//! A Free-List Allocator uses a linked list to track all freed memory blocks.
//! On allocation, it first searches the list for a suitable block (first-fit strategy);
//! if none is found, it falls back to allocating from the unused region.
//! On deallocation, the block is inserted at the head of the list.
//!
//! ```text
//! free_list -> [block A: 64B] -> [block B: 128B] -> [block C: 32B] -> null
//! ```
//!
//! Each free block stores a `FreeBlock` struct at its head (containing block size and next pointer).
//!
//! ## Task
//!
//! Implement `FreeListAllocator`'s `alloc` and `dealloc` methods:
//!
//! ### alloc
//! 1. Traverse the free_list, find the first block with `size >= layout.size()` and proper alignment (first-fit)
//! 2. If found, remove it from the list and return it
//! 3. If not found, allocate from the `bump` region (same as bump allocator)
//!
//! ### dealloc
//! 1. Write `FreeBlock` header info at the freed block
//! 2. Insert it at the head of free_list
//!
//! ## Key Concepts
//!
//! - Intrusive linked list
//! - `*mut T` read/write: `ptr.write(val)` / `ptr.read()`
//! - Memory alignment checks

#![cfg_attr(not(test), no_std)]

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Free block header, stored at the beginning of each free memory block.
struct FreeBlock {
    size: usize,
    next: *mut FreeBlock,
}

pub struct FreeListAllocator {
    heap_end: usize,
    bump_next: AtomicUsize,
    list_locked: AtomicBool,
    free_list: UnsafeCell<*mut FreeBlock>,
}

// SAFETY: The heap is exclusively managed by this allocator. Bump updates
// are atomic, and all accesses to the free list and its nodes are locked.
unsafe impl Send for FreeListAllocator {}
unsafe impl Sync for FreeListAllocator {}

impl FreeListAllocator {
    /// # Safety
    /// The heap must be a valid, writable region reserved exclusively for this
    /// allocator and must remain available for the allocator's lifetime.
    pub unsafe fn new(heap_start: usize, heap_end: usize) -> Self {
        Self {
            heap_end,
            bump_next: AtomicUsize::new(heap_start),
            list_locked: AtomicBool::new(false),
            free_list: UnsafeCell::new(null_mut()),
        }
    }

    fn lock_list(&self) {
        while self.list_locked.compare_exchange_weak(
            false, true, Ordering::Acquire, Ordering::Relaxed,
        ).is_err() {
            core::hint::spin_loop();
        }
    }
}

unsafe impl GlobalAlloc for FreeListAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size().max(core::mem::size_of::<FreeBlock>());
        let align = layout.align().max(core::mem::align_of::<FreeBlock>());
        let mask = align - 1;

        self.lock_list();
        let mut link = self.free_list.get();
        // SAFETY: The lock protects the head and all linked free blocks.
        unsafe {
            while !(*link).is_null() {
                let block = *link;
                if block.addr() & mask == 0 && (*block).size >= size {
                    *link = (*block).next;
                    self.list_locked.store(false, Ordering::Release);
                    return block.cast();
                }
                link = core::ptr::addr_of_mut!((*block).next);
            }
        }
        self.list_locked.store(false, Ordering::Release);

        let mut current = self.bump_next.load(Ordering::Relaxed);
        loop {
            let Some(rounded) = current.checked_add(mask) else {
                return null_mut();
            };
            let aligned = rounded & !mask;
            let Some(end) = aligned.checked_add(size) else {
                return null_mut();
            };
            if end > self.heap_end {
                return null_mut();
            }
            match self.bump_next.compare_exchange_weak(
                current, end, Ordering::Relaxed, Ordering::Relaxed,
            ) {
                Ok(_) => return aligned as *mut u8,
                Err(observed) => current = observed,
            }
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = layout.size().max(core::mem::size_of::<FreeBlock>());
        let block = ptr.cast::<FreeBlock>();

        self.lock_list();
        // SAFETY: The allocation is large and aligned enough for the header.
        // The caller relinquishes the block, and the lock protects the list.
        unsafe {
            let head = self.free_list.get();
            block.write(FreeBlock { size, next: *head });
            *head = block;
        }
        self.list_locked.store(false, Ordering::Release);
    }
}

// ============================================================
// Tests
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    const HEAP_SIZE: usize = 4096;

    fn make_allocator() -> (FreeListAllocator, Vec<u8>) {
        let mut heap = vec![0u8; HEAP_SIZE];
        let start = heap.as_mut_ptr() as usize;
        let alloc = unsafe { FreeListAllocator::new(start, start + HEAP_SIZE) };
        (alloc, heap)
    }

    #[test]
    fn test_alloc_basic() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(32, 8).unwrap();
        let ptr = unsafe { alloc.alloc(layout) };
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_alloc_alignment() {
        let (alloc, _heap) = make_allocator();
        for align in [1, 2, 4, 8, 16] {
            let layout = Layout::from_size_align(8, align).unwrap();
            let ptr = unsafe { alloc.alloc(layout) };
            assert!(!ptr.is_null());
            assert_eq!(ptr as usize % align, 0, "align={align}");
        }
    }

    #[test]
    fn test_dealloc_and_reuse() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(64, 8).unwrap();

        let p1 = unsafe { alloc.alloc(layout) };
        assert!(!p1.is_null());

        // After freeing, the next allocation should reuse the same block
        unsafe { alloc.dealloc(p1, layout) };
        let p2 = unsafe { alloc.alloc(layout) };
        assert!(!p2.is_null());
        assert_eq!(p1, p2, "should reuse the freed block");
    }

    #[test]
    fn test_multiple_alloc_dealloc() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(128, 8).unwrap();

        let p1 = unsafe { alloc.alloc(layout) };
        let p2 = unsafe { alloc.alloc(layout) };
        let p3 = unsafe { alloc.alloc(layout) };
        assert!(!p1.is_null() && !p2.is_null() && !p3.is_null());

        unsafe { alloc.dealloc(p2, layout) };
        unsafe { alloc.dealloc(p1, layout) };

        let q1 = unsafe { alloc.alloc(layout) };
        let q2 = unsafe { alloc.alloc(layout) };
        assert!(!q1.is_null() && !q2.is_null());
    }

    #[test]
    fn test_oom() {
        let (alloc, _heap) = make_allocator();
        let layout = Layout::from_size_align(HEAP_SIZE + 1, 1).unwrap();
        let ptr = unsafe { alloc.alloc(layout) };
        assert!(ptr.is_null(), "should return null when exceeding heap");
    }
}
