//! # Read-Write Lock (Writer-Priority)
//!
//! In this exercise, you will implement a **writer-priority** read-write lock from scratch using atomics.
//! Multiple readers may hold the lock concurrently; a writer holds it exclusively.
//!
//! **Note:** Rust's standard library already provides [`std::sync::RwLock`]. This exercise implements
//! a minimal version for learning the protocol and policy without using the standard one.
//!
//! ## Common policies for read-write locks
//! Different implementations can give different **priority** when both readers and writers are waiting:
//!
//! - **Reader-priority (读者优先)**: New readers are allowed to enter while a writer is waiting, so writers
//!   may be starved if readers keep arriving.
//! - **Writer-priority (写者优先)**: Once a writer is waiting, no new readers are admitted until that writer
//!   has run; this exercise implements this policy.
//! - **Read-write fair (读写公平)**: Requests are served in a fair order (e.g. FIFO or round-robin), so
//!   neither readers nor writers are systematically starved.
//!
//! ## Key Concepts
//! - **Readers**: share access; many threads can hold a read lock at once.
//! - **Writer**: exclusive access; only one writer, and no readers while the writer holds the lock.
//! - **Writer-priority (this implementation)**: when at least one writer is waiting, new readers block
//!   until the writer runs.
//!
//! ## State (single atomic)
//! We use one `AtomicU32`: low 20 bits count readers, the next 11 count waiting writers,
//! and the highest bit marks an active writer.
//! All logic is implemented with compare_exchange and load/store; no use of `std::sync::RwLock`.

use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU32, Ordering};

const READER_MASK: u32 = (1 << 20) - 1;
const ONE_WAITING_WRITER: u32 = 1 << 20;
const WRITER_HOLDING: u32 = 1 << 31;
const WAITING_MASK: u32 = (WRITER_HOLDING - 1) & !READER_MASK;

/// Writer-priority read-write lock. Implemented from scratch; does not use `std::sync::RwLock`.
pub struct RwLock<T> {
    state: AtomicU32,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for RwLock<T> {}
unsafe impl<T: Send + Sync> Sync for RwLock<T> {}

impl<T> RwLock<T> {
    pub const fn new(data: T) -> Self {
        Self {
            state: AtomicU32::new(0),
            data: UnsafeCell::new(data),
        }
    }

    /// Acquire a read lock. Blocks (spins) until no writer holds and no writer is waiting (writer-priority).
    pub fn read(&self) -> RwLockReadGuard<'_, T> {
        loop {
            let state = self.state.load(Ordering::Acquire);
            if state < READER_MASK
                && self.state
                    .compare_exchange(state, state + 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                return RwLockReadGuard { lock: self };
            }
            std::hint::spin_loop();
        }
    }

    /// Acquire the write lock. Blocks until no readers and no other writer.
    pub fn write(&self) -> RwLockWriteGuard<'_, T> {
        // Register before competing for the lock so new readers must wait.
        loop {
            let state = self.state.load(Ordering::Relaxed);
            if state & WAITING_MASK != WAITING_MASK
                && self.state
                    .compare_exchange(
                        state, state + ONE_WAITING_WRITER, Ordering::Relaxed, Ordering::Relaxed,
                    )
                    .is_ok()
            {
                break;
            }
            std::hint::spin_loop();
        }

        loop {
            let state = self.state.load(Ordering::Acquire);
            if state & (READER_MASK | WRITER_HOLDING) == 0 {
                let next = (state - ONE_WAITING_WRITER) | WRITER_HOLDING;
                if self.state
                    .compare_exchange(state, next, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return RwLockWriteGuard { lock: self };
                }
            }
            std::hint::spin_loop();
        }
    }
}

/// Guard for a read lock; releases the read lock on drop.
pub struct RwLockReadGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T> Deref for RwLockReadGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: The guard keeps writers out for the lifetime of the borrow.
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> Drop for RwLockReadGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.state.fetch_sub(1, Ordering::Release);
    }
}

/// Guard for a write lock; releases the write lock on drop.
pub struct RwLockWriteGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T> Deref for RwLockWriteGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: The write guard holds exclusive access to the data.
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> DerefMut for RwLockWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: The guard holds exclusive access and is mutably borrowed.
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T> Drop for RwLockWriteGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.state.fetch_and(!WRITER_HOLDING, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_multiple_readers() {
        let lock = Arc::new(RwLock::new(0u32));
        let mut handles = vec![];
        for _ in 0..10 {
            let l = Arc::clone(&lock);
            handles.push(thread::spawn(move || {
                let g = l.read();
                assert_eq!(*g, 0);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_writer_excludes_readers() {
        let lock = Arc::new(RwLock::new(0u32));
        let lock_w = Arc::clone(&lock);
        let writer = thread::spawn(move || {
            let mut g = lock_w.write();
            *g = 42;
        });
        writer.join().unwrap();
        let g = lock.read();
        assert_eq!(*g, 42);
    }

    #[test]
    fn test_concurrent_reads_after_write() {
        let lock = Arc::new(RwLock::new(Vec::<i32>::new()));
        {
            let mut g = lock.write();
            g.push(1);
            g.push(2);
        }
        let mut handles = vec![];
        for _ in 0..5 {
            let l = Arc::clone(&lock);
            handles.push(thread::spawn(move || {
                let g = l.read();
                assert_eq!(g.len(), 2);
                assert_eq!(&*g, &[1, 2]);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_writes_serialized() {
        let lock = Arc::new(RwLock::new(0u64));
        let mut handles = vec![];
        for _ in 0..10 {
            let l = Arc::clone(&lock);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let mut g = l.write();
                    *g += 1;
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(*lock.read(), 1000);
    }
}
