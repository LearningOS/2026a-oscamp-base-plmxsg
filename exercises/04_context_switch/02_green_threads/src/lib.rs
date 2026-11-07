//! # Green Thread Scheduler (riscv64)
//!
//! In this exercise, you build a simple cooperative (green) thread scheduler on top of context switching.
//! This crate is **riscv64 only**; run with the repo's normal flow (`./check.sh` / `oscamp`) or natively on riscv64.
//!
//! ## Key Concepts
//! - Cooperative vs preemptive scheduling
//! - Thread state: `Ready`, `Running`, `Finished`
//! - `yield_now()`: current thread voluntarily gives up the CPU
//! - Scheduler loop: pick next ready thread and switch to it
//!
//! ## Design
//! Each green thread has its own stack and `TaskContext`. Threads call `yield_now()` to yield.
//! The scheduler round-robins among ready threads. User entry is wrapped by `thread_wrapper`, which
//! calls the entry then marks the thread `Finished` and switches back.

#![cfg(target_arch = "riscv64")]

use core::arch::naked_asm;
use std::cell::Cell;

/// Per-thread stack size. Slightly larger to avoid overflow under QEMU / test harness.
const STACK_SIZE: usize = 1024 * 128;

/// Task context (riscv64); layout must match `01_stack_coroutine::TaskContext` and the asm below.
#[repr(C)]
#[derive(Debug, Default, Clone)]
pub struct TaskContext {
    sp: u64,
    ra: u64,
    s0: u64,
    s1: u64,
    s2: u64,
    s3: u64,
    s4: u64,
    s5: u64,
    s6: u64,
    s7: u64,
    s8: u64,
    s9: u64,
    s10: u64,
    s11: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThreadState {
    Ready,
    Running,
    Finished,
}

struct GreenThread {
    ctx: TaskContext,
    state: ThreadState,
    _stack: Option<Vec<u8>>,
    /// User entry; taken once when the thread is first scheduled and passed to `thread_wrapper`.
    entry: Option<extern "C" fn()>,
}

thread_local! {
    static CURRENT_THREAD_ENTRY: Cell<Option<extern "C" fn()>> = const { Cell::new(None) };
    static SCHEDULER: Cell<*mut Scheduler> = const { Cell::new(std::ptr::null_mut()) };
}

/// Run the user entry once, then return control to the scheduler.
extern "C" fn thread_wrapper() -> ! {
    if let Some(entry) = CURRENT_THREAD_ENTRY.with(Cell::take) {
        entry();
    }
    thread_finished();
    std::process::abort();
}

/// Save current callee-saved regs into `old`, load from `new`, then `ret` to `new.ra`.
/// Zero `a0`/`a1` before `ret` so we don't leak pointers into the new context.
///
/// Must be `#[unsafe(naked)]` to prevent the compiler from generating a prologue/epilogue.
#[unsafe(naked)]
unsafe extern "C" fn switch_context(_old: *mut TaskContext, _new: *const TaskContext) {
    naked_asm!(
        "sd sp, 0(a0)",
        "sd ra, 8(a0)",
        "sd s0, 16(a0)",
        "sd s1, 24(a0)",
        "sd s2, 32(a0)",
        "sd s3, 40(a0)",
        "sd s4, 48(a0)",
        "sd s5, 56(a0)",
        "sd s6, 64(a0)",
        "sd s7, 72(a0)",
        "sd s8, 80(a0)",
        "sd s9, 88(a0)",
        "sd s10, 96(a0)",
        "sd s11, 104(a0)",
        "ld sp, 0(a1)",
        "ld ra, 8(a1)",
        "ld s0, 16(a1)",
        "ld s1, 24(a1)",
        "ld s2, 32(a1)",
        "ld s3, 40(a1)",
        "ld s4, 48(a1)",
        "ld s5, 56(a1)",
        "ld s6, 64(a1)",
        "ld s7, 72(a1)",
        "ld s8, 80(a1)",
        "ld s9, 88(a1)",
        "ld s10, 96(a1)",
        "ld s11, 104(a1)",
        "li a0, 0",
        "li a1, 0",
        "ret",
    );
}

pub struct Scheduler {
    threads: Vec<GreenThread>,
    current: usize,
}

impl Scheduler {
    pub fn new() -> Self {
        let main_thread = GreenThread {
            ctx: TaskContext::default(),
            state: ThreadState::Running,
            _stack: None,
            entry: None,
        };

        Self {
            threads: vec![main_thread],
            current: 0,
        }
    }

    /// Register a new green thread that will run `entry` when first scheduled.
    ///
    /// 1. Allocate a stack of `STACK_SIZE` bytes; compute `stack_top` (high address).
    /// 2. Set up the context: `ra = thread_wrapper` so the first switch jumps to the wrapper;
    ///    `sp` must be 16-byte aligned (e.g. `(stack_top - 16) & !15` to leave headroom).
    /// 3. Push a `GreenThread` with this context, state `Ready`, and `entry` stored for the wrapper to call.
    pub fn spawn(&mut self, entry: extern "C" fn()) {
        let stack = vec![0u8; STACK_SIZE + 15];
        let stack_top = (stack.as_ptr() as usize + stack.len()) & !15;
        let ctx = TaskContext {
            sp: stack_top as u64,
            ra: thread_wrapper as *const () as u64,
            ..TaskContext::default()
        };
        self.threads.push(GreenThread {
            ctx,
            state: ThreadState::Ready,
            _stack: Some(stack),
            entry: Some(entry),
        });
    }

    /// Run until every green thread finishes.
    pub fn run(&mut self) {
        let scheduler = self as *mut Self;
        SCHEDULER.with(|slot| {
            assert!(slot.get().is_null(), "nested schedulers are unsupported");
            slot.set(scheduler);
        });
        unsafe {
            while (&(*scheduler).threads)[1..]
                .iter()
                .any(|thread| thread.state != ThreadState::Finished)
            {
                Self::schedule_next(scheduler);
            }
        }
        SCHEDULER.with(|slot| slot.set(std::ptr::null_mut()));
    }

    fn prepare_switch(&mut self) -> Option<(*mut TaskContext, *const TaskContext)> {
        let current = self.current;
        let count = self.threads.len();
        let next = (1..count)
            .map(|offset| (current + offset) % count)
            .find(|&index| self.threads[index].state == ThreadState::Ready)?;

        let threads = self.threads.as_mut_ptr();
        // SAFETY: The indices are in bounds and refer to distinct threads.
        unsafe {
            let old = threads.add(current);
            let new = threads.add(next);
            if (*old).state == ThreadState::Running {
                (*old).state = ThreadState::Ready;
            }
            (*new).state = ThreadState::Running;
            self.current = next;
            let entry = (*new).entry.take();
            CURRENT_THREAD_ENTRY.with(|slot| slot.set(entry));
            Some((&raw mut (*old).ctx, &raw const (*new).ctx))
        }
    }

    unsafe fn schedule_next(scheduler: *mut Self) {
        // End the mutable borrow before switching stacks.
        let contexts = unsafe { (&mut *scheduler).prepare_switch() };
        if let Some((old, new)) = contexts {
            unsafe { switch_context(old, new) };
        }
    }
}

/// Voluntarily let the next ready thread run.
pub fn yield_now() {
    let scheduler = SCHEDULER.with(Cell::get);
    if !scheduler.is_null() {
        unsafe { Scheduler::schedule_next(scheduler) };
    }
}

/// Mark the current thread as finished and switch away.
fn thread_finished() {
    let scheduler = SCHEDULER.with(Cell::get);
    unsafe {
        let sched = &mut *scheduler;
        sched.threads[sched.current].state = ThreadState::Finished;
        Scheduler::schedule_next(scheduler);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Mutex;

    /// Tests must run serially: the scheduler uses global state (SCHEDULER, CURRENT_THREAD_ENTRY).
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    static EXEC_ORDER: AtomicU32 = AtomicU32::new(0);

    extern "C" fn task_a() {
        EXEC_ORDER.fetch_add(1, Ordering::SeqCst);
        yield_now();
        EXEC_ORDER.fetch_add(10, Ordering::SeqCst);
        yield_now();
        EXEC_ORDER.fetch_add(100, Ordering::SeqCst);
    }

    extern "C" fn task_b() {
        EXEC_ORDER.fetch_add(1, Ordering::SeqCst);
        yield_now();
        EXEC_ORDER.fetch_add(10, Ordering::SeqCst);
    }

    #[test]
    fn test_scheduler_runs_all() {
        let _guard = TEST_LOCK.lock().unwrap();
        EXEC_ORDER.store(0, Ordering::SeqCst);

        let mut sched = Scheduler::new();
        sched.spawn(task_a);
        sched.spawn(task_b);
        sched.run();

        let got = EXEC_ORDER.load(Ordering::SeqCst);
        if got != 122 {
            panic!(
                "EXEC_ORDER: expected 122, got {} (run with --nocapture to see stderr)",
                got
            );
        }
    }

    static SIMPLE_FLAG: AtomicU32 = AtomicU32::new(0);

    extern "C" fn simple_task() {
        SIMPLE_FLAG.store(42, Ordering::SeqCst);
    }

    #[test]
    fn test_single_thread() {
        let _guard = TEST_LOCK.lock().unwrap();
        SIMPLE_FLAG.store(0, Ordering::SeqCst);

        let mut sched = Scheduler::new();
        sched.spawn(simple_task);
        sched.run();

        assert_eq!(SIMPLE_FLAG.load(Ordering::SeqCst), 42);
    }
}
