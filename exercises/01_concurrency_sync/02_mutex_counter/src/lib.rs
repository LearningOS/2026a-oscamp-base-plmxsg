//! # Mutex Shared State
//!
//! In this exercise, you will use `Arc<Mutex<T>>` to safely share and modify data between multiple threads.
//!
//! ## Concepts
//! - `Mutex<T>` mutex protects shared data
//! - `Arc<T>` atomic reference counting enables cross-thread sharing
//! - `lock()` acquires the lock and accesses data

use std::sync::{Arc, Mutex};
use std::thread;

/// Increment a counter concurrently using `n_threads` threads.
/// Each thread increments the counter `count_per_thread` times.
/// Returns the final counter value.
///
/// Hint: Use `Arc<Mutex<usize>>` as the shared counter.
pub fn concurrent_counter(n_threads: usize, count_per_thread: usize) -> usize {
    let counter = Arc::new(Mutex::new(0usize));
    let mut handles = Vec::with_capacity(n_threads);

    for _ in 0..n_threads {
        let counter = Arc::clone(&counter);
        handles.push(thread::spawn(move || {
            for _ in 0..count_per_thread {
                let mut value = counter.lock().unwrap();
                *value += 1;
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let result = *counter.lock().unwrap();
    result
}

/// Add elements to a shared vector concurrently using multiple threads.
/// Each thread pushes its own id (0..n_threads) to the vector.
/// Returns the sorted vector.
///
/// Hint: Use `Arc<Mutex<Vec<usize>>>`.
pub fn concurrent_collect(n_threads: usize) -> Vec<usize> {
    let values = Arc::new(Mutex::new(Vec::with_capacity(n_threads)));
    let mut handles = Vec::with_capacity(n_threads);

    for id in 0..n_threads {
        let values = Arc::clone(&values);
        handles.push(thread::spawn(move || {
            values.lock().unwrap().push(id);
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let mut result = std::mem::take(&mut *values.lock().unwrap());
    result.sort_unstable();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_single_thread() {
        assert_eq!(concurrent_counter(1, 100), 100);
    }

    #[test]
    fn test_counter_multi_thread() {
        assert_eq!(concurrent_counter(10, 100), 1000);
    }

    #[test]
    fn test_counter_zero() {
        assert_eq!(concurrent_counter(5, 0), 0);
    }

    #[test]
    fn test_collect() {
        let result = concurrent_collect(5);
        assert_eq!(result, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_collect_single() {
        assert_eq!(concurrent_collect(1), vec![0]);
    }
}
