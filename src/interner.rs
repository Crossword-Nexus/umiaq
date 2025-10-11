//! String interning to avoid redundant allocations for common variable values.
//!
//! Since variable bindings often repeat (e.g., "a", "b", "ch"), we maintain a
//! global cache of interned strings. Multiple `Rc<str>` pointers can share the
//! same underlying allocation.

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

thread_local! {
    /// Thread-local interner cache.
    /// We use thread_local instead of a global Mutex since the solver is single-threaded.
    static INTERNER: RefCell<HashMap<String, Rc<str>>> = RefCell::new(HashMap::new());
}

/// Intern a string, returning an `Rc<str>`.
///
/// If the string has been interned before, returns a cheap clone of the existing `Rc`.
/// Otherwise, allocates a new `Rc<str>` and caches it.
pub fn intern(s: impl AsRef<str>) -> Rc<str> {
    let s = s.as_ref();

    INTERNER.with(|interner| {
        let mut cache = interner.borrow_mut();

        // Check if already interned
        if let Some(existing) = cache.get(s) {
            return Rc::clone(existing);
        }

        // Not found - create new Rc and cache it
        let rc: Rc<str> = Rc::from(s);
        cache.insert(s.to_string(), Rc::clone(&rc));
        rc
    })
}

/// Clear the interner cache (useful for testing or memory management)
#[cfg(test)]
pub fn clear() {
    INTERNER.with(|interner| {
        interner.borrow_mut().clear();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern_same_string_returns_same_rc() {
        clear();

        let s1 = intern("hello");
        let s2 = intern("hello");

        // Same underlying allocation - Rc::ptr_eq checks pointer equality
        assert!(Rc::ptr_eq(&s1, &s2));
    }

    #[test]
    fn test_intern_different_strings() {
        clear();

        let s1 = intern("hello");
        let s2 = intern("world");

        assert!(!Rc::ptr_eq(&s1, &s2));
        assert_eq!(s1.as_ref(), "hello");
        assert_eq!(s2.as_ref(), "world");
    }

    #[test]
    fn test_intern_with_string() {
        clear();

        let owned = String::from("test");
        let rc1 = intern(&owned);
        let rc2 = intern("test");

        assert!(Rc::ptr_eq(&rc1, &rc2));
    }
}
