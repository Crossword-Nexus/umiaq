use std::fmt;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

pub(crate) const WORD_SENTINEL: char = '*';
const WORD_SENTINEL_INDEX: usize = 26;
const NUM_SLOTS: usize = 27; // 26 letters + 1 sentinel

/// `Bindings` maps a variable name (char) to the string it's bound to.
/// Special variable `'*'` is reserved for the bound word.
///
/// Uses `Rc<str>` for values to avoid expensive string cloning in hot paths.
/// Uses array-based storage instead of `HashMap` since variables are limited to 'A'-'Z' + '*'.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bindings {
    /// Array storage: index 0-25 for 'A'-'Z', index 26 for '*' (word sentinel)
    slots: [Option<Rc<str>>; NUM_SLOTS],
}

impl Default for Bindings {
    fn default() -> Self {
        Self {
            slots: [const { None }; NUM_SLOTS],
        }
    }
}

/// Convert a variable character to an array index
/// 'A' -> 0, 'B' -> 1, ..., 'Z' -> 25, '*' -> 26
///
/// # Safety
/// This function assumes the caller has validated that `c` is 'A'..='Z' or '*'.
/// Invalid characters indicate a bug in the parser/solver logic.
///
/// In debug builds, this will panic on invalid input to catch bugs early.
/// In release builds, invalid characters are logged and mapped to index 0 as a safe fallback.
#[inline]
fn char_to_index(c: char) -> usize {
    debug_assert!(
        matches!(c, 'A'..='Z' | '*'),
        "Invalid variable character: {c} (parser should have validated this)"
    );
    match c {
        'A'..='Z' => (c as u8 - b'A') as usize,
        '*' => WORD_SENTINEL_INDEX,
        // in release builds, handle invalid input gracefully to prevent crashes
        _ => {
            eprintln!("WARNING: Invalid variable character '{c}' in bindings (defaulting to index 0)");
            0 // TODO does this make sense?
        }
    }
}

impl Display for Bindings {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let pairs: Vec<String> = self.iter()
            .map(|(k, v)| format!("{k}→{v}"))
            .collect();
        write!(f, "[{}]", pairs.join(", "))
    }
}

impl Bindings {
    /// Bind a variable to an already-interned value (cheap clone of Rc)
    pub(crate) fn set_rc(&mut self, var_char: char, var_val: Rc<str>) {
        let i = char_to_index(var_char);
        self.slots[i] = Some(var_val);
    }

    /// Retrieve the binding for a variable
    pub(crate) fn get(&self, var_char: char) -> Option<&Rc<str>> {
        let i = char_to_index(var_char);
        self.slots[i].as_ref()
    }

    /// Remove a binding for the given variable (by setting slot to None)
    pub(crate) fn remove(&mut self, var_char: char) {
        let i = char_to_index(var_char);
        self.slots[i] = None;
    }

    /// Assign the word binding to '*'
    pub(crate) fn set_word(&mut self, word: &str) {
        self.slots[WORD_SENTINEL_INDEX] = Some(Rc::from(word));
    }

    /// Retrieve the bound word, if any
    #[must_use]
    pub fn get_word(&self) -> Option<&Rc<str>> {
        self.slots[WORD_SENTINEL_INDEX].as_ref()
    }

    /// Iterate over the bindings (returns owned char since we compute it from index)
    pub(crate) fn iter(&self) -> impl Iterator<Item = (char, &Rc<str>)> {
        self.slots.iter().enumerate().filter_map(|(i, opt)| {
            opt.as_ref().map(|val| {
                let c = if i == WORD_SENTINEL_INDEX {
                    WORD_SENTINEL
                } else {
                    (b'A' + i as u8) as char
                };
                (c, val)
            })
        })
    }

    pub(crate) fn contains_all_vars(&self, vars: &[char]) -> bool {
        vars.iter().all(|&var_char| {
            let i = char_to_index(var_char);
            self.slots[i].is_some()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bindings_set_and_get() {
        let mut b = Bindings::default();
        let val = Rc::from("test");
        b.set_rc('A', Rc::clone(&val));

        assert_eq!(b.get('A'), Some(&val));
        assert_eq!(b.get('B'), None);
    }

    #[test]
    fn test_bindings_remove() {
        let mut b = Bindings::default();
        b.set_rc('A', Rc::from("test"));
        assert!(b.get('A').is_some());

        b.remove('A');
        assert!(b.get('A').is_none());
    }

    #[test]
    fn test_bindings_word_sentinel() {
        let mut b = Bindings::default();
        b.set_word("hello");

        assert_eq!(b.get_word().map(|s| s.as_ref()), Some("hello"));
    }

    #[test]
    fn test_bindings_iter() {
        let mut b = Bindings::default();
        b.set_rc('A', Rc::from("alpha"));
        b.set_rc('B', Rc::from("beta"));
        b.set_word("word");

        let items: Vec<_> = b.iter().collect();
        assert_eq!(items.len(), 3); // A, B, and * (word sentinel)

        let has_a = items.iter().any(|(c, v)| *c == 'A' && v.as_ref() == "alpha");
        let has_b = items.iter().any(|(c, v)| *c == 'B' && v.as_ref() == "beta");
        let has_word = items.iter().any(|(c, v)| *c == WORD_SENTINEL && v.as_ref() == "word");

        assert!(has_a);
        assert!(has_b);
        assert!(has_word);
    }

    #[test]
    fn test_bindings_iter_empty() {
        let b = Bindings::default();
        let items: Vec<_> = b.iter().collect();
        assert!(items.is_empty());
    }

    #[test]
    fn test_contains_all_vars_true() {
        let mut b = Bindings::default();
        b.set_rc('A', Rc::from("a"));
        b.set_rc('B', Rc::from("b"));
        b.set_rc('C', Rc::from("c"));

        assert!(b.contains_all_vars(&['A', 'B']));
        assert!(b.contains_all_vars(&['A', 'B', 'C']));
        assert!(b.contains_all_vars(&['A']));
        assert!(b.contains_all_vars(&[]));
    }

    #[test]
    fn test_contains_all_vars_false() {
        let mut b = Bindings::default();
        b.set_rc('A', Rc::from("a"));
        b.set_rc('B', Rc::from("b"));

        assert!(!b.contains_all_vars(&['A', 'B', 'C']));
        assert!(!b.contains_all_vars(&['D']));
        assert!(!b.contains_all_vars(&['A', 'Z']));
    }

    #[test]
    fn test_bindings_all_26_variables() {
        let mut b = Bindings::default();

        for c in 'A'..='Z' {
            b.set_rc(c, Rc::from(c.to_string().to_lowercase()));
        }

        for c in 'A'..='Z' {
            assert_eq!(Some(Rc::from(c.to_string().to_lowercase())), b.get(c).cloned());
        }

        let items: Vec<_> = b.iter().collect();
        assert_eq!(items.len(), 26);
    }

    #[test]
    fn test_bindings_display() {
        let mut b = Bindings::default();
        b.set_rc('A', Rc::from("alpha"));
        b.set_rc('Z', Rc::from("zeta"));

        let display = format!("{}", b);
        assert!(display.contains("A→alpha"));
        assert!(display.contains("Z→zeta"));
    }

    #[test]
    fn test_bindings_clone() {
        let mut b1 = Bindings::default();
        b1.set_rc('A', Rc::from("test"));
        b1.set_word("word");

        let b2 = b1.clone();

        assert_eq!(b1.get('A'), b2.get('A'));
        assert_eq!(b1.get_word(), b2.get_word());

        // should share same Rc
        assert!(Rc::ptr_eq(b1.get('A').unwrap(), b2.get('A').unwrap()));
    }

    #[test]
    fn test_bindings_equality() {
        let mut b1 = Bindings::default();
        b1.set_rc('A', Rc::from("test"));

        let mut b2 = Bindings::default();
        b2.set_rc('A', Rc::from("test"));

        assert_eq!(b1, b2);
    }

    #[test]
    fn test_char_to_index_bounds() {
        // test that all valid chars map to valid indices
        for c in 'A'..='Z' {
            let i = char_to_index(c);
            assert!(i < 26, "Index {} for '{}' should be < 26", i, c);
        }

        assert_eq!(char_to_index(WORD_SENTINEL), WORD_SENTINEL_INDEX);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Invalid variable character")]
    fn test_char_to_index_invalid_debug() {
        char_to_index('a'); // lowercase should cause panic in debug builds
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_char_to_index_invalid_release() {
        // in release builds, invalid chars fall back to index 0 (variable 'A')
        let idx = char_to_index('a');
        assert_eq!(idx, 0, "Invalid char should fall back to index 0");
    }
}
