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
#[inline]
fn char_to_index(c: char) -> usize {
    match c {
        'A'..='Z' => (c as u8 - b'A') as usize,
        '*' => WORD_SENTINEL_INDEX,
        _ => panic!("Invalid variable character: {c}"),
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
