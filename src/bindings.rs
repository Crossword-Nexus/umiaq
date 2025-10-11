use std::collections::HashMap;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

pub(crate) const WORD_SENTINEL: char = '*';

/// `Bindings` maps a variable name (char) to the string it's bound to.
/// Special variable `'*'` is reserved for the bound word.
///
/// Uses `Rc<str>` for values to avoid expensive string cloning in hot paths.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Bindings {
    map: HashMap<char, Rc<str>>,
}

impl Display for Bindings {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.map.iter().map(|(k, v)| format!("{k}→{v}")).collect::<Vec<_>>().join(", "))
    }
}

impl Bindings {
    /// Bind a variable to a value (takes ownership and converts to Rc<str>)
    pub(crate) fn set(&mut self, var_char: char, var_val: impl Into<Rc<str>>) {
        self.map.insert(var_char, var_val.into());
    }

    /// Bind a variable to an already-interned value (cheap clone of Rc)
    pub(crate) fn set_rc(&mut self, var_char: char, var_val: Rc<str>) {
        self.map.insert(var_char, var_val);
    }

    /// Retrieve the binding for a variable
    pub(crate) fn get(&self, var_char: char) -> Option<&Rc<str>> {
        self.map.get(&var_char)
    }

    /// Remove a binding for the given variable (if it exists)
    pub(crate) fn remove(&mut self, var_char: char) {
        self.map.remove(&var_char);
    }

    /// Assign the word binding to '*'
    pub(crate) fn set_word(&mut self, word: &str) {
        self.map.insert(WORD_SENTINEL, Rc::from(word));
    }

    /// Retrieve the bound word, if any
    #[must_use]
    pub fn get_word(&self) -> Option<&Rc<str>> {
        self.map.get(&WORD_SENTINEL)
    }

    /// Iterate over the bindings
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&char, &Rc<str>)> {
        self.map.iter()
    }

    pub(crate) fn contains_all_vars(&self, vars: &[char]) -> bool {
        vars.iter().all(|var_char| self.map.contains_key(var_char))
    }
}
