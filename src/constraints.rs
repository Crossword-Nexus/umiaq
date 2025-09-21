// constraints.rs
use crate::parser::ParsedForm;
use once_cell::sync::OnceCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Display;
use crate::errors::ParseError;

/// A collection of constraints for variables in a pattern-matching equation.
///
/// This wraps a `HashMap<char, VarConstraint>` where:
/// - Each `char` key is a variable name (e.g., 'A').
/// - The associated `VarConstraint` stores rules about what that variable can match.
#[derive(Debug, Clone, Default)]
pub struct VarConstraints {
    inner: HashMap<char, VarConstraint>,
}

impl VarConstraints {
    // Create a `VarConstraints` map whose internal map is `map`.
    //fn of(map: HashMap<char, VarConstraint>) -> Self { Self { inner: map } }

    /// Insert a complete `VarConstraint` for a variable.
    #[cfg(test)]
    pub(crate) fn insert(&mut self, var_char: char, constraint: VarConstraint) {
        self.inner.insert(var_char, constraint);
    }

    /// Ensure a variable has an entry; create a default constraint if missing.
    /// Returns a mutable reference so the caller can update it in place.
    pub(crate) fn ensure(&mut self, var_char: char) -> &mut VarConstraint {
        self.inner.entry(var_char).or_default()
    }

    pub(crate) fn bounds(&self, var_char: char) -> Bounds {
        self.get(var_char).map_or(Bounds::default(), |vc| vc.bounds)
    }

    /// Ensure an entry exists and return it mutably.
    pub fn ensure_entry_mut(&mut self, var_char: char) -> &mut VarConstraint {
        self.ensure(var_char)
    }

    /// Retrieve a read-only reference to the constraints for a variable, if any.
    pub(crate) fn get(&self, var_char: char) -> Option<&VarConstraint> {
        self.inner.get(&var_char)
    }

    // Iterate over `(variable, constraint)` pairs.
    //fn iter(&self) -> impl Iterator<Item = (&char, &VarConstraint)> { self.inner.iter() }

    // Convenience: number of variables with constraints.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.inner.len()
    }

    // Convenience: true if no constraints are stored.
    // fn is_empty(&self) -> bool { self.inner.is_empty() }
}

/// Pretty, deterministic display (sorted by variable) like:
/// `A: len=[2, 4], form=Some("a*"), not_equal={B,C}`
impl Display for VarConstraints {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut keys: Vec<_> = self.inner.keys().copied().collect();
        keys.sort_unstable();
        for (i, k) in keys.iter().enumerate() {
            let vc = &self.inner[k];
            if i > 0 { writeln!(f)?; }
            write!(f, "{k}: {vc}")?;
        }
        Ok(())
    }
}

/// Simple lower/upper bound for a variable's length.
///
/// - `min_len`: minimum length (always finite)
/// - `max_len_opt`: optional maximum length (`None` means unbounded)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub(crate) min_len: usize, // TODO? minimize direct assignment of min_len and/or max_len_opt (instead using Bounds::of(_no_max) whenever possible/sensible)
    pub(crate) max_len_opt: Option<usize>
}

impl Default for Bounds {
    fn default() -> Self {
        Bounds {
            min_len: VarConstraint::DEFAULT_MIN,
            max_len_opt: Option::default()
        }
    }
}

impl Bounds {
    pub(crate) fn of(min_len: usize, max_len: usize) -> Self {
        Bounds { min_len, max_len_opt: Some(max_len) }
    }

    pub(crate) fn of_unbounded(min_len: usize) -> Self {
        Bounds { min_len, max_len_opt: None }
    }

    // only set what the constraint explicitly provides
    pub(crate) fn constrain_by(&mut self, other: Bounds) -> Result<(), ParseError> {
        self.min_len = self.min_len.max(other.min_len);
        self.max_len_opt = self.max_len_opt
            .min(other.max_len_opt)
            .or(self.max_len_opt) // since None is treated as less than anything
            .or(other.max_len_opt); // since None is treated as less than anything

        // Check for contradictory bounds
        if let Some(mx) = self.max_len_opt && self.min_len > mx {
            return Err(ParseError::ContradictoryBounds { min: self.min_len, max: mx });
        }
        Ok(())
    }
}

impl Display for Bounds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.max_len_opt
            .map(|max_len| { format!("[{},{max_len}]", self.min_len) })
            .unwrap_or(format!("[{},∞)", self.min_len));
        write!(f, "{s}")
    }
}

/// A set of rules restricting what a single variable can match.
///
/// Fields are optional so that constraints can be partial:
/// - `bounds` limit how many characters the variable can bind to.
/// - `form` is an optional sub-pattern the variable's match must satisfy
///   (e.g., `"a*"` means "must start with `a`"; `"*z*"` means "must contain `z`").
/// - `not_equal` lists variables whose matches must *not* be identical to this one.
#[derive(Debug, Clone, Default)]
pub struct VarConstraint {
    pub bounds: Bounds,
    pub form: Option<String>,
    pub not_equal: HashSet<char>,
    pub parsed_form: OnceCell<ParsedForm>,
}

impl VarConstraint {
    pub(crate) const DEFAULT_MIN: usize = 1; // TODO!!!

    /// Set both min and max to the same exact length.
    pub(crate) fn set_exact_len(&mut self, len: usize) {
        self.bounds = Bounds::of(len, len);
    }
    /// Get the parsed form
    pub(crate) fn get_parsed_form(&self) -> Result<Option<&ParsedForm>, Box<ParseError>> {
        let parsed_form_raw = if let Some(f) = &self.form {
            let parsed = self.parsed_form.get_or_try_init(|| f.parse::<ParsedForm>())?;
            Some(parsed)
        } else {
            None
        };

        Ok(parsed_form_raw)
    }

    pub(crate) fn constrain_by(&mut self, other: &VarConstraint) -> Result<(), ParseError> {
        self.bounds.constrain_by(other.bounds)
    }
}

// Implement equality for VarConstraint
impl PartialEq for VarConstraint {
    fn eq(&self, other: &Self) -> bool {
        self.bounds == other.bounds
            && self.form == other.form
            && self.not_equal == other.not_equal
        // ignore parsed_form
    }
}

impl Eq for VarConstraint {}

/// Compact human-readable display for a single `VarConstraint`.
///
/// This is intended for debugging / logs, not for round-tripping.
/// It summarizes:
/// - the allowed length range (e.g., `[3,5]`, `[3,∞)`)
/// - the optional form string (or `*` if absent)
/// - the set of variables it must not equal, in sorted order
impl Display for VarConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Show the "form" string if present, otherwise `-`
        let form_str = self.form.as_deref().unwrap_or("*");

        // Collect the `not_equal` set into a sorted Vec<char> for stable output
        let mut ne: Vec<_> = self.not_equal.iter().copied().collect();
        ne.sort_unstable();
        // Turn it into a string: e.g., ['A','B','C'] → "ABC"
        let ne_str = if ne.is_empty() {
            "∅".to_string()
        } else {
            ne.into_iter().collect::<String>()
        };

        // Final compact output
        write!(f, "len={}; form={form_str}; not_equal={ne_str}", self.bounds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_creates_default() {
        let mut vcs = VarConstraints::default();
        assert!(vcs.get('A').is_none());
        {
            let a = vcs.ensure('A');
            // default created; tweak it
            a.bounds = Bounds::of_unbounded(3);
        }
        assert_eq!(3, vcs.get('A').unwrap().bounds.min_len);
        assert_eq!(1, vcs.len());
    }

    #[test]
    fn insert_and_get_roundtrip() {
        let mut vcs = VarConstraints::default();
        let vc = VarConstraint {
            form: Some("*z*".into()),
            not_equal: ['B', 'C'].into_iter().collect(),
            ..Default::default()
        };
        vcs.insert('A', vc.clone());
        assert_eq!(Some(&vc), vcs.get('A'));
    }

    #[test]
    fn display_var_constraint_is_stable() {
        let mut vc = VarConstraint {
            bounds: Bounds::of(2, 4),
            form: Some("a*".into()),
            ..Default::default()
        };
        vc.not_equal.extend(['C', 'B']); // out of order on purpose
        let shown = vc.to_string();
        // not_equal should be sorted -> {BC}
        assert_eq!("len=[2,4]; form=a*; not_equal=BC", shown);
    }

    #[test]
    fn display_var_constraints_multiline_sorted() {
        let mut vcs = VarConstraints::default();
        let a = VarConstraint { bounds: Bounds::of_unbounded(1), ..Default::default() };
        let b = VarConstraint { form: Some("*x*".into()), ..Default::default() };
        let c = VarConstraint { bounds: Bounds::of(1, 2), ..Default::default() };
        // Insert out of order to verify deterministic sort in Display
        vcs.insert('C', c);
        vcs.insert('A', a);
        vcs.insert('B', b);

        let s = vcs.to_string();
        let lines: Vec<_> = s.lines().collect();

        let expected = vec![
            "A: len=[1,∞); form=*; not_equal=∅",
            "B: len=[1,∞); form=*x*; not_equal=∅", // TODO!!! are we OK not distinguishing between "*" and "≥1"?
            "C: len=[1,2]; form=*; not_equal=∅" // TODO!!! are we OK not distinguishing between "≤2" and "1-2"?
        ];

        assert_eq!(expected, lines);
    }

    // constrain_by tests

    #[test]
    fn both_finite_overlap() {
        let mut a = Bounds::of(1, 5);
        let b = Bounds::of(3, 7);
        a.constrain_by(b).unwrap();
        assert_eq!(a, Bounds::of(3, 5));
    }

    #[test]
    fn both_finite_nested() {
        let mut a = Bounds::of(2, 8);
        let b = Bounds::of(3, 6);
        a.constrain_by(b).unwrap();
        assert_eq!(a, Bounds::of(3, 6));
    }

    #[test]
    fn left_finite_right_unbounded() {
        let mut a = Bounds::of(2, 6);
        let b = Bounds::of_unbounded(4);
        a.constrain_by(b).unwrap();
        assert_eq!(a, Bounds::of(4, 6));
    }

    #[test]
    fn left_unbounded_right_finite() {
        let mut a = Bounds::of_unbounded(5);
        let b = Bounds::of(3, 8);
        a.constrain_by(b).unwrap();
        assert_eq!(a, Bounds::of(5, 8));
    }

    #[test]
    fn both_unbounded() {
        let mut a = Bounds::of_unbounded(1);
        let b = Bounds::of_unbounded(4);
        a.constrain_by(b).unwrap();
        assert_eq!(a, Bounds::of_unbounded(4));
    }

    #[test]
    fn exact_length_intersection() {
        let mut a = Bounds::of(5, 5);
        let b = Bounds::of(3, 7);
        a.constrain_by(b).unwrap();
        assert_eq!(a, Bounds::of(5, 5));
    }

    #[test]
    fn impossible_interval_errs() {
        let mut a = Bounds::of(5, 5);
        let b = Bounds::of(10, 12);
        let result = a.constrain_by(b);
        assert!(matches!(result, Err(ParseError::ContradictoryBounds { .. })));
    }
}
