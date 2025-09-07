use crate::constraints::{VarConstraint, VarConstraints};
use fancy_regex::Regex;
use std::cmp::Reverse;
use std::collections::HashSet;
use std::str::FromStr;
use std::sync::LazyLock;
use crate::comparison_operator::ComparisonOperator;
use crate::errors::ParseError;
use crate::parser::ParsedForm;
use crate::umiaq_char::UmiaqChar;

/// The character that separates forms, in an equation
pub const FORM_SEPARATOR: char = ';';

/// Matches comparative length constraints like `|A|>4`, `|A|<=7`, etc.
/// (Whitespace is permitted around operator.)
static LEN_CMP_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\|([A-Z])\|\s*(<=|>=|=|<|>)\s*(\d+)$").unwrap());

/// Matches inequality constraints like `!=AB`
static NEQ_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^!=([A-Z]+)$").unwrap());

/// Matches complex constraints like `A=(3-5:a*)` with length and/or pattern
// syntax:
//
// complex constraint expression = {variable name}={constraint}
// variable name = A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X | Y | Z
// constraint = ({inner constraint})
//            | inner_constraint
// inner_constraint = {length range}:{literal string}
//                  | {length range}
//                  | {literal string}
// length range = {number}-{number}
//              | {number}-
//              | -{number}
//              | {number}
// number = {digit}
//        | {digit}{number}
// digit = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
// literal string = {literal string component}
//                | {literal string component}{literal string}
// literal string component = {literal string character}
//                          | {dot char}
//                          | {star char}
//                          | {vowel char}
//                          | {consonant char}
//                          | {charset string}
//                          | {anagram string}
// literal string char = a | b | c | d | e | f | g | h | i | j | k | l | m | n | o | p | q | r | s | t | u | v | w | x | y | z
// dot char = .
// star char = *
// vowel char = @
// consonant char = #
// charset string = [{one or more literal string chars}]
// one or more literal string chars = {literal string char}
//               | {literal string char}{charset chars}
// anagram string = /{one or more literal string chars}
#[derive(Debug, Clone)]
/// A single raw form string plus *solver metadata*; **not tokenized**.
/// Use `parse_equation(&pattern.raw_string)` to get `Vec<FormPart>`.
///
/// ## Solver metadata (what it is and why it exists)
/// - `lookup_keys`: `HashSet<char>`
///   - **What:** The subset of this form's variables that also appear in forms
///     that have already been placed earlier in `Patterns::ordered_list`.
///   - **When it's set:** Assigned by `Patterns::ordered_patterns()` *after* the
///     forms have been reordered for solving.
///   - **Why it helps:** During the multi-form join, candidate bindings for this
///     form can be bucketed by the concrete values of these variables and then
///     matched in O(1)/O(log N) time against earlier choices, instead of scanning
///     all candidates. In other words, `lookup_keys` is the *join key* that lets
///     you intersect partial solutions cheaply.
///   - **How it's used:** When you collect matches for each form, you can index
///     (e.g., `HashMap<JoinKey, Vec<Bindings>>`) by the values of `lookup_keys`.
///     Then, when recursing, you fetch only the compatible bucket for the next form.
///
/// Note: `Pattern` intentionally keeps `raw_string` (e.g., "AB", "A~A", "/sett")
/// unparsed; tokenization to `Vec<FormPart>` is deferred to matching time.
///
/// Example:
/// - Input: `"ABC;BC;C"`
/// - Reordering picks `"ABC"` first, then `"BC"`, then `"C"`.
/// - `lookup_keys`:
///     * for `"ABC"`: `None` (first form has nothing prior)
///     * for `"BC"`: `Some({'B','C'})` (overlap with already-chosen variables)
///     * for `"C"`:  `Some({'C'})`
pub struct Pattern {
    /// The raw string representation of the pattern, such as "AB" or "/triangle"
    pub raw_string: String,
    /// Set of variable names that this pattern shares with previously processed ones,
    /// used for optimizing lookups in recursive solving
    pub lookup_keys: HashSet<char>,
    /// Position of this form among *forms only* in the original (display) order.
    /// This is stable and survives reordering/cloning.
    pub original_index: usize,
    /// Determine whether the string is deterministic. Created on init.
    pub(crate) is_deterministic: bool,
    /// The set of variables present in the pattern
    pub(crate) variables: HashSet<char>,
}

/// Implementation for the `Pattern` struct, representing a single pattern string
/// and utilities to extract its variable set.
impl Pattern {
    /// Constructs a new `Pattern` from any type that can be converted into a `String`.
    /// The resulting `lookup_keys` is initialized to `None`.
    /// `original_index` is the index for this `Pattern`'s position in the original equation.
    fn create(string: impl Into<String>, original_index: usize) -> Self {
        let raw_string = string.into();
        // Determine if the pattern is deterministic
        let deterministic = &raw_string.parse::<ParsedForm>()
            .map(|parts| { parts.iter().all(|p| { p.is_deterministic() }) })
            .unwrap_or(false);

        // Get the variables involved
        let vars = raw_string
            .chars()
            .filter(char::is_variable)
            .collect();

        Self {
            raw_string,
            lookup_keys: HashSet::default(),
            original_index,
            is_deterministic: *deterministic,
            variables: vars,
        }
    }

    /// True iff every variable this pattern uses is included in its `lookup_keys`.
    pub(crate) fn all_vars_in_lookup_keys(&self) -> bool {
        self.variables.is_subset(&self.lookup_keys)
    }

    /// Weights for different pattern parts when computing constraint score.
    const SCORE_LITERAL: usize = 3;
    const SCORE_CLASS:   usize = 1; // for @ and #
    const SCORE_DEFAULT: usize = 0;

    /// Get the "constraint score" (name?) of a pattern
    /// The more literals and @# it has, the more constrained it is
    fn constraint_score(&self) -> usize {
        let s = &self.raw_string;
        s.chars()
            .map(|c| {
                if c.is_literal() {
                    Self::SCORE_LITERAL
                } else if c == '@' || c == '#' {
                    Self::SCORE_CLASS
                } else {
                    Self::SCORE_DEFAULT
                }
            })
            .sum()
    }
}

#[derive(Debug, Default)]
/// The **parsed equation** at a structural level: extracted constraints + collected forms +
/// a solver-friendly order. Forms here are still raw strings; tokenize each with
/// `parse_equation` when matching.
///
/// - `list`: all non-constraint forms in original order
/// - `var_constraints`: per-variable rules parsed from things like `|A|=5`, `!=AB`,
///   `A=(3-5:a*)`
/// - `ordered_list`: `list` reordered so that forms with many variables appear
///   earlier and subsequent forms maximize overlap with already-chosen variables.
///   As part of this step, each later form's `lookup_keys` is set to the overlap
///   with the variables seen so far (its *join key*).
///
/// TODO change the name of this struct (since it contains (a list of) `Pattern`s... but also more)
pub struct Patterns {
    /// List of patterns directly extracted from the input string (not constraints)
    // TODO should we keep Vec<Pattern> for each order or just one (likely ordered_list) and use map
    //      (original_to_ordered) when other is needed?
    pub p_list: Vec<Pattern>,
    /// Map of variable names (A-Z) to their associated constraints
    pub var_constraints: VarConstraints,
    /// Reordered list of patterns, optimized for solving (most-constrained first)
    pub ordered_list: Vec<Pattern>,        // solver order
    /// ordered index -> original index
    pub ordered_to_original: Vec<usize>,
    /// original index -> ordered index
    pub original_to_ordered: Vec<usize>,
}

impl Patterns {
    fn build_order_maps(&mut self) {
        let n = self.p_list.len();
        self.ordered_to_original = self
            .ordered_list
            .iter()
            .map(|p| p.original_index)
            .collect();

        self.original_to_ordered = vec![usize::MAX; n];
        for (ordered_ix, &orig_ix) in self.ordered_to_original.iter().enumerate() {
            self.original_to_ordered[orig_ix] = ordered_ix;
        }
    }

    /// Parses the input string into constraint entries and pattern entries.
    /// Recognizes:
    /// - exact length (e.g., `|A|=5`)
    /// - inequality constraints (e.g., `!=AB`)
    /// - complex constraints (e.g., length + pattern) (e.g., `A=(3-5:a*)`)
    ///
    /// Non-constraint entries are added to `self.list` as actual patterns.
    fn set_var_constraints(&mut self, input: &str) {
        let forms: Vec<&str> = input.split(FORM_SEPARATOR).collect();
        // Iterate through all parts of the input string, split by `;`

        let mut next_form_ix = 0; // counts only *forms* we accept

        for form in &forms {
            if let Some(cap) = LEN_CMP_RE.captures(form).unwrap() {
                let var = cap[1].chars().next().unwrap();
                let op  = ComparisonOperator::from_str(&cap[2]).unwrap(); // TODO better error handling
                let n   = cap[3].parse::<usize>().unwrap();
                let vc  = self.var_constraints.ensure(var);

                match op {
                    ComparisonOperator::EQ => vc.set_exact_len(n),
                    ComparisonOperator::NE => {}
                    ComparisonOperator::LE => vc.max_length = Some(n),
                    ComparisonOperator::GE => vc.min_length = n,
                    ComparisonOperator::LT => vc.max_length = n.checked_sub(1),   // n-1 (None if n==0)
                    ComparisonOperator::GT => vc.min_length = n + 1, // TODO? check for overflow
                }
            } else if let Some(cap) = NEQ_RE.captures(form).unwrap() {
                // Extract all variables from inequality constraint
                // !=α (where α is a string of at least 2 distinct variables) means that any pair of
                //     variables in α are not equal
                // Examples:
                // * !=AB means A != B
                // * !=ABC means A != B, A != C, B != C
                let vars: Vec<char> = cap[1].chars().collect();
                for &v in &vars {
                    let var_constraint = self.var_constraints.ensure(v);
                    var_constraint.not_equal = vars.iter().copied().filter(|&x| x != v).collect();
                }
            } else if let Ok((var, cc_vc)) = get_complex_constraint(form) {
                let var_constraint = self.var_constraints.ensure(var);

                // TODO! test instances where neither min_length is none and where neither max_length is none
                // only set what the constraint explicitly provides
                var_constraint.min_length = var_constraint.min_length.max(cc_vc.min_length);
                var_constraint.max_length = var_constraint.max_length.min(cc_vc.max_length).or(var_constraint.max_length).or(cc_vc.max_length);
                if let Some(f) = cc_vc.form {
                    if let Some(old_form) = &var_constraint.form {
                        if *old_form != f {
                            // TODO error? somehow combine forms?
                        }
                    } else {
                        var_constraint.form = Some(f);
                    }
                }
            } else { // TODO? avoid swallowing error?
                // We only want to add a form if it is parseable
                // Specifically, things like |AB|=7 should not be picked up here
                // TODO do we check for those separately?
                // TODO avoid calling parse_form twice on the same form? (here and in solve_equation)
                if let Ok(_parsed) = form.parse::<ParsedForm>() {
                    self.p_list.push(Pattern::create(*form, next_form_ix));
                    next_form_ix += 1;
                } else {
                    // TODO throw exception
                }
            }
        }
    }

    // TODO is this the right way to order things?
    /// Reorders the list of patterns to improve solving efficiency.
    /// First selects the pattern with the most variables,
    /// then repeatedly selects the next pattern with the most overlap with those already chosen.
    /// This helps early patterns prune the solution space.
    fn ordered_patterns(&self) -> Vec<Pattern> {
        let mut p_list = self.p_list.clone();
        let mut ordered = Vec::with_capacity(p_list.len());

        // Reusable tiebreak tail: (constraint_score desc, deterministic asc, original_index asc)
        // Note: Reverse(bool) makes false > true under max_by_key, i.e., ascending by bool.
        let tie_tail = |p: &Pattern| (p.constraint_score(), Reverse(p.is_deterministic), p.original_index);

        // First pick: most variables; tiebreak by tail.
        let first_ix = p_list
            .iter()
            .enumerate()
            .max_by_key(|(_, p)| (p.variables.len(), tie_tail(p)))
            .map(|(i, _)| i)
            .unwrap();

        let first = p_list.remove(first_ix);
        ordered.push(first);

        while !p_list.is_empty() {
            // Vars already "seen"
            let found_vars: HashSet<char> = ordered
                .iter()
                .flat_map(|p| p.variables.iter().copied())
                .collect();

            // Next pick: minimize difference; tiebreak by tail.
            let (ix, mut next_p) = p_list
                .iter()
                .enumerate()
                .min_by_key(|(_, p)| {
                    let var_diff = p.variables.difference(&found_vars).count();
                    (var_diff, tie_tail(p))
                })
                .map(|(i, p)| (i, p.clone()))
                .unwrap();

            // Assign join keys for the chosen pattern
            let lookup_keys: HashSet<char> = next_p.variables
                .intersection(&found_vars)
                .copied()
                .collect();
            next_p.lookup_keys = lookup_keys;

            p_list.remove(ix);
            ordered.push(next_p);
        }

        ordered
    }

    /// Number of forms (from `ordered_list`)
    pub(crate) fn len(&self) -> usize {
        self.ordered_list.len()
    }

    /// Iterate over forms in solver-friendly order
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, Pattern> {
        self.ordered_list.iter()
    }

    /* -- Several unused functions but maybe someday?
     *    /// Convenience (often handy with `len`)
     *    fn is_empty(&self) -> bool {
     *        self.ordered_list.is_empty()
     *    }
     *
     *    /// Iterate in original (display) order
     *    pub(crate) fn iter_original(&self) -> std::slice::Iter<'_, Pattern> {
     *        self.list.iter()
     *    }
     *
     *    /// Map a solver index to the original index
     *    pub(crate) fn original_ix(&self, ordered_ix: usize) -> usize {
     *        self.ordered_to_original[ordered_ix]
     *    }
     *
     *    /// Map an original index to the solver index (if it was placed)
     *    pub(crate) fn ordered_ix(&self, original_ix: usize) -> Option<usize> {
     *        let ix = self.original_to_ordered.get(original_ix).copied()?;
     *        (ix != usize::MAX).then_some(ix)
     *    }
     */
}

impl FromStr for Patterns {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut patterns = Patterns::default();
        patterns.set_var_constraints(s);
        patterns.ordered_list = patterns.ordered_patterns();
        // populate original_to_ordered and ordered_to_original
        patterns.build_order_maps();
        Ok(patterns)
    }
}

// TODO? do this via regex?
// e.g., A=(3-:x*)
fn get_complex_constraint(form: &str) -> Result<(char, VarConstraint), Box<ParseError>> {
    let top_parts = form.split('=').collect::<Vec<_>>();
    if top_parts.len() != 2 {
        return Err(Box::new(ParseError::InvalidComplexConstraint { str: format!("expected 1 equals sign (not {})", top_parts.len()) }));
    }

    let var_str = top_parts[0];
    if var_str.len() != 1 {
        return Err(Box::new(ParseError::InvalidComplexConstraint { str: format!("expected 1 character (as the variable) to the left of \"=\" (not {})", var_str.len()) }));
    }

    let var = var_str.chars().next().unwrap();

    let constraint_str = top_parts[1].to_string();

    // remove outer parentheses if they are there
    let inner_constraint_str = if constraint_str.starts_with('(') && constraint_str.ends_with(')') {
        let mut chars = constraint_str.chars();
        chars.next();
        chars.next_back();
        chars.as_str()
    } else {
        constraint_str.as_str()
    };

    let constraint_halves = inner_constraint_str.split(':').collect::<Vec<_>>();
    let (len_range, literal_constraint_str) = match constraint_halves.len() {
        2 => {
            let len_range = parse_length_range(constraint_halves[0])?;
            (Some(len_range), Some(constraint_halves[1]))
        },
        1 => {
            match parse_length_range(constraint_halves[0]) {
                Ok(len_range) => (Some(len_range), None),
                Err(_) => (None, Some(constraint_halves[0]))
            }
        }
        _ => return Err(Box::new(ParseError::InvalidComplexConstraint { str: format!("too many colons--0 or 1 expected (not {})", constraint_halves.len() - 1) }))
    };

    let vc = VarConstraint {
        // TODO!!!? instead of `len_range` as `Option<(usize, Option<usize>)`, maybe create a richer
        // type--say `LenRange`--instead of `(usize, Option<usize>)` whose default is (equiv. to)
        // `(VarConstraint::DEFAULT_MIN, None)`... and then we'd avoid the outer `Option`, using
        // `LenRange`'s default instead of `None`
        min_length: len_range.map_or(VarConstraint::DEFAULT_MIN, |(lrl, _)| lrl),
        max_length: len_range.and_then(|(_, lru)| lru),
        form: literal_constraint_str.map(ToString::to_string),
        not_equal: HashSet::default(),
        ..Default::default()
    };

    Ok((var, vc))
}

/// Enable `for p in &patterns { ... }`.
///
/// Why `&Patterns` and not `Patterns`?
/// - `for x in collection` desugars to `IntoIterator::into_iter(collection)`.
/// - If we implement `IntoIterator` for **`Patterns`**, iteration would *consume* (move) the
///   whole `Patterns`, which we don't want here.
/// - Implementing it for **`&Patterns`** lets you iterate **by reference** without moving.
impl<'a> IntoIterator for &'a Patterns {
    type Item = &'a Pattern;
    type IntoIter = std::slice::Iter<'a, Pattern>;

    fn into_iter(self) -> Self::IntoIter {
        // Delegate to the slice iterator over the underlying Vec
        self.ordered_list.iter()
    }
}

/// Parses a string like "3-5", "-5", "3-", or "3" into min and max length values.
/// Returns `((min, max_opt))`.
fn parse_length_range(input: &str) -> Result<(usize, Option<usize>), Box<ParseError>> {
    let parts: Vec<_> = input.split('-').map(|part| part.parse::<usize>().ok()).collect();
    if parts.is_empty() || (parts.len() == 1 && parts[0].is_none()) || parts.len() > 2 {
        return Err(Box::new(ParseError::InvalidLengthRange { input: input.to_string() }))
    }
    // TODO!!! is there a better way to do this?
    let min = parts.first().unwrap().unwrap_or(VarConstraint::DEFAULT_MIN);
    let max = *parts.last().unwrap();
    Ok((min, max))
}

#[cfg(test)]
mod tests {
    use crate::constraints::VarConstraint;
    use super::*;

    #[test]
    fn test_basic_pattern_and_constraints() {
        let patterns = "AB;|A|=3;!=AB;B=(2:b*)".parse::<Patterns>().unwrap();

        // Test raw pattern list
        assert_eq!(vec!["AB".to_string()], patterns.p_list.iter().map(|p| p.raw_string.clone()).collect::<Vec<_>>());

        // Test constraints
        let a = patterns.var_constraints.get('A').unwrap();

        let expected_a = VarConstraint {
            min_length: 3,
            max_length: Some(3),
            form: None,
            not_equal: HashSet::from_iter(['B']),
            ..Default::default()
        };
        assert_eq!(expected_a, a.clone());

        let b = patterns.var_constraints.get('B').unwrap();
        let expected_b = VarConstraint {
            min_length: 2,
            max_length: Some(2),
            form: Some("b*".to_string()),
            not_equal: HashSet::from_iter(['A']),
            ..Default::default()
        };
        assert_eq!(expected_b, b.clone());
    }

    #[test]
    fn test_complex_re_len_only() {
        let patterns = "A;A=(6)".parse::<Patterns>().unwrap();

        let expected = VarConstraint {
            min_length: 6,
            max_length: Some(6),
            form: None,
            ..Default::default()
        };
        assert_eq!(expected, patterns.var_constraints.get('A').unwrap().clone());
    }

    #[test]
    fn test_complex_re_lit_only() {
        let patterns = "A;A=(g*)".parse::<Patterns>().unwrap();

        let expected = VarConstraint {
            min_length: VarConstraint::DEFAULT_MIN,
            max_length: None,
            form: Some("g*".to_string()),
            ..Default::default()
        };
        assert_eq!(expected, patterns.var_constraints.get('A').unwrap().clone());
    }
    #[test]
    fn test_complex_re() {
        let patterns = "A;A=(3-4:x*)".parse::<Patterns>().unwrap();

        let expected = VarConstraint {
            min_length: 3,
            max_length: Some(4),
            form: Some("x*".to_string()),
            ..Default::default()
        };
        assert_eq!(expected, patterns.var_constraints.get('A').unwrap().clone());
    }

    #[test]
    fn test_complex_re_unbounded_max_len() {
        let patterns = "A;A=(3-:x*)".parse::<Patterns>().unwrap();

        let expected = VarConstraint {
            min_length: 3,
            max_length: None,
            form: Some("x*".to_string()),
            ..Default::default()
        };
        assert_eq!(expected, patterns.var_constraints.get('A').unwrap().clone());
    }

    #[test]
    fn test_complex_re_unbounded_min_len() {
        let patterns = "A;A=(-4:x*)".parse::<Patterns>().unwrap();

        let expected = VarConstraint {
            min_length: VarConstraint::DEFAULT_MIN,
            max_length: Some(4),
            form: Some("x*".to_string()),
            ..Default::default()
        };
        assert_eq!(expected, patterns.var_constraints.get('A').unwrap().clone());
    }

    #[test]
    fn test_complex_re_exact_len() {
        let patterns = "A;A=(6:x*)".parse::<Patterns>().unwrap();

        let expected = VarConstraint {
            min_length: 6,
            max_length: Some(6),
            form: Some("x*".to_string()),
            ..Default::default()
        };
        assert_eq!(expected, patterns.var_constraints.get('A').unwrap().clone());
    }

    #[test]
    fn test_ordered_patterns() {
        let input = "ABC;BC;C";
        let patterns = input.parse::<Patterns>().unwrap();

        let vars: Vec<HashSet<char>> = patterns.ordered_list.iter().map(|p| p.variables.clone()).collect();

        assert!((&vars[0]).len() >= (&vars[1]).len());
        assert!((&vars[1]).intersection(&vars[0]).count() >= (&vars[2]).intersection(&vars[0]).count());
    }

    #[test]
    fn test_parse_length_range() {
        assert_eq!((2, Some(3)), parse_length_range("2-3").unwrap());
        assert_eq!((VarConstraint::DEFAULT_MIN, Some(3)), parse_length_range("-3").unwrap());
        assert_eq!((1, None), parse_length_range("1-").unwrap());
        assert_eq!((7, Some(7)), parse_length_range("7").unwrap());
        assert!(matches!(*parse_length_range("").unwrap_err(), ParseError::InvalidLengthRange { input } if input == "" ));
        assert!(matches!(*parse_length_range("1-2-3").unwrap_err(), ParseError::InvalidLengthRange { input } if input == "1-2-3" ));
    }

    #[test]
    fn test_len_gt() {
        let patterns = "|A|>4;A".parse::<Patterns>().unwrap();
        let a = patterns.var_constraints.get('A').unwrap();
        assert_eq!(a.min_length, 5);
        assert_eq!(a.max_length, None);
    }

    #[test]
    fn test_len_ge() {
        let patterns = "|A|>=4;A".parse::<Patterns>().unwrap();
        let a = patterns.var_constraints.get('A').unwrap();
        assert_eq!(a.min_length, 4);
        assert_eq!(a.max_length, None);
    }

    #[test]
    fn test_len_lt() {
        let patterns = "|A|<4;A".parse::<Patterns>().unwrap();
        let a = patterns.var_constraints.get('A').unwrap();
        // For <4, max becomes 3; <1 would become None via checked_sub
        assert_eq!(a.min_length, VarConstraint::DEFAULT_MIN);
        assert_eq!(a.max_length, Some(3));
    }

    #[test]
    fn test_len_le() {
        let patterns = "|A|<=4;A".parse::<Patterns>().unwrap();
        let a = patterns.var_constraints.get('A').unwrap();
        assert_eq!(a.min_length, VarConstraint::DEFAULT_MIN);
        assert_eq!(a.max_length, Some(4));
    }

    #[test]
    fn test_len_equality_then_complex_form_only() {
        // Equality first, then a complex constraint that only specifies a form
        let patterns = "A;|A|=7;A=(x*a)".parse::<Patterns>().unwrap();
        let a = patterns.var_constraints.get('A').unwrap().clone();

        let expected = VarConstraint {
            min_length: 7,
            max_length: Some(7),
            form: Some("x*a".to_string()),
            ..Default::default()
        };

        assert_eq!(expected, a);
    }

    #[test]
    /// Verify constraint_score calculation and all_vars_in_lookup_keys logic.
    fn test_constraint_score_and_all_vars_in_lookup_keys() {
        let p1 = Pattern::create("abc", 0); // all literals
        assert_eq!(p1.constraint_score(), 9);

        let p2 = Pattern::create("A@", 1); // var + class
        assert_eq!(p2.constraint_score(), 1);

        let p3 = Pattern::create("#B", 2); // class + var
        assert_eq!(p3.constraint_score(), 1);

        let mut p4 = Pattern::create("AB", 3);
        assert!(!p4.all_vars_in_lookup_keys());
        p4.lookup_keys = HashSet::from_iter(['A', 'B']);
        assert!(p4.all_vars_in_lookup_keys());
    }

    #[test]
    /// Ensure parse_length_range rejects malformed or nonsensical inputs.
    fn test_parse_length_range_invalid_cases() {
        assert!(matches!(
            *parse_length_range("--").unwrap_err(),
            ParseError::InvalidLengthRange { .. }
        ));
        assert!(matches!(
            *parse_length_range("abc").unwrap_err(),
            ParseError::InvalidLengthRange { .. }
        ));
        assert!(matches!(
            *parse_length_range("1-2-3").unwrap_err(),
            ParseError::InvalidLengthRange { .. }
        ));
    }

    #[test]
    /// Ensure get_complex_constraint returns errors for malformed inputs.
    fn test_get_complex_constraint_invalid_cases() {
        // no '='
        assert!(matches!(
            *get_complex_constraint("A").unwrap_err(),
            ParseError::InvalidComplexConstraint { .. }
        ));
        // too many '='
        assert!(matches!(
            *get_complex_constraint("A=B=C").unwrap_err(),
            ParseError::InvalidComplexConstraint { .. }
        ));
        // lhs not length 1
        assert!(matches!(
            *get_complex_constraint("AB=3").unwrap_err(),
            ParseError::InvalidComplexConstraint { .. }
        ));
    }

    #[test]
    /// Verify merging of min/max constraints with a literal form.
    fn test_merge_constraints_len_and_form() {
        // |A|>=5 and A=(3-7:abc) -> min should be 5, max should be 7, form = abc
        let patterns = "A;|A|>=5;A=(3-7:abc)".parse::<Patterns>().unwrap();
        let a = patterns.var_constraints.get('A').unwrap();
        assert_eq!(a.min_length, 5);
        assert_eq!(a.max_length, Some(7));
        assert_eq!(a.form.as_deref(), Some("abc"));
    }

    #[test]
    /// Check that !=ABC constraint gives correct not_equal sets for each variable.
    fn test_not_equal_constraint_three_vars() {
        let patterns = "ABC;!=ABC".parse::<Patterns>().unwrap();
        let a = patterns.var_constraints.get('A').unwrap();
        let b = patterns.var_constraints.get('B').unwrap();
        let c = patterns.var_constraints.get('C').unwrap();

        assert_eq!(a.not_equal, HashSet::from_iter(['B','C']));
        assert_eq!(b.not_equal, HashSet::from_iter(['A','C']));
        assert_eq!(c.not_equal, HashSet::from_iter(['A','B']));
    }

    #[test]
    /// Test ordering tie-breakers: constraint_score and deterministic flag.
    fn test_ordered_patterns_tiebreak_constraint_score_and_deterministic() {
        // Xz has var + literal (score 3), X just var
        let input = "Xz;X".parse::<Patterns>().unwrap();
        assert_eq!(input.ordered_list[0].raw_string, "Xz");

        // Deterministic vs non-deterministic: "AB" (det) vs "A.B" (non-det)
        let input2 = "A.B;AB".parse::<Patterns>().unwrap();
        assert_eq!(input2.ordered_list[0].raw_string, "A.B");
    }

    #[test]
    /// Confirm that IntoIterator yields ordered_list without consuming Patterns.
    fn test_into_iterator_yields_ordered_list() {
        let patterns = "AB;BC".parse::<Patterns>().unwrap();
        let from_iter: Vec<String> = (&patterns).into_iter().map(|p| p.raw_string.clone()).collect();
        let ordered: Vec<String> = patterns.ordered_list.iter().map(|p| p.raw_string.clone()).collect();
        assert_eq!(from_iter, ordered);
    }

    #[test]
    /// Verify that build_order_maps produces true inverses.
    fn test_build_order_maps_inverse() {
        let patterns = "AB;BC;C".parse::<Patterns>().unwrap();
        for (ordered_ix, &orig_ix) in patterns.ordered_to_original.iter().enumerate() {
            let roundtrip = patterns.original_to_ordered[orig_ix];
            assert_eq!(ordered_ix, roundtrip);
        }
    }

}
