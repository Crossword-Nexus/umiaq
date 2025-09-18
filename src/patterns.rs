use crate::comparison_operator::ComparisonOperator;
use crate::constraints::{Bounds, VarConstraint, VarConstraints};
use crate::errors::ParseError;
use crate::parser::ParsedForm;
use crate::umiaq_char::UmiaqChar;
use fancy_regex::Regex;
use std::cmp::Reverse;
use std::collections::HashSet;
use std::str::FromStr;
use std::sync::LazyLock;
use crate::joint_constraints::{propagate_joint_to_var_bounds, JointConstraint, JointConstraints};
use crate::parser::prefilter::build_prefilter_regex;

/// The character that separates forms, in an equation
pub const FORM_SEPARATOR: char = ';';

/// Matches comparative length constraints like `|A|>4`, `|A|<=7`, etc.
/// (Whitespace is permitted around operator.)
static LEN_CMP_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\|([A-Z])\|\s*(<=|>=|=|<|>)\s*(\d+)$").unwrap());

/// Matches inequality constraints like `!=AB`
static NEQ_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^!=([A-Z]+)$").unwrap());

/// Classification of a single input "form" string into one of the
/// supported categories.
///
/// Each variant corresponds to a different type of constraint or pattern
/// that the solver recognizes. This is produced by the `classify_form`
/// helper and consumed by `EquationContext::set_var_constraints`.
enum FormKind {
    /// A length constraint on a single variable, e.g. `|A|=5` or `|B|>3`.
    /// Carries the variable character, the comparison operator, and the
    /// numeric bound.
    LenConstraint(char, ComparisonOperator, usize),

    /// An inequality constraint among a set of variables, e.g. `AB≠CD`.
    /// Stores the set of variable characters that must differ.
    NeqConstraint(Vec<char>),

    /// A more complex constraint derived from parsing (such as a
    /// restricted sub-form). Associates a variable with the specific
    /// `VarConstraint` that applies to it.
    ComplexConstraint(char, VarConstraint),

    /// A "joint" constraint, e.g. `|AB|=7`, where the combined length
    /// of multiple variables is restricted.
    JointConstraint(JointConstraint),

    /// A normal parsed pattern (non-constraint), represented as a
    /// `ParsedForm` ready for matching against candidate words.
    Pattern(ParsedForm),
}



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
///     * for `"BC"`: `Some({'B','C'})` (default to original order)
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
    #[cfg(test)]
    pub fn create(s: &str, index: usize) -> Self {
        let parsed = s.parse::<ParsedForm>().unwrap();
        Self::from_parsed(parsed, s, index)
    }

    /// Construct a `Pattern` from an already-parsed `ParsedForm`.
    ///
    /// This avoids reparsing the raw string (as `create` does),
    /// which is useful when classification has already produced
    /// a `ParsedForm`.
    ///
    /// # Arguments
    ///
    /// * `parsed` — the `ParsedForm` representation of the pattern.
    /// * `raw` — the original string form, kept for display/debugging.
    /// * `original_index` — the index of this pattern in the input order.
    ///
    /// # Behavior
    ///
    /// - `is_deterministic` is computed by checking whether all parts
    ///   of the parsed form are deterministic.
    /// - `variables` is the set of variable characters referenced
    ///   in the raw string.
    /// - `lookup_keys` is initialized empty; it will be populated
    ///   later during solver setup.
    pub fn from_parsed(parsed: ParsedForm, raw: &str, original_index: usize) -> Self {
        // Check whether every part of the parsed form is deterministic.
        let deterministic = parsed.iter().all(|p| p.is_deterministic());

        // Collect all variable characters that appear in the raw string.
        let vars = raw.chars().filter(char::is_variable).collect();

        Self {
            raw_string: raw.to_string(),
            lookup_keys: HashSet::default(), // filled during indexing
            original_index,
            is_deterministic: deterministic,
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

/// Attempt to classify a raw input string into a specific kind of
/// constraint or pattern.
///
/// This function is the central parser for forms. It tries each
/// constraint type in turn, and if none match, falls back to treating
/// the input as a `ParsedForm` pattern.
///
/// The result is wrapped in a [`FormKind`] enum, which can then be
/// dispatched on by `EquationContext::set_var_constraints`.
///
/// # Order of checks
/// 1. Length constraints, e.g. `|A|=5` or `|B|>3`.
/// 2. Inequality constraints, e.g. `AB≠CD`.
/// 3. Complex constraints (`get_complex_constraint`).
/// 4. Joint constraints, e.g. `|AB|=7`.
/// 5. Plain parsed forms (patterns).
///
/// If none of these succeed, returns a `ParseError::InvalidInput`.
///
/// # Errors
/// - Returns `ParseError` if the form is invalid or cannot be parsed
///   as any recognized type.
fn classify_form(form: &str) -> Result<FormKind, Box<ParseError>> {
    // 1. Check for a simple length comparison constraint: |A|=5
    if let Some(cap) = LEN_CMP_RE.captures(form).unwrap() {
        let var_char = cap[1].chars().next().unwrap();
        let op = ComparisonOperator::from_str(&cap[2])?;
        let n = cap[3].parse::<usize>()?;
        return Ok(FormKind::LenConstraint(var_char, op, n));
    }

    // 2. Check for inequality constraints: e.g. AB≠CD
    if let Some(cap) = NEQ_RE.captures(form).unwrap() {
        let vars: Vec<_> = cap[1].chars().collect();
        return Ok(FormKind::NeqConstraint(vars));
    }

    // 3. Complex constraints (delegate to helper)
    if let Ok((var_char, vc)) = get_complex_constraint(form) {
        return Ok(FormKind::ComplexConstraint(var_char, vc));
    }

    // 4. Joint constraints: |AB|=7
    if let Ok(jc) = form.parse::<JointConstraint>() {
        return Ok(FormKind::JointConstraint(jc));
    }

    // 5. Fallback: try to parse as a pattern
    if let Ok(parsed) = form.parse::<ParsedForm>() {
        return Ok(FormKind::Pattern(parsed));
    }

    // Nothing matched → invalid form
    Err(Box::new(ParseError::InvalidInput {
        str: form.to_string(),
    }))
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
pub struct EquationContext {
    /// List of patterns directly extracted from the input string (not constraints)
    // TODO should we keep Vec<Pattern> for each order or just one (likely ordered_list) and use map
    //      (original_to_ordered) when other is needed?
    pub p_list: Vec<Pattern>,
    pub parsed_forms: Vec<ParsedForm>,
    /// Map of variable names (A-Z) to their associated constraints
    pub var_constraints: VarConstraints,
    pub joint_constraints: JointConstraints,
    /// Reordered list of patterns, optimized for solving (most-constrained first)
    pub ordered_list: Vec<Pattern>,        // solver order
    /// ordered index -> original index
    pub ordered_to_original: Vec<usize>,
    /// original index -> ordered index
    pub original_to_ordered: Vec<usize>,
}

impl EquationContext {
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

    /// Parses a semicolon-separated string of forms into constraints and patterns,
    /// and records them in this `EquationContext`.
    ///
    /// Each form is first classified by [`classify_form`] into a [`FormKind`],
    /// then dispatched here:
    ///
    /// - [`FormKind::LenConstraint`] — variable length constraints like `|A|=5`, `|B|>3`.
    /// - [`FormKind::NeqConstraint`] — inequality constraints like `AB≠CD`.
    /// - [`FormKind::ComplexConstraint`] — compound constraints (e.g. bounded length plus a sub-pattern).
    /// - [`FormKind::JointConstraint`] — constraints across multiple variables, e.g. `|AB|=7`.
    /// - [`FormKind::Pattern`] — ordinary parsed forms that become candidate patterns.
    ///
    /// # Errors
    /// - Returns [`ParseError`] if a form cannot be parsed into any recognized kind.
    /// - Returns [`ParseError::ConflictingConstraint`] if a variable is given
    ///   multiple incompatible constraints.
    fn set_var_constraints(&mut self, input: &str) -> Result<(), Box<ParseError>> {
        // Split the input on semicolon (FORM_SEPARATOR). Each fragment is
        // either a constraint or a pattern.
        let forms: Vec<_> = input.split(FORM_SEPARATOR).collect();
        let mut next_form_ix = 0;

        for form in &forms {
            match classify_form(form)? {
                // --- Length constraint on a single variable (e.g. |A|=5, |B|>3) ---
                FormKind::LenConstraint(var_char, op, n) => {
                    let vc = self.var_constraints.ensure(var_char);
                    match op {
                        ComparisonOperator::EQ => vc.set_exact_len(n),
                        ComparisonOperator::NE => {} // handled separately as NeqConstraint
                        ComparisonOperator::LE => vc.bounds.max_len_opt = Some(n),
                        ComparisonOperator::GE => vc.bounds.min_len = n,
                        ComparisonOperator::LT => vc.bounds.max_len_opt = n.checked_sub(1),
                        ComparisonOperator::GT => {
                            vc.bounds.min_len = n.checked_add(1).ok_or_else(|| {
                                Box::new(ParseError::InvalidInput {
                                    str: form.to_string(),
                                })
                            })?
                        }
                    }
                }

                // --- Inequality constraint among variables (e.g. AB≠CD) ---
                FormKind::NeqConstraint(vars) => {
                    for &var_char in &vars {
                        let vc = self.var_constraints.ensure(var_char);
                        // For each variable, store the set of other variables it must differ from.
                        vc.not_equal = vars.iter().copied().filter(|&x| x != var_char).collect();
                    }
                }

                // --- Complex constraint: bounds + embedded pattern ---
                FormKind::ComplexConstraint(var_char, cc_vc) => {
                    let vc = self.var_constraints.ensure(var_char);
                    vc.constrain_by(&cc_vc);

                    // If the complex constraint carries a sub-form, ensure consistency
                    // with any prior form assigned to this variable.
                    if let Some(f) = cc_vc.form {
                        if let Some(old_form) = &vc.form {
                            if *old_form != f {
                                return Err(Box::new(ParseError::ConflictingConstraint {
                                    var_char,
                                    older: old_form.clone(),
                                    newer: f,
                                }));
                            }
                        } else {
                            vc.form = Some(f);
                        }
                    }
                }

                // --- Joint constraint (e.g. |AB|=7) ---
                FormKind::JointConstraint(jc) => {
                    self.joint_constraints.add(jc);
                }

                // --- Regular pattern (not a constraint) ---
                FormKind::Pattern(parsed) => {
                    // Wrap into a Pattern object, reusing the already-parsed form.
                    self.p_list.push(Pattern::from_parsed(parsed, form, next_form_ix));
                    next_form_ix += 1;
                }
            }
        }

        Ok(())
    }

    // TODO is this the right way to order things?
    /// Reorders the list of patterns to improve solving efficiency.
    ///
    /// Strategy:
    /// - First pick: choose the pattern with the most variables (desc).
    /// - Subsequent picks: choose the pattern with the fewest *new* variables (asc).
    ///
    /// Tie-breakers (applied in order):
    /// 1. Higher `constraint_score` first
    /// 2. Non-deterministic before deterministic
    /// 3. Lower original index first
    fn ordered_patterns(&self) -> Vec<Pattern> {
        let mut p_list = self.p_list.clone();
        let mut ordered = Vec::with_capacity(p_list.len());

        while !p_list.is_empty() {
            // Vars already "seen" in previously chosen patterns
            let found_vars = ordered
                .iter()
                .flat_map(|p: &Pattern| p.variables.iter().copied())
                .collect();

            // Unified scoring function
            let get_score = |p: &Pattern| {
                if ordered.is_empty() {
                    // First pick: more vars is better
                    (
                        p.variables.len(),
                        p.constraint_score(),
                        Reverse(p.is_deterministic),
                        Reverse(p.original_index),
                    )
                } else {
                    // Subsequent picks: fewer new vars is better,
                    // so we negate the count (to maximize a "negative number").
                    (
                        usize::MAX - p.variables.difference(&found_vars).count(), // TODO? avoid MAX
                        p.constraint_score(),
                        Reverse(p.is_deterministic),
                        Reverse(p.original_index),
                    )
                }
            };

            // Select the best candidate
            let (ix, _) = p_list
                .iter()
                .enumerate()
                .max_by_key(|(_, p)| get_score(p))
                .unwrap();

            let mut chosen = p_list.remove(ix);

            if !ordered.is_empty() {
                // Assign join keys only after the first pick
                chosen.lookup_keys = chosen.variables.intersection(&found_vars).copied().collect();
            }

            ordered.push(chosen);
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

impl FromStr for EquationContext {
    type Err = Box<ParseError>;

    /// Build a fully-initialized `EquationContext` from a raw equation string.
    ///
    /// This constructor wraps all the setup work that used to live in
    /// `solve_equation` (steps 4–7):
    ///
    /// - Calls [`EquationContext::set_var_constraints`] to populate
    ///   patterns (`p_list`), per-variable constraints, and joint constraints.
    /// - Parses each pattern string once into a `ParsedForm` and stores
    ///   them in `parsed_forms`.
    /// - Upgrades each `ParsedForm`'s regex prefilter to respect any
    ///   variable constraints (e.g. turn `.+` into `g.*`).
    /// - Propagates joint constraints into the per-variable constraints
    ///   to tighten variable length bounds.
    /// - Builds the solver ordering and index maps.
    ///
    /// After this returns, the `EquationContext` is ready for solving:
    /// all constraints and forms are parsed and upgraded, and ordering is set.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut ctx = EquationContext::default();

        // Step 1: Parse the input string into patterns + constraints.
        // This fills in `p_list`, `var_constraints`, and `joint_constraints`.
        ctx.set_var_constraints(s)?;

        // Step 2: Parse each raw pattern string into a `ParsedForm`.
        // These are kept parallel to `p_list` in `parsed_forms`.
        ctx.parsed_forms = ctx
            .p_list
            .iter()
            .map(|p| p.raw_string.parse::<ParsedForm>())
            .collect::<Result<_, _>>()?;

        // Step 3: Upgrade prefilters for each parsed form using variable constraints.
        // For example, if |A| has form `g*`, prefilter `.+` can be tightened to `g.*`.
        for pf in &mut ctx.parsed_forms {
            let upgraded = build_prefilter_regex(pf, &ctx.var_constraints)?;
            pf.prefilter = upgraded;
        }

        // Step 4: Apply joint constraints to tighten per-variable bounds.
        // E.g. from |AB|=7 we infer additional length limits for |A| and |B|.
        if !ctx.joint_constraints.is_empty() {
            propagate_joint_to_var_bounds(&mut ctx.var_constraints, &ctx.joint_constraints);
        }

        // Step 5: Establish solving order and index maps.
        ctx.ordered_list = ctx.ordered_patterns();
        ctx.build_order_maps();

        Ok(ctx)
    }
}


// TODO? do this via regex?
// e.g., A=(3-:x*)
fn get_complex_constraint(form: &str) -> Result<(char, VarConstraint), Box<ParseError>> {
    if let Some((var_str, constraint_str)) = form.split_once('=') {
        if var_str.len() == 1 {
            if constraint_str.contains('=') {
                return Err(Box::new(ParseError::InvalidComplexConstraint { str: format!("expected 1 equals sign (not {})", form.chars().filter(|c| *c == '=').count()) }));
            }

            let var_char = var_str.chars().next().unwrap();

            // remove outer parentheses if they are there
            let inner_constraint_str = if constraint_str.starts_with('(') && constraint_str.ends_with(')') {
                let mut chars = constraint_str.chars();
                chars.next();
                chars.next_back();
                chars.as_str()
            } else {
                constraint_str
            };

            let (len_range, literal_constraint_str) = if let Some((len_range_raw, literal_constraint_str)) = inner_constraint_str.split_once(':') {
                if literal_constraint_str.contains(':') { // too many colons
                    return Err(Box::new(ParseError::InvalidComplexConstraint { str: format!("too many colons--0 or 1 expected (not {})", inner_constraint_str.chars().filter(|c| *c == ':').count()) }));
                }
                let len_range = parse_length_range(len_range_raw)?;
                (len_range, Some(literal_constraint_str))
            } else {
                match parse_length_range(inner_constraint_str) {
                    Ok(len_range) => (len_range, None),
                    Err(_) => (Bounds::default(), Some(inner_constraint_str))
                }
            };

            let vc = VarConstraint {
                bounds: len_range,
                form: literal_constraint_str.map(ToString::to_string),
                not_equal: HashSet::default(),
                ..Default::default()
            };

            Ok((var_char, vc))
        } else {
            Err(Box::new(ParseError::InvalidComplexConstraint { str: format!("expected 1 character (as the variable) to the left of \"=\" (not {})", var_str.len()) }))
        }
    } else {
        Err(Box::new(ParseError::InvalidComplexConstraint { str: format!("expected 1 equals sign (not {})", form.chars().filter(|c| *c == '=').count()) }))
    }
}

/// Enable `for p in &patterns { ... }`.
///
/// Why `&Patterns` and not `Patterns`?
/// - `for x in collection` desugars to `IntoIterator::into_iter(collection)`.
/// - If we implement `IntoIterator` for **`Patterns`**, iteration would *consume* (move) the
///   whole `Patterns`, which we don't want here.
/// - Implementing it for **`&Patterns`** lets you iterate **by reference** without moving.
impl<'a> IntoIterator for &'a EquationContext {
    type Item = &'a Pattern;
    type IntoIter = std::slice::Iter<'a, Pattern>;

    fn into_iter(self) -> Self::IntoIter {
        // Delegate to the slice iterator over the underlying Vec
        self.ordered_list.iter()
    }
}

/// Parses a string like "3-5", "-5", "3-", or "3" into min and max length values.
/// Returns `((min, max_opt))`.
fn parse_length_range(input: &str) -> Result<Bounds, Box<ParseError>> {
    let parts: Vec<_> = input.split('-').map(|part| part.parse::<usize>().ok()).collect();
    if parts.is_empty() || (parts.len() == 1 && parts[0].is_none()) || parts.len() > 2 {
        return Err(Box::new(ParseError::InvalidLengthRange { input: input.to_string() }))
    }
    // TODO!!! is there a better way to do this?
    let min = parts.first().unwrap().unwrap_or(VarConstraint::DEFAULT_MIN);
    let max = *parts.last().unwrap();
    Ok(max.map_or(Bounds::of_unbounded(min), |u| Bounds::of(min, u)))
}

#[cfg(test)]
mod tests {
    use crate::constraints::VarConstraint;
    use super::*;

    #[test]
    fn test_basic_pattern_and_constraints() {
        let patterns = "AB;|A|=3;!=AB;B=(2:b*)".parse::<EquationContext>().unwrap();

        // Test raw pattern list
        assert_eq!(vec!["AB".to_string()], patterns.p_list.iter().map(|p| p.raw_string.clone()).collect::<Vec<_>>());

        // Test constraints
        let a = patterns.var_constraints.get('A').unwrap();

        let expected_a = VarConstraint {
            bounds: Bounds::of(3, 3),
            form: None,
            not_equal: HashSet::from_iter(['B']),
            ..Default::default()
        };
        assert_eq!(expected_a, a.clone());

        let b = patterns.var_constraints.get('B').unwrap();
        let expected_b = VarConstraint {
            bounds: Bounds::of(2, 2),
            form: Some("b*".to_string()),
            not_equal: HashSet::from_iter(['A']),
            ..Default::default()
        };
        assert_eq!(expected_b, b.clone());
    }

    #[test]
    fn test_complex_re_len_only() {
        let patterns = "A;A=(6)".parse::<EquationContext>().unwrap();

        let expected = VarConstraint {
            bounds: Bounds::of(6, 6),
            form: None,
            ..Default::default()
        };
        assert_eq!(expected, patterns.var_constraints.get('A').unwrap().clone());
    }

    #[test]
    fn test_complex_re_lit_only() {
        let patterns = "A;A=(g*)".parse::<EquationContext>().unwrap();

        let expected = VarConstraint {
            bounds: Bounds::of_unbounded(VarConstraint::DEFAULT_MIN),
            form: Some("g*".to_string()),
            ..Default::default()
        };
        assert_eq!(expected, patterns.var_constraints.get('A').unwrap().clone());
    }
    #[test]
    fn test_complex_re() {
        let patterns = "A;A=(3-4:x*)".parse::<EquationContext>().unwrap();

        let expected = VarConstraint {
            bounds: Bounds::of(3, 4),
            form: Some("x*".to_string()),
            ..Default::default()
        };
        assert_eq!(expected, patterns.var_constraints.get('A').unwrap().clone());
    }

    #[test]
    fn test_complex_re_unbounded_max_len() {
        let patterns = "A;A=(3-:x*)".parse::<EquationContext>().unwrap();

        let expected = VarConstraint {
            bounds: Bounds::of_unbounded(3),
            form: Some("x*".to_string()),
            ..Default::default()
        };
        assert_eq!(expected, patterns.var_constraints.get('A').unwrap().clone());
    }

    #[test]
    fn test_complex_re_unbounded_min_len() {
        let patterns = "A;A=(-4:x*)".parse::<EquationContext>().unwrap();

        let expected = VarConstraint {
            bounds: Bounds::of(VarConstraint::DEFAULT_MIN, 4),
            form: Some("x*".to_string()),
            ..Default::default()
        };
        assert_eq!(expected, patterns.var_constraints.get('A').unwrap().clone());
    }

    #[test]
    fn test_complex_re_exact_len() {
        let patterns = "A;A=(6:x*)".parse::<EquationContext>().unwrap();

        let expected = VarConstraint {
            bounds: Bounds::of(6, 6),
            form: Some("x*".to_string()),
            ..Default::default()
        };
        assert_eq!(expected, patterns.var_constraints.get('A').unwrap().clone());
    }

    #[test]
    /// Test that ordering leaves the list unchanged when no reordering is needed.
    fn test_ordered_patterns() {
        let input = "ABC;BC;C";
        let patterns = input.parse::<EquationContext>().unwrap();

        let actual: Vec<_> = patterns
            .ordered_list
            .iter()
            .map(|p| p.raw_string.clone())
            .collect();

        let expected = vec!["ABC".to_string(), "BC".to_string(), "C".to_string()];
        assert_eq!(actual, expected);
    }


    #[test]
    fn test_parse_length_range() {
        assert_eq!(Bounds::of(2, 3), parse_length_range("2-3").unwrap());
        assert_eq!(Bounds::of(VarConstraint::DEFAULT_MIN, 3), parse_length_range("-3").unwrap());
        assert_eq!(Bounds::of_unbounded(1), parse_length_range("1-").unwrap());
        assert_eq!(Bounds::of(7, 7), parse_length_range("7").unwrap());
        assert!(matches!(*parse_length_range("").unwrap_err(), ParseError::InvalidLengthRange { input } if input.is_empty() ));
        assert!(matches!(*parse_length_range("1-2-3").unwrap_err(), ParseError::InvalidLengthRange { input } if input == "1-2-3" ));
    }

    #[test]
    fn test_len_gt() {
        let patterns = "|A|>4;A".parse::<EquationContext>().unwrap();
        let a = patterns.var_constraints.get('A').unwrap();
        assert_eq!(a.bounds.min_len, 5);
        assert_eq!(a.bounds.max_len_opt, None);
    }

    #[test]
    fn test_len_ge() {
        let patterns = "|A|>=4;A".parse::<EquationContext>().unwrap();
        let a = patterns.var_constraints.get('A').unwrap();
        assert_eq!(a.bounds.min_len, 4);
        assert_eq!(a.bounds.max_len_opt, None);
    }

    #[test]
    fn test_len_lt() {
        let patterns = "|A|<4;A".parse::<EquationContext>().unwrap();
        let a = patterns.var_constraints.get('A').unwrap();
        // For <4, max becomes 3; <1 would become None via checked_sub
        assert_eq!(a.bounds.min_len, VarConstraint::DEFAULT_MIN);
        assert_eq!(a.bounds.max_len_opt, Some(3));
    }

    #[test]
    fn test_len_le() {
        let patterns = "|A|<=4;A".parse::<EquationContext>().unwrap();
        let a = patterns.var_constraints.get('A').unwrap();
        assert_eq!(a.bounds.min_len, VarConstraint::DEFAULT_MIN);
        assert_eq!(a.bounds.max_len_opt, Some(4));
    }

    #[test]
    fn test_len_equality_then_complex_form_only() {
        // Equality first, then a complex constraint that only specifies a form
        let patterns = "A;|A|=7;A=(x*a)".parse::<EquationContext>().unwrap();
        let a = patterns.var_constraints.get('A').unwrap().clone();

        let expected = VarConstraint {
            bounds: Bounds::of(7, 7),
            form: Some("x*a".to_string()),
            ..Default::default()
        };

        assert_eq!(expected, a);
    }

    #[test]
    /// Verify `constraint_score` calculation and `all_vars_in_lookup_keys` logic—all literals.
    fn test_constraint_score_and_all_vars_in_lookup_keys_all_literals() {
        assert_eq!(Pattern::create("abc", 0).constraint_score(), 9);
    }

    #[test]
    /// Verify `constraint_score` calculation and `all_vars_in_lookup`_keys logic—var + class.
    fn test_constraint_score_and_all_vars_in_lookup_keys_var_class() {
        assert_eq!(Pattern::create("A@", 1).constraint_score(), 1);
    }

    #[test]
    /// Verify `constraint_score` calculation and `all_vars_in_lookup_keys` logic—class + var.
    fn test_constraint_score_and_all_vars_in_lookup_keys_class_var() {
        assert_eq!(Pattern::create("#B", 2).constraint_score(), 1);
    }

    #[test]
    /// Verify `all_vars_in_lookup_keys` logic.
    fn test_constraint_score_and_all_vars_in_lookup_keys() {
        let mut p = Pattern::create("AB", 3);
        assert!(!p.all_vars_in_lookup_keys());
        p.lookup_keys = HashSet::from_iter(['B']);
        assert!(!p.all_vars_in_lookup_keys());
        p.lookup_keys = HashSet::from_iter(['A', 'B']);
        assert!(p.all_vars_in_lookup_keys());
    }

    #[test]
    /// Ensure parse_length_range rejects malformed or nonsensical inputs—dashes pair.
    fn test_parse_length_range_invalid_cases_dashes_pair() {
        assert!(matches!(
            *parse_length_range("--").unwrap_err(),
            ParseError::InvalidLengthRange{ input } if input == "--"
        ));
    }

    #[test]
    /// Ensure parse_length_range rejects malformed or nonsensical inputs—just letters.
    fn test_parse_length_range_invalid_cases_just_letters() {
        assert!(matches!(
            *parse_length_range("abc").unwrap_err(),
            ParseError::InvalidLengthRange { input } if input == "abc"
        ));
    }

    #[test]
    /// Ensure parse_length_range rejects malformed or nonsensical inputs—1-2-3-.
    fn test_parse_length_range_invalid_cases_1_2_3() {
        assert!(matches!(
            *parse_length_range("1-2-3").unwrap_err(),
            ParseError::InvalidLengthRange { input } if input == "1-2-3"
        ));
    }

    #[test]
    /// Ensure get_complex_constraint returns errors for malformed inputs—no '='.
    fn test_get_complex_constraint_invalid_cases_no_equals() {
        assert!(matches!(
            *get_complex_constraint("A").unwrap_err(),
            ParseError::InvalidComplexConstraint { str } if str == "expected 1 equals sign (not 0)"
        ));
    }

    #[test]
    /// Ensure get_complex_constraint returns errors for malformed inputs—too many '='s
    fn test_get_complex_constraint_invalid_cases_too_many_equals() {
        assert!(matches!(
            *get_complex_constraint("A=B=C").unwrap_err(),
            ParseError::InvalidComplexConstraint { str } if str == "expected 1 equals sign (not 2)"
        ));
    }

    #[test]
    /// Ensure get_complex_constraint returns errors for malformed inputs—lhs too long.
    fn test_get_complex_constraint_invalid_cases_lhs_too_long() {
        assert!(matches!(
            *get_complex_constraint("AB=3").unwrap_err(),
            ParseError::InvalidComplexConstraint { str } if str == "expected 1 character (as the variable) to the left of \"=\" (not 2)"
        ));
    }

    #[test]
    /// Verify merging of min/max constraints with a literal form.
    fn test_merge_constraints_len_and_form() {
        // |A|>=5 and A=(3-7:abc) -> min should be 5, max should be 7, form = abc
        let patterns = "A;|A|>=5;A=(3-7:abc)".parse::<EquationContext>().unwrap();
        let a = patterns.var_constraints.get('A').unwrap();
        assert_eq!(a.bounds.min_len, 5);
        assert_eq!(a.bounds.max_len_opt, Some(7));
        assert_eq!(a.form.as_deref(), Some("abc"));
    }

    #[test]
    /// Check that !=ABC constraint gives correct `not_equal` sets for each variable.
    fn test_not_equal_constraint_three_vars() {
        let patterns = "ABC;!=ABC".parse::<EquationContext>().unwrap();
        let a = patterns.var_constraints.get('A').unwrap();
        let b = patterns.var_constraints.get('B').unwrap();
        let c = patterns.var_constraints.get('C').unwrap();

        assert_eq!(a.not_equal, HashSet::from_iter(['B','C']));
        assert_eq!(b.not_equal, HashSet::from_iter(['A','C']));
        assert_eq!(c.not_equal, HashSet::from_iter(['A','B']));
    }

    #[test]
    /// Test ordering tiebreakers: `constraint_score`.
    fn test_ordered_patterns_tiebreak_constraint_score() {
        // "Xz" has var + literal (score 3), "X" just var
        let patterns = "Xz;X".parse::<EquationContext>().unwrap();
        assert_eq!(patterns.ordered_list.iter().map(|p| p.raw_string.clone()).collect::<Vec<_>>(), vec!["Xz", "X"]);
    }

    #[test]
    /// Test ordering tiebreakers: deterministic flag.
    fn test_ordered_patterns_tiebreak_deterministic() {
        // deterministic vs non-deterministic: "AB" (det) vs "A.B" (non-det)
        let patterns = "A.B;AB".parse::<EquationContext>().unwrap();
        assert_eq!(patterns.ordered_list.iter().map(|p| p.raw_string.clone()).collect::<Vec<_>>(), vec!["A.B", "AB"]);
    }

    #[test]
    /// Confirm that `IntoIterator` yields `ordered_list` without consuming `Patterns`.
    fn test_into_iterator_yields_ordered_list() {
        let patterns = "AB;BC".parse::<EquationContext>().unwrap();
        let from_iter: Vec<_> = (&patterns).into_iter().map(|p| p.raw_string.clone()).collect();
        let ordered: Vec<_> = patterns.ordered_list.iter().map(|p| p.raw_string.clone()).collect();
        assert_eq!(from_iter, ordered);
    }

    #[test]
    /// Verify that `build_order_maps` produces true inverses.
    fn test_build_order_maps_inverse() {
        let patterns = "AB;BC;C".parse::<EquationContext>().unwrap();
        for (ordered_ix, &orig_ix) in patterns.ordered_to_original.iter().enumerate() {
            let roundtrip = patterns.original_to_ordered[orig_ix];
            assert_eq!(ordered_ix, roundtrip);
        }
    }

    #[test]
    /// Ensure conflicting complex constraints for the same variable produce an error.
    fn test_conflicting_complex_constraints_error() {
        assert!(matches!(
            *"A=(1-5:k*);A=(5-6:a*);A".parse::<EquationContext>().unwrap_err(),
            ParseError::ConflictingConstraint { var_char, older,  newer } if var_char == 'A' && older == "k*" && newer == "a*" )
        );
    }
}
