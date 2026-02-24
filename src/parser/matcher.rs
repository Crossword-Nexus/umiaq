use crate::bindings::Bindings;
use crate::constraints::{VarConstraint, VarConstraints};
use crate::errors::ParseError;
use crate::interner;
use crate::joint_constraints::JointConstraints;
use crate::umiaq_char::UmiaqChar;
use log::error;

use super::form::{FormPart, ParsedForm};

/// If `prefix` is indeed a prefix of `chars`, return a slice pointing to
/// the remaining portion of `chars` (the part after the prefix).
/// Otherwise, return `None`.
fn get_rest_if_valid_prefix<'a>(prefix: &str, chars: &'a [char]) -> Option<&'a [char]> {
    // Ensure we have enough characters left to match the prefix
    if chars.len() >= prefix.len()
        // Compare each char of `chars` with each char of `prefix`
        && chars.iter().take(prefix.len()).copied().eq(prefix.chars())
    {
        // Success: return the suffix of `chars` that follows the prefix
        Some(&chars[prefix.len()..])
    } else {
        // Failure: not a prefix
        None
    }
}

/// Helper to reverse a bound value if the part is `RevVar`.
fn get_reversed_or_not(first: &FormPart, var_val: &str) -> String {
    if matches!(first, FormPart::RevVar(_)) {
        var_val.chars().rev().collect()
    } else {
        var_val.to_owned()
    }
}

/// Validate whether a candidate binding value is allowed under a `VarConstraint`.
///
/// Checks:
/// 1. If length constraints are present, enforce them
/// 2. If `form` is present, the value must itself match that form.
/// 3. The value must not equal any variable listed in `not_equal` that is already bound.
fn is_valid_binding(
    var_val: &str,
    constraints: &VarConstraint,
    bindings: &Bindings,
) -> Result<bool, Box<ParseError>> {
    // 1. Length checks (if configured)
    if var_val.len() < constraints.bounds.min_len
        || constraints.bounds.max_len_opt.is_some_and(|max_len| var_val.len() > max_len)
    {
        return Ok(false);
    }

    // 2. Apply nested-form constraint if present (use cached parse)
    if let Some(parsed) = constraints.get_parsed_form()? && !match_equation_exists(
        var_val,
        parsed,
        &VarConstraints::default(),
        &JointConstraints::default(),
    )? {
        return Ok(false);
    }

    // 3. Check "not equal" constraints
    for &other in &constraints.not_equal {
        if let Some(existing) = bindings.get(other) && existing.as_ref() == var_val {
            return Ok(false)
        }
    }

    Ok(true)
}

/// Return `true` if at least one binding satisfies the equation.
///
/// # Errors
/// Returns `ParseError::PrefilterFailed` if the regex prefilter fails to execute.
/// Returns `ParseError::AnagramCheckFailed` if anagram validation encounters an error.
pub fn match_equation_exists(
    entry: &str,
    parts: &ParsedForm,
    constraints: &VarConstraints,
    joint_constraints: &JointConstraints,
) -> Result<bool, Box<ParseError>> {
    let mut results: Vec<Bindings> = Vec::with_capacity(1); // at most 1 result when all_matches=false
    match_equation_internal(entry, parts, false, &mut results, constraints, joint_constraints)?;
    Ok(!results.is_empty())
}

/// Return all bindings that satisfy the equation.
///
/// # Errors
/// Returns `ParseError::PrefilterFailed` if the regex prefilter fails to execute.
/// Returns `ParseError::AnagramCheckFailed` if anagram validation encounters an error.
pub fn match_equation_all(
    entry: &str,
    parts: &ParsedForm,
    constraints: &VarConstraints,
    joint_constraints: &JointConstraints,
) -> Result<Vec<Bindings>, Box<ParseError>> {
    let mut results: Vec<Bindings> = Vec::new();
    match_equation_internal(entry, parts, true, &mut results, constraints, joint_constraints)?;
    Ok(results)
}

/// Core entry point for the backtracking search.
///
/// This function coordinates matching an `entry` against a parsed form:
/// - **Prefilter step:** quickly reject the entry if it cannot possibly match,
///   using the compiled regex stored in the `ParsedForm`.
/// - **Recursive search:** initialize a `HelperParams` context and invoke
///   its `recurse` method, which handles binding variables, matching literals,
///   wildcards, and enforcing constraints.
/// - **Result collection:** all successful bindings are pushed into `results`.
///   If `all_matches` is false, the recursion will stop after the first match.
///
/// Arguments:
/// - `entry`: the candidate entry to test.
/// - `parsed_form`: the form (sequence of `FormPart`s) we are matching against.
/// - `all_matches`: if `true`, collect all possible bindings; if `false`, stop
///   once a single valid binding is found.
/// - `results`: accumulator for successful variable bindings.
/// - `constraints`: variable-level constraints to enforce during binding.
/// - `joint_constraints`: constraints that involve multiple variables together.
///
/// # Errors
/// Returns `ParseError::PrefilterFailed` if the regex prefilter fails to execute.
/// Returns `ParseError::AnagramCheckFailed` if anagram validation encounters an error.
fn match_equation_internal(
    entry: &str,
    parsed_form: &ParsedForm,
    all_matches: bool,
    results: &mut Vec<Bindings>,
    constraints: &VarConstraints,
    joint_constraints: &JointConstraints,
) -> Result<(), Box<ParseError>> {
    // === PREFILTER STEP ===
    // Use the regex prefilter on the parsed form to quickly discard entries
    // that cannot possibly match. This helps avoid expensive recursive searching.
    // Instead of silently treating errors as "no match", propagate them.
    match parsed_form.prefilter.is_match(entry) {
        Ok(false) => return Ok(()), // No match, return early
        Err(e) => {
            error!("Prefilter regex failed for entry '{entry}': {e}");
            return Err(Box::new(ParseError::PrefilterFailed(e)));
        }
        Ok(true) => { /* Continue matching */ }
    }

    // === INITIALIZE SEARCH CONTEXT ===
    // Create a helper context with mutable references to bindings and results,
    // plus configuration flags and constraints.
    let mut hp = HelperParams {
        bindings: &mut Bindings::default(),
        results,
        all_matches,
        entry,
        constraints,
        joint_constraints,
    };

    // === RECURSIVE SEARCH ===
    // Convert the entry to a Vec<char> for indexed access, and start recursion.
    hp.recurse(&entry.chars().collect::<Vec<_>>(), &parsed_form.parts)?;

    Ok(())
}


// Helper params for recursion
struct HelperParams<'a> {
    bindings: &'a mut Bindings,
    results: &'a mut Vec<Bindings>,
    all_matches: bool,
    entry: &'a str,
    constraints: &'a VarConstraints,
    joint_constraints: &'a JointConstraints,
}

impl HelperParams<'_> {
    /// Attempt to bind a variable (`var_name`) to some prefix of `chars` and continue recursion.
    ///
    /// This explores all candidate substring lengths in `[min_len, max_len]`,
    /// where the range is determined both by the variable’s declared constraints
    /// (if any) and by how many characters remain in `chars`.
    ///
    /// For each candidate length `l`:
    /// - Take the first `l` characters from `chars` as a potential binding.
    /// - If the pattern form is a `RevVar`, reverse those characters.
    /// - Check the candidate against the variable’s constraint (if present) using
    ///   [`is_valid_binding`].
    /// - If valid, temporarily record the binding, recurse on the remainder of `chars`,
    ///   and then backtrack (remove the binding) if we’re still searching.
    ///
    /// Returns `true` as soon as any candidate path leads to a successful match;
    /// otherwise returns `false`.
    ///
    /// # Notes
    /// - Errors from [`is_valid_binding`] are currently swallowed (treated as “invalid”).
    ///   A future refactor could bubble them up instead of discarding them.
    /// - This method mutates `self.bindings` as part of the search, always restoring
    ///   the state on backtracking.
    fn try_bind_var(
        &mut self,
        var_name: char,
        chars: &[char],
        rest: &[FormPart],
        first: &FormPart,
    ) -> Result<bool, Box<ParseError>> {
        // Compute min and max candidate lengths from constraints and availability.
        let min_len = self
            .constraints
            .get(var_name)
            .map_or(VarConstraint::DEFAULT_MIN, |vc| vc.bounds.min_len);
        let max_len_cfg = self
            .constraints
            .get(var_name)
            .and_then(|vc| vc.bounds.max_len_opt)
            .unwrap_or(chars.len());

        let avail = chars.len();

        if min_len > avail {
            // Not enough characters left to satisfy the minimum length.
            return Ok(false);
        }

        // Cap the maximum so we never slice past the available characters.
        let capped_max = std::cmp::min(max_len_cfg, avail);

        for l in min_len..=capped_max {
            // Slice off a candidate binding of length `l`.
            let candidate_chars = &chars[..l];

            // Reverse if this is a RevVar; otherwise leave as-is.
            let var_val_string: String = if matches!(first, FormPart::RevVar(_)) {
                candidate_chars.iter().rev().collect()
            } else {
                candidate_chars.iter().collect()
            };

            // Intern the string to avoid duplicate allocations
            let var_val = interner::intern(var_val_string);

            // Apply any variable-specific constraint.
            // Propagate errors instead of swallowing them
            if let Some(c) = self.constraints.get(var_name) &&
                !is_valid_binding(var_val.as_ref(), c, self.bindings)? {
                continue; // This binding is invalid, try next length
            }

            // Tentatively record the binding (using interned Rc).
            self.bindings.set_rc(var_name, var_val);

            // Recurse on the remaining characters.
            // If `all_matches` is false, stop once we've found one match.
            let retval = self.recurse(&chars[l..], rest)? && !self.all_matches;

            if retval {
                return Ok(true); // Found a match, stop searching
            }

            // Backtrack (remove the binding) and continue to next length
            self.bindings.remove(var_name);
        }

        Ok(false) // No valid binding found
    }


    /// Recursive backtracking matcher.
    ///
    /// Attempts to match the slice of `chars` against the remaining `parts`.
    /// On success, it may push a completed `Bindings` into `self.results`.
    /// Stops early if `all_matches` is false and one valid match is found.
    ///
    /// # Errors
    /// Returns `ParseError::AnagramCheckFailed` if anagram validation encounters an error.
    fn recurse(&mut self, chars: &[char], parts: &[FormPart]) -> Result<bool, Box<ParseError>> {
        // Base case: no parts left
        if parts.is_empty() {
            if chars.is_empty() {
                // Check the joint constraints (if any)
                if self.joint_constraints.all_satisfied(self.bindings) {
                    let mut full_result = self.bindings.clone();
                    full_result.set_entry(self.entry);
                    self.results.push(full_result);
                    return Ok(!self.all_matches); // Stop early if only one match needed
                }
            }
            return Ok(false);
        }

        // safe: parts is non-empty (checked above), so indexing [0] and slicing [1..] are valid
        debug_assert!(!parts.is_empty(), "parts must be non-empty after early return check");
        let (first, rest) = (&parts[0], &parts[1..]);

        match first {
            FormPart::Lit(s) => {
                // Literal match (case-insensitive, stored lowercase)
                match get_rest_if_valid_prefix(s, chars) {
                    Some(rest_chars) => self.recurse(rest_chars, rest),
                    None => Ok(false),
                }
            }
            FormPart::Star => {
                // Zero-or-more wildcard; try all possible splits
                for i in 0..=chars.len() {
                    if self.recurse(&chars[i..], rest)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }

            // Combined vowel, consonant, charset, dot cases
            FormPart::Dot => self.take_if(chars, rest, |_| true),
            FormPart::Vowel => self.take_if(chars, rest, char::is_vowel),
            FormPart::Consonant => self.take_if(chars, rest, char::is_consonant),
            FormPart::Charset(s) => self.take_if(chars, rest, |c| s.contains(c)),

            FormPart::Anagram(ag) => {
                // Match if the next len chars are an anagram of target
                let len = ag.len;
                if chars.len() >= len {
                    // Propagate anagram check errors instead of swallowing them
                    match ag.is_anagram(&chars[..len]) {
                        Ok(true) => self.recurse(&chars[len..], rest),
                        Ok(false) => Ok(false),
                        Err(e) => {
                            error!("Anagram check failed during matching: {e}");
                            Err(Box::new(ParseError::AnagramCheckFailed(e)))
                        }
                    }
                } else {
                    Ok(false)
                }
            }

            FormPart::Var(var_name) | FormPart::RevVar(var_name) => {
                if let Some(var_val) = self.bindings.get(*var_name) {
                    // Already bound: must match exactly
                    match get_rest_if_valid_prefix(&get_reversed_or_not(first, var_val), chars) {
                        Some(rest_chars) => self.recurse(rest_chars, rest),
                        None => Ok(false),
                    }
                } else {
                    // Not bound yet: try binding to all possible lengths
                    self.try_bind_var(*var_name, chars, rest, first)
                }
            }
        }
    }

    /// Try to consume exactly one char from `chars` if present, and apply `pred` to it.
    ///
    /// - If the predicate passes, recurse with the rest of the chars.
    /// - Otherwise return false (dead end).
    ///
    /// Covers Dot (any char), Vowel, Consonant, Charset, etc.
    ///
    /// # Errors
    /// Propagates errors from recursive matching.
    fn take_if(
        &mut self,
        chars: &[char],
        rest: &[FormPart],
        pred: impl Fn(&char) -> bool,
    ) -> Result<bool, Box<ParseError>> {
        match chars.split_first() {
            Some((c, rest_chars)) if pred(c) => self.recurse(rest_chars, rest),
            _ => Ok(false),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_palindrome_matching() {
        let pf = "A~A".parse::<ParsedForm>().unwrap();
        assert!(match_equation_exists("noon", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        assert!(!match_equation_exists("radar", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        assert!(!match_equation_exists("test", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
    }

    #[test]
    fn test_match_equation_exists() {
        let pf = "A~A[rstlne]/jon@#.*".parse::<ParsedForm>().unwrap();
        assert!(match_equation_exists("aaronjudge", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        assert!(!match_equation_exists("noon", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        assert!(!match_equation_exists("toon", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
    }

    #[test]
    fn test_literal_matching() {
        let pf = "abc".parse::<ParsedForm>().unwrap();
        assert!(match_equation_exists("abc", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        assert!(!match_equation_exists("xyz", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
    }

    mod edge_cases {
        use super::*;

        #[test]
        fn test_many_consecutive_wildcards() {
            // min len: 7 (1+5+1)
            let pf = "A.....B".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("testing", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(!match_equation_exists("test", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap()); // only 4 chars, need 7+
        }

        #[test]
        fn test_multiple_variable_wildcards() {
            let pf = "A*B*C*D".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("abcd", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap()); // min case: each var is 1 char
            assert!(match_equation_exists("testing", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_pattern_longer_than_entry() {
            // min len: 10
            let pf = "ABCDEFGHIJ".parse::<ParsedForm>().unwrap();
            assert!(!match_equation_exists("cat", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_all_wildcards_pattern() {
            let pf = "*****".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("test", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("a", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap()); // all wildcards empty
        }

        #[test]
        fn test_alternating_vars_and_wildcards() {
            // min len: 5
            let pf = "A.B.C".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("abcde", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(!match_equation_exists("abcd", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap()); // only 4 chars
        }

        #[test]
        fn test_reverse_operator() {
            let pf = "~A".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("test", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap()); // A = "tset"

            let pf2 = "A~B".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("testing", &pf2, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_palindrome_structure() {
            let pf = "A~A~B".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("abbac", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap()); // A="ab", ~A="ba", ~B="c" (so B="c")
            assert!(!match_equation_exists("test", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_single_char_entry() {
            let pf = "A".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("a", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());

            let pf2 = "A*".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("a", &pf2, &VarConstraints::default(), &JointConstraints::default()).unwrap()); // A="a", wildcard empty
        }

        #[test]
        fn test_very_long_entry() {
            let long_entry = "a".repeat(1000);
            let pf = "A".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists(&long_entry, &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());

            let pf2 = "A*B".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists(&long_entry, &pf2, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_backtracking_intensive_pattern() {
            let pf = "A*B*C".parse::<ParsedForm>().unwrap();
            let all_matches = match_equation_all("aaaa", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap();
            // TODO? count?
            assert!(!all_matches.is_empty());
        }

        #[test]
        fn test_repeated_variable_same_value() {
            let pf = "ABA".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("catc", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap()); // A="c", B="at"
            assert!(!match_equation_exists("catd", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_minimum_length_patterns() {
            let pf = "ABCD".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("abcd", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap()); // exact minimum
            assert!(!match_equation_exists("abc", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap()); // too short
        }

        #[test]
        fn test_wildcard_at_boundaries() {
            let pf = "*A".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("test", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());

            let pf2 = "A*".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("test", &pf2, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        // Anagram edge cases
        #[test]
        fn test_anagram_with_duplicate_letters() {
            // /aab means any permutation of a,a,b
            let pf = "/aab".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("aab", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("aba", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("baa", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            // Wrong count: has two b's instead of one
            assert!(!match_equation_exists("abb", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            // Wrong count: only one 'a'
            assert!(!match_equation_exists("abc", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_anagram_single_char() {
            let pf = "/a".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("a", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(!match_equation_exists("b", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_anagram_all_same_letter() {
            let pf = "/aaa".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("aaa", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(!match_equation_exists("aab", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_anagram_shorter_than_entry() {
            // Anagram of 3 chars, but entry has more
            let pf = "/abc".parse::<ParsedForm>().unwrap();
            // Entry "abcd" has 4 chars, anagram needs exactly 3
            assert!(!match_equation_exists("abcd", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_anagram_longer_than_entry() {
            // Anagram of 5 chars, but entry only has 3
            let pf = "/abcde".parse::<ParsedForm>().unwrap();
            assert!(!match_equation_exists("abc", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        // Charset edge cases
        #[test]
        fn test_charset_basic() {
            let pf = "[abc]".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("a", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("b", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("c", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(!match_equation_exists("d", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(!match_equation_exists("ab", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_charset_with_range() {
            // [a-c] should match a, b, or c
            let pf = "[a-c]".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("a", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("b", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("c", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(!match_equation_exists("d", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(!match_equation_exists("ab", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_charset_multiple_in_pattern() {
            let pf = "[abc][xyz]".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("ax", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("by", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("cz", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(!match_equation_exists("ad", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(!match_equation_exists("dx", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_charset_with_variables() {
            let pf = "A[xyz]B".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("axb", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap()); // A="a", B="b"
            assert!(match_equation_exists("ayb", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("testxing", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap()); // A="test", B="ing"
            assert!(!match_equation_exists("tab", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap()); // middle char 'a' not in [xyz]
        }

        #[test]
        fn test_charset_single_char() {
            let pf = "[a]".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("a", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(!match_equation_exists("b", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_negated_charset_matching() {
            let pf = "[^abc]".parse::<ParsedForm>().unwrap();
            assert!(!match_equation_exists("a", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(!match_equation_exists("b", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(!match_equation_exists("c", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("d", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("z", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_negated_charset_range_matching() {
            let pf = "[^a-c]".parse::<ParsedForm>().unwrap();
            assert!(!match_equation_exists("a", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(!match_equation_exists("c", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("d", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_negated_charset_complex_matching() {
            let pf = "[^pb-mr]".parse::<ParsedForm>().unwrap();
            // all 14 excluded chars
            for s in ["p", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "r"] {
                assert!(!match_equation_exists(s, &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap(),
                    "{s} should not match");
            }
            // all 12 included chars
            for s in ["a", "n", "o", "q", "s", "t", "u", "v", "w", "x", "y", "z"] {
                assert!(match_equation_exists(s, &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap(),
                    "{s} should match");
            }
        }

        // Vowel and consonant edge cases
        #[test]
        fn test_vowel_matching() {
            let pf = "@".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("a", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("e", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("i", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("o", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("u", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("y", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(!match_equation_exists("b", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_consonant_matching() {
            let pf = "#".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("b", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("c", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("z", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(!match_equation_exists("a", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(!match_equation_exists("e", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_mixed_wildcards_vowels_consonants() {
            // Pattern: consonant, vowel, consonant
            let pf = "#@#".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("cat", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("bed", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(match_equation_exists("cab", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
            assert!(!match_equation_exists("ate", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap()); // starts with vowel
            assert!(!match_equation_exists("tea", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap()); // ends with vowel
        }

        // Constraint edge cases
        #[test]
        fn test_variable_with_not_equal_constraint() {
            let pf = "AB".parse::<ParsedForm>().unwrap();
            let mut constraints = VarConstraints::default();
            // Create constraint: A != B (need to set on both variables for bidirectional check)
            let mut constraint_a = VarConstraint::default();
            constraint_a.not_equal.insert('B');
            constraints.insert('A', constraint_a);

            let mut constraint_b = VarConstraint::default();
            constraint_b.not_equal.insert('A');
            constraints.insert('B', constraint_b);

            assert!(match_equation_exists("ab", &pf, &constraints, &JointConstraints::default()).unwrap()); // A="a", B="b" (different)
            assert!(!match_equation_exists("aa", &pf, &constraints, &JointConstraints::default()).unwrap()); // A="a", B="a" (violates A!=B)
        }

        #[test]
        fn test_variable_with_length_constraint() {
            let pf = "AB".parse::<ParsedForm>().unwrap();
            let mut constraints = VarConstraints::default();
            // Constraint: |A| >= 2
            let mut constraint_a = VarConstraint::default();
            constraint_a.bounds.min_len = 2;
            constraints.insert('A', constraint_a);

            assert!(match_equation_exists("abc", &pf, &constraints, &JointConstraints::default()).unwrap()); // A="ab", B="c"
            assert!(!match_equation_exists("ab", &pf, &constraints, &JointConstraints::default()).unwrap()); // A can't be just "a" (min 2)
        }

        #[test]
        fn test_variable_with_max_length_constraint() {
            let pf = "AB".parse::<ParsedForm>().unwrap();
            let mut constraints = VarConstraints::default();
            // Constraint: |A| <= 2
            let mut constraint_a = VarConstraint::default();
            constraint_a.bounds.max_len_opt = Some(2);
            constraints.insert('A', constraint_a);

            assert!(match_equation_exists("abc", &pf, &constraints, &JointConstraints::default()).unwrap()); // A="ab", B="c"
            assert!(match_equation_exists("abcd", &pf, &constraints, &JointConstraints::default()).unwrap()); // A="ab", B="cd"
            // But can't have A take more than 2 chars
            let all = match_equation_all("abcd", &pf, &constraints, &JointConstraints::default()).unwrap();
            // Check that no solution has A with length > 2
            for binding in &all {
                if let Some(a_val) = binding.get('A') {
                    assert!(a_val.len() <= 2, "A should have max length 2, got {}", a_val.len());
                }
            }
        }

        // Recursion depth tests
        #[test]
        fn test_deep_recursion_many_variables() {
            // Pattern with many variables forces deep recursion
            let pf = "ABCDEFGHIJ".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("abcdefghij", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }

        #[test]
        fn test_deep_recursion_many_wildcards() {
            // Many wildcards force backtracking
            let pf = "*.*.*.*.*".parse::<ParsedForm>().unwrap();
            assert!(match_equation_exists("testing", &pf, &VarConstraints::default(), &JointConstraints::default()).unwrap());
        }
    }
}
