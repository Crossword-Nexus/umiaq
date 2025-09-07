use crate::bindings::Bindings;
use crate::constraints::{VarConstraint, VarConstraints};
use crate::joint_constraints::JointConstraints;
use crate::umiaq_char::UmiaqChar;

use super::form::{FormPart, ParsedForm};

/// Validate whether a candidate binding value is allowed under a `VarConstraint`.
///
/// Checks:
/// 1. If length constraints are present, enforce them
/// 2. If `form` is present, the value must itself match that form.
/// 3. The value must not equal any variable listed in `not_equal` that is already bound.
fn is_valid_binding(val: &str, constraints: &VarConstraint, bindings: &Bindings) -> bool {
    // 1. Length checks (if configured)
    if val.len() < constraints.min_length
        || constraints.max_length.is_some_and(|max_len| val.len() > max_len)
    {
        return false;
    }

    // 2. Apply nested-form constraint if present (use cached parse)
    if let Some(parsed) = constraints.get_parsed_form()
        && !match_equation_exists(val, parsed, &VarConstraints::default(), JointConstraints::default())
    {
        return false;
    }

    // 3. Check "not equal" constraints
    for &other in &constraints.not_equal {
        if let Some(existing) = bindings.get(other) && existing == val {
            return false;
        }
    }

    true
}

/// Return `true` if at least one binding satisfies the equation.
#[must_use]
pub fn match_equation_exists(
    word: &str,
    parts: &ParsedForm,
    constraints: &VarConstraints,
    joint_constraints: JointConstraints,
) -> bool {
    let mut results: Vec<Bindings> = Vec::new();
    match_equation_internal(word, parts, false, &mut results, constraints, joint_constraints);
    results.into_iter().next().is_some()
}

/// Return all bindings that satisfy the equation.
#[must_use]
pub fn match_equation_all(
    word: &str,
    parts: &ParsedForm,
    constraints: &VarConstraints,
    joint_constraints: JointConstraints,
) -> Vec<Bindings> {
    let mut results: Vec<Bindings> = Vec::new();
    match_equation_internal(word, parts, true, &mut results, constraints, joint_constraints);
    results
}

/// Core backtracking search that tries to match `word` against `parts`.
///
/// - Uses the compiled `prefilter` on `ParsedForm` (already upgraded upstream).
/// - Recursively attempts to bind variables and match literals/wildcards.
/// - Stops early if `all_matches` is false and a single match is found.
fn match_equation_internal(
    word: &str,
    parsed_form: &ParsedForm,
    all_matches: bool,
    results: &mut Vec<Bindings>,
    constraints: &VarConstraints,
    joint_constraints: JointConstraints,
) {
    /// Helper to reverse a bound value if the part is `RevVar`.
    fn get_reversed_or_not(first: &FormPart, val: &str) -> String {
        if matches!(first, FormPart::RevVar(_)) {
            val.chars().rev().collect()
        } else {
            val.to_owned()
        }
    }

    /// Try to consume exactly one char if present, and apply `pred` to it.
    /// - If the predicate passes, recurse with the rest of the chars.
    /// - Otherwise return false (dead end).
    ///
    /// Covers Dot (any char), Vowel, Consonant, Charset, etc.
    fn take_if(
        chars: &[char],
        rest: &[FormPart],
        hp: &mut HelperParams,
        pred: impl Fn(&char) -> bool,
    ) -> bool {
        chars
            .split_first()
            .is_some_and(|(c, rest_chars)| pred(c) && helper(rest_chars, rest, hp))
    }

    // --- Inner recursive matcher -------------------------------------------------
    fn helper(chars: &[char], parts: &[FormPart], hp: &mut HelperParams) -> bool {
        // Base case: no parts left
        if parts.is_empty() {
            if chars.is_empty() {
                // Check the joint constraints (if any)
                if hp.joint_constraints.all_satisfied(hp.bindings) {
                    let mut full_result = hp.bindings.clone();
                    full_result.set_word(hp.word);
                    hp.results.push(full_result);
                    return !hp.all_matches; // Stop early if only one match needed
                }
            }
            return false;
        }

        let (first, rest) = (&parts[0], &parts[1..]);

        match first {
            FormPart::Lit(s) => {
                // Literal match (case-insensitive, stored lowercase)
                is_prefix(s, chars, rest, hp)
            }
            FormPart::Star => {
                // Zero-or-more wildcard; try all possible splits
                (0..=chars.len()).any(|i| helper(&chars[i..], rest, hp))
            }

            // Combined vowel, consonant, charset, dot cases
            FormPart::Dot => take_if(chars, rest, hp, |_| true),
            FormPart::Vowel => take_if(chars, rest, hp, char::is_vowel),
            FormPart::Consonant => take_if(chars, rest, hp, char::is_consonant),
            FormPart::Charset(s) => take_if(chars, rest, hp, |c| s.contains(c)),

            FormPart::Anagram(ag) => {
                // Match if the next len chars are an anagram of target
                let len = ag.len;
                chars.len() >= len
                    && ag.is_anagram(&chars[..len]).unwrap() // TODO! handle error
                    && helper(&chars[len..], rest, hp)
            }

            FormPart::Var(var_name) | FormPart::RevVar(var_name) => {
                if let Some(bound_val) = hp.bindings.get(*var_name) {
                    // Already bound: must match exactly
                    is_prefix(&get_reversed_or_not(first, bound_val), chars, rest, hp)
                } else {
                    // Not bound yet: try binding to all possible lengths
                    // To prune the search space, apply length constraints up front
                    let min_len = hp
                        .constraints
                        .get(*var_name)
                        .map_or(VarConstraint::DEFAULT_MIN, |vc| vc.min_length);
                    let max_len_cfg = hp
                        .constraints
                        .get(*var_name)
                        .and_then(|vc| vc.max_length)
                        .unwrap_or(chars.len());

                    let avail = chars.len();

                    // If the minimum exceeds what we have left, this path can't work
                    if min_len > avail {
                        false
                    } else {
                        // Never try to slice past what's actually available
                        let capped_max = std::cmp::min(max_len_cfg, avail);

                        (min_len..=capped_max).any(|l| {
                            let candidate_chars = &chars[..l];

                            let bound_val: String = if matches!(first, FormPart::RevVar(_)) {
                                candidate_chars.iter().rev().collect()
                            } else {
                                candidate_chars.iter().collect()
                            };

                            // Apply variable-specific constraints
                            let valid = hp
                                .constraints
                                .get(*var_name).is_none_or(|c| {
                                    is_valid_binding(&bound_val, c, hp.bindings)
                                });

                            if valid {
                                hp.bindings.set(*var_name, bound_val);
                                let retval = helper(&chars[l..], rest, hp) && !hp.all_matches;
                                if !retval {
                                    // Backtrack only when continuing the search.
                                    hp.bindings.remove(*var_name);
                                }
                                retval
                            } else {
                                false
                            }
                        })
                    }
                }
            }
        }
    }

    /// Returns true if `prefix` is a prefix of `chars`
    fn is_prefix(
        prefix: &str,
        chars: &[char],
        rest: &[FormPart],
        helper_params: &mut HelperParams,
    ) -> bool {
        let prefix_len = prefix.len();
        if chars.len() >= prefix_len
            && chars[..prefix_len]
                .iter()
                .copied()
                .zip(prefix.chars())
                .all(|(a, b)| a == b)
        {
            helper(&chars[prefix_len..], rest, helper_params)
        } else {
            false
        }
    }

    // === PREFILTER STEP ===
    if !parsed_form.prefilter.is_match(word).unwrap_or(false) {
        return;
    }

    // Normalize word and start recursive matching
    let mut hp = HelperParams {
        bindings: &mut Bindings::default(),
        results,
        all_matches,
        word,
        constraints,
        joint_constraints,
    };

    helper(&word.chars().collect::<Vec<_>>(), &parsed_form.parts, &mut hp);
}

// Helper params for recursion
struct HelperParams<'a> {
    bindings: &'a mut Bindings,
    results: &'a mut Vec<Bindings>,
    all_matches: bool,
    word: &'a str,
    constraints: &'a VarConstraints,
    joint_constraints: JointConstraints,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_palindrome_matching() {
        let pf = "A~A".parse::<ParsedForm>().unwrap();
        assert!(match_equation_exists("noon", &pf, &VarConstraints::default(), JointConstraints::default()));
        assert!(!match_equation_exists("radar", &pf, &VarConstraints::default(), JointConstraints::default()));
        assert!(!match_equation_exists("test", &pf, &VarConstraints::default(), JointConstraints::default()));
    }

    #[test]
    fn test_match_equation_exists() {
        let pf = "A~A[rstlne]/jon@#.*".parse::<ParsedForm>().unwrap();
        assert!(match_equation_exists("aaronjudge", &pf, &VarConstraints::default(), JointConstraints::default()));
        assert!(!match_equation_exists("noon", &pf, &VarConstraints::default(), JointConstraints::default()));
        assert!(!match_equation_exists("toon", &pf, &VarConstraints::default(), JointConstraints::default()));
    }

    #[test]
    fn test_literal_matching() {
        let pf = "abc".parse::<ParsedForm>().unwrap();
        assert!(match_equation_exists("abc", &pf, &VarConstraints::default(), JointConstraints::default()));
        assert!(!match_equation_exists("xyz", &pf, &VarConstraints::default(), JointConstraints::default()));
    }
}
