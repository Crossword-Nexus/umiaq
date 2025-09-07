use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use fancy_regex::Regex;

use crate::constraints::VarConstraints;
use crate::errors::ParseError;
use crate::parser::utils::letter_to_num;
use crate::umiaq_char::{CONSONANTS, NUM_POSSIBLE_VARIABLES, VOWELS};

use super::form::{FormPart, ParsedForm};

/// Global, lazily initialized cache of compiled regexes.
///
/// - `OnceLock` ensures the cache is created at most once, on first use.
/// - We wrap the `HashMap` in a `Mutex` to provide **interior mutability** and
///   **thread safety**. A plain `HashMap` isn't thread-safe and cannot be
///   mutated through a shared reference; `Mutex` gives us a safe, exclusive
///   handle when inserting or reading.
///
/// Locking strategy:
/// - We hold the `Mutex` only while accessing the map (lookups/inserts).
/// - We compile outside the lock to keep contention low, with a "double-check"
///   before insert to avoid duplicate work in rare races.
/// - `Regex` clones are cheap (internally ref-counted), so we release the lock quickly.
static REGEX_CACHE: OnceLock<Mutex<HashMap<String, Regex>>> = OnceLock::new();

/// Return a compiled `Regex` for `pattern`, caching the result.
pub(crate) fn get_regex(pattern: &str) -> Result<Regex, Box<fancy_regex::Error>> {
    let cache = REGEX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    if let Some(re) = cache.lock().unwrap().get(pattern).cloned() {
        return Ok(re);
    }

    // Compile outside the lock.
    let compiled = Regex::new(pattern)?;

    // Insert with a double-check in case another thread inserted it meanwhile.
    let mut guard = cache.lock().unwrap();
    if let Some(existing) = guard.get(pattern).cloned() {
        return Ok(existing);
    }
    guard.insert(pattern.to_string(), compiled.clone());
    Ok(compiled)
}

/// Convert a sequence of `FormPart`s into a regex string, with optional
/// variable constraints applied.
///
/// - If `constraints` is `None`, variables are rendered as plain `.+`,
///   or as capture groups/backreferences when they repeat.
/// - If `constraints` is `Some`, then for the **first occurrence** of a
///   variable with an attached form constraint, we inject a lookahead
///   such as `(?=x.*a).+` (or `(?=x.*a)(.+)` if it’s multi-use).
/// - Reversed variables (`~A`) are always rendered as `.+`,
///   since reversing constraint regexes is not practical.
/// - Other `FormPart` variants (literals, wildcards, charsets, anagrams)
///   are handled uniformly, regardless of constraints.
fn render_parts_to_regex(
    parts: &[FormPart],
    constraints: Option<&VarConstraints>,
) -> Result<String, Box<ParseError>> {
    use std::fmt::Write;

    // Count how many times each variable / revvar occurs
    let (var_counts, rev_var_counts) = get_var_and_rev_var_counts(parts)?;

    // Bookkeeping for assigning capture-group indices
    let mut var_to_backreference_num = [0; NUM_POSSIBLE_VARIABLES];
    let mut rev_var_to_backreference_num = [0; NUM_POSSIBLE_VARIABLES];
    let mut backreference_index = 0;

    let mut regex_str = String::new();

    for part in parts {
        match part {
            // --- Variable handling (with optional constraints) ---
            FormPart::Var(c) => {
                let idx = uc_letter_to_num(*c)?;
                let occurs_many = var_counts[idx] > 1;
                let already_has_group = var_to_backreference_num[idx] != 0;

                // Inline constraint form if present
                let lookahead = constraints
                    .and_then(|cs| cs.get(*c))
                    .and_then(|vc| vc.get_parsed_form())
                    // Constraint forms are guaranteed to be var-free
                    .map(|pf| render_parts_to_regex(&pf.parts, None))
                    .transpose()?;

                if already_has_group {
                    // Subsequent occurrences → backreference
                    let _ = write!(regex_str, "\\{}", var_to_backreference_num[idx]);
                } else if occurs_many {
                    // First of multiple occurrences → capture group
                    backreference_index += 1;
                    var_to_backreference_num[idx] = backreference_index;
                    if let Some(nested) = lookahead {
                        // Capture group with constraint
                        let _ = write!(regex_str, "(?={nested})(.+)");
                    } else {
                        regex_str.push_str("(.+)");
                    }
                } else if let Some(nested) = lookahead {
                    // Single-use variable with constraint
                    let _ = write!(regex_str, "(?={nested}).+");
                } else {
                    // Single-use variable, no constraint
                    regex_str.push_str(".+");
                }
            }

            // --- Reversed variable (no constraints supported) ---
            FormPart::RevVar(c) => {
                let idx = uc_letter_to_num(*c)?;
                let occurs_many = rev_var_counts[idx] > 1;
                let already_has_group = rev_var_to_backreference_num[idx] != 0;

                if already_has_group {
                    let _ = write!(regex_str, "\\{}", rev_var_to_backreference_num[idx]);
                } else if occurs_many {
                    backreference_index += 1;
                    rev_var_to_backreference_num[idx] = backreference_index;
                    regex_str.push_str("(.+)");
                } else {
                    regex_str.push_str(".+");
                }
            }

            // --- Other parts (shared behavior) ---
            FormPart::Lit(s) => regex_str.push_str(&fancy_regex::escape(s)),
            FormPart::Dot => regex_str.push('.'),
            FormPart::Star => regex_str.push_str(".*"),
            FormPart::Vowel => { let _ = write!(regex_str, "[{VOWELS}]"); }
            FormPart::Consonant => { let _ = write!(regex_str, "[{CONSONANTS}]"); }
            FormPart::Charset(chars) => {
                regex_str.push('[');
                for c in chars { regex_str.push(*c); }
                regex_str.push(']');
            }
            FormPart::Anagram(ag) => {
                let len = ag.len;
                let class = fancy_regex::escape(ag.as_string.as_str());
                let _ = write!(regex_str, "[{class}]{{{len}}}");
            }
        }
    }

    Ok(regex_str)
}

/// Convert a parsed form into a regex string without constraints.
///
/// This is a thin wrapper over `render_parts_to_regex` with `constraints = None`.
pub(crate) fn form_to_regex_str(parts: &[FormPart]) -> Result<String, Box<ParseError>> {
    render_parts_to_regex(parts, None)
}

/// Convert a parsed form into a regex string, applying variable constraints.
///
/// This is a thin wrapper over `render_parts_to_regex` with `constraints = Some(vcs)`.
pub(crate) fn form_to_regex_str_with_constraints(
    parts: &[FormPart],
    constraints: &VarConstraints,
) -> Result<String, Box<ParseError>> {
    render_parts_to_regex(parts, Some(constraints))
}

// 'A' -> 0, 'B' -> 1, ..., 'Z' -> 25
fn uc_letter_to_num(c: char) -> Result<usize, Box<ParseError>> { letter_to_num(c, 'A' as usize) }

// Count occurrences of vars and revvars to decide capture/backref scheme.
fn get_var_and_rev_var_counts(
    parts: &[FormPart],
) -> Result<([usize; NUM_POSSIBLE_VARIABLES], [usize; NUM_POSSIBLE_VARIABLES]), Box<ParseError>> {
    let mut var_counts = [0; NUM_POSSIBLE_VARIABLES];
    let mut rev_var_counts = [0; NUM_POSSIBLE_VARIABLES];
    for part in parts {
        match part {
            FormPart::Var(c) => var_counts[uc_letter_to_num(*c)?] += 1,
            FormPart::RevVar(c) => rev_var_counts[uc_letter_to_num(*c)?] += 1,
            _ => (),
        }
    }
    Ok((var_counts, rev_var_counts))
}

/// True if any `Var` in `parts` has a `.form` constraint we can inline.
pub(crate) fn has_inlineable_var_form(parts: &[FormPart], constraints: &VarConstraints) -> bool {
    parts.iter().any(|p| match p {
        FormPart::Var(c) => constraints.get(*c).and_then(|vc| vc.get_parsed_form()).is_some(),
        _ => false,
    })
}

/// Try to improve the prefilter for this (form, constraints) pair by building a constraint-aware
/// regex (with lookaheads) if possible; otherwise, reuse the already-cached `ParsedForm.prefilter`.
pub(crate) fn build_prefilter_regex(
    parsed_form: &ParsedForm,
    vcs: &VarConstraints,
) -> Result<Regex, Box<ParseError>> {
    if has_inlineable_var_form(&parsed_form.parts, vcs) {
        let anchored = format!("^{}$", form_to_regex_str_with_constraints(&parsed_form.parts, vcs)?);
        Ok(get_regex(&anchored).unwrap_or_else(|_| parsed_form.prefilter.clone()))
    } else {
        Ok(parsed_form.prefilter.clone()) // TODO DRY w/2 lines above
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{VarConstraint, VarConstraints};

    #[test]
    fn test_constraint_prefilter_string_single_use() {
        let pf = "A".parse::<ParsedForm>().unwrap();
        let mut vcs = VarConstraints::default();
        let mut vc = VarConstraint::default();
        vc.form = Some("x*a".to_string());
        vcs.insert('A', vc);
        let re_str = form_to_regex_str_with_constraints(&pf.parts, &vcs).unwrap();
        assert_eq!(re_str, "(?=x.*a).+");
    }

    #[test]
    fn test_prefilter_upgrade_prunes_nonmatching_words() {
        let mut pf = "A".parse::<ParsedForm>().unwrap();
        let mut vcs = VarConstraints::default();
        let mut vc = VarConstraint::default();
        vc.form = Some("x*a".to_string());
        vcs.insert('A', vc);

        assert!(pf.prefilter.is_match("abba").unwrap());
        let upgraded = build_prefilter_regex(&pf, &vcs).unwrap();
        pf.prefilter = upgraded;

        assert!(pf.prefilter.is_match("xya").unwrap());
        assert!(!pf.prefilter.is_match("abba").unwrap());
    }
}
