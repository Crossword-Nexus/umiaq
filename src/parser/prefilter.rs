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

    // check cache first; if lock is poisoned, recover and continue
    if let Ok(guard) = cache.lock() {
        if let Some(re) = guard.get(pattern).cloned() {
            return Ok(re);
        }
    }
    // if lock was poisoned, we just compile without caching

    // Compile outside the lock.
    let compiled = Regex::new(pattern)?;

    // Insert with a double-check in case another thread inserted it meanwhile.
    // if lock is poisoned, we still return the compiled regex (but don't cache it)
    if let Ok(mut guard) = cache.lock() {
        if let Some(existing) = guard.get(pattern).cloned() {
            return Ok(existing);
        }
        guard.insert(pattern.to_string(), compiled.clone());
    }
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

                if already_has_group {
                    // Subsequent occurrences → backreference
                    let _ = write!(regex_str, "\\{}", var_to_backreference_num[idx]);
                } else if occurs_many {
                    // First of multiple occurrences → capture group
                    backreference_index += 1;
                    var_to_backreference_num[idx] = backreference_index;
                    // Inline constraint form if present
                    if let Some(nested) = get_lookahead(constraints, *c)? {
                        // Capture group with constraint
                        let _ = write!(regex_str, "(?={nested})(.+)");
                    } else {
                        regex_str.push_str("(.+)");
                    }
                // Inline constraint form if present
                } else if let Some(nested) = get_lookahead(constraints, *c)? {
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

fn get_lookahead(constraints: Option<&VarConstraints>, c: char) -> Result<Option<String>, Box<ParseError>> {
    let lookahead_raw = match constraints.and_then(|cs| cs.get(c)) {
        Some(vc) => {
            if let Some(parsed) = vc.get_parsed_form()? {
                Some(render_parts_to_regex(&parsed.parts, None)?)
            } else {
                None
            }
        }
        None => None,
    };

    Ok(lookahead_raw)
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
fn uc_letter_to_num(c: char) -> Result<usize, Box<ParseError>> {
    letter_to_num(c, 'A' as usize).map_err(|_| {
        Box::new(ParseError::InvalidVariableName {
            var: c.to_string()
        })
    })
}

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
pub(crate) fn has_inlineable_var_form(
    parts: &[FormPart],
    constraints: &VarConstraints,
) -> Result<bool, Box<ParseError>> {
    for part in parts {
        if let FormPart::Var(c) = part
            && let Some(vc) = constraints.get(*c)
            && vc.get_parsed_form()?.is_some() {
                return Ok(true);
            }
    }
    Ok(false)
}


/// Try to improve the prefilter for this (form, constraints) pair by building a constraint-aware
/// regex (with lookaheads) if possible; otherwise, reuse the already-cached `ParsedForm.prefilter`.
pub(crate) fn build_prefilter_regex(
    parsed_form: &ParsedForm,
    vcs: &VarConstraints,
) -> Result<Regex, Box<ParseError>> {
    // Decide which regex string to use
    let regex_str = if has_inlineable_var_form(&parsed_form.parts, vcs)? {
        // inlineable: regenerate from parts
        format!(
            "^{}$",
            form_to_regex_str_with_constraints(&parsed_form.parts, vcs)?
        )
    } else {
        // not inlineable: just reuse prefilter string
        parsed_form.prefilter.as_str().to_string()
    };

    // TODO perhaps get_regex shouldn't be throwing exceptions on cases that we shouldn't be panicking on
    // Compile the upgraded regex; if compilation fails (e.g., due to complex lookaheads),
    // fall back to the existing prefilter--this is safe because the existing prefilter
    // is less specific but still correct
    Ok(get_regex(&regex_str).unwrap_or_else(|e| {
        debug_assert!(false, "Failed to compile upgraded prefilter regex: {}. Falling back to original.", e);
        parsed_form.prefilter.clone()
    }))
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{VarConstraint, VarConstraints};

    #[test]
    fn test_constraint_prefilter_string_single_use() {
        let pf = "A".parse::<ParsedForm>().unwrap();
        let mut vcs = VarConstraints::default();
        let vc = VarConstraint { form: Some("x*a".to_string()), ..Default::default() };
        vcs.insert('A', vc);
        let re_str = form_to_regex_str_with_constraints(&pf.parts, &vcs).unwrap();
        assert_eq!(re_str, "(?=x.*a).+");
    }

    #[test]
    fn test_prefilter_upgrade_prunes_nonmatching_words() {
        let mut pf = "A".parse::<ParsedForm>().unwrap();
        let mut vcs = VarConstraints::default();
        let vc = VarConstraint { form: Some("x*a".to_string()), ..Default::default() };
        vcs.insert('A', vc);

        assert!(pf.prefilter.is_match("abba").unwrap());
        let upgraded = build_prefilter_regex(&pf, &vcs).unwrap();
        pf.prefilter = upgraded;

        assert!(pf.prefilter.is_match("xya").unwrap());
        assert!(!pf.prefilter.is_match("abba").unwrap());
    }
}
