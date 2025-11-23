use std::collections::HashSet;
use crate::constraints::{Bounds, VarConstraint};
use crate::errors::ParseError;

/// Parse a complex constraint of the form `X=(...)`.
///
/// The right-hand side may be optionally wrapped in parentheses. Inside:
///
/// - **With a colon** (`len_range:literal`):
///   - The part before the colon must be a valid length range (e.g., `5-7` or `3`).
///   - The part after the colon is treated as a literal constraint string.
///   - Exactly one colon is allowed; more than one yields `InvalidComplexConstraint`.
///
/// - **Without a colon**:
///   - First try to parse the whole string as a length range. If successful, this
///     becomes a pure length constraint.
///   - Otherwise, treat the string as a literal constraint, with default bounds.
///
/// Returns the variable name (single char) and a `VarConstraint` carrying the
/// parsed `Bounds` and optional literal form string.
///
/// Errors:
/// - `InvalidComplexConstraint` if the input is malformed (e.g., no `=`, no variable,
///   too many colons, or an unparsable length range when one was expected).
pub(crate) fn get_complex_constraint(form: &str) -> Result<(char, VarConstraint), Box<ParseError>> {
    // check that `form` has at least 1 '='
    let (var_raw, rhs_raw) = form.split_once('=').ok_or_else(|| {
        Box::new(ParseError::InvalidComplexConstraint {
            str: "expected 1 equals sign (not 0)".to_string(),
        })
    })?;

    // check that `form` had at most 1 '='
    if rhs_raw.contains('=') {
        let eq_count = form.chars().filter(|&c| c == '=').count();
        return Err(Box::new(ParseError::InvalidComplexConstraint {
            str: format!("expected 1 equals sign (not {eq_count})"),
        }));
    }

    // Find the variable (ensuring there is only one)
    let mut chars = var_raw.chars();
    let var_char = chars.next().ok_or_else(|| {
        Box::new(ParseError::InvalidComplexConstraint { str: form.to_string() })
    })?;
    if chars.next().is_some() {
        return Err(Box::new(ParseError::InvalidComplexConstraint { str: form.to_string() }));
    }
    let var = var_char;

    // complex constraints must have surrounding parentheses
    let rhs = rhs_raw.trim();
    let rhs = rhs
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| {
            Box::new(ParseError::InvalidComplexConstraint {
                str: format!("A complex constraint requires parentheses: perhaps use '{var_char}=({rhs})' not '{form}'"),
            })
        })?;

    let (bounds, form_str_opt) = if let Some((lhs, rhs_after_colon)) = rhs.split_once(':') {
        if rhs_after_colon.contains(':') {
            return Err(Box::new(ParseError::InvalidComplexConstraint {
                str: format!(
                    "too many colons--0 or 1 expected (not {})",
                    rhs.chars().filter(|c| *c == ':').count()
                ),
            }));
        }
        let bounds = parse_length_range(lhs)?;
        (bounds, Some(rhs_after_colon.to_string()))
    } else {
        match parse_length_range(rhs) {
            Ok(bounds) => (bounds, None),
            Err(_) => (Bounds::default(), Some(rhs.to_string())),
        }
    };

    let vc = VarConstraint {
        bounds,
        form: form_str_opt,
        not_equal: HashSet::default(),
        ..Default::default()
    };

    Ok((var, vc))
}

/// Parse a string into a `Bounds` representing a length constraint.
///
/// Accepted forms:
/// - `"N"`
///   A single number.
///   Example: `"5"` → `[5, 5]`.
///
/// - `"N-M"`
///   An explicit bounded range from `N` to `M` (inclusive). Both sides must be integers.
///   Example: `"4-7"` → `[4, 7]`.
///
/// - `"N-"`
///   An open-ended range starting from `N`.
///   Example: `"4-"` → `[4, ∞)`.
///
/// - `"-M"`
///   A range from the default minimum up to `M` (inclusive).
///   Example: `"-7"` → `[DEFAULT_MIN, 7]`.
///
/// Rejected forms:
/// - Non-numeric tokens (e.g., `"4-5a"`, `"foo"`, `"foo-7"`).
/// - More than one dash (e.g., `"4-5-6"`).
/// - Empty string.
///
/// Errors return `ParseError::InvalidLengthRange` with the original input.
pub(crate) fn parse_length_range(input: &str) -> Result<Bounds, Box<ParseError>> {
    // Split on dash, but preserve raw strings so we can distinguish
    // between `""` (empty) and `"junk"` (invalid).
    let raw_parts: Vec<&str> = input.split('-').collect();

    // Basic validation: must have 1 or 2 parts.
    if raw_parts.is_empty() || raw_parts.len() > 2 {
        return Err(Box::new(ParseError::InvalidLengthRange { input: input.to_string() }));
    }

    match raw_parts.as_slice() {
        // Case: single number, e.g., "5"
        [single] => {
            let n = single.parse::<usize>().map_err(|_| {
                Box::new(ParseError::InvalidLengthRange { input: input.to_string() })
            })?;
            Ok(Bounds::of(n, n))
        }

        // Case: two pieces, e.g., "lhs-rhs"
        [lhs, rhs] => {
            // LHS: either a number or empty (default min).
            let min = if lhs.is_empty() {
                VarConstraint::DEFAULT_MIN
            } else {
                lhs.parse::<usize>().map_err(|_| {
                    Box::new(ParseError::InvalidLengthRange { input: input.to_string() })
                })?
            };

            // RHS: either a number or empty (open-ended).
            if rhs.is_empty() {
                // "N-" → open ended upper bound
                Ok(Bounds::of_unbounded(min))
            } else {
                let max = rhs.parse::<usize>().map_err(|_| {
                    Box::new(ParseError::InvalidLengthRange { input: input.to_string() })
                })?;
                Ok(Bounds::of(min, max))
            }
        }

        _ => Err(Box::new(ParseError::InvalidLengthRange { input: input.to_string() })),
    }
}

#[cfg(test)]
mod tests {
    use crate::patterns::EquationContext;
    use super::*;

    #[test]
    fn test_parse_length_range() {
        assert_eq!(Bounds::of(2, 3), parse_length_range("2-3").unwrap());
        assert_eq!(Bounds::of(5, 5), parse_length_range("5").unwrap());
        assert_eq!(Bounds::of(VarConstraint::DEFAULT_MIN, 3), parse_length_range("-3").unwrap());
        assert_eq!(Bounds::of_unbounded(1), parse_length_range("1-").unwrap());
        assert_eq!(Bounds::of(7, 7), parse_length_range("7").unwrap());
        assert!(matches!(*parse_length_range("").unwrap_err(), ParseError::InvalidLengthRange { input } if input.is_empty() ));
        assert!(matches!(*parse_length_range("1-2-3").unwrap_err(), ParseError::InvalidLengthRange { input } if input == "1-2-3" ));
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
    fn test_parse_length_range_valid_cases() {
        // Exact bounded range
        assert_eq!(Bounds::of(2, 3), parse_length_range("2-3").unwrap());
        // Open start: "-3" → [DEFAULT_MIN, 3]
        assert_eq!(
            Bounds::of(VarConstraint::DEFAULT_MIN, 3),
            parse_length_range("-3").unwrap()
        );
        // Open end: "1-" → [1, ∞)
        assert_eq!(Bounds::of_unbounded(1), parse_length_range("1-").unwrap());
        // Single number: "7" → exactly 7
        assert_eq!(Bounds::of(7, 7), parse_length_range("7").unwrap());
    }

    #[test]
    fn test_parse_length_range_invalid_cases() {
        // Empty string
        assert!(matches!(
            *parse_length_range("").unwrap_err(),
            ParseError::InvalidLengthRange { input } if input.is_empty()
        ));

        // Too many dashes
        assert!(matches!(
            *parse_length_range("1-2-3").unwrap_err(),
            ParseError::InvalidLengthRange { input } if input == "1-2-3"
        ));

        // Garbage tokens
        assert!(matches!(
            *parse_length_range("foo").unwrap_err(),
            ParseError::InvalidLengthRange { input } if input == "foo"
        ));

        assert!(matches!(
            *parse_length_range("4-foo").unwrap_err(),
            ParseError::InvalidLengthRange { input } if input == "4-foo"
        ));

        assert!(matches!(
            *parse_length_range("foo-7").unwrap_err(),
            ParseError::InvalidLengthRange { input } if input == "foo-7"
        ));

        assert!(matches!(
            *parse_length_range("4-5a").unwrap_err(),
            ParseError::InvalidLengthRange { input } if input == "4-5a"
        ));
    }

    #[test]
    fn test_complex_constraint_len_and_form() {
        // A=(3-5:g@*) → variable A has bounds [3,5], form "g@*"
        let eq = "A;A=(3-5:g@*)".parse::<EquationContext>().unwrap();
        let a = eq.var_constraints.get('A').unwrap();
        assert_eq!(a.bounds, Bounds::of(3, 5));
        assert_eq!(a.form.as_deref(), Some("g@*"));
    }

    #[test]
    fn test_complex_constraint_form_only() {
        // A=(r*) → variable A has unbounded length (default min), form "r*"
        let eq = "A;A=(r*)".parse::<EquationContext>().unwrap();
        let a = eq.var_constraints.get('A').unwrap();
        assert_eq!(a.bounds, Bounds::of_unbounded(VarConstraint::DEFAULT_MIN));
        assert_eq!(a.form.as_deref(), Some("r*"));
    }

    #[test]
    fn test_complex_constraint_len_only() {
        // A=(6) → variable A has exact length 6, no form
        let eq = "A;A=(6)".parse::<EquationContext>().unwrap();
        let a = eq.var_constraints.get('A').unwrap();
        assert_eq!(a.bounds, Bounds::of(6, 6));
        assert_eq!(a.form, None);
    }

    #[test]
    fn test_complex_constraint_open_start() {
        // A=(-4:x*) → variable A has bounds [DEFAULT_MIN,4], form "x*"
        let eq = "A;A=(-4:x*)".parse::<EquationContext>().unwrap();
        let a = eq.var_constraints.get('A').unwrap();
        assert_eq!(a.bounds, Bounds::of(VarConstraint::DEFAULT_MIN, 4));
        assert_eq!(a.form.as_deref(), Some("x*"));
    }

    #[test]
    fn test_complex_constraint_open_end() {
        // A=(2-:z*) → variable A has bounds [2,∞), form "z*"
        let eq = "A;A=(2-:z*)".parse::<EquationContext>().unwrap();
        let a = eq.var_constraints.get('A').unwrap();
        assert_eq!(a.bounds, Bounds::of_unbounded(2));
        assert_eq!(a.form.as_deref(), Some("z*"));
    }

}
