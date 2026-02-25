use super::prefilter::{form_to_regex_str, get_regex};
use crate::errors::ParseError;
use crate::parser::utils::letter_to_num;
use crate::umiaq_char::{ALPHABET_SIZE, LITERAL_CHARS, VARIABLE_CHARS};
use fancy_regex::Regex;
use nom::bytes::complete::is_a;
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::one_of,
    combinator::{map, opt},
    multi::many1,
    sequence::preceded,
    IResult,
    Parser,
};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::str::FromStr;

/// Parser result type: input, output, with our custom `ParseError`
pub type PResult<'a, O> = IResult<&'a str, O, Box<ParseError>>;

/// Represents a single parsed token (component) from a "form" string.
#[derive(Debug, Clone, PartialEq)]
pub enum FormPart {
    Var(char),              // 'A': uppercase A–Z variable reference
    RevVar(char),           // '~A': reversed variable reference
    Lit(String),            // 'abc': literal lowercase sequence (lowercase)
    Dot,                    // '.' wildcard: exactly one letter
    Star,                   // '*' wildcard: zero or more letters
    Vowel,                  // '@' wildcard: any vowel (aeiouy)
    Consonant,              // '#' wildcard: any consonant (bcdf...xz)
    Charset(HashSet<char>), // '[abc]': any of the given letters
    Anagram(Alphagram),     // '/abc': any permutation of the given letters
}

impl FormPart {
    pub(crate) fn is_deterministic(&self) -> bool {
        matches!(self, FormPart::Var(_) | FormPart::RevVar(_) | FormPart::Lit(_))
    }

    pub(crate) fn anagram_of(s: &str) -> Result<FormPart, Box<ParseError>> {
        Ok(FormPart::Anagram(s.parse::<Alphagram>()?))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Alphagram {
    char_counts: [u8; ALPHABET_SIZE],
    pub(crate) as_string: String, // for regexes, pretty printing // TODO? default to alpha order (alphagram)
    pub(crate) len: usize
}

// 'a' -> 0, 'b' -> 1, ..., 'z' -> 25
fn lc_letter_to_num(c: char) -> Result<usize, Box<ParseError>> {
    letter_to_num(c, 'a' as usize).map_err(|_| {
        Box::new(ParseError::InvalidLowercaseChar {
            invalid_char: c
        })
    })
}

impl Alphagram {
    pub(crate) fn is_anagram(&self, other_entry: &[char]) -> Result<bool, Box<ParseError>> {
        if self.len != other_entry.len() {
            return Ok(false);
        }

        let mut char_counts = self.char_counts;
        for &c in other_entry {
            let c_as_num = lc_letter_to_num(c)?;
            if char_counts[c_as_num] == 0 {
                return Ok(false);
            }
            char_counts[c_as_num] -= 1;
        }

        Ok(char_counts.iter().all(|&count| count == 0))
    }
}

impl FromStr for Alphagram {
    type Err = Box<ParseError>;

    // NB: throws error if lowercase_entry contains anything but lowercase letters
    fn from_str(lowercase_entry: &str) -> Result<Self, Self::Err> {
        let mut len = 0;
        let mut char_counts = [0u8; ALPHABET_SIZE];
        for c in lowercase_entry.chars() {
            let c_as_num = lc_letter_to_num(c).map_err(|e| {
                // Wrap the lower-level error with anagram context
                if let ParseError::InvalidLowercaseChar { invalid_char } = *e {
                    Box::new(ParseError::InvalidAnagramChars {
                        anagram: lowercase_entry.to_string(),
                        invalid_char
                    })
                } else {
                    e
                }
            })?;
            char_counts[c_as_num] += 1;
            len += 1;
        }

        Ok(Alphagram { char_counts, as_string: lowercase_entry.to_string(), len })
    }
}

/// A `Vec` of `FormPart`s along with a compiled regex prefilter.
#[derive(Debug, Clone)]
pub struct ParsedForm {
    pub parts: Vec<FormPart>,
    pub prefilter: Regex,
}

impl ParsedForm {
    fn of(parts: Vec<FormPart>) -> Result<Self, Box<ParseError>> {
        // Build the base regex string from tokens only (no var constraints).
        let regex_str = form_to_regex_str(&parts)?;
        let anchored = format!("^{regex_str}$");
        let prefilter = get_regex(&anchored)?;

        Ok(ParsedForm { parts, prefilter })
    }

    // Return an iterator over the form parts
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, FormPart> {
        self.parts.iter()
    }

    /// If this form is deterministic, build the concrete entry using `env`.
    /// Returns `None` if any required var is unbound or if a nondeterministic part is present.
    pub(crate) fn materialize_deterministic_with_env(
        &self,
        env: &HashMap<char, Rc<str>>,
    ) -> Option<String> {
        self.iter()
            .map(|part| match part {
                FormPart::Lit(s) => Some(s.clone()),
                FormPart::Var(var_char) => Some(env.get(var_char)?.as_ref().to_string()),
                FormPart::RevVar(var_char) => Some(env.get(var_char)?.chars().rev().collect()),
                _ => None, // stop at first nondeterministic token
            })
            .collect::<Option<String>>()
    }
}

// Enable `for part in &parsed_form { ... }`
impl<'a> IntoIterator for &'a ParsedForm {
    type Item = &'a FormPart;
    type IntoIter = std::slice::Iter<'a, FormPart>;
    fn into_iter(self) -> Self::IntoIter { self.parts.iter() }
}

impl FromStr for ParsedForm {
    type Err = Box<ParseError>;

    /// Parse a form string into a `ParsedForm` object.
    ///
    /// Walks the input, consuming tokens one at a time with `equation_part`.
    fn from_str(raw_form: &str) -> Result<Self, Self::Err> {
        let mut rest = raw_form;
        let mut parts = Vec::new();

        while !rest.is_empty() {
            match equation_part(rest) {
                Ok((next, part)) => {
                    parts.push(part);
                    rest = next;
                }
                Err(nom::Err::Failure(e)) => {
                    // bubble up the specific ParseError
                    return Err(e);
                }
                Err(_) => {
                    // fall back to generic ParseFailure for other cases
                    return Err(Box::new(ParseError::InvalidInput {
                        str: rest.to_string(),
                        reason: format!("illegal character '{}' in pattern", rest.chars().next().unwrap_or('?'))
                    }));
                }
            }
        }


        if parts.is_empty() {
            return Err(Box::new(ParseError::EmptyForm));
        }

        ParsedForm::of(parts)
    }
}

// === Token parsers ===

fn var_ref(input: &'_ str) -> PResult<'_, FormPart> {
    map(one_of(VARIABLE_CHARS), FormPart::Var).parse(input)
}
fn rev_ref(input: &'_ str) -> PResult<'_, FormPart> {
    map(preceded(tag("~"), one_of(VARIABLE_CHARS)), FormPart::RevVar).parse(input)
}
fn literal(input: &'_ str) -> PResult<'_, FormPart> {
    map(many1(one_of(LITERAL_CHARS)), |chars| {
        FormPart::Lit(chars.into_iter().collect())
    })
    .parse(input)
}
fn dot(input: &'_ str) -> PResult<'_, FormPart> { parser_one_char_inner(input, ".", FormPart::Dot) }
fn star(input: &'_ str) -> PResult<'_, FormPart> { parser_one_char_inner(input, "*", FormPart::Star) }
fn vowel(input: &'_ str) -> PResult<'_, FormPart> { parser_one_char_inner(input, "@", FormPart::Vowel) }
fn consonant(input: &'_ str) -> PResult<'_, FormPart> { parser_one_char_inner(input, "#", FormPart::Consonant) }

// single-char tokens share the same shape
fn parser_one_char_inner<'a>(
    input: &'a str,
    tag_str: &'static str,
    form_part: FormPart
) -> PResult<'a, FormPart> {
    map(
        tag(tag_str),
        move |_| form_part.clone()
    ).parse(input)
}

/// Expands a raw charset body string (like "abc" or "a-e") into a set of characters.
///
/// This function handles:
/// - Individual characters: "abc" -> {a, b, c}
/// - Ranges: "a-e" -> {a, b, c, d, e}
/// - Mixed: "ax-z" -> {a, x, y, z}
///
/// # Errors
/// - Returns `ParseError::InvalidCharsetRange` if a range start is greater than its end (e.g., "z-a").
/// - Returns `ParseError::DanglingCharsetDash` if a dash appears at the end of the string without an end character.
fn expand_charset(body: &str) -> Result<HashSet<char>, Box<ParseError>> {
    let mut chars = HashSet::new();
    let mut iter = body.chars().peekable();

    while let Some(start) = iter.next() {
        if iter.peek() == Some(&'-') {
            iter.next(); // consume '-'
            match iter.next() {
                // Found a range like 'a-e'
                Some(end) if start <= end => chars.extend(start..=end),
                Some(end) => return Err(Box::new(ParseError::InvalidCharsetRange(start, end))),
                None => return Err(Box::new(ParseError::DanglingCharsetDash)),
            }
        } else {
            // Found a single character
            chars.insert(start);
        }
    }

    Ok(chars)
}

/// Parses a character set token from the input, supporting both positive and negated sets.
///
/// Charsets are enclosed in square brackets `[...]`. Supported syntax:
/// - **Positive set**: `[abc]` matches 'a', 'b', or 'c'.
/// - **Negated set**: `[^abc]` or `[!abc]` matches any lowercase letter *except* 'a', 'b', or 'c'.
/// - **Ranges**: `[a-e]` or `[^a-e]` uses inclusive ranges.
///
/// Negation is always calculated relative to the full 'a' through 'z' lowercase alphabet.
/// Empty charsets `[]` are not allowed.
fn charset(input: &'_ str) -> PResult<'_, FormPart> {
    // 1. Consume opening bracket '['
    let (input, _) = tag("[")(input)?;

    // 2. Check for optional negation prefix '^' or '!'
    let (input, negated) = opt(alt((tag("^"), tag("!")))).parse(input)?;

    // 3. Consume characters and ranges within the charset
    let (input, body) = is_a("-abcdefghijklmnopqrstuvwxyz").parse(input)?;

    // 4. Consume closing bracket ']'
    let (input, _) = tag("]")(input)?;

    // 5. Expand ranges (e.g., "a-e" into 'a', 'b', 'c', 'd', 'e')
    let mut chars = expand_charset(body).map_err(nom::Err::Failure)?;

    // 6. If negated, invert the set against the full 'a'-'z' alphabet
    if negated.is_some() {
        chars = ('a'..='z').filter(|c| !chars.contains(c)).collect();
    }

    Ok((input, FormPart::Charset(chars)))
}

fn anagram(input: &'_ str) -> PResult<'_, FormPart> {
    let (input, _) = tag("/")(input)?;
    let (input, chars) = many1(one_of(LITERAL_CHARS)).parse(input)?;
    let anagram_str = chars.into_iter().collect::<String>();
    let anagram_part = FormPart::anagram_of(&anagram_str).map_err(nom::Err::Failure)?;
    Ok((input, anagram_part))
}

fn equation_part(input: &'_ str) -> PResult<'_, FormPart> {
    alt((rev_ref, var_ref, anagram, charset, literal, dot, star, vowel, consonant)).parse(input)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_empty_form_error() {
        assert!(matches!(*"".parse::<ParsedForm>().unwrap_err(), ParseError::EmptyForm));
    }

    #[test] fn test_parse_failure_error() {
        assert!(matches!(*"[".parse::<ParsedForm>().unwrap_err(), ParseError::InvalidInput { ref str, .. } if str == "[" ));
    }

    #[test] fn test_parse_form_basic() {
        let parsed_form = "abc".parse::<ParsedForm>().unwrap();
        assert_eq!(vec![FormPart::Lit("abc".to_string())], parsed_form.parts);
    }

    #[test] fn test_parse_form_variable() {
        assert_eq!(vec![FormPart::Var('A')], "A".parse::<ParsedForm>().unwrap().parts);
    }

    #[test] fn test_parse_form_reversed_variable() {
        assert_eq!(vec![FormPart::RevVar('A')], "~A".parse::<ParsedForm>().unwrap().parts);
    }

    #[test] fn test_parse_form_wildcards() {
        let parts = ".*@#".parse::<ParsedForm>().unwrap().parts;
        assert_eq!(vec![FormPart::Dot, FormPart::Star, FormPart::Vowel, FormPart::Consonant], parts);
    }

    #[test] fn test_parse_form_charset() {
        assert_eq!(vec![FormPart::Charset(HashSet::from(['a', 'b', 'c']))], "[abc]".parse::<ParsedForm>().unwrap().parts);
    }

    #[test] fn test_parse_form_anagram() {
        assert_eq!(vec![FormPart::anagram_of("abc").unwrap()], "/abc".parse::<ParsedForm>().unwrap().parts);
    }

    // only lowercase is allowed
    #[test]
    fn test_parse_form_anagram_bad_char() {
        assert_eq!(
            FormPart::anagram_of("aBc").unwrap_err().to_string(),
            "Anagram constraint \"aBc\" contains invalid character 'B' (only a-z allowed)"
        );
    }

    #[test]
    fn test_is_anagram_negative_case() {
        let ag = FormPart::anagram_of("abc").unwrap();
        if let FormPart::Anagram(agi) = ag {
            let entry: Vec<char> = "abd".chars().collect();
            assert!(!agi.is_anagram(&entry).unwrap());
        }
    }


    #[test]
    fn test_expand_charset_range() {
        // [a-e] should expand to a, b, c, d, e
        let set = expand_charset("a-e").unwrap();
        let expected: HashSet<char> = HashSet::from(['a', 'b', 'c', 'd', 'e']);
        assert_eq!(set, expected);
    }

    #[test]
    fn test_expand_charset_mixed() {
        // [ax-z] should contain a, x, y, z
        let set = expand_charset("ax-z").unwrap();
        let expected: HashSet<char> = HashSet::from(['a', 'x', 'y', 'z']);
        assert_eq!(set, expected);
    }

    #[test]
    fn test_expand_charset_dangling_dash() {
        let err = expand_charset("a-").unwrap_err();
        assert!(matches!(*err,ParseError::DanglingCharsetDash));
    }

    #[test]
    fn test_parsed_form_dangling_dash() {
        assert!(matches!(*"[a-]".parse::<ParsedForm>().unwrap_err(), ParseError::DanglingCharsetDash));
    }

    mod edge_cases {
        use super::*;

        #[test]
        fn test_very_long_pattern() {
            let long_pattern = "a".repeat(150);
            let result = long_pattern.parse::<ParsedForm>();
            assert!(result.is_ok());
            let parsed = result.unwrap();
            assert_eq!(parsed.parts.len(), 1);
            if let FormPart::Lit(s) = &parsed.parts[0] {
                assert_eq!(s.len(), 150);
            } else {
                panic!("Expected literal");
            }
        }

        #[test]
        fn test_very_long_variable_sequence() {
            let pattern = "A".repeat(20) + &"B".repeat(20) + &"C".repeat(20);
            let result = pattern.parse::<ParsedForm>();
            assert!(result.is_ok());
            let parsed = result.unwrap();
            assert_eq!(parsed.parts.len(), 60);
        }

        #[test]
        fn test_charset_invalid_range() {
            let result = "[z-a]".parse::<ParsedForm>();
            assert!(result.is_err());
            assert!(matches!(*result.unwrap_err(), ParseError::InvalidCharsetRange('z', 'a')));
        }

        #[test]
        fn test_charset_empty() {
            let result = "[]".parse::<ParsedForm>();
            assert!(result.is_err());
        }

        #[test]
        fn test_unclosed_charset() {
            let result = "[abc".parse::<ParsedForm>();
            assert!(result.is_err());
        }

        #[test]
        fn test_anagram_with_uppercase() {
            let result = "/ABC".parse::<ParsedForm>();
            assert!(result.is_err());
        }

        #[test]
        fn test_anagram_with_numbers() {
            let result = "/abc123".parse::<ParsedForm>();
            assert!(result.is_err());
        }

        #[test]
        fn test_anagram_empty() {
            let result = "/".parse::<ParsedForm>();
            assert!(result.is_err());
        }

        #[test]
        fn test_reverse_operator_without_variable() {
            let result = "~".parse::<ParsedForm>();
            assert!(result.is_err());
        }

        #[test]
        fn test_reverse_operator_with_literal() {
            let result = "~abc".parse::<ParsedForm>();
            assert!(result.is_err());
        }

        #[test]
        fn test_mixed_case_string() {
            let result = "aBc".parse::<ParsedForm>();
            assert!(result.is_ok());
            let parsed = result.unwrap();
            assert_eq!(parsed.parts.len(), 3);
            assert_eq!(parsed.parts[0], FormPart::Lit("a".to_string()));
            assert_eq!(parsed.parts[1], FormPart::Var('B'));
            assert_eq!(parsed.parts[2], FormPart::Lit("c".to_string()));
        }

        #[test]
        fn test_numbers_in_pattern() {
            let result = "abc123".parse::<ParsedForm>();
            assert!(result.is_err());
        }

        #[test]
        fn test_special_chars_in_pattern() {
            let result = "abc!def".parse::<ParsedForm>();
            assert!(result.is_err());
        }

        #[test]
        fn test_whitespace_in_pattern() {
            let result = "abc def".parse::<ParsedForm>();
            assert!(result.is_err());
        }

        #[test]
        fn test_tab_in_pattern() {
            let result = "abc\tdef".parse::<ParsedForm>();
            assert!(result.is_err());
        }

        #[test]
        fn test_newline_in_pattern() {
            let result = "abc\ndef".parse::<ParsedForm>();
            assert!(result.is_err());
        }

        #[test]
        fn test_unicode_in_literal() {
            let result = "café".parse::<ParsedForm>();
            assert!(result.is_err());
        }

        #[test]
        fn test_unicode_in_variable() {
            let result = "Ä".parse::<ParsedForm>();
            assert!(result.is_err());
        }

        #[test]
        fn test_charset_with_unicode() {
            let result = "[aé]".parse::<ParsedForm>();
            assert!(result.is_err());
        }

        #[test]
        fn test_deeply_nested_charsets() {
            let result = "[a-c][d-f][g-i][j-l]".parse::<ParsedForm>();
            assert!(result.is_ok());
            let parsed = result.unwrap();
            assert_eq!(parsed.parts.len(), 4);
        }

        #[test]
        fn test_all_wildcard_types() {
            let result = ".*@#".parse::<ParsedForm>();
            assert!(result.is_ok());
            let parsed = result.unwrap();
            assert_eq!(parsed.parts, vec![
                FormPart::Dot,
                FormPart::Star,
                FormPart::Vowel,
                FormPart::Consonant
            ]);
        }

        #[test]
        fn test_complex_mixed_pattern() {
            let result = "A~Babc[d-f].*@#/ghi".parse::<ParsedForm>();
            assert!(result.is_ok());
            let parsed = result.unwrap();
            assert_eq!(parsed.parts.len(), 9);
        }

        #[test]
        fn test_charset_single_char() {
            let result = "[a]".parse::<ParsedForm>();
            assert!(result.is_ok());
            let parsed = result.unwrap();
            if let FormPart::Charset(chars) = &parsed.parts[0] {
                assert_eq!(chars.len(), 1);
                assert!(chars.contains(&'a'));
            }
        }

        #[test]
        fn test_charset_multiple_ranges() {
            let result = "[a-ce-g]".parse::<ParsedForm>();
            assert!(result.is_ok());
            let parsed = result.unwrap();
            if let FormPart::Charset(chars) = &parsed.parts[0] {
                assert_eq!(chars.len(), 6);
                assert!(chars.contains(&'a'));
                assert!(chars.contains(&'b'));
                assert!(chars.contains(&'c'));
                assert!(chars.contains(&'e'));
                assert!(chars.contains(&'f'));
                assert!(chars.contains(&'g'));
                assert!(!chars.contains(&'d'));
            }
        }

        #[test]
        fn test_negated_charset_basic() {
            let result = "[^abc]".parse::<ParsedForm>();
            assert!(result.is_ok());
            let parsed = result.unwrap();
            if let FormPart::Charset(chars) = &parsed.parts[0] {
                assert_eq!(chars.len(), 23);
                assert!(!chars.contains(&'a'));
                assert!(!chars.contains(&'b'));
                assert!(!chars.contains(&'c'));
                assert!(chars.contains(&'d'));
                assert!(chars.contains(&'z'));
            } else {
                panic!("expected Charset");
            }
        }

        #[test]
        fn test_negated_charset_range() {
            let result = "[^a-c]".parse::<ParsedForm>();
            assert!(result.is_ok());
            let parsed = result.unwrap();
            if let FormPart::Charset(chars) = &parsed.parts[0] {
                assert_eq!(chars.len(), 23);
                assert!(!chars.contains(&'a'));
                assert!(!chars.contains(&'b'));
                assert!(!chars.contains(&'c'));
                assert!(chars.contains(&'d'));
                assert!(chars.contains(&'z'));
            } else {
                panic!("expected Charset");
            }
        }

        #[test]
        fn test_negated_charset_complex() {
            // [^pb-mr] excludes p, b–m (12 chars), r → complement is 12 chars: anoqs-z
            let result = "[^pb-mr]".parse::<ParsedForm>();
            assert!(result.is_ok());
            let parsed = result.unwrap();
            if let FormPart::Charset(chars) = &parsed.parts[0] {
                assert_eq!(chars.len(), 12);
                for c in ['p', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'r'] {
                    assert!(!chars.contains(&c), "{c} should be excluded");
                }
                for c in ['a', 'n', 'o', 'q', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'] {
                    assert!(chars.contains(&c), "{c} should be included");
                }
            } else {
                panic!("expected Charset");
            }
        }

        #[test]
        fn test_negated_charset_bang() {
            let result = "[!abc]".parse::<ParsedForm>();
            assert!(result.is_ok());
            let parsed = result.unwrap();
            if let FormPart::Charset(chars) = &parsed.parts[0] {
                assert_eq!(chars.len(), 23);
                assert!(!chars.contains(&'a'));
                assert!(!chars.contains(&'b'));
                assert!(!chars.contains(&'c'));
                assert!(chars.contains(&'d'));
            } else {
                panic!("expected Charset");
            }
        }

        #[test]
        fn test_anagram_same_letters_repeated() {
            let result = "/aabbcc".parse::<ParsedForm>();
            assert!(result.is_ok());
            let parsed = result.unwrap();
            if let FormPart::Anagram(ag) = &parsed.parts[0] {
                assert_eq!(ag.len, 6);
                assert_eq!(ag.as_string, "aabbcc");
            }
        }

        #[test]
        fn test_materialize_deterministic_simple() {
            let parsed = "A".parse::<ParsedForm>().unwrap();
            let mut env = HashMap::new();
            env.insert('A', Rc::from("test"));
            let result = parsed.materialize_deterministic_with_env(&env);
            assert_eq!(result, Some("test".to_string()));
        }

        #[test]
        fn test_materialize_deterministic_with_literal() {
            let parsed = "Atest".parse::<ParsedForm>().unwrap();
            let mut env = HashMap::new();
            env.insert('A', Rc::from("pre"));
            let result = parsed.materialize_deterministic_with_env(&env);
            assert_eq!(result, Some("pretest".to_string()));
        }

        #[test]
        fn test_materialize_deterministic_with_reverse() {
            let parsed = "A~A".parse::<ParsedForm>().unwrap();
            let mut env = HashMap::new();
            env.insert('A', Rc::from("ab"));
            let result = parsed.materialize_deterministic_with_env(&env);
            assert_eq!(result, Some("abba".to_string()));
        }

        #[test]
        fn test_materialize_nondeterministic_returns_none() {
            let parsed = "A*B".parse::<ParsedForm>().unwrap();
            let mut env = HashMap::new();
            env.insert('A', Rc::from("test"));
            env.insert('B', Rc::from("ing"));
            let result = parsed.materialize_deterministic_with_env(&env);
            assert_eq!(result, None);
        }

        #[test]
        fn test_materialize_unbound_variable_returns_none() {
            let parsed = "AB".parse::<ParsedForm>().unwrap();
            let mut env = HashMap::new();
            env.insert('A', Rc::from("test"));
            let result = parsed.materialize_deterministic_with_env(&env);
            // B is unbound
            assert_eq!(result, None);
        }

        #[test]
        fn test_is_anagram_positive() {
            // "abc" is anagram of "bca"
            let ag = FormPart::anagram_of("abc").unwrap();
            if let FormPart::Anagram(agi) = ag {
                let entry: Vec<char> = "bca".chars().collect();
                assert!(agi.is_anagram(&entry).unwrap());
            }
        }

        #[test]
        fn test_is_anagram_different_length() {
            let ag = FormPart::anagram_of("abc").unwrap();
            if let FormPart::Anagram(agi) = ag {
                let entry: Vec<char> = "ab".chars().collect();
                assert!(!agi.is_anagram(&entry).unwrap());
            }
        }

        #[test]
        fn test_is_anagram_different_chars() {
            let ag = FormPart::anagram_of("abc").unwrap();
            if let FormPart::Anagram(agi) = ag {
                let entry: Vec<char> = "xyz".chars().collect();
                assert!(!agi.is_anagram(&entry).unwrap());
            }
        }

        #[test]
        fn test_is_deterministic() {
            assert!(FormPart::Var('A').is_deterministic());
            assert!(FormPart::RevVar('A').is_deterministic());
            assert!(FormPart::Lit("abc".to_string()).is_deterministic());
            assert!(!FormPart::Dot.is_deterministic());
            assert!(!FormPart::Star.is_deterministic());
            assert!(!FormPart::Vowel.is_deterministic());
            assert!(!FormPart::Consonant.is_deterministic());
        }
    }
}
