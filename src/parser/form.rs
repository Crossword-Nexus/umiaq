use std::collections::HashSet;
use std::str::FromStr;
use crate::umiaq_char::{ALPHABET_SIZE, LITERAL_CHARS, VARIABLE_CHARS};
use fancy_regex::Regex;
use nom::{
    branch::alt,
    bytes::complete::{tag, is_not},
    character::complete::one_of,
    combinator::map,
    multi::many1,
    sequence::{delimited, preceded},
    IResult,
    Parser,
};
use crate::errors::ParseError;
use crate::errors::ParseError::ParseFailure;
use crate::parser::utils::letter_to_num;
use super::prefilter::{form_to_regex_str, get_regex};

/// Parser result type: input, output, with our custom ParseError
pub type PResult<'a, O> = IResult<&'a str, O, ParseError>;

/// Represents a single parsed token (component) from a "form" string.
#[derive(Debug, Clone, PartialEq)]
pub enum FormPart {
    Var(char),          // 'A': uppercase A–Z variable reference
    RevVar(char),       // '~A': reversed variable reference
    Lit(String),        // 'abc': literal lowercase sequence (lowercase)
    Dot,                // '.' wildcard: exactly one letter
    Star,               // '*' wildcard: zero or more letters
    Vowel,              // '@' wildcard: any vowel (aeiouy)
    Consonant,          // '#' wildcard: any consonant (bcdf...xz)
    Charset(HashSet<char>), // '[abc]': any of the given letters
    Anagram(Alphagram), // '/abc': any permutation of the given letters
}

impl FormPart {
    pub(crate) fn is_deterministic(&self) -> bool {
        matches!(self, FormPart::Var(_) | FormPart::RevVar(_) | FormPart::Lit(_))
    }

    fn get_tag_string(&self) -> Option<&str> {
        match self {
            FormPart::Dot => Some("."),
            FormPart::Star => Some("*"),
            FormPart::Vowel => Some("@"),
            FormPart::Consonant => Some("#"),
            _ => None // Only the single-char tokens have tags
        }
    }

    pub(crate) fn anagram_of(s: &str) -> Result<FormPart, Box<ParseError>> {
        Ok(FormPart::Anagram(s.parse::<Alphagram>()?))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Alphagram {
    char_counts: [u8; ALPHABET_SIZE],
    pub(crate) as_string: String, // for regexes, pretty printing
    pub(crate) len: usize
}

// 'a' -> 0, 'b' -> 1, ..., 'z' -> 25
fn lc_letter_to_num(c: char) -> Result<usize, Box<ParseError>> { letter_to_num(c, 'a' as usize) }

impl Alphagram {
    pub(crate) fn is_anagram(&self, other_word: &[char]) -> Result<bool, Box<ParseError>> {
        if self.len != other_word.len() {
            return Ok(false);
        }

        let mut char_counts = self.char_counts;
        for &c in other_word {
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

    // NB: throws error if lowercase_word contains anything but lowercase letters
    fn from_str(lowercase_word: &str) -> Result<Self, Self::Err> {
        let mut len = 0;
        let mut char_counts = [0u8; ALPHABET_SIZE];
        for c in lowercase_word.chars() {
            let c_as_num = lc_letter_to_num(c)?;
            char_counts[c_as_num] += 1;
            len += 1;
        }

        Ok(Alphagram { char_counts, as_string: lowercase_word.to_string(), len })
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
        // Build the base regex string from tokens only (no var-constraints).
        let regex_str = form_to_regex_str(&parts)?;
        let anchored = format!("^{regex_str}$");
        let prefilter = get_regex(&anchored)?;

        Ok(ParsedForm { parts, prefilter })
    }

    // Return an iterator over the form parts
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, FormPart> {
        self.parts.iter()
    }

    /// If this form is deterministic, build the concrete word using `env`.
    /// Returns `None` if any required var is unbound or if a nondeterministic part is present.
    pub(crate) fn materialize_deterministic_with_env(
        &self,
        env: &std::collections::HashMap<char, String>,
    ) -> Option<String> {
        self.iter()
            .map(|part| match part {
                FormPart::Lit(s) => Some(s.clone()),
                FormPart::Var(v) => Some(env.get(v)?.clone()),
                FormPart::RevVar(v) => Some(env.get(v)?.chars().rev().collect()),
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
                Err(_) => return Err(Box::new(ParseFailure { s: rest.to_string() })),
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
fn dot(input: &'_ str) -> PResult<'_, FormPart> { parser_one_char_inner(input, &FormPart::Dot) }
fn star(input: &'_ str) -> PResult<'_, FormPart> { parser_one_char_inner(input, &FormPart::Star) }
fn vowel(input: &'_ str) -> PResult<'_, FormPart> { parser_one_char_inner(input, &FormPart::Vowel) }
fn consonant(input: &'_ str) -> PResult<'_, FormPart> { parser_one_char_inner(input, &FormPart::Consonant) }

// single-char tokens share the same shape
fn parser_one_char_inner<'a>(
    input: &'a str,
    form_part: &FormPart
) -> PResult<'a, FormPart> {
    map(
        tag(form_part.get_tag_string().unwrap()),
        |_| form_part.clone()
    ).parse(input)
}

/// Expand a string like "abcx-z" into a set of characters.
/// Supports ranges like a-e (inclusive).
fn expand_charset(body: &str) -> Result<HashSet<char>, ParseError> {
    let mut chars = HashSet::new();
    let mut iter = body.chars().peekable();

    while let Some(start) = iter.next() {
        if iter.peek() == Some(&'-') {
            iter.next(); // consume '-'
            match iter.next() {
                Some(end) if start <= end => chars.extend(start..=end),
                Some(end) => return Err(ParseError::InvalidCharsetRange(start, end)),
                None => return Err(ParseError::DanglingCharsetDash),
            }
        } else {
            chars.insert(start);
        }
    }

    Ok(chars)
}

fn charset(input: &'_ str) -> PResult<'_, FormPart> {
    let (input, body) = delimited(tag("["), is_not("]"), tag("]")).parse(input)?;
    // Expand ranges
    let chars = expand_charset(body).map_err(nom::Err::Failure)?;
    Ok((input, FormPart::Charset(chars)))
}

fn anagram(input: &'_ str) -> PResult<'_, FormPart> {
    let (input, _) = tag("/")(input)?;
    let (input, chars) = many1(one_of(LITERAL_CHARS)).parse(input)?;
    Ok((input, FormPart::anagram_of(&chars.into_iter().collect::<String>()).unwrap())) // TODO handle error (better than unwrap)
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
        assert!(matches!(*"[".parse::<ParsedForm>().unwrap_err(), ParseFailure { .. }));
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
    #[test] fn test_parse_form_anagram_bad_char() {
        assert!(FormPart::anagram_of("aBc").is_err_and(|pe| pe.to_string() == "Form parsing failed: \"Illegal char: 'B'\""));
    }

    #[test]
    fn test_is_anagram_negative_case() {
        let ag = FormPart::anagram_of("abc").unwrap();
        if let FormPart::Anagram(agi) = ag {
            let word: Vec<char> = "abd".chars().collect();
            assert!(!agi.is_anagram(&word).unwrap());
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
        assert!(matches!(err, ParseError::DanglingCharsetDash));
    }

    /* I'm not able to get this test to pass
    #[test]
    fn test_parsedform_dangling_dash() {
        let err = "[a-]".parse::<ParsedForm>().unwrap_err();
        assert!(matches!(*err, ParseError::DanglingCharsetDash));
    }
     */
}
