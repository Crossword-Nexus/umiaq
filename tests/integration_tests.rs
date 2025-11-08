//! Integration tests for the Umiaq crossword solver.
//!
//! These tests verify the complete pipeline from equation parsing through solving
//! to result validation, using realistic word lists and complex constraint scenarios.

use std::collections::HashSet;
use std::fs;

use umiaq::solver::{solve_equation, SolveResult, SolveStatus, SolverError};
use umiaq::errors::ParseError;

/// Load the test word list from fixtures
fn load_test_word_list() -> Vec<String> {
    let content = fs::read_to_string("tests/fixtures/test_wordlist.txt")
        .expect("Failed to read test word list");

    content
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            line.split(';')
                .next()
                .expect("Invalid word list format")
                .to_lowercase()
        })
        .collect()
}

/// Helper to convert Vec<String> to Vec<&str>
fn as_str_slice(words: &[String]) -> Vec<&str> {
    words.iter().map(|s| s.as_str()).collect()
}

/// Helper to extract just the words from a solution
fn solution_words(result: &SolveResult) -> Vec<Vec<String>> {
    result
        .solutions
        .iter()
        .map(|bindings_vec| {
            bindings_vec
                .iter()
                .filter_map(|b| b.get_word().map(|rc| rc.to_string()))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod simple_equations {
    use super::*;

    #[test]
    fn test_single_literal_pattern() {
        let words = vec!["able", "acid", "area"];
        let result = solve_equation("able", &words, 10).unwrap();

        assert_eq!(result.solutions.len(), 1);
        assert_eq!(solution_words(&result), vec![vec!["able"]]);
        // Word list is exhausted after finding the one match
        assert_eq!(result.status, SolveStatus::WordListExhausted);
    }

    #[test]
    fn test_single_variable_pattern() {
        let words = vec!["a", "be", "cat", "dogs"];
        let result = solve_equation("A", &words, 10).unwrap();

        // Should match all 4 words
        assert_eq!(result.solutions.len(), 4);
        let words_found: HashSet<String> = solution_words(&result)
            .into_iter()
            .flatten()
            .collect();
        assert!(words_found.contains("a"));
        assert!(words_found.contains("be"));
        assert!(words_found.contains("cat"));
        assert!(words_found.contains("dogs"));
    }

    #[test]
    fn test_pattern_with_wildcard() {
        let words = vec!["cat", "bat", "rat", "car", "bar"];
        let result = solve_equation(".at", &words, 10).unwrap();

        assert_eq!(result.solutions.len(), 3);
        let words_found: HashSet<String> = solution_words(&result)
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(words_found, HashSet::from(["cat".to_string(), "bat".to_string(), "rat".to_string()]));
    }

    #[test]
    fn test_pattern_with_literal_and_variable() {
        let words = vec!["able", "axle", "agile", "ankle", "angle"];
        let result = solve_equation("aAle", &words, 10).unwrap();

        // All 5 words match: a[X]le where X can be b, x, gi, nk, or ng
        assert_eq!(result.solutions.len(), 5);
        let words_found: HashSet<String> = solution_words(&result)
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(words_found, HashSet::from([
            "able".to_string(),
            "axle".to_string(),
            "agile".to_string(),
            "ankle".to_string(),
            "angle".to_string()
        ]));
    }

    #[test]
    fn test_even_length_palindrome_pattern() {
        let words = vec!["noon", "deed", "level", "radar", "test"];
        let result = solve_equation("A~A", &words, 10).unwrap();

        assert_eq!(result.solutions.len(), 2);
        let words_found: HashSet<String> = solution_words(&result)
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(words_found, HashSet::from(["noon".to_string(), "deed".to_string()]));
    }

    #[test]
    fn test_anagram_pattern() {
        let word_list = load_test_word_list();

        // Verify "area" exists in word list (it's in our test fixtures)
        assert!(word_list.contains(&"area".to_string()), "Test word list should contain 'area'");

        let words = as_str_slice(&word_list);

        // /area is anagram pattern - should find words with letters a,e,r,a
        let result = solve_equation("/area", &words, 10).unwrap();

        // At minimum, "area" itself should be found
        assert!(result.solutions.len() > 0);
        let words_found: HashSet<String> = solution_words(&result)
            .into_iter()
            .flatten()
            .collect();
        assert!(words_found.contains("area"));
    }
}

#[cfg(test)]
mod multi_pattern_equations {
    use super::*;

    #[test]
    fn test_two_patterns_shared_variable() {
        let words = vec!["ab", "a", "ba", "b"];
        let result = solve_equation("Ab;A", &words, 10).unwrap();

        // "Ab" means A followed by literal 'b'
        // Only "ab" matches with A="a"
        // Then "A" with A="a" matches word "a"
        // Solution: ["ab", "a"]
        assert_eq!(result.solutions.len(), 1);
        assert_eq!(solution_words(&result), vec![vec!["ab", "a"]]);
    }

    #[test]
    fn test_three_patterns_chain() {
        let words = vec!["ab", "ba", "bc", "cb"];
        let result = solve_equation("AB;BA;BC", &words, 10).unwrap();

        // Should find exactly 6 solutions:
        // ab•ba•ba, ab•ba•bc, ba•ab•ab, bc•cb•cb, cb•bc•ba, cb•bc•bc
        assert_eq!(result.solutions.len(), 6);

        let solutions = solution_words(&result);
        let solutions_set: HashSet<Vec<String>> = solutions.into_iter().collect();

        assert!(solutions_set.contains(&vec!["ab".to_string(), "ba".to_string(), "ba".to_string()]));
        assert!(solutions_set.contains(&vec!["ab".to_string(), "ba".to_string(), "bc".to_string()]));
        assert!(solutions_set.contains(&vec!["ba".to_string(), "ab".to_string(), "ab".to_string()]));
        assert!(solutions_set.contains(&vec!["bc".to_string(), "cb".to_string(), "cb".to_string()]));
        assert!(solutions_set.contains(&vec!["cb".to_string(), "bc".to_string(), "ba".to_string()]));
        assert!(solutions_set.contains(&vec!["cb".to_string(), "bc".to_string(), "bc".to_string()]));
    }

    #[test]
    fn test_multiple_solutions() {
        let words = vec!["inch", "chin"];
        let result = solve_equation("AB;BA", &words, 10).unwrap();

        // inch/chin: A=in, B=ch OR A=ch, B=in
        assert_eq!(result.solutions.len(), 2);
        let solutions_set: HashSet<Vec<String>> = solution_words(&result).into_iter().collect();
        assert!(solutions_set.contains(&vec!["inch".to_string(), "chin".to_string()]));
        assert!(solutions_set.contains(&vec!["chin".to_string(), "inch".to_string()]));
    }
}

#[cfg(test)]
mod constraint_tests {
    use super::*;

    #[test]
    fn test_length_constraint_exact() {
        let words = vec!["a", "ab", "abc", "abcd"];
        let result = solve_equation("A;|A|=2", &words, 10).unwrap();

        assert_eq!(result.solutions.len(), 1);
        assert_eq!(solution_words(&result), vec![vec!["ab"]]);
    }

    #[test]
    fn test_length_constraint_range() {
        let words = vec!["a", "ab", "abc", "abcd", "abcde"];
        let result = solve_equation("A;|A|>=2;|A|<=3", &words, 10).unwrap();

        assert_eq!(result.solutions.len(), 2);
        let words_found: HashSet<String> = solution_words(&result)
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(words_found, HashSet::from(["ab".to_string(), "abc".to_string()]));
    }

    #[test]
    fn test_inequality_constraint() {
        let words = vec!["inch", "chin", "noon", "deed"];
        let result = solve_equation("AB;BA;!=AB", &words, 10).unwrap();

        // Should find inch/chin and chin/inch but not noon/noon or deed/deed
        assert_eq!(result.solutions.len(), 2);
        let solutions_set: HashSet<Vec<String>> = solution_words(&result).into_iter().collect();
        assert!(solutions_set.contains(&vec!["inch".to_string(), "chin".to_string()]));
        assert!(solutions_set.contains(&vec!["chin".to_string(), "inch".to_string()]));
    }

    #[test]
    fn test_form_constraint() {
        let words = vec!["able", "axle", "area", "beta"];
        let result = solve_equation("A;A=(a*)", &words, 10).unwrap();

        assert_eq!(result.solutions.len(), 3);
        let words_found: HashSet<String> = solution_words(&result)
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(words_found, HashSet::from(["able".to_string(), "axle".to_string(), "area".to_string()]));
    }

    #[test]
    fn test_complex_constraint() {
        let words = vec!["a", "ab", "abc", "ace", "are", "be"];
        let result = solve_equation("A;A=(2-3:a*)", &words, 10).unwrap();

        // Length 2-3, starts with 'a'
        // Should match: ab, abc, ace, are
        // Should NOT match: a (too short), be (doesn't start with 'a')
        assert_eq!(result.solutions.len(), 4);
        let words_found: HashSet<String> = solution_words(&result)
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(words_found, HashSet::from(["ab".to_string(), "abc".to_string(), "ace".to_string(), "are".to_string()]));
    }

    #[test]
    fn test_joint_constraint() {
        let words = vec!["inch", "chin", "chess", "test", "able"];
        let result = solve_equation("AB;BA;|AB|=4", &words, 10).unwrap();

        // inch (i+nch=4) + chin (ch+in=4) ✓
        // chin (ch+in=4) + inch (in+ch=4) ✓
        assert_eq!(result.solutions.len(), 2);
        let solutions_set: HashSet<Vec<String>> = solution_words(&result).into_iter().collect();
        assert!(solutions_set.contains(&vec!["inch".to_string(), "chin".to_string()]));
        assert!(solutions_set.contains(&vec!["chin".to_string(), "inch".to_string()]));
    }

    #[test]
    fn test_multiple_constraints_combined() {
        let word_list = load_test_word_list();
        let words = as_str_slice(&word_list);

        // Pattern with multiple constraint types: length 4, starts with 'b'
        // Note: Can't have multiple conflicting form constraints on the same variable
        let result = solve_equation("A;|A|=4;A=(b*)", &words, 10).unwrap();

        // Verify all results match constraints
        let solutions = solution_words(&result);
        assert!(!solutions.is_empty(), "Should find at least one solution");

        for sol in &solutions {
            for word in sol {
                assert_eq!(word.len(), 4, "Word '{}' should be length 4", word);
                assert!(word.starts_with('b'), "Word '{}' should start with 'b'", word);
            }
        }
    }
}

#[cfg(test)]
mod error_cases {
    use super::*;

    #[test]
    fn test_parse_error_invalid_syntax() {
        let words = vec!["test"];
        let result = solve_equation("A;|A|", &words, 10);

        assert!(result.is_err());
    }

    #[test]
    fn test_contradictory_bounds() {
        let words = vec!["test"];
        let result = solve_equation("A;|A|>10;|A|<5", &words, 10);

        // Should fail during parsing due to contradictory constraints
        assert!(result.is_err());
        let e = result.unwrap_err();
        // Verify it's a parse error containing contradictory bounds
        if let SolverError::ParseFailure(parse_err) = e {
            assert!(matches!(*parse_err, ParseError::ContradictoryBounds { min: 11, max: 4 }));
        } else {
            panic!("Expected ParseFailure error, got: {:?}", e);
        }
    }

    #[test]
    fn test_no_solutions_found() {
        let words = vec!["able", "baker"];
        let result = solve_equation("xyz", &words, 10).unwrap();

        // No matches for literal "xyz"
        assert_eq!(result.solutions.len(), 0);
        assert_eq!(result.status, SolveStatus::WordListExhausted);
    }

    #[test]
    fn test_empty_word_list() {
        let words: Vec<&str> = vec![];
        let result = solve_equation("A", &words, 10).unwrap();

        assert_eq!(result.solutions.len(), 0);
        assert_eq!(result.status, SolveStatus::WordListExhausted);
    }

    #[test]
    fn test_no_matching_words_for_constraints() {
        let words = vec!["test", "able", "word"];
        // Looking for a word ending with 'x' and a word starting with 'x'
        // None exist in this word list
        let result = solve_equation("Ax;xB", &words, 10).unwrap();

        assert_eq!(result.solutions.len(), 0);
        assert_eq!(result.status, SolveStatus::WordListExhausted);
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn test_very_long_pattern() {
        let words = vec!["abcdefghij"];
        let result = solve_equation("ABCDEFGHIJ", &words, 10).unwrap();

        assert_eq!(result.solutions.len(), 1);
        assert_eq!(solution_words(&result), vec![vec!["abcdefghij"]]);
    }

    #[test]
    fn test_many_patterns() {
        let words = vec!["a", "b", "c"];
        // 10 patterns, each matching single letter
        let result = solve_equation("A;B;C;D;E;F;G;H;I;J", &words, 10).unwrap();

        // 3^10 = 59049 total combinations, but we only request 10
        assert_eq!(result.solutions.len(), 10);
        assert_eq!(result.status, SolveStatus::FoundEnough);
    }

    #[test]
    fn test_early_stop_on_limit() {
        let words = vec!["a", "b", "c", "d", "e"];
        let result = solve_equation("A", &words, 3).unwrap();

        // Should stop at exactly 3 results
        assert_eq!(result.solutions.len(), 3);
        assert_eq!(result.status, SolveStatus::FoundEnough);
    }

    #[test]
    fn test_word_list_exhausted() {
        let words = vec!["a", "b"];
        let result = solve_equation("A", &words, 100).unwrap();

        // Requested 100 but only 2 available
        assert_eq!(result.solutions.len(), 2);
        assert_eq!(result.status, SolveStatus::WordListExhausted);
    }
}

#[cfg(test)]
mod realistic_scenarios {
    use super::*;

    #[test]
    fn test_crossword_intersection() {
        let word_list = load_test_word_list();
        let words = as_str_slice(&word_list);

        // Simulates two crossing words in a crossword
        // Pattern 1: bABC (literal 'b' + variables A, B, C)
        // Pattern 2: ADEF (variables A, D, E, F)
        // Both patterns share variable A
        let result = solve_equation("bABC;ADEF", &words, 5).unwrap();

        // Verify all solutions have the correct intersection
        let solutions = solution_words(&result);
        assert!(!solutions.is_empty(), "Should find at least one solution");

        for sol in &solutions {
            assert_eq!(sol.len(), 2, "Each solution should have 2 words");
            let first = &sol[0];
            let second = &sol[1];
            assert!(first.starts_with('b'), "First word should start with 'b'");
            // First letter of second word should match second letter of first word (both are A)
            assert_eq!(
                second.chars().nth(0),
                first.chars().nth(1),
                "Second word '{}' first letter should match first word '{}' second letter",
                second,
                first
            );
        }
    }

    #[test]
    fn test_themed_puzzle() {
        let word_list = load_test_word_list();
        let words = as_str_slice(&word_list);

        // All words must start with 'b' (themed)
        let result = solve_equation("bA;bB;bC", &words, 3).unwrap();

        let solutions = solution_words(&result);
        assert_eq!(solutions.len(), 3);

        for sol in solutions {
            assert_eq!(sol.len(), 3, "Each solution should have 3 words");
            // Verify all words in each solution start with 'b'
            assert!(sol.iter().all(|w| w.starts_with('b')));
        }
    }

    #[test]
    fn test_complex_real_world() {
        let word_list = load_test_word_list();
        let words = as_str_slice(&word_list);

        // Complex equation with multiple patterns and constraints
        // AB and BA with |A|>=2, |B|>=2, !=AB
        // This finds pairs like "inch"/"chin" where the parts can be swapped
        let result = solve_equation(
            "AB;BA;|A|>=2;|B|>=2;!=AB",
            &words,
            10
        ).unwrap();

        // Should find at least some solutions
        assert!(!result.solutions.is_empty(), "Should find at least one solution");

        // Verify each solution has 2 bindings (one for each pattern)
        for solution in &result.solutions {
            assert_eq!(solution.len(), 2, "Each solution should have 2 bindings");
        }

        // Verify we got actual words in the solutions
        let solutions = solution_words(&result);
        for sol in &solutions {
            assert_eq!(sol.len(), 2, "Each solution should have 2 words");
            // Both words should exist in our word list
            assert!(word_list.iter().any(|w| w == &sol[0]), "Word '{}' should be in word list", sol[0]);
            assert!(word_list.iter().any(|w| w == &sol[1]), "Word '{}' should be in word list", sol[1]);
        }
    }
}
