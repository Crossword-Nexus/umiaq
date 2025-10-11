//! The main solver for pattern-matching equations against word lists.
//!
//! # Error Handling
//!
//! The solver uses [`SolverError`] with three variants:
//!
//! - S001: `ParseFailure` (Pattern parsing failed (wraps [`ParseError`]))
//! - S002: `NoPatterns` (Equation has only constraints, no patterns to solve)
//! - S003: `MaterializationError` (Internal error during solution construction)
//!
//! Each error has a `code()`, optional `help()`, and `display_detailed()` method.
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! use umiaq::solver;
//!
//! let words = vec!["cat", "dog", "catalog", "dogma"];
//! let result = solver::solve_equation("A*B", &words, 10)?;
//!
//! println!("Found {} solutions", result.solutions.len());
//! for solution in result.solutions {
//!     println!("{}", solver::solution_to_string(&solution)?);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Handling Errors with Detailed Messages
//!
//! ```
//! use umiaq::solver::{self, SolverError};
//!
//! let words = vec!["test"];
//! match solver::solve_equation("", &words, 10) {
//!     Ok(result) => println!("Success: {} solutions", result.solutions.len()),
//!     Err(e) => {
//!         // Show detailed error with code and help
//!         eprintln!("{}", e.display_detailed());
//!         // Error code: S001
//!         // Help text: Example: Use 'A*B' or '*cat*' instead of empty string
//!     }
//! }
//! ```
//!
//! ## Checking Solve Status
//!
//! ```
//! use umiaq::solver::{self, SolveStatus};
//!
//! let words = vec!["cat", "dog"];
//! let result = solver::solve_equation("A*", &words, 100)?;
//!
//! match result.status {
//!     SolveStatus::FoundEnough => println!("Found all requested results"),
//!     SolveStatus::WordListExhausted => println!("Searched entire word list"),
//!     SolveStatus::TimedOut { elapsed } => {
//!         println!("Timed out after {:?}", elapsed);
//!     }
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::bindings::{Bindings, WORD_SENTINEL};
use crate::errors::ParseError;
use crate::joint_constraints::JointConstraints;
use crate::parser::{match_equation_all, ParsedForm};
use crate::patterns::{Pattern, EquationContext};
use crate::errors::ParseError::ParseFailure;
use instant::Instant;
use std::collections::hash_map::{DefaultHasher, Entry};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::time::Duration;

// The amount of time (in seconds) we allow the query to run
const TIME_BUDGET: u64 = 30;
// The initial number of words from the word list we look through
const DEFAULT_BATCH_SIZE: usize = 10_000;
// A constant to split up items in our hashes
const HASH_SPLIT: u16 = 0xFFFFu16;

/// Status of the solver run.
#[derive(Debug, Clone, PartialEq)]
pub enum SolveStatus {
    /// Solver ran through the entire word list without reaching the requested number.
    WordListExhausted,

    /// Solver stopped early because the requested number of results was found.
    FoundEnough,

    /// Solver stopped because the time budget expired. Contains the elapsed time.
    TimedOut { elapsed: Duration },
}

/// Successful solver run (even if it stopped early).
#[derive(Debug, Clone)]
pub struct SolveResult {
    /// Solutions discovered (may be fewer than requested if timed out).
    pub solutions: Vec<Vec<Bindings>>,
    /// Status indicating whether we finished or timed out.
    pub status: SolveStatus,
}

impl SolveResult {
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.solutions.len()
    }
}

impl IntoIterator for SolveResult {
    type Item = Vec<Bindings>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.solutions.into_iter()
    }
}

/// Unified error type for the solver pipeline.
///
/// This consolidates the different error sources we encounter when parsing
/// equations, materializing solutions, or respecting time budgets, so that
/// callers only need to handle a single `Result<_, SolverError>`.
#[derive(Debug, thiserror::Error)]
pub enum SolverError {
    /// Failure during parsing of the equation string into an `EquationContext`.
    ///
    /// These originate from the parser (`ParseError`), which we box to keep the
    /// error type size stable.
    #[error("parse failure: {0}")]
    ParseFailure(#[from] Box<ParseError>),

    /// The equation contained only constraints and no patterns,
    /// so there is nothing to solve.
    #[error("no patterns to solve (constraints only)")]
    NoPatterns,

    /// Failure while materializing a candidate binding into a concrete solution.
    ///
    /// This generally indicates that an internal constraint check failed during
    /// `recursive_join` or related routines.
    #[error("materialization error: {context}")]
    MaterializationError { context: String },
}

impl SolverError {
    /// Returns the error code for this error variant
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            SolverError::ParseFailure(_) => "S001",
            SolverError::NoPatterns => "S002",
            SolverError::MaterializationError { .. } => "S003",
        }
    }

    /// Returns a short description of this error type (for documentation)
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            SolverError::ParseFailure(_) => "Pattern parsing failed",
            SolverError::NoPatterns => "Equation has only constraints, no patterns to solve",
            SolverError::MaterializationError { .. } => "Internal error during solution construction",
        }
    }

    /// Returns detailed explanation of this error type (for documentation)
    #[must_use]
    pub fn details(&self) -> &'static str {
        match self {
            SolverError::ParseFailure(_) => "The input pattern could not be parsed. This wraps an underlying ParseError (see Parse Errors section for specific error codes).",
            SolverError::NoPatterns => "The equation contains only constraints (like `|A|=3`) but no actual patterns to match against words. Add at least one pattern like `A*B` or `*cat*`.",
            SolverError::MaterializationError { .. } => "This indicates an internal solver error where a pattern matched but constraints could not be satisfied during solution materialization. This is usually a bug in the solver logic.",
        }
    }

    /// Returns a helpful suggestion for this error
    #[must_use]
    pub fn help(&self) -> Option<&'static str> {
        match self {
            SolverError::NoPatterns => Some("Add at least one pattern to solve. Example: 'A*B' or '*cat*;*dog*'"),
            SolverError::MaterializationError { .. } => Some("This is an internal error. The pattern matched but constraints could not be satisfied."),
            SolverError::ParseFailure(_) => None, // ParseError has its own help
        }
    }

    /// Formats the error with code and optional help text
    #[must_use]
    pub fn display_detailed(&self) -> String {
        match self {
            SolverError::ParseFailure(pe) => {
                // delegate to ParseError's detailed display
                format!("{}\n  caused by: {}", self.code(), pe.display_detailed())
            }
            _ => {
                crate::errors::format_error_with_code_and_help(&self.to_string(), self.code(), self.help())
            }
        }
    }
}

/// Bucket key for indexing candidates by the subset of variables that must agree.
/// - `None` means "no lookup constraints for this pattern" (Python's `words[i][None]`).
/// - When present, we store a *sorted* `(var_char, var_val)` list so the key is deterministic
///   and implements `Eq`/`Hash` naturally. This mirrors Python's
///   `frozenset(dict(...).items())`, but with a stable order.
/// - The sort happens once when we construct the key, not on hash/compare.
pub type LookupKey = Vec<(char, Rc<str>)>;

/// Context for a `recursive_join` call
struct JoinCtx<'a> {
    num_results_requested: usize,
    word_set: &'a HashSet<&'a str>,
    joint_constraints: &'a JointConstraints,
    budget: &'a TimeBudget,
}

/// All candidates for one pattern ("bucketed" by `LookupKey`).
/// - `buckets`: groups candidate bindings that share the same values for the
///   pattern's `lookup_keys` (variables that must align with previously chosen patterns).
/// - `count`: mirrors Python's `word_counts[i]` and is used to stop early when a global cap
///   per-pattern is reached (e.g., `MAX_WORD_COUNT`). We track it here to avoid recomputing.
#[derive(Debug, Default)]
#[derive(Clone)]
pub struct CandidateBuckets {
    /// Mapping from lookup key -> all bindings that fit that key
    pub buckets: HashMap<LookupKey, Vec<Bindings>>,
    /// Total number of bindings added for this pattern (across all keys)
    pub count: usize,
}

/// Put the results in uppercase and separated with a bullet
///
/// # Errors
///
/// Will return `Box<ParseError>` if the bound word cannot be found for one of the `Bindings`
/// objects in `solution`.
pub fn solution_to_string(solution: &[Bindings]) -> Result<String, Box<ParseError>> {
    let str = solution.iter()
        .map(|b| b.get_word().ok_or_else(|| Box::new(ParseFailure { s : format!("cannot find solution in bindings {b}") })).map(|c| c.to_ascii_uppercase()))
        .collect::<Result<Vec<_>, _>>()?
        .join(" • ");

    Ok(str)
}

/// Build a stable key for a full solution (bindings in **pattern order**).
/// Prefer the whole word if present (`WORD_SENTINEL`). Fall back to sorted `(var_char, var_val)` pairs.
fn solution_key(solution: &[Bindings]) -> u64 {
    let mut hasher = DefaultHasher::new();

    for b in solution {
        // Try whole-word first (fast + canonical)
        if let Some(w) = b.get_word() {
            w.hash(&mut hasher);
        } else {
            // this should never happen
            panic!("solution_key: no '*' binding found in solution: {solution:?}");
            /*
            // Fall back: hash all (var_char, var_val) pairs sorted by var
            let mut pairs: Vec<(char, String)> =
                b.iter().map(|(var_char, var_val)| (*var_char, var_val.clone())).collect();
            pairs.sort_unstable_by_key(|(var_char, _)| *var_char);
            for (var_char, string) in pairs {
                var_char.hash(&mut hasher);
                string.hash(&mut hasher);
            }
            */
        }
        // Separator between patterns to avoid ambiguity like ["ab","c"] vs ["a","bc"]
        HASH_SPLIT.hash(&mut hasher);
    }

    hasher.finish()
}

/// Simple helper to enforce a wall-clock time limit.
///
/// Usage:
/// ```ignore
///  let budget = TimeBudget::new(Duration::from_secs(30));
///  while !budget.expired() {
///    // do some work
///  }
/// ```
///
/// You can also query how much time is left (`remaining()`).
struct TimeBudget {
    start: Instant,   // when the budget began
    limit: Duration,  // maximum allowed elapsed time
}

impl TimeBudget {
    /// Create a new budget that lasts for `limit` (e.g., 30 seconds).
    fn new(limit: Duration) -> Self {
        Self { start: Instant::now(), limit }
    }

    /// How long this budget has been running.
    fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Returns true if the allowed time has fully elapsed.
    fn expired(&self) -> bool {
        self.start.elapsed() >= self.limit
    }

    // Returns the remaining time before expiration, or zero if the budget is already used up.
    // Unused for now but it may be useful later
    // fn remaining(&self) -> Duration {self.limit.saturating_sub(self.start.elapsed())}
}

macro_rules! timed_stop {
    // For functions that return Result<(), E>
    ($budget:expr) => {
        if $budget.expired() {
            return Ok(());
        }
    };
    // For functions that return Result<T, E>; caller passes the Ok(...) expr to return
    ($budget:expr, $ret_expr:expr) => {
        if $budget.expired() {
            return Ok($ret_expr);
        }
    };
}

/// Build a lookup key from an environment (`HashMap`) given a set of key variables.
///
/// Returns `None` if any required key variable is missing in `env`.
/// Otherwise returns a sorted `LookupKey`.
fn lookup_key_from_env(
    env: &HashMap<char, Rc<str>>,
    keys: &HashSet<char>,
) -> Option<LookupKey> {
    debug_assert!(
        keys.iter().all(char::is_ascii_uppercase),
        "All key variables must be one of uppercase A-Z"
    );

    let mut pairs: Vec<(char, Rc<str>)> = Vec::with_capacity(keys.len());
    for &var_char in keys {
        match env.get(&var_char) {
            Some(var_val) => {
                debug_assert!(!var_val.is_empty(), "Variable bindings must be non-empty strings");
                pairs.push((var_char, Rc::clone(var_val)));
            }
            None => return None, // required key missing
        }
    }

    pairs.sort_unstable_by_key(|(c, _)| *c);
    debug_assert!(
        pairs.windows(2).all(|w| w[0].0 < w[1].0),
        "Sorted pairs must be strictly increasing"
    );
    Some(pairs)
}

/// Build the deterministic lookup key for a binding given the pattern's lookup vars.
///
/// Returns a `LookupKey` (alias for `Vec<(char, Rc<str>)>`) sorted by `var_char`.
///
/// Conventions:
/// - If `keys` is empty, returns an empty `LookupKey` (unkeyed bucket).
/// - If any required key is missing in the binding, also returns an empty `LookupKey`;
///   callers must check `keys.is_empty()` to distinguish this case.
///   TODO: perhaps avoid pushing this responsibility on callers (via an enum return type?)
/// - Otherwise, returns the full normalized key.
fn lookup_key_for_binding(
    binding: &Bindings,
    keys: &HashSet<char>,
) -> LookupKey {
    debug_assert!(
        keys.iter().all(char::is_ascii_uppercase),
        "All key variables must be one of uppercase A-Z"
    );

    let mut pairs: Vec<(char, Rc<str>)> = Vec::with_capacity(keys.len());
    for &var_char in keys {
        match binding.get(var_char) {
            Some(var_val) => {
                debug_assert!(!var_val.is_empty(), "Variable bindings must be non-empty strings");
                pairs.push((var_char, Rc::clone(var_val)));
            }
            None => return Vec::new(), // required key missing
        }
    }

    pairs.sort_unstable_by_key(|(c, _)| *c);
    debug_assert!(
        pairs.windows(2).all(|w| w[0].0 < w[1].0),
        "Sorted pairs must be strictly increasing"
    );
    pairs
}

/// Push a binding into the appropriate bucket and bump the count.
fn push_binding(words: &mut [CandidateBuckets], i: usize, key: LookupKey, binding: Bindings) {
    debug_assert!(i < words.len(), "Pattern index {} out of bounds (len={})", i, words.len());
    debug_assert!(
        binding.get_word().is_some(),
        "Binding must have a word set before being pushed"
    );

    let old_count = words[i].count;
    words[i].buckets.entry(key).or_default().push(binding);
    words[i].count += 1;

    debug_assert_eq!(
        words[i].count, old_count + 1,
        "Count must increment by exactly 1"
    );
}

/// Scan a slice of candidate words against all patterns in the equation context,
/// materializing any matching bindings into per-pattern buckets.
///
/// - Iterates through `word_list` starting at `start_idx`, up to `batch_size` words
///   (or the end of the list).
/// - For each word, applies length prefilters and variable constraints from
///   `equation_context`, pushing any resulting bindings into `words[i]`.
/// - May stop early when the [`TimeBudget`] is exhausted; in that case it returns
///   the number of words processed so far (and any bindings already pushed remain
///   in `words`).
///
/// # Arguments
/// * `word_list` — master list of candidate words (scanned by index range)
/// * `start_idx` — starting index in `word_list` for this batch
/// * `batch_size` — maximum number of words to scan in this batch
/// * `equation_context` — parsed equation state with patterns, constraints, and hints
/// * `words` — mutable slice of per-pattern candidate buckets to be populated
/// * `budget` — wall-clock time budget to respect
///
/// # Returns
/// The number of words consumed (≤ `batch_size`). If the time budget expires,
/// this count reflects the partial progress made before stopping. Callers that
/// wish to surface a timeout should detect early completion (e.g., by checking
/// `budget.expired()`) and convert that condition into `SolverError::Timeout`
/// at a higher level.
fn scan_batch(
    word_list: &[&str],
    start_idx: usize,
    batch_size: usize,
    equation_context: &EquationContext,
    words: &mut [CandidateBuckets],
    budget: &TimeBudget,
) -> Result<usize, SolverError> {
    // precondition checks
    debug_assert!(
        start_idx <= word_list.len(),
        "start_idx ({}) must be <= word_list.len() ({})",
        start_idx, word_list.len()
    );
    debug_assert_eq!(
        words.len(), equation_context.len(),
        "words slice length must match equation_context pattern count"
    );
    debug_assert!(batch_size > 0, "batch_size must be positive");

    let mut i_word = start_idx;
    let end = start_idx.saturating_add(batch_size).min(word_list.len());

    debug_assert!(
        end <= word_list.len(),
        "end ({}) must be <= word_list.len() ({})",
        end, word_list.len()
    );

    while i_word < end {
        timed_stop!(budget, i_word);

        let word = word_list[i_word];

        for (i, p) in equation_context.iter().enumerate() {
            // No per-pattern cap anymore

            // Skip deterministic fully keyed forms
            if p.is_deterministic && p.all_vars_in_lookup_keys() {
                continue;
            }
            // Cheap length prefilter
            if !&equation_context.scan_hints[i].is_word_len_possible(word.len()) {
                continue;
            }

            let matches = match_equation_all(
                word,
                &equation_context.parsed_forms[i],
                &equation_context.var_constraints,
                &equation_context.joint_constraints,
            );

            for binding in matches {
                timed_stop!(budget, i_word);

                let key = lookup_key_for_binding(&binding, &p.lookup_keys);

                // If a required key is missing, skip
                if key.is_empty() && !p.lookup_keys.is_empty() {
                    continue;
                }

                push_binding(words, i, key, binding);
            }
        }

        i_word += 1;
    }

    Ok(i_word)
}

struct RecursiveJoinParameters<'a> {
    candidate_buckets: &'a CandidateBuckets,
    lookup_keys: &'a HashSet<char>,
    pattern: &'a Pattern,
    parsed_form: &'a ParsedForm,
}

/// Depth-first recursive join of per-pattern candidate buckets into full solutions.
///
/// This mirrors `recursive_filter` from `umiaq.py`. We walk patterns in order and at each step
/// select only the bucket of candidates whose shared variables agree with what we've already
/// chosen (`env`).
///
/// Parameters:
/// - `selected`: the partial solution (one chosen `Bindings` per pattern so far).
/// - `env`: the accumulated variable → value environment from earlier choices.
/// - `results`: completed solutions (each is a `Vec<Binding>`, one per pattern).
/// - `ctx`: join context containing the requested result count, word set, constraints, and budget.
/// - `seen`: set of solution hashes to avoid duplicates.
/// - `rjp`: remaining patterns to process (candidate buckets, lookup keys, pattern info).
///
/// Return:
/// - This function mutates `results` and stops early once `results` contains
///   `num_results_requested` values.
/// # Errors
/// Returns `Err(SolverError::Timeout)` if the time budget expires during the join.
/// In that case, partial solutions already found remain in `results`.
fn recursive_join(
    selected: &mut Vec<Bindings>,
    env: &mut HashMap<char, Rc<str>>,
    results: &mut Vec<Vec<Bindings>>,
    ctx: &JoinCtx,
    seen: &mut HashSet<u64>,
    rjp: &[RecursiveJoinParameters],
) -> Result<(), SolverError> {
    recursive_join_inner(
        selected,
        env,
        results,
        ctx,
        seen,
        rjp,
        rjp.len(),
    )
}

fn recursive_join_inner(
    selected: &mut Vec<Bindings>,
    env: &mut HashMap<char, Rc<str>>,
    results: &mut Vec<Vec<Bindings>>,
    ctx: &JoinCtx,
    seen: &mut HashSet<u64>,
    rjp: &[RecursiveJoinParameters],
    total_patterns: usize, // for debug assert
) -> Result<(), SolverError> {
    // Invariant: selected.len() + rjp.len() == total_patterns
    // (we've processed 'selected' patterns and have 'rjp' remaining)
    debug_assert_eq!(
        selected.len() + rjp.len(), total_patterns,
        "selected ({}) + remaining ({}) must equal total patterns ({})",
        selected.len(), rjp.len(), total_patterns
    );

    // Stop if we've met the requested quota of full solutions.
    if results.len() >= ctx.num_results_requested {
        return Ok(());
    }

    timed_stop!(ctx.budget);

    if let Some(rjp_cur) = rjp.first() {
        // ---- FAST PATH: deterministic + fully keyed ----------------------------
        let p = &rjp_cur.pattern;
        if p.is_deterministic && p.all_vars_in_lookup_keys() {
            // The word is fully determined by literals + already-bound vars in `env`.
            let Some(expected) = rjp_cur.parsed_form
                .materialize_deterministic_with_env(env)
            else {
                return Err(SolverError::MaterializationError {
                    context: format!(
                        "failed to materialize deterministic pattern with variables: {:?}\n  \
                        environment: {:?}\n  \
                        pattern depth: {}/{}",
                        p.variables,
                        env,
                        selected.len() + 1,
                        total_patterns
                    )
                });
            };

            if !ctx.word_set.contains(expected.as_str()) {
                // This branch cannot succeed — prune immediately.
                return Ok(());
            }

            // Build a minimal Bindings for this pattern:
            // - include WORD_SENTINEL (whole word)
            // - include only vars that belong to this pattern (they must already be in env)
            let mut binding = Bindings::default();
            binding.set_word(&expected);
            for &var_char in &p.variables {
                // safe: all vars in lookup_keys must be in env by construction
                if let Some(var_val) = env.get(&var_char) {
                    binding.set_rc(var_char, Rc::clone(var_val));
                } else {
                    // this should never happen--indicates a logic error in solver
                    debug_assert!(
                        false,
                        "Variable '{var_char}' in lookup_keys but not in env--solver invariant violated"
                    );
                }
            }

            selected.push(binding);
            recursive_join_inner(selected, env, results, ctx, seen, &rjp[1..], total_patterns)?;
            selected.pop();
            return Ok(()); // IMPORTANT: skip normal enumeration path
        }
        // ------------------------------------------------------------------------

        // Decide which bucket of candidates to iterate for pattern `idx`.
        //
        // We must create the deterministic key
        //   `Some(sorted_pairs)` using the current `env` and fetch that bucket.
        //   (This includes the case keys.is_empty() → key is `Some([])`.)
        let bucket_candidates_opt: Option<&Vec<Bindings>> = {
            timed_stop!(ctx.budget);
            // Build lookup key from current environment
            let Some(key) = lookup_key_from_env(env, rjp_cur.lookup_keys) else {
                // If any required var isn't bound yet, there can be no matches for this branch.
                return Ok(());
            };

            rjp_cur.candidate_buckets.buckets.get(&key)
        };

        // If there are no candidates in that bucket, dead-end this branch.
        let Some(bucket_candidates) = bucket_candidates_opt else {
            return Ok(());
        };

        // Try each candidate binding for this pattern.
        for cand in bucket_candidates {

            timed_stop!(ctx.budget);

            if results.len() >= ctx.num_results_requested {
                break; // stop early if we've already met the quota
            }

            // Defensive compatibility check: if a variable is already in `env`,
            // its value must match the candidate. This *should* already be true
            // because we selected the bucket using the shared vars—but keep this
            // in case upstream bucketing logic ever changes.
            if cand.iter().filter(|(var_char, _)| *var_char != WORD_SENTINEL).any(|(var_char, var_val)| env.get(&var_char).is_some_and(|prev| prev != var_val)) {
                continue;
            }

            // Extend `env` with any *new* bindings from this candidate (don't overwrite).
            // Track what we added so we can backtrack cleanly.
            let mut added_vars: Vec<char> = vec![];
            for (var_char, var_val) in cand.iter() {
                if var_char == WORD_SENTINEL {
                    continue;
                }
                if let Entry::Vacant(e) = env.entry(var_char) {
                    e.insert(Rc::clone(var_val));
                    added_vars.push(var_char);
                }
            }

            // Choose this candidate for pattern `idx` and recurse for `idx + 1`.
            selected.push(cand.clone());
            recursive_join_inner(selected, env, results, ctx, seen, &rjp[1..], total_patterns)?;
            selected.pop();

            // Backtrack: remove only what we added at this level.
            for k in added_vars {
                env.remove(&k);
            }
        }
    } else {
        // Base case: if we've placed all patterns, `selected` is a full solution.
        if ctx.joint_constraints.all_strictly_satisfied_for_parts(selected)
            && seen.insert(solution_key(selected)) {
            results.push(selected.clone());
        }
    }

    Ok(())
}

/// Read in an equation string and return results from the word list
///
/// This orchestrates parsing the input string into an [`EquationContext`],
/// scanning candidate words in adaptive batches, and recursively joining
/// variable bindings into full solutions. Results are accumulated until either
/// the requested number of solutions is found or the [`TimeBudget`] expires.
///
/// # Arguments
/// - `input`: equation string to parse (e.g. `"AB;BC;CA"`).
/// - `word_list`: slice of candidate words to match against.
/// - `num_results_requested`: optional cap on how many solutions to return
///   (use `None` to search exhaustively).
///
/// # Returns
/// A vector of fully materialized [`Solution`]s, up to the requested limit.
///
/// # Errors
/// Returns a [`SolverError`] if:
/// - the equation string cannot be parsed (`ParseFailure`),
/// - a binding cannot be materialized into a valid solution (`MaterializationError`),
/// - or the solver exceeds its wall-clock time budget (`Timeout`).
///
/// On timeout, any partial solutions discovered before expiration are discarded
/// and the error is bubbled up to the caller, so the end user sees an explicit
/// failure rather than a silently truncated result set.
pub fn solve_equation(
    input: &str,
    word_list: &[&str],
    num_results_requested: usize,
) -> Result<SolveResult, SolverError> {
    // Precondition: validate input at API boundary
    if input.is_empty() {
        return Err(SolverError::ParseFailure(Box::new(ParseError::EmptyForm)));
    }
    debug_assert!(
        num_results_requested > 0,
        "num_results_requested must be positive"
    );
    debug_assert!(
        word_list.iter().all(|w| !w.is_empty()),
        "All words in word list must be non-empty"
    );

    // 1. Make a hash set version of our word list
    let word_list_as_set = word_list.iter().copied().collect();

    // 2. Parse the input equation string into our `EquationContext` struct.
    //    This holds each pattern string, its parsed form, and its `lookup_keys` (shared vars).
    let equation_context = input.parse::<EquationContext>()?;

    // If there are no patterns, propagate a NoPatterns error.
    if equation_context.len() == 0 {
        return Err(SolverError::NoPatterns);
    }

    // 3. Prepare storage for candidate buckets, one per pattern.
    //    `CandidateBuckets` tracks (a) the bindings bucketed by shared variable values, and
    //    (b) a count so we can stop early if a pattern gets too many matches.
    // Mutable because we fill buckets/counts during the scan phase.
    let mut words: Vec<CandidateBuckets> = Vec::with_capacity(equation_context.len());
    for _ in &equation_context {
        words.push(CandidateBuckets::default());
    }

    // 4. Pull out some data from equation_context
    let lookup_keys = &equation_context.lookup_keys;
    let parsed_forms = &equation_context.parsed_forms;
    let joint_constraints = &equation_context.joint_constraints;

    // 5. Iterate through every candidate word.
    let budget = TimeBudget::new(Duration::from_secs(TIME_BUDGET));

    let mut results: Vec<Vec<Bindings>> = vec![];
    let mut selected: Vec<Bindings> = vec![];
    let mut env: HashMap<char, Rc<str>> = HashMap::new();

    // scan_pos tracks how far into the word list we've scanned.
    let mut scan_pos = 0;

    // Global set of fingerprints for already-emitted solutions.
    // Ensures we don't return duplicate solutions across scan/join rounds.
    let mut seen: HashSet<u64> = HashSet::new();

    // batch_size controls how many words to scan this round (adaptive).
    let mut batch_size = DEFAULT_BATCH_SIZE;

    // High-level solver driver. Alternates between:
    //   1. scanning more words from the dictionary into candidate buckets
    //   2. recursively joining those buckets into full solutions
    // Continues until either we have enough results, the word list is exhausted,
    // or the time budget expires.
    #[cfg(debug_assertions)]
    let mut iteration = 0;

    while results.len() < num_results_requested
        && scan_pos < word_list.len()
        && !budget.expired()
    {
        #[cfg(debug_assertions)]
        {
            iteration += 1;
            if iteration % 10 == 0 {
                eprintln!(
                    "[solver] iteration {}: scan_pos={}/{}, batch_size={}, results={}/{}, elapsed={:.2}s",
                    iteration, scan_pos, word_list.len(), batch_size,
                    results.len(), num_results_requested,
                    budget.elapsed().as_secs_f64()
                );
            }
        }

        // 1. Scan the next batch_size words into candidate buckets.
        // Each candidate binding is grouped by its lookup key so later joins are fast.
        let new_pos = scan_batch(
            word_list,
            scan_pos,
            batch_size,
            &equation_context,
            &mut words,
            &budget,
        )?;
        scan_pos = new_pos;

        if budget.expired() {
            break;
        }

        // 2. Attempt to build full solutions from the candidates accumulated so far.
        // This may rediscover old partials, so we use `seen` at the base case
        // to ensure only truly new solutions are added to `results`.
        let rjp = words.iter().zip(lookup_keys.iter()).zip(equation_context.ordered_list.iter()).zip(parsed_forms.iter())
            .map(|(((candidate_buckets, lookup_keys), p), parsed_form)| {
                RecursiveJoinParameters {
                    candidate_buckets,
                    lookup_keys,
                    pattern: p,
                    parsed_form,
                }
            }).collect::<Vec<_>>();

        // Set up the context for `recursive_join`
        let ctx = JoinCtx {
            num_results_requested,
            word_set: &word_list_as_set,
            joint_constraints,
            budget: &budget,
        };

        // Call `recursive_join`
        let rj_result = recursive_join(
            &mut selected,
            &mut env,
            &mut results,
            &ctx,
            &mut seen,
            &rjp,
        );
        rj_result?;

        // We exit early in two cases
        // 1. We've hit the number of results requested
        // 2. We have no more words to scan
        if results.len() >= num_results_requested ||
            scan_pos >= word_list.len() {
            break;
        }

        if budget.expired() {
            break;
        }

        // Grow the batch size for the next round
        // TODO: magic number, maybe adaptive resizing?
        batch_size = batch_size.saturating_mul(2);
    }

    // Determine status before consuming results
    let status = if budget.expired() {
        SolveStatus::TimedOut { elapsed: budget.elapsed() }
    } else if results.len() >= num_results_requested {
        SolveStatus::FoundEnough
    } else {
        SolveStatus::WordListExhausted
    };

    // ---- Reorder solutions back to original form order ----
    let reordered = results.into_iter().map(|mut solution| {
        let mut reordered_solution = Vec::with_capacity(solution.len());
        for original_i in 0..solution.len() {
            let ordered_i = equation_context.original_to_ordered[original_i];
            // Move elements by swapping with default values
            reordered_solution.push(std::mem::take(&mut solution[ordered_i]));
        }
        reordered_solution
    }).collect::<Vec<_>>();

    // Postcondition: verify solution structure is consistent
    debug_assert!(
        reordered.len() <= num_results_requested,
        "Number of solutions ({}) must not exceed requested ({})",
        reordered.len(), num_results_requested
    );
    debug_assert!(
        reordered.iter().all(|sol| sol.len() == equation_context.len()),
        "Each solution must have exactly {} bindings (one per pattern)",
        equation_context.len()
    );
    debug_assert!(
        reordered.iter().all(|sol| sol.iter().all(|b| b.get_word().is_some())),
        "All bindings in solutions must have a word set"
    );

    Ok(SolveResult { solutions: reordered, status })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_equation() {
        let word_list: Vec<&str> = vec!["lax", "tax", "lox"];
        let input = "l.x".to_string();
        let results = solve_equation(&input, &word_list, 5).unwrap();
        println!("{results:?}");
        assert_eq!(2, results.len());
    }

    #[test]
    fn test_solve_anagrams() {
        let word_list: Vec<&str> = vec!["integral", "altering", "gallant", "alter"];
        let input = "/triangle".to_string();
        let results = solve_equation(&input, &word_list, 5).unwrap();
        println!("{results:?}");
        assert_eq!(2, results.len());
    }

    #[test]
    fn test_solve_equation2() {
        let word_list: Vec<&str> = vec!["inch", "chin", "dada", "test", "ab"];
        let input = "AB;BA;|A|=2;|B|=2;!=AB".to_string();
        let results = solve_equation(&input, &word_list, 5).unwrap();
        println!("{results:?}");
        assert_eq!(2, results.len());
    }

    #[test]
    fn test_solve_equation3() {
        let word_list = vec!["inch", "chin", "dada", "test", "sky", "sly"];
        let input = "AkB;AlB".to_string();
        let results = solve_equation(&input, &word_list, 5).unwrap();

        let mut sky_bindings = Bindings::default();
        sky_bindings.set_rc('A', Rc::from("s"));
        sky_bindings.set_rc('B', Rc::from("y"));
        sky_bindings.set_word("sky");

        let mut sly_bindings = Bindings::default();
        sly_bindings.set_rc('A', Rc::from("s"));
        sly_bindings.set_rc('B', Rc::from("y"));
        sly_bindings.set_word("sly");
        // NB: this could give a false negative if SLY comes out before SKY (since we presumably shouldn't care about the order), so...
        // TODO allow order independence for equality... perhaps create a richer struct than just Vec<Bindings> that has a notion of order-independent equality
        let expected = vec![vec![sky_bindings, sly_bindings]];
        assert_eq!(expected, results.solutions);
    }

    #[test]
    fn test_solve_equation_joint_constraints() {
        let word_list = vec!["inch", "chin", "chess", "chortle"];
        let input = "ABC;CD;|ABCD|=7".to_string();
        let results = solve_equation(&input, &word_list, 5).unwrap();
        println!("{results:?}");
        let mut inch_bindings = Bindings::default();
        inch_bindings.set_rc('A', Rc::from("i"));
        inch_bindings.set_rc('B', Rc::from("n"));
        inch_bindings.set_rc('C', Rc::from("ch"));
        inch_bindings.set_word("inch");

        let mut chess_bindings = Bindings::default();
        chess_bindings.set_rc('C', Rc::from("ch"));
        chess_bindings.set_rc('D', Rc::from("ess"));
        chess_bindings.set_word("chess");
        let expected = vec![vec![inch_bindings, chess_bindings]];
        assert_eq!(expected, results.solutions);
    }

    #[test]
    fn test_solution_to_string_good() {
        let mut b1 = Bindings::default();
        b1.set_word("AA");
        let mut b2 = Bindings::default();
        b2.set_word("AB");

        let bindings_list = vec![b1, b2];
        let actual = solution_to_string(&bindings_list);

        assert_eq!("AA • AB", actual.unwrap());
    }

    #[test]
    fn test_solution_to_string_bad() {
        let mut b1 = Bindings::default();
        b1.set_word("CC");
        let mut b2 = Bindings::default();
        b2.set_rc('A', Rc::from("a"));

        let bindings_list = vec![b1, b2];
        let actual = solution_to_string(&bindings_list);

        assert!(matches!(*actual.unwrap_err(), ParseFailure { s } if s == "cannot find solution in bindings [A→a]" ));
    }

    #[test]
    fn test_fully_bound() {
        // Toy word list: short, predictable words
        let wl = vec!["atime", "btime", "ab"];

        // Equation has two deterministic patterns Atime, Btime, and then AB
        let eq = "Atime;Btime;AB";

        // Solve with a small limit to ensure it runs to completion
        let solve_result = solve_equation(eq, &wl, 5)
            .expect("equation should not trigger MaterializationError");

        let mut expected_atime_bindings = Bindings::default();
        expected_atime_bindings.set_rc('A', Rc::from("a"));
        expected_atime_bindings.set_word("atime");

        let mut expected_btime_bindings = Bindings::default();
        expected_btime_bindings.set_rc('B', Rc::from("b"));
        expected_btime_bindings.set_word("btime");

        let mut expected_ab_bindings = Bindings::default();
        expected_ab_bindings.set_rc('A', Rc::from("a"));
        expected_ab_bindings.set_rc('B', Rc::from("b"));
        expected_ab_bindings.set_word("ab");

        let expected_bindings_list = vec![expected_atime_bindings, expected_btime_bindings, expected_ab_bindings];

        assert_eq!(1, solve_result.len());
        let sol = solve_result.solutions.get(0).unwrap();

        // we don't care about order, so we allow the actual result to be a list in a different
        // order than the expected list
        // we avoid using HashSet or BTree since Bindings has neither Hash nor Ord trait
        // NB: uses that the 3 expected objects are distinct (for example, the following code could
        // match [x,x,y] to [x,y,y] (e.g.))
        // (n^2... but for n=3 (and it's a test))
        assert_eq!(3, sol.len());
        expected_bindings_list.iter().for_each(|bindings| {
            assert!(sol.contains(bindings))
        });
    }

    #[test]
    fn test_malformed_pattern_returns_error() {
        let words = vec!["TEST"];
        let solver_error = solve_equation("BAD(PATTERN", &words, 10).unwrap_err();
        // verify we get a parse error (could be wrapped in ClauseParseError)
        if let SolverError::ParseFailure(bpe) = solver_error {
            // accept either direct InvalidInput or wrapped in ClauseParseError
            match &*bpe {
                ParseError::InvalidInput { str } if str == "BAD(PATTERN" => {
                    // direct InvalidInput is OK
                }
                ParseError::ClauseParseError { clause, source } if clause == "BAD(PATTERN" => {
                    // wrapped in ClauseParseError--verify the source is also InvalidInput
                    assert!(
                        matches!(**source, ParseError::InvalidInput { ref str } if str == "BAD(PATTERN"),
                        "Expected source to be InvalidInput with 'BAD(PATTERN', got: {:?}", source
                    );
                }
                other => panic!("Expected InvalidInput or ClauseParseError with 'BAD(PATTERN', got: {:?}", other)
            }
        } else {
            panic!("Expected ParseFailure, got: {:?}", solver_error)
        }
    }

    #[test]
    fn test_solve_constraints_only_returns_error() {
        let wl: Vec<&str> = vec!["cat", "dog"];
        let res = solve_equation("|A|=3", &wl, 10);
        assert!(matches!(res, Err(SolverError::NoPatterns)));
    }

    mod error_tests {
        use super::*;

        /// Test that all `SolverError` variants have valid error codes
        #[test]
        fn test_error_codes_are_valid() {
            let parse_err = SolverError::ParseFailure(Box::new(ParseError::EmptyForm));
            assert_eq!(parse_err.code(), "S001");

            let no_patterns_err = SolverError::NoPatterns;
            assert_eq!(no_patterns_err.code(), "S002");

            let mat_err = SolverError::MaterializationError {
                context: "test".to_string(),
            };
            assert_eq!(mat_err.code(), "S003");
        }

        /// Test that error help messages are helpful and non-empty
        #[test]
        fn test_error_help_messages_are_helpful() {
            let no_patterns_err = SolverError::NoPatterns;
            let help = no_patterns_err.help();
            assert!(help.is_some(), "NoPatterns should have help text");
            assert!(
                help.unwrap().len() > 20,
                "Help text should be reasonably detailed"
            );
            assert!(
                help.unwrap().contains("pattern"),
                "Help should mention 'pattern'"
            );

            let mat_err = SolverError::MaterializationError {
                context: "test".to_string(),
            };
            let mat_help = mat_err.help();
            assert!(mat_help.is_some(), "MaterializationError should have help text");
        }

        /// Test that `display_detailed` includes error code and help
        #[test]
        fn test_display_detailed_format() {
            let no_patterns_err = SolverError::NoPatterns;
            let detailed = no_patterns_err.display_detailed();

            // should include error code
            assert!(
                detailed.contains("S002"),
                "Detailed display should include error code"
            );

            // should include help text
            assert!(
                detailed.contains("pattern"),
                "Detailed display should include help text"
            );
        }

        /// Test that `ParseFailure` error chains are properly constructed
        #[test]
        fn test_parse_failure_error_chain() {
            let words = vec!["test"];
            let result = solve_equation("INVALID(", &words, 10);

            match result {
                Err(SolverError::ParseFailure(parse_err)) => {
                    // Verify the error chain is accessible
                    let detailed = SolverError::ParseFailure(parse_err).display_detailed();
                    assert!(
                        detailed.contains("S001"),
                        "ParseFailure should have S001 code"
                    );
                    assert!(
                        detailed.contains("caused by"),
                        "ParseFailure should show error chain"
                    );

                    // The underlying ParseError should have its own code
                    assert!(
                        detailed.contains("E0"),
                        "Should contain ParseError code (E0xx)"
                    );
                }
                _ => panic!("Expected ParseFailure error"),
            }
        }

        /// Test that `MaterializationError` includes context
        #[test]
        fn test_materialization_error_context() {
            let err = SolverError::MaterializationError {
                context: "pattern depth: 2/3, environment: {A: \"test\"}".to_string(),
            };

            let display = err.to_string();
            assert!(
                display.contains("pattern depth"),
                "MaterializationError should include context"
            );
            assert!(
                display.contains("environment"),
                "MaterializationError should include environment info"
            );
        }

        /// Test that `NoPatterns` error has correct message
        #[test]
        fn test_no_patterns_error_message() {
            let err = SolverError::NoPatterns;
            let msg = err.to_string();

            assert!(
                msg.contains("no patterns"),
                "NoPatterns message should mention 'no patterns'"
            );
            assert!(
                msg.contains("constraints"),
                "NoPatterns message should mention 'constraints'"
            );
        }

        /// Test that empty input returns appropriate error
        #[test]
        fn test_empty_input_error() {
            let words = vec!["test"];
            let result = solve_equation("", &words, 10);

            match result {
                Err(SolverError::ParseFailure(parse_err)) => {
                    assert!(
                        matches!(*parse_err, ParseError::EmptyForm),
                        "Empty input should produce EmptyForm error"
                    );
                }
                _ => panic!("Expected ParseFailure with EmptyForm"),
            }
        }

        /// Test that error display is consistent with debug
        #[test]
        fn test_error_display_consistency() {
            let err = SolverError::NoPatterns;

            let display = err.to_string();
            let debug = format!("{:?}", err);

            // debug should contain the variant name
            assert!(
                debug.contains("NoPatterns"),
                "Debug should show variant name"
            );

            // display should be user-friendly
            assert!(
                !display.contains("NoPatterns"),
                "Display should not expose enum variant names"
            );
        }
    }

}
