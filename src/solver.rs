use crate::bindings::{Bindings, WORD_SENTINEL};
use crate::errors::{MaterializationError, ParseError};
use crate::joint_constraints::JointConstraints;
use crate::parser::{match_equation_all, ParsedForm};
use crate::patterns::{Pattern, EquationContext};
use crate::errors::ParseError::ParseFailure;
use instant::Instant;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
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
    /// Solver ran to completion (word list exhausted or requested number found).
    Complete,
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

    /// Failure while materializing a candidate binding into a concrete solution.
    ///
    /// This generally indicates that an internal constraint check failed during
    /// `recursive_join` or related routines.
    #[error("materialization error: {0}")]
    MaterializationError(#[from] MaterializationError),
}



/// Simple `TimeoutError` struct
#[derive(Debug)]
pub struct TimeoutError {
    pub elapsed: Duration,
}

impl std::fmt::Display for TimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "solver timed out after {:.3}s", self.elapsed.as_secs_f64())
    }
}

impl std::error::Error for TimeoutError {}


/// Bucket key for indexing candidates by the subset of variables that must agree.
/// - `None` means "no lookup constraints for this pattern" (Python's `words[i][None]`).
/// - When present, we store a *sorted* `(var_char, var_val)` list so the key is deterministic
///   and implements `Eq`/`Hash` naturally. This mirrors Python's
///   `frozenset(dict(...).items())`, but with a stable order.
/// - The sort happens once when we construct the key, not on hash/compare.
pub type LookupKey = Vec<(char, String)>;

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

/// Build the deterministic lookup key for a binding given the pattern's lookup vars.
///
/// Returns a `LookupKey` (alias for `Vec<(char, String)>`) sorted by `var_char`.
///
/// Conventions:
/// - If `keys` is empty, returns an empty `LookupKey` (unkeyed bucket).
/// - If any required key is missing in the binding, also returns an empty `LookupKey`;
///   callers must check `keys.is_empty()` to distinguish this case.
/// - Otherwise, returns the full normalized key.
fn lookup_key_for_binding(
    binding: &Bindings,
    keys: HashSet<char>,
) -> LookupKey {
    let mut pairs: Vec<(char, String)> = Vec::with_capacity(keys.len());
    for var_char in keys {
        match binding.get(var_char) {
            Some(var_val) => pairs.push((var_char, var_val.clone())),
            None => return Vec::new(), // required key missing
        }
    }

    pairs.sort_unstable_by_key(|(c, _)| *c);
    pairs
}



/// Push a binding into the appropriate bucket and bump the count.
fn push_binding(words: &mut [CandidateBuckets], i: usize, key: LookupKey, binding: Bindings) {
    words[i].buckets.entry(key).or_default().push(binding);
    words[i].count += 1;
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
/// `budget.expired()` or by comparing the return value against the planned end)
/// and convert that condition into `SolverError::Timeout` at a higher level.
fn scan_batch(
    word_list: &[&str],
    start_idx: usize,
    batch_size: usize,
    equation_context: &EquationContext,
    words: &mut [CandidateBuckets],
    budget: &TimeBudget,
) -> Result<usize, SolverError> {

    let mut i_word = start_idx;
    let end = start_idx.saturating_add(batch_size).min(word_list.len());

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
                equation_context.joint_constraints.clone(),
            );

            for binding in matches {
                timed_stop!(budget, i_word);
                let key = lookup_key_for_binding(&binding, p.lookup_keys.clone());

                // If a required key is missing, skip
                if key.is_empty() && !p.lookup_keys.is_empty() {
                    continue;
                }

                push_binding(words, i, key, binding.clone());
            }
        }

        i_word += 1;
    }

    Ok(i_word)
}



struct RecursiveJoinParameters {
    candidate_buckets: CandidateBuckets,
    lookup_keys: HashSet<char>,
    pattern: Pattern,
    parsed_form: ParsedForm,
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
/// - `num_results_requested`: cap on how many full solutions to collect.
/// - `rjp.candidate_buckets`: per-pattern candidate buckets (what you built during scanning).
/// - `rjp.lookup_keys`: for each pattern, which variables must agree with previously chosen
///   patterns.
///
/// Return:
/// - This function mutates `results` and stops early once `results` contains
///   `num_results_requested` values.
/// # Errors
/// Returns `Err(SolverError::Timeout)` if the time budget expires during the join.
/// In that case, partial solutions already found remain in `results`.
fn recursive_join(
    selected: &mut Vec<Bindings>,
    env: &mut HashMap<char, String>,
    results: &mut Vec<Vec<Bindings>>,
    num_results_requested: usize,
    word_list_as_set: &HashSet<&str>,
    joint_constraints: JointConstraints,
    seen: &mut HashSet<u64>,
    rjp: &[RecursiveJoinParameters],
    budget: &TimeBudget,
) -> Result<(), SolverError> {
    // Stop if we've met the requested quota of full solutions.
    if results.len() >= num_results_requested {
        return Ok(());
    }

    timed_stop!(budget);

    if let Some(rjp_cur) = rjp.first() {
        // ---- FAST PATH: deterministic + fully keyed ----------------------------
        let p = &rjp_cur.pattern;
        if p.is_deterministic && p.all_vars_in_lookup_keys() {
            // The word is fully determined by literals + already-bound vars in `env`.
            let Some(expected) = rjp_cur.parsed_form
                .materialize_deterministic_with_env(env)
            else {
                return Err(SolverError::MaterializationError(
                    MaterializationError("deterministic materialization failed".to_string()),
                ));
            };

            if !word_list_as_set.contains(expected.as_str()) {
                // This branch cannot succeed — prune immediately.
                return Ok(());
            }

            // Build a minimal Bindings for this pattern:
            // - include WORD_SENTINEL (whole word)
            // - include only vars that belong to this pattern (they must already be in env)
            let mut binding = Bindings::default();
            binding.set_word(&expected);
            for &var_char in &p.variables {
                // safe to unwrap because all vars are in lookup_keys ⇒ must be in env
                if let Some(var_val) = env.get(&var_char) {
                    binding.set(var_char, var_val.clone());
                }
            }

            selected.push(binding);
            recursive_join(selected, env, results, num_results_requested, word_list_as_set, joint_constraints, seen, &rjp[1..], budget)?;
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
            // Build (var_char, var_val) pairs from env using the set of shared vars.
            // NOTE: HashSet iteration order is arbitrary — we sort the pairs below
            // so the final key is stable/deterministic.
            let mut pairs: Vec<(char, String)> = Vec::with_capacity(rjp_cur.lookup_keys.len());
            for &var_char in &rjp_cur.lookup_keys {
                timed_stop!(budget);
                if let Some(var_val) = env.get(&var_char) {
                    pairs.push((var_char, var_val.clone()));
                } else {
                    // If any required var isn't bound yet, there can be no matches for this branch.
                    return Ok(());
                }
            }
            // Deterministic key: sort by the variable name.
            pairs.sort_unstable_by_key(|(c, _)| *c);

            rjp_cur.candidate_buckets.buckets.get(&pairs)
        };

        // If there are no candidates in that bucket, dead-end this branch.
        let Some(bucket_candidates) = bucket_candidates_opt else {
            return Ok(());
        };

        // Try each candidate binding for this pattern.
        for cand in bucket_candidates {

            timed_stop!(budget);

            if results.len() >= num_results_requested {
                break; // stop early if we've already met the quota
            }

            // Defensive compatibility check: if a variable is already in `env`,
            // its value must match the candidate. This *should* already be true
            // because we selected the bucket using the shared vars—but keep this
            // in case upstream bucketing logic ever changes.
            if cand.iter().filter(|(var_char, _)| **var_char != WORD_SENTINEL).any(|(var_char, var_val)| env.get(var_char).is_some_and(|prev| prev != var_val)) {
                continue;
            }

            // Extend `env` with any *new* bindings from this candidate (don't overwrite).
            // Track what we added so we can backtrack cleanly.
            let mut added_vars: Vec<char> = vec![];
            for (var_char, var_val) in cand.iter() {
                if *var_char == WORD_SENTINEL {
                    continue;
                }
                if !env.contains_key(var_char) {
                    env.insert(*var_char, var_val.clone());
                    added_vars.push(*var_char);
                }
            }

            // Choose this candidate for pattern `idx` and recurse for `idx + 1`.
            selected.push(cand.clone());
            recursive_join(selected, env, results, num_results_requested, word_list_as_set, joint_constraints.clone(), seen, &rjp[1..], budget)?;
            selected.pop();

            // Backtrack: remove only what we added at this level.
            for k in added_vars {
                env.remove(&k);
            }
        }
    } else {
        // Base case: if we've placed all patterns, `selected` is a full solution.
        if joint_constraints.all_strictly_satisfied_for_parts(selected) && seen.insert(solution_key(selected)) {
            results.push(selected.clone());
        }
    }

    Ok(())
}


/// Top-level entry point to solve an equation against a word list.
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
    // 1. Make a hash set version of our word list
    let word_list_as_set = word_list.iter().copied().collect();

    // 2. Parse the input equation string into our `EquationContext` struct.
    //    This holds each pattern string, its parsed form, and its `lookup_keys` (shared vars).
    let equation_context = input.parse::<EquationContext>()?;

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
    let joint_constraints = equation_context.joint_constraints.clone();

    // 5. Iterate through every candidate word.
    let budget = TimeBudget::new(Duration::from_secs(TIME_BUDGET));

    let mut results: Vec<Vec<Bindings>> = vec![];
    let mut selected: Vec<Bindings> = vec![];
    let mut env: HashMap<char, String> = HashMap::new();

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
    while results.len() < num_results_requested
        && scan_pos < word_list.len()
        && !budget.expired()
    {
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

        // TODO? add a time budget check

        // 2. Attempt to build full solutions from the candidates accumulated so far.
        // This may rediscover old partials, so we use `seen` at the base case
        // to ensure only truly new solutions are added to `results`.
        let rjp = words.iter().zip(lookup_keys.iter()).zip(equation_context.ordered_list.iter()).zip(parsed_forms.iter())
            .map(|(((candidate_buckets, lookup_keys), p), parsed_form)| {
                RecursiveJoinParameters {
                    candidate_buckets: candidate_buckets.clone(),
                    lookup_keys: lookup_keys.clone(),
                    pattern: p.clone(),
                    parsed_form: parsed_form.clone(),
                }
            }).collect::<Vec<_>>();
        let rj_result = recursive_join(
            &mut selected,
            &mut env,
            &mut results,
            num_results_requested,
            &word_list_as_set,
            joint_constraints.clone(),
            &mut seen,
            &rjp,
            &budget
        );
        rj_result?;

        // We exit early in two cases
        // 1. We've hit the number of results requested
        // 2. We have no more words to scan
        if results.len() >= num_results_requested ||
            scan_pos >= word_list.len() {
            break;
        }

        // TODO? Add another time budget check

        // Grow the batch size for the next round
        // TODO: magic number, maybe adaptive resizing?
        batch_size = batch_size.saturating_mul(2);
    }

    // ---- Reorder solutions back to original form order ----
    let reordered = results.iter().map(|solution| {
        (0..solution.len()).map(|original_i| {
            solution.clone()[equation_context.original_to_ordered[original_i]].clone()
        }).collect::<Vec<_>>()
    }).collect::<Vec<_>>();

    // Return up to `num_results_requested` reordered solutions
    let status = if budget.expired() {
        SolveStatus::TimedOut { elapsed: budget.elapsed() }
    } else {
        SolveStatus::Complete
    };

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
        sky_bindings.set('A', "s".to_string());
        sky_bindings.set('B', "y".to_string());
        sky_bindings.set_word("sky".to_string().as_ref());

        let mut sly_bindings = Bindings::default();
        sly_bindings.set('A', "s".to_string());
        sly_bindings.set('B', "y".to_string());
        sly_bindings.set_word("sly".to_string().as_ref());
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
        inch_bindings.set('A', "i".to_string());
        inch_bindings.set('B', "n".to_string());
        inch_bindings.set('C', "ch".to_string());
        inch_bindings.set_word("inch".to_string().as_ref());

        let mut chess_bindings = Bindings::default();
        chess_bindings.set('C', "ch".to_string());
        chess_bindings.set('D', "ess".to_string());
        chess_bindings.set_word("chess".to_string().as_ref());
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
        b2.set('A', 'a'.to_string());

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
        expected_atime_bindings.set('A', "a".to_string());
        expected_atime_bindings.set_word("atime".to_string().as_ref());

        let mut expected_btime_bindings = Bindings::default();
        expected_btime_bindings.set('B', "b".to_string());
        expected_btime_bindings.set_word("btime".to_string().as_ref());

        let mut expected_ab_bindings = Bindings::default();
        expected_ab_bindings.set('A', "a".to_string());
        expected_ab_bindings.set('B', "b".to_string());
        expected_ab_bindings.set_word("ab".to_string().as_ref());

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
        let result = solve_equation("BAD(PATTERN", &words, 10);
        assert!(result.is_err(), "Expected parse failure but got success");
    }

    #[test]
    fn test_materialization_error() {
        let words = vec!["a", "b"];
        // Form forces an impossible variable overlap
        let result = solve_equation("AB;A;B;???", &words, 10);
        assert!(result.is_err(), "Expected error but got success");
    }
}
