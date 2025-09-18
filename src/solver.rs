use crate::bindings::{Bindings, WORD_SENTINEL};
use crate::errors::{MaterializationError, ParseError};
use crate::joint_constraints::{propagate_joint_to_var_bounds, JointConstraints};
use crate::parser::prefilter::build_prefilter_regex;
use crate::parser::{match_equation_all, ParsedForm};
use crate::patterns::{Pattern, EquationContext};
use crate::scan_hints::{form_len_hints_pf, PatternLenHints};

use crate::constraints::VarConstraints;
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
            // TODO: throw an error if it does
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
/// TODO: consider using a countdown timer with one "time left" parameter
struct TimeBudget {
    start: Instant,   // when the budget began
    limit: Duration,  // maximum allowed elapsed time
}

impl TimeBudget {
    /// Create a new budget that lasts for `limit` (e.g., 30 seconds).
    fn new(limit: Duration) -> Self {
        Self { start: Instant::now(), limit }
    }

    /// Returns true if the allowed time has fully elapsed.
    fn expired(&self) -> bool {
        self.start.elapsed() >= self.limit
    }

    // Returns the remaining time before expiration, or zero if the budget is already used up.
    // Unused for now but it may be useful later
    // fn remaining(&self) -> Duration {self.limit.saturating_sub(self.start.elapsed())}
}


/// Build the deterministic lookup key for a binding given the pattern's lookup vars.
/// Returns:
///   - None: pattern has no lookup constraints (unkeyed bucket)
///   - Some(vec): concrete key (sorted by `var_char`)
///   - Some(empty vec): sentinel meaning "required key missing" → caller should skip
fn lookup_key_for_binding(
    binding: &Bindings,
    keys: HashSet<char>,
) -> LookupKey {
    // Collect (var_char, var_val) for all required keys; bail out immediately if any is missing.
    let mut pairs: Vec<(char, String)> = Vec::with_capacity(keys.len());
    for var_char in keys {
        match binding.get(var_char) {
            Some(var_val) => pairs.push((var_char, var_val.clone())),
            None => return Vec::new(), // "impossible" sentinel; caller will skip // TODO is this right?
        }
    }

    // Normalize key order for stable hashing/equality
    pairs.sort_unstable_by_key(|(c, _)| *c);

    pairs
}


/// Push a binding into the appropriate bucket and bump the count.
fn push_binding(words: &mut [CandidateBuckets], i: usize, key: LookupKey, binding: Bindings) {
    words[i].buckets.entry(key).or_default().push(binding);
    words[i].count += 1;
}

/// Scan a slice of the word list and incrementally fill candidate buckets.
/// Returns a pair containing (in order) the new scan position and a boolean stating if time is up.
// TODO reword last sentence to be umm better
fn scan_batch(
    word_list: &[&str],
    start_idx: usize,
    batch_size: usize,
    equation_context: &EquationContext,
    parsed_forms: &[ParsedForm],
    scan_hints: &[PatternLenHints],
    var_constraints: &VarConstraints,
    joint_constraints: &JointConstraints,
    words: &mut [CandidateBuckets],
    budget: &TimeBudget,
) -> (usize, bool) {
    let mut i_word = start_idx;
    let end = start_idx.saturating_add(batch_size).min(word_list.len());

    while i_word < end {
        // TODO: have this timeout bubble all the way up
        if budget.expired() {
            return (i_word, true);
        }

        let word = word_list[i_word];

        for (i, p) in equation_context.iter().enumerate() {
            // No per-pattern cap anymore

            // Skip deterministic fully keyed forms
            if p.is_deterministic && p.all_vars_in_lookup_keys() {
                continue;
            }
            // Cheap length prefilter
            if !scan_hints[i].is_word_len_possible(word.len()) {
                continue;
            }

            let matches = match_equation_all(
                word,
                &parsed_forms[i],
                var_constraints,
                joint_constraints.clone(),
            );

            for binding in matches {
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

    (i_word, false)
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
fn recursive_join(
    selected: &mut Vec<Bindings>,
    env: &mut HashMap<char, String>,
    results: &mut Vec<Vec<Bindings>>,
    num_results_requested: usize,
    word_list_as_set: &HashSet<&str>,
    joint_constraints: JointConstraints,
    seen: &mut HashSet<u64>,
    rjp: &[RecursiveJoinParameters]
) -> Result<(), MaterializationError> {
    // Stop if we've met the requested quota of full solutions.
    if results.len() >= num_results_requested {
        return Ok(());
    }

    if let Some(rjp_cur) = rjp.first() {
        // ---- FAST PATH: deterministic + fully keyed ----------------------------
        let p = &rjp_cur.pattern;
        if p.is_deterministic && p.all_vars_in_lookup_keys() {
            // The word is fully determined by literals + already-bound vars in `env`.
            let Some(expected) = rjp_cur.parsed_form.materialize_deterministic_with_env(env) else { return Err(MaterializationError) };

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
            recursive_join(selected, env, results, num_results_requested, word_list_as_set, joint_constraints, seen, &rjp[1..])?;
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
            recursive_join(selected, env, results, num_results_requested, word_list_as_set, joint_constraints.clone(), seen, &rjp[1..])?;
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


/// Read in an equation string and return results from the word list
///
/// - `input`: equation in our pattern syntax (e.g., `"AB;BA;|A|=2;..."`)
/// - `word_list`: list of candidate words to test.
///   Note that we require (but do not enforce!) that all words be lowercase.
///   TODO: should we enforce this?
/// - `num_results_requested`: maximum number of *final* results to return
///
/// Returns:
/// - A `Vec` of solutions, each solution being a `Vec<Binding>` where each `Binding`
///   maps variable names (chars) to concrete substrings they were bound to in that solution.
///
/// # Errors
///
/// Will return a `ParseError` if a form cannot be parsed.
// TODO? add more detail in Errors section
pub fn solve_equation(input: &str, word_list: &[&str], num_results_requested: usize) -> Result<Vec<Vec<Bindings>>, Box<ParseError>> {
    // 0. Make a hash set version of our word list
    let word_list_as_set = word_list.iter().copied().collect();

    // 1. Parse the input equation string into our `Patterns` struct.
    //    This holds each pattern string, its parsed form, and its `lookup_keys` (shared vars).
    let equation_context = input.parse::<EquationContext>()?;

    // 2. Build per-pattern lookup key specs (shared vars) for the join
    let lookup_keys: Vec<_> =
        equation_context.iter().map(|p| p.lookup_keys.clone()).collect();

    // 3. Prepare storage for candidate buckets, one per pattern.
    //    `CandidateBuckets` tracks (a) the bindings bucketed by shared variable values, and
    //    (b) a count so we can stop early if a pattern gets too many matches.
    // Mutable because we fill buckets/counts during the scan phase.
    let mut words: Vec<CandidateBuckets> = Vec::with_capacity(equation_context.len());
    for _ in &equation_context {
        words.push(CandidateBuckets::default());
    }

    // TODO: move a bunch of the next few steps into EquationContext creation

    // 4. Parse each pattern's string form once into a `ParsedForm` (essentially a `Vec` of
    //    `FormPart`s). These are index-aligned with `equation_context`.

    let mut parsed_forms: Vec<_> = equation_context
        .iter()
        .map(|p| {
            let raw_form = &p.raw_string;
            raw_form.parse::<ParsedForm>()
        })
        .collect::<Result<_, _>>()?;

    // 5. Pull out the per-variable constraints collected from the equation.
    let mut var_constraints = equation_context.var_constraints.clone();

    // 6. Upgrade prefilters once per form (only if it helps)
    // Specifically, if a variable has a "form" (like `g*`), we upgrade its prefilter
    // from `.+` to `g.*`
    // TODO: why not do this when constructing Patterns?
    for pf in &mut parsed_forms {
        let upgraded = build_prefilter_regex(pf, &var_constraints)?;
        pf.prefilter = upgraded;
    }

    // 7. Get the joint constraints and use them to tighten per-variable constraints
    // This gets length bounds on variables (from the joint constraints)
    let joint_constraints = equation_context.joint_constraints.clone();
    propagate_joint_to_var_bounds(&mut var_constraints, &joint_constraints);

    // 8. Build cheap, per-form length hints once (index-aligned with equation_context/parsed_forms)
    // The hints are length bounds for each form
    let scan_hints: Vec<_> = parsed_forms
        .iter()
        .map(|pf| form_len_hints_pf(pf, &equation_context.var_constraints, &joint_constraints.clone()))
        .collect();

    // 9. Iterate through every candidate word.
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
        let (new_pos, _is_time_up) = scan_batch(
            word_list,
            scan_pos,
            batch_size,
            &equation_context,
            &parsed_forms,
            &scan_hints,
            &var_constraints,
            &joint_constraints,
            &mut words,
            &budget,
        );
        scan_pos = new_pos;

        // Respect the TimeBudget
        if budget.expired() { break; }

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
            &rjp
        );
        if let Err(e) = rj_result {
            // e is a `MaterializationError` // TODO check/enforce this?
            return Err(Box::new(ParseFailure { s: e.to_string() }))
        }

        // We exit early in three cases
        // 1. We've hit the number of results requested
        // 2. The time is up
        // 3. We have no more words to scan
        if results.len() >= num_results_requested ||
            budget.expired() ||
            scan_pos >= word_list.len() {
            break;
        }

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
    Ok(reordered)
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
        assert_eq!(expected, results);
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
        assert_eq!(expected, results);
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
        let sols = solve_equation(eq, &wl, 5)
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

        assert_eq!(1, sols.len());
        let sol = sols.get(0).unwrap();

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
}
