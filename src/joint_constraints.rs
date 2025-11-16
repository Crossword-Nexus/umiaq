use crate::bindings::Bindings;
use crate::comparison_operator::ComparisonOperator;
use crate::constraints::{Bounds, VarConstraint, VarConstraints};
use crate::errors::ParseError;
use fancy_regex::{escape, Regex};
use log::debug;
use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;
use std::sync::LazyLock;

#[cfg(test)]
use crate::patterns::FORM_SEPARATOR;
#[cfg(test)]
use std::collections::HashMap;

/// Compact representation of the relation between (sum) and (target).
///
/// We encode three mutually exclusive outcomes as bits:
/// - LT (sum < target)  -> 0b001
/// - EQ (sum == target) -> 0b010
/// - GT (sum > target)  -> 0b100
///
/// Compound operators (<=, >=, !=) are unions of these bits.
/// Evaluation can then be done as: `rel.allows(total.cmp(&target))`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelMask {
    mask: u8,
}

impl RelMask {
    pub const LT: Self = Self { mask: 0b001 };
    pub const EQ: Self = Self { mask: 0b010 };
    pub const GT: Self = Self { mask: 0b100 };

    pub const LE: Self = Self { mask: Self::LT.mask | Self::EQ.mask }; // <=
    pub const GE: Self = Self { mask: Self::GT.mask | Self::EQ.mask }; // >=
    pub const NE: Self = Self { mask: Self::LT.mask | Self::GT.mask }; // !=

    /// Return true if this mask allows the given ordering outcome.
    #[inline]
    pub(crate) fn allows(self, ord: Ordering) -> bool {
        let bit = match ord {
            Ordering::Less    => 0b001,
            Ordering::Equal   => 0b010,
            Ordering::Greater => 0b100,
        };
        (self.mask & bit) != 0
    }
}

impl fmt::Display for RelMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match *self {
            RelMask::LT => "<",
            RelMask::EQ => "=",
            RelMask::GT => ">",
            RelMask::LE => "<=",
            RelMask::GE => ">=",
            RelMask::NE => "!=",
            _ => "?", // TODO handle some other way?
        };
        write!(f, "{s}")
    }
}

impl FromStr for RelMask {
    type Err = ParseError;

    /// Parse an operator token into a `RelMask`.
    /// Accepted: "=", "!=", "<=", ">=", "<", ">".
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(
            match ComparisonOperator::from_str(s)? {
                ComparisonOperator::EQ => Self::EQ,
                ComparisonOperator::NE => Self::NE,
                ComparisonOperator::LE => Self::LE,
                ComparisonOperator::GE => Self::GE,
                ComparisonOperator::LT => Self::LT,
                ComparisonOperator::GT => Self::GT,
            }
        )
    }
}

/// Joint length constraint like `|ABC| <= 7`.
///
/// - `vars`  : the participating variable names (A–Z). Duplicates **do** count toward the sum.
/// - `target`: RHS integer to compare against.
/// - `rel`   : operator, stored as a relation mask (see `RelMask`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JointConstraint {
    pub vars: Vec<char>,   // e.g., ['A','B','C']
    pub target: usize,     // e.g., 7
    pub rel: RelMask,      // operator as data
}

impl FromStr for JointConstraint {
    type Err = ParseError;

    /// Attempt to parse a `JointConstraint` from a string.
    ///
    /// This uses the existing `parse_joint_len` helper, which recognizes
    /// joint length expressions such as `|AB|=7`.
    ///
    /// - On success, returns the parsed `JointConstraint`.
    /// - On failure (`parse_joint_len` returns `None`), produces a
    ///   `ParseError::InvalidInput` with the offending string.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Delegate to the existing parser.
        // `parse_joint_len` returns `Option<JointConstraint>`,
        // so map `None` into a `ParseError`.
        parse_joint_len(s).ok_or_else(|| {
            ParseError::InvalidInput {
                str: s.to_string(),
            }
        })
    }
}

impl JointConstraint {
    /// Check satisfaction against current `bindings`.
    ///
    /// **Mid-search semantics** (by design): if **any** referenced var is unbound,
    /// we return `true` (no opinion yet). This keeps partial assignments alive.
    ///
    /// If you need a *final* strict check, run this only after all vars are bound,
    /// or add a separate strict method that returns `false` when some vars are unbound.
    #[inline]
    pub(crate) fn is_satisfied_by(&self, bindings: &Bindings) -> bool {
        // If not all vars are bound, skip this check for now.
        if bindings.contains_all_vars(&self.vars) {
            // Sum the lengths of the bound strings for the referenced vars.
            // safe: unwrap is guaranteed to succeed because contains_all_vars returned true
            let total: usize = self.vars.iter()
                .map(|var_char| {
                    let binding = bindings.get(*var_char);
                    debug_assert!(binding.is_some(), "var '{var_char}' must be bound after contains_all_vars check");
                    binding.expect("var must be bound after contains_all_vars check").len()
                })
                .sum();

            // Compare once via Ordering -> mask test.
            self.rel.allows(total.cmp(&self.target))
        } else {
            true
        }
    }

    /// Check this constraint against a *solution row* represented as a slice of Bindings
    /// (one Bindings per form/pattern). Duplicates in `vars` count toward the sum.
    /// Returns `false` if *any* referenced var is unbound across `parts`.
    pub fn is_strictly_satisfied_by_parts(&self, parts: &[Bindings]) -> bool {
        let mut total = 0;
        for var_len in self.vars.iter().map(|var_char| resolve_var_len(parts, *var_char)) {
            let Some(len) = var_len else { return false };
            total += len;
        }
        self.rel.allows(total.cmp(&self.target))
    }

    // --- Test-only convenience for asserting behavior without needing real `Bindings`.
    //     This keeps tests independent of crate::bindings internals.
    #[cfg(test)]
    fn is_satisfied_by_map(&self, map: &HashMap<char, String>) -> bool {
        if !self.vars.iter().all(|var_char| map.contains_key(var_char)) {
            true
        } else {
            let total: usize = self.vars.iter().map(|var_char| map.get(var_char).unwrap().len()).sum();
            self.rel.allows(total.cmp(&self.target))
        }
    }
}

impl fmt::Display for JointConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let vars: String = self.vars.iter().collect();
        write!(f, "|{}| {} {}", vars, self.rel, self.target)
    }
}


/// Helper: find the current length of variable `var_char` across a slice of `Bindings`.
/// Returns `None` if `var_char` is unbound in all Bindings in `parts`.
#[inline]
fn resolve_var_len(parts: &[Bindings], var_char: char) -> Option<usize> {
    parts.iter().find_map(|bindings| bindings.get(var_char).map(|s| s.len()))
}

/// Regex pattern for joint length constraints like `|AB|=7`.
/// Built dynamically from `ComparisonOperator::ALL` to ensure single source of truth.
static JOINT_LEN_PATTERN: LazyLock<String> = LazyLock::new(|| {
    let all_ops_string = ComparisonOperator::all_as_strings()
        .iter()
        .map(|s| escape(s))
        .collect::<Vec<_>>()
        .join("|");
    format!(r"^\|(?<vars>[A-Z]{{2,}})\| *(?<op>{all_ops_string}) *(?<len>\d+)$")
});

/// Matches joint length constraints like `|AB|=7`
///
/// NB: This regex is validated at WASM startup in `wasm::validate_internal_regexes()`.
/// If a new `LazyLock<Regex>` is added, add it there too!
pub(crate) static JOINT_LEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&JOINT_LEN_PATTERN)
        .unwrap_or_else(|e| panic!(
            "BUG: Failed to compile JOINT_LEN_RE regex pattern '{}': {e}.", *JOINT_LEN_PATTERN
        )));

/// Parse a single joint-length expression that **starts at** a `'|'`. Returns `None` on invalid
/// input.
///
/// Shape: `|VARS| OP NUMBER`
///  - `VARS`  : at least **two** ASCII uppercase letters (A–Z).
///  - `OP`    : one of `<=`, `>=`, `!=`, `<`, `>`, `=` (NB: two-char ops checked first).
///  - `NUMBER`: one or more ASCII digits (base 10).
///
/// Returns `None` instead of propagating errors because this function is called speculatively
/// during pattern parsing on clauses that may or may not be joint constraints. Invalid syntax
/// simply means "not a joint constraint", not a fatal error. Only regex compilation errors
/// (at initialization) are true errors that should panic.
fn parse_joint_len(expr: &str) -> Option<JointConstraint> {
    if let Ok(Some(captures)) = JOINT_LEN_RE.captures(expr) {
        let vars_match = captures.name("vars")?;
        let target_match = captures.name("len")?;
        let op_str_match = captures.name("op")?;

        let vars = vars_match.as_str().chars().collect();
        let Ok(target) = target_match.as_str().parse() else { return None };
        let Ok(rel) = RelMask::from_str(op_str_match.as_str()) else { return None };

        Some(JointConstraint { vars, target, rel })
    } else {
        None
    }
}

/// Container for many joint constraints (useful as a field on your puzzle/parse).
#[derive(Debug, Default, Clone)]
pub struct JointConstraints {
    as_vec: Vec<JointConstraint>
}

impl IntoIterator for JointConstraints {
    type Item = JointConstraint;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_vec.into_iter()
    }
}

impl JointConstraints {
    /// Return an iterator over the joint constraints
    pub(crate) fn iter(&self) -> impl Iterator<Item = &JointConstraint> {
        self.as_vec.iter()
    }

    /// Parse all joint constraints from an equation string by splitting on your
    /// `FORM_SEPARATOR` (i.e., ';'), feeding each part through `parse_joint_len`.
    #[cfg(test)]
    pub(crate) fn parse_equation(equation: &str) -> JointConstraints {
        let jc_vec = equation.split(FORM_SEPARATOR).filter_map(|part| {
            parse_joint_len(part.trim())
        }).collect();

        JointConstraints { as_vec: jc_vec }
    }

    /// Insert a new `JointConstraint` into this collection.
    ///
    /// This is a thin wrapper around `Vec::push` on the internal
    /// storage. Keeping the push logic behind a method preserves
    /// encapsulation: callers don’t need to know or rely on the fact
    /// that `JointConstraints` is backed by a `Vec`
    pub(crate) fn add(&mut self, jc: JointConstraint) {
        // Delegate to the underlying Vec implementation.
        self.as_vec.push(jc);
    }


    pub(crate) fn is_empty(&self) -> bool {
        self.as_vec.is_empty()
    }

    /// Number of joint constraints.
    pub(crate) fn len(&self) -> usize {
        self.as_vec.len()
    }

    /// Return true iff **every** joint constraint is satisfied w.r.t. `bindings`.
    ///
    /// Mid-search semantics: a constraint with unbound vars returns `true`
    /// (see `JointConstraint::is_satisfied_by`), so this is safe to call
    /// during search as a "non-pruning check".
    pub(crate) fn all_satisfied(&self, bindings: &Bindings) -> bool {
        self.as_vec.iter().all(|jc| jc.is_satisfied_by(bindings))
    }

    /// True iff **every** joint constraint is satisfied w.r.t. a slice of `Bindings`.
    /// Requires all referenced variables to be bound.
    pub fn all_strictly_satisfied_for_parts(&self, parts: &[Bindings]) -> bool {
        self.as_vec.iter().all(|jc| jc.is_strictly_satisfied_by_parts(parts))
    }

    // Test-only helper mirroring `all_satisfied` over a plain map.
    #[cfg(test)]
    fn all_satisfied_map(
        &self,
        map: &HashMap<char, String>
    ) -> bool {
        self.as_vec.iter().all(|jc| jc.is_satisfied_by_map(map))
    }

    #[cfg(test)]
    pub(crate) fn of(as_vec: Vec<JointConstraint>) -> JointConstraints {
        JointConstraints { as_vec }
    }
}

impl fmt::Display for JointConstraints {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.as_vec.is_empty() {
            write!(f, "(none)")
        } else {
            for (i, jc) in self.as_vec.iter().enumerate() {
                if i > 0 {
                    writeln!(f)?;
                }
                write!(f, "{jc}")?;
            }
            Ok(())
        }
    }
}

/// Attempt to tighten per-variable length bounds using information from joint constraints.
///
/// # Overview
///
/// This propagation step converts joint constraints (equalities over groups of variables)
/// into tighter individual variable bounds, potentially dramatically reducing the search space.
///
/// # Example: Exact Propagation
///
/// ```text
/// Joint constraint: |ABCDEFGHIJKLMN| = 14
/// Default per-var bounds: each ≥ 1
/// Sum of mins: 14 × 1 = 14 = target
/// Conclusion: Every variable must be exactly length 1
/// ```
///
/// This allows the solver to avoid exploring assignments where any variable is length > 1.
///
/// # Example: Interval Tightening
///
/// ```text
/// joint constraint: |ABC| = 10
/// initial bounds: A ∈ [1,∞), B ∈ [1,∞), C ∈ [1,∞)
///
/// for variable A:
///   • min for |A| = max(1, 10 - ("∞" + "∞")) = 1  (i.e., no tightening)
///   • max for |A| = min("∞", 10 - (1 + 1)) = 8
///
/// result: |A| ∈ [1,8], |B| ∈ [1,8], |C| ∈ [1,8]
/// ```
///
/// This reduces the search space from "∞"³ to 8³ = 512 possibilities.
///
/// # Circuit-Breaker Pattern
///
/// Before attempting potentially expensive search, this function fails fast when constraints are
/// provably unsatisfiable:
///
/// - if `sum(mins) > target` (minimum required exceeds target)
/// - if `sum(maxes) < target` (maximum possible falls short of target)
///
/// Example contradiction:
/// ```text
/// |AB| = 5 with |A| ≥ 3, |B| ≥ 4
/// sum of mins: 3 + 4 = 7 > 5
/// ```
///
/// # Algorithm
///
/// 1. **Early fail**: if `sum(mins) > T` or `sum(maxes) < T`, return `ContradictoryBounds`
/// 2. **Exact by mins**: if `sum(mins) == T`, set all vars to their min (exact)
/// 3. **Exact by maxes**: if `sum(maxes) == T` (finite), set all vars to their max (exact)
/// 4. **Generic tightening**: otherwise, for each variable Vi:
///    - new min for Vi = max(current min, T - Σ other maxes)
///    - new max for Vi = min(current max, T - Σ other mins)
///    - if new min > new max, fail with `ContradictoryBounds`
///
/// # Soundness
///
/// This propagation is **sound** (never removes feasible solutions). It only tightens
/// bounds based on provable arithmetic constraints.
///
/// # Performance Impact
///
/// This is a critical optimization that often eliminates large amounts of search space:
/// - **Best case**: reduces "∞ⁿ" search space to finite bounds
/// - **Common case**: tightens [1,∞) to [1,k] for small k
/// - **Worst case**: no tightening (sum(mins) < T < sum(maxes))
///
/// Especially effective for long chains like `|ABCDEFGH| = 15` where default bounds
/// would allow exponential combinations.
///
/// # Errors
///
/// Returns `Err(ContradictoryBounds)` if the constraints are provably unsatisfiable.
pub fn propagate_joint_to_var_bounds(vcs: &mut VarConstraints, jcs: &JointConstraints) -> Result<(), Box<ParseError>> {
    for jc in jcs.iter() {
        if jc.rel != RelMask::EQ { continue; }

        // Cache per-var (min,max) and aggregate sums
        let mut sum_min = 0;
        let mut sum_max_opt = Some(0);

        let mut mins = Vec::with_capacity(jc.vars.len());
        let mut maxes = Vec::with_capacity(jc.vars.len());

        for &var_char in &jc.vars {
            let bounds = vcs.bounds(var_char);
            sum_min += bounds.min_len;

            // Track finite sum of maxes; if any is unbounded, the group max is unbounded.
            sum_max_opt = sum_max_opt.and_then(|a| bounds.max_len_opt.map(|u| a + u));

            mins.push((var_char, bounds.min_len));
            if let Some(u) = bounds.max_len_opt {
                maxes.push((var_char, u));
            }
        }

        // Fail fast: if constraints are provably unsatisfiable, fail immediately
        if sum_min > jc.target {
            // sum of minimums exceeds target (impossible to satisfy)
            let bounds_str: Vec<String> = mins.iter()
                .map(|(v, min)| format!("|{v}|>={min}"))
                .collect();
            return Err(Box::new(ParseError::JointConstraintContradiction {
                constraint: format!("|{}|={}", jc.vars.iter().collect::<String>(), jc.target),
                reason: format!(
                    "sum of minimum lengths ({}) exceeds target ({}). Individual constraints: {}",
                    sum_min, jc.target, bounds_str.join(", ")
                ),
            }));
        }
        if let Some(sum_max) = sum_max_opt && sum_max < jc.target {
            // sum of maximums is less than target (impossible to satisfy)
            let bounds_str: Vec<String> = maxes.iter()
                .map(|(v, max)| format!("|{v}|<={max}"))
                .collect();
            return Err(Box::new(ParseError::JointConstraintContradiction {
                constraint: format!("|{}|={}", jc.vars.iter().collect::<String>(), jc.target),
                reason: format!(
                    "sum of maximum lengths ({}) is less than target ({}). Individual constraints: {}",
                    sum_max, jc.target, bounds_str.join(", ")
                ),
            }));
        }

        if sum_min == jc.target {
            // Case 1: exact by mins
            debug!("Joint constraint propagation: sum of mins equals target {} for vars {:?}, setting exact bounds", jc.target, jc.vars);
            for (var_char, min_len) in mins {
                vcs.ensure_entry_mut(var_char).set_exact_len(min_len);
            }
        } else if let Some(sum_max) = sum_max_opt && sum_max == jc.target {
            // Case 2: exact by finite maxes
            for (var_char, u) in maxes {
                vcs.ensure_entry_mut(var_char).set_exact_len(u);
            }
        } else {
            // Case 3: generic tightening
            for &var_char in &jc.vars {
                let bounds = vcs.bounds(var_char);

                // Σ other mins
                let sum_other_min = jc.vars
                    .iter()
                    .filter(|&&w| w != var_char)
                    .map(|&w| vcs.bounds(w).min_len)
                    .sum();

                // Σ other finite maxes (None if any is unbounded)
                let mut sum_other_max_opt = Some(0);
                for &w in jc.vars.iter().filter(|&&w| w != var_char) {
                    let w_bounds = vcs.bounds(w);
                    if w_bounds.max_len_opt.is_none() {
                        sum_other_max_opt = None;
                    }
                    sum_other_max_opt = sum_other_max_opt.and_then(|a| w_bounds.max_len_opt.map(|w| a + w));
                    if sum_other_max_opt.is_none() {
                        break;
                    }
                }

                // Calculate lower bound from joint constraint
                // Note: Use saturating_sub because during iterative propagation, intermediate
                // bounds might appear contradictory but resolve after further tightening.
                // The final contradictioncheck happens below (new_min > new_max).
                let lower_from_joint = sum_other_max_opt.map_or(VarConstraint::DEFAULT_MIN, |s| {
                    if s > jc.target {
                        debug!(
                            "Joint constraint |{:?}|={}: sum of other maxes ({}) > target (will saturate to 0)",
                            jc.vars, jc.target, s
                        );
                    }
                    jc.target.saturating_sub(s)
                });
                let upper_from_joint = jc.target.saturating_sub(sum_other_min);

                // Tighten and store
                let new_min = bounds.min_len.max(lower_from_joint);
                let new_max = bounds.max_len_opt.unwrap_or(upper_from_joint).min(upper_from_joint);

                // fail fast: check for contradictory bounds
                if new_min > new_max {
                    return Err(Box::new(ParseError::JointConstraintContradiction {
                        constraint: format!("|{}|={}", jc.vars.iter().collect::<String>(), jc.target),
                        reason: format!(
                            "variable '{var_char}' would need |{var_char}|>={new_min} and |{var_char}|<={new_max}, which is impossible. \
                             This occurs when the joint constraint forces contradictory bounds on a variable."
                        ),
                    }));
                }

                let e = vcs.ensure_entry_mut(var_char);
                e.bounds = Bounds::of(new_min, new_max);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patterns::FORM_SEPARATOR;
    use crate::constraints::VarConstraints;
    use crate::constraints::Bounds;

    #[test]
    fn propagate_exact_by_mins_all_explicit() {
        // |AB| = 5, with A.min=2, B.min=3
        let mut vcs = VarConstraints::default();
        vcs.ensure_entry_mut('A').bounds = Bounds::of_unbounded(2);
        vcs.ensure_entry_mut('B').bounds = Bounds::of_unbounded(3);

        let jc = JointConstraint { vars: vec!['A','B'], target: 5, rel: RelMask::EQ };
        let jcs = JointConstraints::of(vec![jc]);

        propagate_joint_to_var_bounds(&mut vcs, &jcs).unwrap();

        assert_eq!(vcs.bounds('A'), Bounds::of(2, 2));
        assert_eq!(vcs.bounds('B'), Bounds::of(3, 3));
    }

    #[test]
    fn propagate_exact_by_mins_with_implicit_default() {
        // |ABC| = 7, with A.min=3, B.min=None (implicit default=1), C.min=3
        let mut vcs = VarConstraints::default();
        vcs.ensure_entry_mut('A').bounds = Bounds::of_unbounded(3);
        // B left unconstrained -> min_length=None
        vcs.ensure_entry_mut('C').bounds = Bounds::of_unbounded(3);

        let jc = JointConstraint { vars: vec!['A','B','C'], target: 7, rel: RelMask::EQ };
        let jcs = JointConstraints::of(vec![jc]);

        propagate_joint_to_var_bounds(&mut vcs, &jcs).unwrap();

        // All should be exact, B should lock to default=1
        assert_eq!(vcs.bounds('A'), Bounds::of(3, 3));
        assert_eq!(vcs.bounds('B'), Bounds::of(1, 1));
        assert_eq!(vcs.bounds('C'), Bounds::of(3, 3));
    }

    #[test]
    fn propagate_no_exact_when_sum_min_lt_target() {
        // |ABC| = 8, A.min=3, B.min=1 (from lack of explicit min), C.min=3 → sum_min=7 < 8
        let mut vcs = VarConstraints::default();
        vcs.ensure_entry_mut('A').bounds = Bounds::of_unbounded(3);
        vcs.ensure_entry_mut('C').bounds = Bounds::of_unbounded(3);

        let jc = JointConstraint { vars: vec!['A','B','C'], target: 8, rel: RelMask::EQ };
        let jcs = JointConstraints::of(vec![jc]);

        propagate_joint_to_var_bounds(&mut vcs, &jcs).unwrap();

        // Nothing should be forced exact
        assert_eq!(vcs.bounds('A'), Bounds::of(3, 4));
        assert_eq!(vcs.bounds('B'), Bounds::of(VarConstraint::DEFAULT_MIN, 2)); // TODO!!! VC::D_M or just "1"?
        assert_eq!(vcs.bounds('C'), Bounds::of(3, 4));
    }

    #[test]
    fn propagate_exact_by_maxes() {
        // |AB| = 7, with A.max=4, B.max=3
        let mut vcs = VarConstraints::default();
        vcs.ensure_entry_mut('A').bounds = Bounds::of(VarConstraint::DEFAULT_MIN, 4);
        vcs.ensure_entry_mut('B').bounds = Bounds::of(VarConstraint::DEFAULT_MIN, 3);

        let jc = JointConstraint { vars: vec!['A','B'], target: 7, rel: RelMask::EQ };
        let jcs = JointConstraints::of(vec![jc]);

        propagate_joint_to_var_bounds(&mut vcs, &jcs).unwrap();

        assert_eq!(vcs.bounds('A'), Bounds::of(4, 4)); // exact=4
        assert_eq!(vcs.bounds('B'), Bounds::of(3, 3)); // exact=3
    }

    #[test]
    fn rel_mask_from_str_and_allows() {
        assert_eq!(RelMask::EQ, RelMask::from_str("=").unwrap());
        assert_eq!(RelMask::NE, RelMask::from_str("!=").unwrap());
        assert_eq!(RelMask::LE, RelMask::from_str("<=").unwrap());
        assert_eq!(RelMask::GE, RelMask::from_str(">=").unwrap());
        assert_eq!(RelMask::LT, RelMask::from_str("<").unwrap());
        assert_eq!(RelMask::GT, RelMask::from_str(">").unwrap());
        assert!(RelMask::from_str("INVALID123").is_err_and(|pe| {
            pe.to_string() == "Form parsing failed: \"Invalid comparison operator 'INVALID123' (expected: =, !=, <=, >=, <, >)\""
        }));
        assert!(!RelMask::EQ.allows(Ordering::Less));
        assert!(RelMask::EQ.allows(Ordering::Equal));
        assert!(!RelMask::EQ.allows(Ordering::Greater));
        assert!(RelMask::NE.allows(Ordering::Less));
        assert!(!RelMask::NE.allows(Ordering::Equal));
        assert!(RelMask::NE.allows(Ordering::Greater));
        assert!(RelMask::LE.allows(Ordering::Less));
        assert!(RelMask::LE.allows(Ordering::Equal));
        assert!(!RelMask::LE.allows(Ordering::Greater));
        assert!(!RelMask::GE.allows(Ordering::Less));
        assert!(RelMask::GE.allows(Ordering::Equal));
        assert!(RelMask::GE.allows(Ordering::Greater));
        assert!(RelMask::LT.allows(Ordering::Less));
        assert!(!RelMask::LT.allows(Ordering::Equal));
        assert!(!RelMask::LT.allows(Ordering::Greater));
        assert!(!RelMask::GT.allows(Ordering::Less));
        assert!(!RelMask::GT.allows(Ordering::Equal));
        assert!(RelMask::GT.allows(Ordering::Greater));
    }

    #[test]
    fn parse_joint_len_basic() {
        // Basic equality
        let jc = parse_joint_len("|AB|=7").expect("should parse");
        assert_eq!(jc.vars, vec!['A','B']);
        assert_eq!(jc.target, 7);
        assert_eq!(jc.rel, RelMask::EQ);
    }

    #[test]
    fn parse_joint_len_basic_with_spaces() {
        // Whitespace tolerated; two-char op
        let jc2 = parse_joint_len("|ABC|  <=   10").expect("should parse");
        assert_eq!(jc2.vars, vec!['A','B','C']);
        assert_eq!(jc2.target, 10);
        assert_eq!(jc2.rel, RelMask::LE);
    }

    #[test]
    fn parse_joint_len_single_var() {
        assert!(parse_joint_len("|A|=3").is_none());
    }

    #[test]
    fn parse_joint_len_lowercase() {
        assert!(parse_joint_len("|Ab|=3").is_none());
    }

    #[test]
    fn parse_joint_len_start_with_pipe() {
        assert!(parse_joint_len("foo |AB|=3").is_none());
    }

    #[test]
    fn parse_joint_len_end_with_number() {
        // Reject noise after length constraint
        assert!(parse_joint_len("|AB|=3x").is_none());
    }

    #[test]
    fn parse_joint_constraints_from_equation() {
        let sep = FORM_SEPARATOR; // could be ';' or some other separator

        // Build an equation with two constraints and a non-constraint chunk.
        let equation = format!("|AB|=3{sep}foo{sep}|BC|<=5");

        let jc_vec = JointConstraints::parse_equation(&equation).into_iter().collect::<Vec<_>>();

        assert_eq!(jc_vec.len(), 2);

        assert_eq!(jc_vec[0].vars, vec!['A', 'B']);
        assert_eq!(jc_vec[0].target, 3);
        assert_eq!(jc_vec[0].rel, RelMask::EQ);

        assert_eq!(jc_vec[1].vars, vec!['B', 'C']);
        assert_eq!(jc_vec[1].target, 5);
        assert_eq!(jc_vec[1].rel, RelMask::LE);
    }

    #[test]
    fn is_satisfied_mid_search_semantics() {
        // |AB| = 5
        let jc = JointConstraint { vars: vec!['A','B'], target: 5, rel: RelMask::EQ };

        let mut map = HashMap::from([('A', "HI".to_string())]); // len 2
        // 'B' unbound -> should return true (skip mid-search)
        assert!(jc.is_satisfied_by_map(&map));

        // Bind B (len 3) => total 5 -> satisfied
        map.insert('B', "YOU".to_string());
        assert!(jc.is_satisfied_by_map(&map));

        // Change B to length 4 => total 6 -> violated
        map.insert('B', "YOUR".to_string());
        assert!(!jc.is_satisfied_by_map(&map));
    }

    #[test]
    fn joint_constraints_all_satisfied_map_variant() {
        let jcs = JointConstraints::of(
            vec![
                JointConstraint { vars: vec!['A', 'B'], target: 6, rel: RelMask::LE }, // len(A)+len(B) <= 6
                JointConstraint { vars: vec!['B', 'C'], target: 3, rel: RelMask::GE }, // len(B)+len(C) >= 3
            ]
        );

        let mut map = HashMap::from([
            ('A', "NO".to_string()), // 2
            ('B', "YES".to_string()), // 3
            ('C', "X".to_string())] // 1
        );

        // (2+3) <= 6  AND  (3+1) >= 3  => true
        assert!(jcs.all_satisfied_map(&map));

        // Make B longer → first constraint fails
        map.insert('B', "LONGER".to_string()); // 6
        // (2+6) <= 6  is false  → overall false
        assert!(!jcs.all_satisfied_map(&map));
    }

    #[test]
    fn propagate_overlapping_equalities_ab_bc() {
        // Constraints: |AB| = 3  and  |BC| = 6
        let mut vcs = VarConstraints::default();

        let jc1 = JointConstraint { vars: vec!['A','B'], target: 3, rel: RelMask::EQ };
        let jc2 = JointConstraint { vars: vec!['B','C'], target: 6, rel: RelMask::EQ };
        let jcs = JointConstraints::of(vec![jc1, jc2]);

        // Create the variable constraints
        propagate_joint_to_var_bounds(&mut vcs, &jcs).unwrap();

        // At this stage we expect consistent tightening.
        let a_bounds = vcs.bounds('A');
        let b_bounds = vcs.bounds('B');
        let c_bounds = vcs.bounds('C');
        let c_max = c_bounds.max_len_opt;

        // A cannot exceed 2, since then B would have to be less than 1 (from A+B=3)
        assert_eq!(a_bounds.max_len_opt.unwrap(), 2);
        // A should be at least the default min
        assert_eq!(a_bounds.min_len, VarConstraint::DEFAULT_MIN);

        // B is between 1 and 2, since A+B=3 and both ≥1
        assert_eq!(b_bounds.min_len, VarConstraint::DEFAULT_MIN);
        assert_eq!(b_bounds.max_len_opt.unwrap(), 2);

        // C must be at least 4, since B≤2 and B+C=6
        assert_eq!(c_bounds.min_len, 4);
        // And at most 5, since B≥1
        assert_eq!(c_max.unwrap(), 5);
    }

    /// Test failing fast on provably unsatisfiable constraints
    #[test]
    fn test_fast_fail_on_unsatisfiable_constraints() {
        // setup: A has min_len=3, B has min_len=3, but |A+B|=5
        // (impossible--3+3=6 > 5)
        let mut vcs = VarConstraints::default();
        vcs.ensure_entry_mut('A').bounds = Bounds::of(3, 10);
        vcs.ensure_entry_mut('B').bounds = Bounds::of(3, 10);

        let jc = JointConstraint {
            vars: vec!['A', 'B'],
            rel: RelMask::EQ,
            target: 5,
        };
        let jcs = JointConstraints::of(vec![jc]);

        let result = propagate_joint_to_var_bounds(&mut vcs, &jcs);
        assert!(result.is_err(), "Should fail on unsatisfiable constraints: sum_min=6 > target=5");

        match result.unwrap_err().as_ref() {
            ParseError::JointConstraintContradiction { constraint, reason } => {
                assert_eq!(constraint, "|AB|=5");
                assert_eq!(reason, "sum of minimum lengths (6) exceeds target (5). Individual constraints: |A|>=3, |B|>=3");
            }
            other => panic!("Expected JointConstraintContradiction, got: {:?}", other),
        }
    }

    /// Test failing fast when maximum is too small
    #[test]
    fn test_fast_fail_when_max_too_small() {
        // setup: A has max=2, B has max=2, but |A+B|=6
        // (impossible--2+2=4 < 6)
        let mut vcs = VarConstraints::default();
        vcs.ensure_entry_mut('A').bounds = Bounds::of(1, 2);
        vcs.ensure_entry_mut('B').bounds = Bounds::of(1, 2);

        let jc = JointConstraint {
            vars: vec!['A', 'B'],
            rel: RelMask::EQ,
            target: 6,
        };
        let jcs = JointConstraints::of(vec![jc]);

        let result = propagate_joint_to_var_bounds(&mut vcs, &jcs);
        assert!(result.is_err(), "Should fail when sum of maxes < target");

        match result.unwrap_err().as_ref() {
            ParseError::JointConstraintContradiction { constraint, reason } => {
                assert_eq!(constraint, "|AB|=6");
                assert_eq!(reason, "sum of maximum lengths (4) is less than target (6). Individual constraints: |A|<=2, |B|<=2");
            }
            other => panic!("Expected JointConstraintContradiction, got: {:?}", other),
        }
    }

    mod edge_cases {
        use super::*;

        #[test]
        fn test_very_large_target_value() {
            let mut vcs = VarConstraints::default();
            vcs.ensure_entry_mut('A').bounds = Bounds::of_unbounded(1);
            vcs.ensure_entry_mut('B').bounds = Bounds::of_unbounded(1);

            let jc = JointConstraint {
                vars: vec!['A', 'B'],
                rel: RelMask::EQ,
                target: 1000,
            };
            let jcs = JointConstraints::of(vec![jc]);

            let result = propagate_joint_to_var_bounds(&mut vcs, &jcs);
            assert!(result.is_ok(), "large target values should be handled");
        }

        #[test]
        fn test_multiple_overlapping_constraints() {
            // |AB|=5, |BC|=6, |AC|=7
            // TODO would be nice to reduce to |A|=3,|B|=2,|C|=4 as soon as possible...
            let mut vcs = VarConstraints::default();

            let jcs = JointConstraints::of(vec![
                JointConstraint { vars: vec!['A', 'B'], rel: RelMask::EQ, target: 5 },
                JointConstraint { vars: vec!['B', 'C'], rel: RelMask::EQ, target: 6 },
                JointConstraint { vars: vec!['A', 'C'], rel: RelMask::EQ, target: 7 },
            ]);

            let result = propagate_joint_to_var_bounds(&mut vcs, &jcs);
            assert!(result.is_ok(), "overlapping constraints should be handled");
        }

        #[test]
        fn test_inequality_greater_than() {
            let mut vcs = VarConstraints::default();
            vcs.ensure_entry_mut('A').bounds = Bounds::of(1, 10);
            vcs.ensure_entry_mut('B').bounds = Bounds::of(1, 10);

            let jc = JointConstraint {
                vars: vec!['A', 'B'],
                rel: RelMask::GT,
                target: 5,
            };
            let jcs = JointConstraints::of(vec![jc]);

            let result = propagate_joint_to_var_bounds(&mut vcs, &jcs);
            assert!(result.is_ok());
        }

        #[test]
        fn test_inequality_less_than() {
            let mut vcs = VarConstraints::default();
            vcs.ensure_entry_mut('A').bounds = Bounds::of(1, 10);
            vcs.ensure_entry_mut('B').bounds = Bounds::of(1, 10);

            let jc = JointConstraint {
                vars: vec!['A', 'B'],
                rel: RelMask::LT,
                target: 10,
            };
            let jcs = JointConstraints::of(vec![jc]);

            let result = propagate_joint_to_var_bounds(&mut vcs, &jcs);
            assert!(result.is_ok());
        }

        #[test]
        fn test_constraint_with_many_variables() {
            // Constraint spanning many variables: |ABCDEFGH|=20
            let mut vcs = VarConstraints::default();
            let vars = vec!['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];

            let jc = JointConstraint {
                vars: vars.clone(),
                rel: RelMask::EQ,
                target: 20,
            };
            let jcs = JointConstraints::of(vec![jc]);

            let result = propagate_joint_to_var_bounds(&mut vcs, &jcs);
            assert!(result.is_ok(), "many-variable constraints should be handled");
        }

        #[test]
        fn test_constraint_on_single_variable() {
            let mut vcs = VarConstraints::default();

            let jc = JointConstraint {
                vars: vec!['A'],
                rel: RelMask::EQ,
                target: 5,
            };
            let jcs = JointConstraints::of(vec![jc]);

            let result = propagate_joint_to_var_bounds(&mut vcs, &jcs);
            assert!(result.is_ok());

            assert_eq!(vcs.bounds('A'), Bounds::of(5, 5));
        }

        // TODO? have propagate_joint_to_var_bounds catch contradictions like these?
        // #[test]
        // fn test_contradictory_constraints_on_same_vars() {
        //     // |AB|=5 and |AB|=7 are contradictory
        //     let mut vcs = VarConstraints::default();
        //     vcs.ensure_entry_mut('A').bounds = Bounds::of(1, 10);
        //     vcs.ensure_entry_mut('B').bounds = Bounds::of(1, 10);
        //
        //     let jcs = JointConstraints::of(vec![
        //         JointConstraint { vars: vec!['A', 'B'], rel: RelMask::EQ, target: 5 },
        //         JointConstraint { vars: vec!['A', 'B'], rel: RelMask::EQ, target: 7 },
        //     ]);
        //
        //     let result = propagate_joint_to_var_bounds(&mut vcs, &jcs);
        //     assert!(result.is_err());
        // }

        #[test]
        fn test_redundant_consistent_constraints() {
            let mut vcs = VarConstraints::default();

            let jcs = JointConstraints::of(vec![
                JointConstraint { vars: vec!['A', 'B'], rel: RelMask::EQ, target: 5 },
                JointConstraint { vars: vec!['A', 'B'], rel: RelMask::EQ, target: 5 },
            ]);

            let result = propagate_joint_to_var_bounds(&mut vcs, &jcs);
            assert!(result.is_ok(), "redundant but consistent constraints should succeed");
        }

        #[test]
        fn test_constraint_with_zero_target() {
            // |AB|=0 is impossible (variables must be nonempty)
            let mut vcs = VarConstraints::default();

            let jc = JointConstraint {
                vars: vec!['A', 'B'],
                rel: RelMask::EQ,
                target: 0,
            };
            let jcs = JointConstraints::of(vec![jc]);

            let result = propagate_joint_to_var_bounds(&mut vcs, &jcs);
            assert!(result.is_err(), "zero target with nonempty variables should fail");
        }

        #[test]
        fn test_not_equal_constraint() {
            let mut vcs = VarConstraints::default();

            let jc = JointConstraint {
                vars: vec!['A', 'B'],
                rel: RelMask::NE,
                target: 5,
            };
            let jcs = JointConstraints::of(vec![jc]);

            let result = propagate_joint_to_var_bounds(&mut vcs, &jcs);
            assert!(result.is_ok());
        }
    }
}
