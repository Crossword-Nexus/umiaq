// src/scan_hints.rs
// -----------------------------------------------------------------------------
// Fast, form-local length hints for prefiltering during the scan phase.
//
// This module computes (min_len, max_len?) for a single ParsedForm, taking into account:
//   • fixed tokens in the form (i.e., literals, '.', '@', '#', [charset], /anagram)
//   • frequencies of variables that appear in the form (Var, RevVar)
//   • unary per-variable bounds from VarConstraints (normalized: min≥1, max=∞ if unset)
//   • joint group constraints from JointConstraints that refer ONLY to vars
//     present in THIS form (e.g., "|AB|=6").
//
// The result can be used as a cheap prefilter: if a candidate word's length does
// not satisfy these bounds, you can skip calling the heavy matcher altogether.
//
// Design notes:
// - We intentionally do *not* attempt global propagation across multiple forms.
//   These hints are computed per-form, once per equation.
// - We do not try to detect infeasibility of the full constraint set; if min>max
//   emerges it simply means "no candidates" for that form.
// - "!=" (not-equal) group relations are ignored for tightening because they do
//   not produce a contiguous interval--we conservatively skip tightening on them.
// - Presence of '*' in the form makes the form's max length unbounded for the
//   hint's purposes (even if unary-var maxima are finite), because '*' can soak
//   an arbitrary number of extra characters.
// -----------------------------------------------------------------------------

use std::cmp::max;
use std::collections::{HashMap, HashSet};
use crate::constraints::{VarConstraint, VarConstraints};
use crate::joint_constraints::{JointConstraint, JointConstraints, RelMask};
use crate::parser::{FormPart, ParsedForm};

/// Resulting per-form hints.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PatternLenHints {
    /// Lower bound on the form's length.
    pub min_len: usize,
    /// Upper bound on the form's length. `None` = unbounded above.
    pub max_len: Option<usize>,
}

/// Small enum for `weighted_extreme_for_t`
#[derive(Clone, Copy, Debug)]
enum Extreme { Min, Max }

impl PatternLenHints {
    /// Quick check for a candidate word length against this hint.
    pub(crate) fn is_word_len_possible(&self, len: usize) -> bool {
        len >= self.min_len && self.max_len.is_none_or(|max_len| len <= max_len)
    }
}

/// A joint constraint over variables *restricted to this form* considered
/// as a contiguous total length bound for their raw lengths (not weighted).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupLenConstraint {
    pub vars: Vec<char>,          // e.g., ['A','B'] for |AB|
    pub total_min: usize,         // inclusive
    pub total_max: Option<usize>, // inclusive, None => unbounded
}

/// Convert a crate-level `JointConstraint` to a `GroupLenConstraint` interval.
/// Returns `None` for incompatible/non-interval relations (e.g., NE) or empty.
fn group_from_joint(jc: &JointConstraint) -> Option<GroupLenConstraint> {
    // Map RelMask to [min,max] on the target.
    // Note: For LT/GT we avoid underflow/overflow.
    let (tmin, tmax_opt) = match jc.rel {
        RelMask::EQ => (jc.target, Some(jc.target)),
        RelMask::LE => (0, Some(jc.target)),
        RelMask::LT => (0, Some(max(0, jc.target - 1))),
        RelMask::GE => (jc.target, None),
        RelMask::GT => (jc.target.saturating_add(1), None),
        _ => return None // NE (or unusual mask combos) don't give a single interval — skip tightening.
    };

    // Basic sanity: empty interval ⇒ None
    if let Some(tmax) = tmax_opt && tmin > tmax {
        None
    } else {
        Some(GroupLenConstraint {
            vars: jc.vars.clone(),
            total_min: tmin,
            total_max: tmax_opt,
        })
    }
}

/// Compute per-form hints from a `ParsedForm` *and* the equation's constraints.
/// These are just length bounds for a parsed form
///
/// - `vcs`: the full equation's `VarConstraints` (we'll only read vars present in form)
/// - `jcs`: the equation's `JointConstraints` (we'll filter to constraints whose
///   variable set is a subset of the form's variables)
pub(crate) fn form_len_hints_pf(
    parts: &ParsedForm,
    vcs: &VarConstraints,
    jcs: &JointConstraints,
) -> PatternLenHints {
    #[derive(Clone, Copy, Debug)]
    struct Bounds {
        li: usize,
        ui: Option<usize>
    }

    // 1. Scan tokens: accumulate fixed_base, detect '*', and count var frequencies
    let mut fixed_base = 0;
    let mut has_star = false;
    let mut var_frequency = HashMap::new();

    for p in parts {
        match p {
            FormPart::Star => has_star = true,
            FormPart::Dot | FormPart::Vowel | FormPart::Consonant | FormPart::Charset(_) => fixed_base += 1,
            FormPart::Lit(s) => fixed_base += s.len(),
            FormPart::Anagram(ag) => fixed_base += ag.len,
            FormPart::Var(v) | FormPart::RevVar(v) => *var_frequency.entry(*v).or_insert(0) += VarConstraint::DEFAULT_MIN,
        }
    }

    // Exact case (exit early): no variables and no star ⇒ exact fixed length
    if var_frequency.is_empty() && !has_star {
        return PatternLenHints {
            min_len: fixed_base,
            max_len: Some(fixed_base),
        };
    }

    // 2. Pull unary bounds just for vars in this form
    let mut vars: Vec<char> = var_frequency.keys().copied().collect();
    vars.sort_unstable();

    let bounds_map = &vars
        .iter()
        .map(|&v| {
            let (li, ui) = vcs.bounds(v);
            (v, Bounds { li, ui })
        })
        .collect::<HashMap<char, Bounds>>();

    let get_weight = |v: char| *var_frequency.get(&v).unwrap_or(&0);

    // Baseline min/max ignoring groups
    let mut weighted_min = {
        let sum = vars
            .iter()
            .map(|&v| get_weight(v) * bounds_map[&v].li)
            .sum::<usize>();
        fixed_base + sum
    };

    // small helper function for maxes
    let sum_weighted_ui_opt = |vars: &[char]| -> Option<usize> {
        if has_star {
            return None;
        }
        vars.iter().copied().try_fold(0usize, |acc, v| {
            bounds_map[&v].ui.map(|u| acc + get_weight(v) * u)
        })
    };

    let sum_opt = sum_weighted_ui_opt(&vars);
    let mut weighted_max = sum_opt.map(|s| fixed_base + s);

    // 3. Tighten with group constraints valid for this form
    for g in &group_constraints_for_form(parts, jcs) {
        #[derive(Clone, Copy)]
        struct Row {
            w: usize,
            li: usize,
            ui: Option<usize>
        }

        /// Compute the weighted extremal sum at fixed total `t` over the rows,
        /// where each row contributes `w * len_i`, and `len_i ∈ [li, ui]`.
        /// If `minimize` is true, distribute remaining length to cheaper weights first;
        /// otherwise to most expensive first.
        ///
        /// `sum_li` is Σ li; `sum_ui_opt` is Σ ui if all ui are finite, else None (∞).
        fn weighted_extreme_for_t(
            rows: &[Row],
            sum_li: usize,
            sum_ui_opt: Option<usize>, // Some(sum_ui) if ALL ui are finite; None if ANY ui is "unbounded"
            t: usize,
            extreme: Extreme,
        ) -> Option<usize> {
            // Feasibility checks
            if t < sum_li {
                return None;
            }
            if let Some(su) = sum_ui_opt && t > su {
                return None;
            }

            // Base cost at lower bounds
            let base_weighted = rows.iter().map(|r| r.w.saturating_mul(r.li)).sum::<usize>();
            let mut rem = t - sum_li;
            if rem == 0 {
                return Some(base_weighted);
            }

            // Greedy: assign remaining letters to cheapest (Min) or priciest (Max) first.
            // We still honor each row's individual capacity (ui - li). A row is "unbounded"
            // iff r.ui == VarConstraint::DEFAULT_MAX.
            let mut order: Vec<&Row> = rows.iter().collect();
            match extreme {
                Extreme::Min => order.sort_unstable_by_key(|r| r.w),              // cheapest first
                Extreme::Max => order.sort_unstable_by_key(|r| std::cmp::Reverse(r.w)),     // priciest first
            }

            let mut extra = 0usize;
            for r in order {
                // Per-row capacity above li
                let cap = if let Some(u) = r.ui {
                    u.saturating_sub(r.li).min(rem)
                } else {
                    rem
                };

                if cap > 0 {
                    extra = extra.saturating_add(r.w.saturating_mul(cap));
                    rem -= cap;
                    // Invariant: `rem` never goes negative because `cap <= rem` at each step.
                    // A debug_assert at the end checks that rem reaches 0 if t was feasible.
                    if rem == 0 {
                        break;
                    }
                }
            }

            // If we reach here with rem != 0, `t` wasn't feasible to begin with.
            // TODO: throw an error?
            if rem != 0 {
                return None;
            }

            debug_assert_eq!(rem, 0);
            Some(base_weighted.saturating_add(extra))
        }

        if g.vars.is_empty() {
            continue;
        }

        // Intersect with the form's variables (only consider vars that appear in this form)
        let mut gvars: Vec<char> = g
            .vars
            .iter()
            .copied()
            .filter(|v| var_frequency.contains_key(v))
            .collect();
        gvars.sort_unstable();
        if gvars.is_empty() {
            continue;
        }

        // Build rows and Σ li / Σ ui (finite-only)
        let mut rows: Vec<Row> = Vec::with_capacity(gvars.len());
        let mut sum_li: usize = 0;
        let mut sum_ui_opt: Option<usize> = Some(0);
        for &v in &gvars {
            let b = bounds_map[&v];
            rows.push(Row {
                w: get_weight(v),
                li: b.li,
                ui: b.ui
            });
            sum_li += b.li;
            sum_ui_opt = if b.ui.is_none() {
                None
            } else {
                sum_ui_opt.and_then(|a| b.ui.map(|u| a + u))
            };
        }

        let weighted_min_for_t =
            |t: usize| weighted_extreme_for_t(&rows, sum_li, sum_ui_opt, t, Extreme::Min);
        let weighted_max_for_t =
            |t: usize| weighted_extreme_for_t(&rows, sum_li, sum_ui_opt, t, Extreme::Max);

        // ---- Account for group vars that are NOT in this form ------------------
        // They eat into the group's total before we allocate to in-form vars.
        // outside_form_min = Σ (min of vars outside the form)
        // outside_form_max_opt = Σ (finite max of vars outside the form), None if any is ∞
        let (outside_form_min, outside_form_max_opt) = g
            .vars
            .iter()
            .filter(|v| !var_frequency.contains_key(v))
            .fold((0usize, Some(0usize)), |(min_acc, max_acc_opt), &v| {
                let (li, ui) = vcs.bounds(v);
                let min_acc = min_acc + li;
                let max_acc_opt = ui.and_then(|u| max_acc_opt.map(|a| a + u));
                (min_acc, max_acc_opt)
            });

        // Effective totals for the in-form part of this group:
        // - For the LOWER bound, outside takes as much as possible (use outside_form_max if finite).
        // - For the UPPER bound, outside takes as little as possible (use outside_form_min).
        // re 0: if outside can be arbitrarily large, in-form lower could be 0
        let tmin_eff = outside_form_max_opt.map_or(0, |of_max| g.total_min.saturating_sub(of_max));
        let tmax_eff_opt = g
            .total_max
            .map(|tmax| tmax.saturating_sub(outside_form_min));

        // Evaluate endpoints of the adjusted interval for in-form vars.
        let gmin_w = weighted_min_for_t(tmin_eff);
        let gmax_w = tmax_eff_opt.and_then(weighted_max_for_t);

        // Combine with outside-of-group contributions (vars in this form but not in this group)
        let outside: Vec<char> = vars.iter().copied().filter(|v| !gvars.contains(v)).collect();

        let outside_min = outside
            .iter()
            .map(|&v| get_weight(v) * bounds_map[&v].li)
            .sum::<usize>();

        let outside_max_opt = sum_weighted_ui_opt(&outside);

        if let Some(gmin) = gmin_w {
            weighted_min = weighted_min.max(fixed_base + gmin + outside_min);
        }

        // Candidate upper bound from this group + outside
        let candidate_upper = match (gmax_w, outside_max_opt) {
            (Some(gm), Some(om)) => Some(fixed_base + gm + om),
            _ => None,
        };

        // Combine with the running upper bound:
        weighted_max = match (weighted_max, candidate_upper) {
            (Some(cur), Some(cand)) => Some(cur.min(cand)),
            (None, Some(cand)) => Some(cand),
            (Some(cur), None) => Some(cur),
            (None, None) => None,
        };
    }

    PatternLenHints {
        min_len: weighted_min,
        max_len: weighted_max,
    }
}

/// Build the list of group constraints (as contiguous intervals) that are *scoped
/// to this form*: every referenced variable must appear in the form.
fn group_constraints_for_form(form: &ParsedForm, jcs: &JointConstraints) -> Vec<GroupLenConstraint> {
    if jcs.is_empty() {
        vec![]
    } else {
        let present: HashSet<char> = form.iter().filter_map(|p| match p {
            FormPart::Var(v) | FormPart::RevVar(v) => Some(*v),
            _ => None,
        }).collect();

        jcs.clone().into_iter()
            // ← revert to ANY overlap so constraints like |AB|=6 still inform A-only forms
            .filter(|jc| jc.vars.iter().any(|v| present.contains(v)))
            .filter_map(|jc: JointConstraint| group_from_joint(&jc))
            .collect()
    }
}


// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::FormPart;

    // Minimal helper ParsedForm for tests without pulling regex machinery.
    fn pf(parts: Vec<FormPart>) -> ParsedForm {
        ParsedForm {
            parts,
            prefilter: fancy_regex::Regex::new("^.*$").unwrap(),
        }
    }

    #[test]
    fn no_vars_no_star_exact() {
        let form = pf(vec![
            FormPart::Lit("AB".into()),
            FormPart::Dot,
            FormPart::anagram_of("xy").unwrap(),
        ]);
        let vcs = VarConstraints::default();
        let hints = form_len_hints_pf(&form, &vcs, &JointConstraints::default());

        let expected = PatternLenHints {
            min_len: 5,
            max_len: Some(5),
        };
        assert_eq!(expected, hints);
        assert!(hints.is_word_len_possible(5));
        assert!(!hints.is_word_len_possible(4));
    }

    #[test]
    fn star_unbounded_max() {
        let form = pf(vec![FormPart::Lit("HEL".into()), FormPart::Star, FormPart::Var('A')]);
        let mut vcs = VarConstraints::default();

        // A in [2,4]
        let a = vcs.ensure('A');
        a.min_length = 2;
        a.max_length = Some(4);

        let hints = form_len_hints_pf(&form, &vcs, &JointConstraints::default());

        let expected = PatternLenHints {
            min_len: 5,
            max_len: None,
        };
        assert_eq!(expected, hints);
    }

    #[test]
    fn unary_bounds_only() {
        // A . B ; base=1
        let form = pf(vec![FormPart::Var('A'), FormPart::Dot, FormPart::Var('B')]);
        let mut vcs = VarConstraints::default();

        let a = vcs.ensure('A');
        a.min_length = 2;
        a.max_length = Some(3);

        let b = vcs.ensure('B');
        b.min_length = 1;
        b.max_length = Some(5);

        let hints = form_len_hints_pf(&form, &vcs, &JointConstraints::default());

        let expected = PatternLenHints {
            min_len: 4,
            max_len: Some(9),
        };
        assert_eq!(expected, hints);
    }

    #[test]
    fn group_eq_on_ab_with_weights() {
        // Form: A B A   weights wA=2, wB=1
        let form = pf(vec![FormPart::Var('A'), FormPart::Var('B'), FormPart::Var('A')]);
        let vcs = VarConstraints::default();

        // Build a JointConstraints equivalent to |AB|=6
        let jc = JointConstraint {
            vars: vec!['A', 'B'],
            target: 6,
            rel: RelMask::EQ,
        };
        let jcs = JointConstraints::of(vec![jc]);

        let hints = form_len_hints_pf(&form, &vcs, &jcs);

        // Explanation:
        // - base = 0
        // - wA=2, wB=1
        // At fixed |AB|=6, minimizing weighted length puts as much as possible on cheaper B,
        // maximizing puts as much as possible on A.
        let expected = PatternLenHints {
            min_len: 7,
            max_len: Some(11),
        };
        assert_eq!(expected, hints);
    }

    #[test]
    fn group_eq_plus_unary_forces_exact() {
        // . A B . ; base=2 ; |AB|=6 ; |A| fixed to 2
        let form = pf(vec![
            FormPart::Dot,
            FormPart::Var('A'),
            FormPart::Var('B'),
            FormPart::Dot,
        ]);
        let mut vcs = VarConstraints::default();

        let a = vcs.ensure('A');
        a.min_length = 2;
        a.max_length = Some(2);

        let jc = JointConstraint {
            vars: vec!['A', 'B'],
            target: 6,
            rel: RelMask::EQ,
        };
        let jcs = JointConstraints::of(vec![jc]);
        let hints = form_len_hints_pf(&form, &vcs, &jcs);

        let expected = PatternLenHints {
            min_len: 8,
            max_len: Some(8),
        };
        assert_eq!(expected, hints);
    }

    #[test]
    fn ranged_group_bounds() {
        // A B with unary: A in [1,5], B in [1,10]; group |AB| in [4,6]
        let form = pf(vec![FormPart::Var('A'), FormPart::Var('B')]);
        let mut vcs = VarConstraints::default();

        let a = vcs.ensure('A');
        a.min_length = 1;
        a.max_length = Some(5);

        let b = vcs.ensure('B');
        b.min_length = 1;
        b.max_length = Some(10);

        let g1 = JointConstraint {
            vars: vec!['A', 'B'],
            target: 4,
            rel: RelMask::GE,
        };
        let g2 = JointConstraint {
            vars: vec!['A', 'B'],
            target: 6,
            rel: RelMask::LE,
        };
        let jcs = JointConstraints::of(vec![g1, g2]);
        let hints = form_len_hints_pf(&form, &vcs, &jcs);

        let expected = PatternLenHints {
            min_len: 4,
            max_len: Some(6),
        };
        assert_eq!(expected, hints);
    }

    #[test]
    fn star_blocks_exact_even_with_exact_groups() {
        // A*B ; |AB|=6
        let form = pf(vec![FormPart::Var('A'), FormPart::Star, FormPart::Var('B')]);
        let jc = JointConstraint {
            vars: vec!['A', 'B'],
            target: 6,
            rel: RelMask::EQ,
        };
        let jcs = JointConstraints::of(vec![jc]);
        let vcs = VarConstraints::default();
        let hints = form_len_hints_pf(&form, &vcs, &jcs);

        // star contributes 0 to min_len
        let expected = PatternLenHints {
            min_len: 6,
            max_len: None,
        };
        assert_eq!(expected, hints);
    }

    #[test]
    fn group_hints_apply_to_single_var() {
        let form = pf(vec![FormPart::Var('A')]);
        let jc = JointConstraint {
            vars: vec!['A', 'B'],
            target: 6,
            rel: RelMask::EQ,
        };
        let jcs = JointConstraints::of(vec![jc]);
        let vcs = VarConstraints::default();
        let hints = form_len_hints_pf(&form, &vcs, &jcs);

        // With |AB|=6 and only A present in this form:
        // - outside_form_min = min(B)
        // - outside_form_max may be ∞ (default), so tmin_eff becomes 0.
        // Here the normalized defaults are A∈[1,∞), B∈[1,∞) → we get
        // min_len = 1 (from A's min), and an effective upper bound of 5 (6 - min(B)).
        let expected = PatternLenHints {
            min_len: VarConstraint::DEFAULT_MIN,
            max_len: Some(5),
        };
        assert_eq!(expected, hints);
    }

    #[test]
    fn ranged_group_bounds_unary_pushes_min() {
        // A in [3,5], B in [2,10]; |AB| in [4,6]
        // sum_li = 3+2 = 5, so the group’s GE(4) is weaker than unary mins.
        // Max is still capped by LE(6).
        let form = pf(vec![FormPart::Var('A'), FormPart::Var('B')]);
        let mut vcs = VarConstraints::default();

        let a = vcs.ensure('A');
        a.min_length = 3;
        a.max_length = Some(5);

        let b = vcs.ensure('B');
        b.min_length = 2;
        b.max_length = Some(10);

        let g1 = JointConstraint { vars: vec!['A','B'], target: 4, rel: RelMask::GE };
        let g2 = JointConstraint { vars: vec!['A','B'], target: 6, rel: RelMask::LE };
        let jcs = JointConstraints::of(vec![g1, g2]);

        let hints = form_len_hints_pf(&form, &vcs, &jcs);

        // Without the group we’d have [5, 15]; the group tightens max to 6, min stays 5.
        let expected = PatternLenHints { min_len: 5, max_len: Some(6) };
        assert_eq!(expected, hints);
    }

    #[test]
    fn ranged_group_bounds_with_outside_var() {
        // Form has only A,B. Group is |ABC| in [10, 12].
        // C is outside the form with C in [3,4], so:
        // - tmin_eff = 10 - max(C) = 10 - 4 = 6
        // - tmax_eff = 12 - min(C) = 12 - 3 = 9
        // A in [2,5], B in [1,7].
        let form = pf(vec![FormPart::Var('A'), FormPart::Var('B')]);
        let mut vcs = VarConstraints::default();

        let a = vcs.ensure('A');
        a.min_length = 2;
        a.max_length = Some(5);

        let b = vcs.ensure('B');
        b.min_length = 1;
        b.max_length = Some(7);

        let c = vcs.ensure('C');
        c.min_length = 3;
        c.max_length = Some(4);

        let ge = JointConstraint { vars: vec!['A','B','C'], target: 10, rel: RelMask::GE };
        let le = JointConstraint { vars: vec!['A','B','C'], target: 12, rel: RelMask::LE };
        let jcs = JointConstraints::of(vec![ge, le]);

        let hints = form_len_hints_pf(&form, &vcs, &jcs);

        // Base (no group): min = 2+1 = 3, max = 5+7 = 12.
        // With group & outside C: A+B must be in [6,9].
        // So final form length is [6,9].
        let expected = PatternLenHints { min_len: 6, max_len: Some(9) };
        assert_eq!(expected, hints);
    }


}
