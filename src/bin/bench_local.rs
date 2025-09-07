//! `bench_local.rs` — quick local timing runner (no Criterion)
//!
//! PURPOSE
//! -------
//! - Fast, ad-hoc timing for a handful of patterns on *your* machine.
//! - Loads the word list once, then runs each pattern several times and reports the median.
//! - Always requests 100 results per pattern (by design, to keep comparisons simple).
//! - Optionally shows a *static* time per pattern (from Qat), and the delta %.
//!
//! HOW TO RUN
//! ----------
//! - Optimized build:                `cargo run --bin bench_local --release`
//! - Multiple repeats:               `cargo run --bin bench_local --release -- -r 5`
//! - Print a few solutions:          `cargo run --bin bench_local --release -- -p 5`
//! - See all flags:                  `cargo run --bin bench_local -- --help`
//!
//! NOTES
//! -----
//! - This is *not* Criterion. It's quick and convenient, not statistically rigorous.
//! - Use the same machine and `--release` for more comparable numbers.
//! - Patterns + optional Qat times live in `cases()` below.
//! - I/O (printing) is kept outside the timed section.
//! - One warm-up run per pattern is done (not included in timing).
//! - We report the *median* over repeats (more robust than mean for small _N_).

use clap::Parser;
use std::hint::black_box;
use std::time::Instant;

use umiaq::bindings::Bindings;
use umiaq::solver;
use umiaq::wordlist;

/// Simple local benchmark runner: load word list once, time several patterns.
/// Each case is a pattern + optional Qat time; name = pattern; always requests 100 results.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to the word list file (word;score per line)
    #[arg(
        short,
        long,
        default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/data/spreadthewordlist.dict")
    )]
    wordlist: String,

    /// Minimum score filter
    #[arg(short = 'm', long, default_value_t = 50)]
    min_score: i32,

    /// Maximum allowed word length
    #[arg(short = 'L', long, default_value_t = 21)]
    max_len: usize,

    /// Number of repeats per pattern (use >1 to reduce noise; median is reported)
    #[arg(short = 'r', long = "repeats", default_value_t = 1)]
    num_repeats: usize,

    /// Print up to this many solutions per pattern (0 = print none)
    #[arg(short = 'p', long = "print", default_value_t = 0)]
    print_limit: usize,
}

/// The fixed number of results we request per pattern.
/// Keeping this constant across cases makes local comparisons simpler.
const NUM_RESULTS: usize = 100;

/// A benchmark case: the Umiaq pattern and an optional Qat time (seconds).
/// Set `qat_s` by running the query in Qat against the Broda list.
#[derive(Clone)]
struct Case {
    pattern: &'static str,
    qat_s: Option<f64>,
}

/// Edit/add new patterns here. The summary will display the pattern text as the "name".
/// The `qat_s` values below are results from Qat
fn get_cases() -> Vec<Case> {
    vec![
        Case { pattern: "AB;BA;|A|=2;|B|=2;!=AB", qat_s: Some(0.260) },
        // Note: this one is slow (for now?)
        Case { pattern: "Atime;Btime;Ctime;ABC", qat_s: Some(6.240) },
        Case { pattern: "AB;CD;AC;BD", qat_s: Some(0.300) },
        Case { pattern: "A*BCD;AC*BD;DCB*A", qat_s: Some(4.100) },
        Case { pattern: "AkB;AlB", qat_s: Some(2.360) },
        Case { pattern: "l*x", qat_s: Some(0.420) },
        Case {
            pattern: "ABCDEFGHIJKLMN;|ABCDEFGHIJKLMN|=14;!=ABCDEFGHIJKLMN",
            qat_s: Some(0.340)
        },
        Case { pattern: "A@B;A#B;!=AB;B=(g.*)", qat_s: Some(24.420) },
    ]
}

/// Small helper: robust central tendency for small samples.
fn median(mut xs: Vec<f64>) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

const MAX_PATTERN_LEN: usize = 48;

fn main() -> std::io::Result<()> {
    /// One row in the benchmark summary: (pattern name, elapsed seconds,
    /// number of results, optional reference time, optional other time).
    type SummaryRow = (String, f64, usize, Option<f64>, Option<f64>);

    let cli = Cli::parse();

    // Load the word list once. This I/O is *not* included in per-pattern timing.
    eprintln!("Loading word list from: {}", cli.wordlist);
    let t_load = Instant::now();
    let wl = wordlist::WordList::load_from_path(&cli.wordlist, cli.min_score)?;
    let load_secs = t_load.elapsed().as_secs_f64();
    eprintln!("Loaded {} words in {:.3}s", wl.entries.len(), load_secs);

    // Keep references to avoid reallocating strings during benchmarks.
    let words_ref: Vec<&str> = wl.entries.iter().map(String::as_str).collect();

    let cases = get_cases();
    // Store (pattern, median_seconds, solutions_last_run, qat_s, delta_pct_opt) for the summary.
    let mut summary: Vec<SummaryRow> = Vec::with_capacity(cases.len());

    for (idx, case) in cases.iter().enumerate() {
        let pattern = case.pattern;
        eprintln!("\n[{:02}] {}", idx + 1, pattern);

        // One *warm-up* execution per pattern to "touch" code paths / caches.
        // We intentionally ignore its timing.
        let _warmup = solver::solve_equation(pattern, &words_ref, NUM_RESULTS)
            .expect("solver warm-up failed");

        // Repeat the timed runs and collect durations.
        let mut times = Vec::with_capacity(cli.num_repeats);
        let mut last_solutions: Vec<Vec<Bindings>> = Vec::new();

        for rep in 0..cli.num_repeats {
            // Keep only the *core* operation inside the timed region.
            let t_solve = Instant::now();
            let solutions = solver::solve_equation(black_box(pattern), &words_ref, NUM_RESULTS)
                .expect("solver failed");
            let solve_secs = t_solve.elapsed().as_secs_f64();

            // Prevent the compiler from proving the result unused and eliding work.
            let _keep = black_box(solutions.len());

            times.push(solve_secs);
            last_solutions = solutions;

            eprintln!(
                "  run {:>2}/{:>2}: {:.3}s ({} solutions)",
                rep + 1,
                cli.num_repeats,
                solve_secs,
                last_solutions.len()
            );
        }

        // Prefer median for small N--it's less sensitive to noisy outliers.
        let med = median(times);

        // Optionally print a few solutions from the *last* run (outside timing).
        if cli.print_limit > 0 && !last_solutions.is_empty() {
            for sol in last_solutions.iter().take(cli.print_limit) {
                let display = solution_to_string(sol);
                println!("{display}");
            }
        }

        // Compute delta % vs. Qat value, if provided.
        let delta_pct = case.qat_s.and_then(|exp| {
            if exp > 0.0 {
                Some((med - exp) / exp * 100.0)
            } else {
                None
            }
        });

        eprintln!(
            "  → median {:.3}s over {} run(s); last run produced {} {}.{}",
            med,
            cli.num_repeats,
            last_solutions.len(),
            pluralizer(last_solutions.len(), "solution".into(), None),
            match (case.qat_s, delta_pct) {
                (Some(exp), Some(dp)) => format!(" (Qat {exp:.3}s, Δ = {dp:+.1}%)"),
                _ => String::new(),
            }
        );

        summary.push((
            pattern.to_string(),
            med,
            last_solutions.len(),
            case.qat_s,
            delta_pct,
        ));
    }

    // Compact summary at the end for a quick scan across all patterns.
    eprintln!("\n==== Summary ====");
    eprintln!(
        "{:<MAX_PATTERN_LEN$} | {:>10} | {:>11} | {:>10} | {:>8}",
        "pattern", "median (s)", "# solutions", "qat (s)", "Δ %"
    );
    eprintln!(
        "{:-<MAX_PATTERN_LEN$}-+-{:-<10}-+-{:-<11}-+-{:-<10}-+-{:-<8}",
        "", "", "", "", ""
    );
    for (pat, med, num_solutions, qat, delta_pct) in &summary {
        // Trim very long patterns for readability in the summary.
        let display = if pat.len() > MAX_PATTERN_LEN {
            // "- 1" for the "…"
            format!("{}…", pat.chars().take(MAX_PATTERN_LEN - 1).collect::<String>())
        } else {
            pat.clone()
        };
        let qat_str = qat.map(|x| format!("{x:.1}")).unwrap_or_else(|| "—".into());
        let dp_str = delta_pct
            .map(|x| format!("{x:+.1}"))
            .unwrap_or_else(|| "—".into());
        eprintln!(
            "{display:<MAX_PATTERN_LEN$} | {med:>10.3} | {num_solutions:>11} | {qat_str:>10} | {dp_str:>8}"
        );
    }

    Ok(())
}

// TODO DRY w/main.rs solution_to_string
// Put the results in uppercase and separated with a bullet
fn solution_to_string(solution: &[Bindings]) -> String {
    solution.iter()
        .map(|b| b.get_word().unwrap().to_ascii_uppercase())
        .collect::<Vec<_>>()
        .join(" • ")
}

// TODO? put this elsewhere
fn pluralizer(count: usize, singular: String, plural: Option<String>) -> String {
    if count == 1 {
        singular
    } else {
        plural.unwrap_or(singular + "s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pluralizer() {
        assert_eq!(pluralizer(0, "diameter".into(), None), "diameters");
        assert_eq!(pluralizer(1, "diameter".into(), None), "diameter");
        assert_eq!(pluralizer(2, "diameter".into(), None), "diameters");
        assert_eq!(pluralizer(99, "diameter".into(), None), "diameters");
        assert_eq!(pluralizer(0, "radius".into(), Some("radii".into())), "radii");
        assert_eq!(pluralizer(1, "radius".into(), Some("radii".into())), "radius");
        assert_eq!(pluralizer(2, "radius".into(), Some("radii".into())), "radii");
        assert_eq!(pluralizer(99, "radius".into(), Some("radii".into())), "radii");
    }
}
