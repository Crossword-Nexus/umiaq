use std::process::ExitCode;
use clap::Parser;
use std::time::Instant;

use umiaq::solver;
use umiaq::solver::SolveStatus;
use umiaq::entry_list;

/// Umiaq equation solver
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// The equation to solve (e.g., "AB;BA;|A|=2;|B|=2;!=AB")
    equation: String,

    /// Path to the entry list file (entry;score per line)
    #[arg(
        short,
        long,
        default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/data/spreadthewordlist.dict")
    )]
    entry_list: String,

    /// Minimum score filter
    #[arg(short = 'm', long, default_value_t = 50)]
    min_score: i32,

    /// Maximum number of results to return
    #[arg(short = 'n', long, default_value_t = 100)]
    num_results_requested: usize,
}

/// Entry point of the Umiaq CLI solver.
///
/// Delegates to [`try_main`], catching any errors and printing them
/// in a user-friendly way before exiting with code 1.
fn main() -> ExitCode {

    // Set up logging
    let debug_enabled = std::env::var("UMIAQ_DEBUG").is_ok();
    umiaq::log::init_logger(debug_enabled);

    log::info!("Starting Umiaq solver");

    if let Err(e) = try_main() {
        // Print the error message to stderr, with detailed formatting if it's a SolverError
        if let Some(solver_err) = e.downcast_ref::<solver::SolverError>() {
            eprintln!("Error: {}", solver_err.display_detailed());
        } else {
            eprintln!("Error: {e}");
        }
        // Exit explicitly with a nonzero code so scripts can detect failure
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// Core application logic for the Umiaq CLI solver.
///
/// Steps:
/// 1. Parse CLI arguments with Clap.
/// 2. Load the entry list from disk, applying the minimum score filter.
/// 3. Solve the given pattern against the entry list.
/// 4. Print each solution on stdout.
/// 5. Print performance metrics (timings, counts) on stderr.
///
/// Returns `Ok(())` on success or an error (e.g., invalid pattern,
/// failed regex parse, missing entry-list file) which bubbles up to [`main`].
fn try_main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command-line arguments
    let cli = Cli::parse();

    // 1. Load the entry list from disk, filtering out low-score entries
    let t_load = Instant::now();
    let entry_list = entry_list::EntryList::load_from_path(&cli.entry_list, cli.min_score)?;
    let load_secs = t_load.elapsed().as_secs_f64();

    // Build a Vec<&str> of entry references for the solver
    let entries_ref: Vec<_> = entry_list.entries.iter().map(String::as_str).collect();

    // 2. Solve the equation against the entry list
    let t_solve = Instant::now();
    let solve_result = solver::solve_equation(&cli.equation, &entries_ref, cli.num_results_requested)?;
    let solve_secs = t_solve.elapsed().as_secs_f64();

    // 3. Print each solution on stdout
    for solution in &solve_result.solutions {
        println!("{}", solver::solution_to_string(solution)?);
    }

    match solve_result.status {
        SolveStatus::TimedOut { elapsed } => {
            eprintln!("⚠️  Timed out after {:.1}s; some solutions may not have been returned", elapsed.as_secs_f64());
        }
        SolveStatus::FoundEnough => {
            eprintln!("✓ Stopped after finding {}/{} requested solutions", solve_result.solutions.len(), cli.num_results_requested);
        }
        SolveStatus::EntryListExhausted => {
            eprintln!("✓ Entry list exhausted (no more solutions)");
        }
    }

    // 4. Print diagnostics (entry-list size, timings, number of results) to stderr
    eprintln!(
        "Loaded {} entries in {:.3}s; solved in {:.3}s ({} tuples).",
        entry_list.entries.len(),
        load_secs,
        solve_secs,
        solve_result.solutions.len()
    );

    Ok(())
}

