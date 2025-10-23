use log::LevelFilter;

#[cfg(not(target_arch = "wasm32"))]
use env_logger;
#[cfg(target_arch = "wasm32")]
use console_log;

/// Initialize unified logging for Umiaq.
///
/// # Behavior
/// - **Native (CLI):** respects `debug_enabled` or `RUST_LOG`.
/// - **WASM:** always logs at `Debug` level (so all messages appear in the console).
pub fn init_logger(debug_enabled: bool) {
    #[cfg(target_arch = "wasm32")]
    {
        // Always log everything in the browser for better visibility.
        console_log::init_with_level(log::Level::Debug)
            .expect("failed to initialize console_log");
        log::info!("WASM logger initialized (always DEBUG level)");
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::env;
        let level = if debug_enabled {
            LevelFilter::Debug
        } else {
            LevelFilter::Info
        };

        let mut builder = env_logger::Builder::new();
        builder
            .filter(None, level)
            .format_timestamp(None)
            .format_module_path(false)
            .format_target(false);

        // Let RUST_LOG override our defaults if explicitly set
        if let Ok(spec) = env::var("RUST_LOG") {
            builder.parse_filters(&spec);
        }

        builder.init();
        log::info!("Native logger initialized at {level:?} level");
    }
}
