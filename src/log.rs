#[cfg(not(target_arch = "wasm32"))]
use log::LevelFilter;
#[cfg(not(target_arch = "wasm32"))]
use env_logger;
#[cfg(target_arch = "wasm32")]
use console_log;

/// Initialize unified logging for Umiaq.
///
/// # Behavior
/// - **Native (CLI):** respects `debug_enabled` or `RUST_LOG`.
/// - **WASM:** uses `Debug` level if `debug_enabled` is true, otherwise `Info` level.
pub fn init_logger(debug_enabled: bool) {
    #[cfg(target_arch = "wasm32")]
    {
        let level = if debug_enabled {
            log::Level::Debug
        } else {
            log::Level::Info
        };

        match console_log::init_with_level(level) {
            Ok(_) => {
                log::info!("WASM logger initialized at {level:?} level");
            }
            Err(e) => {
                // If console_log fails, try to log error via web_sys and continue.
                // This provides graceful degradation rather than crashing the module.
                let msg = format!("Failed to initialize console_log: {}. Logging will be unavailable.", e);
                web_sys::console::error_1(&msg.into());
            }
        }
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
