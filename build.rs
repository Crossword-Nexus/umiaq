use std::process::Command;

fn main() {
    // Capture git commit hash at build time
    let output_result = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output();

    let git_hash = match output_result {
        Ok(output) if output.status.success() => {
            String::from_utf8(output.stdout)
                .unwrap_or_else(|_| "unknown".to_owned())
                .trim()
                .to_owned()
        }
        _ => "unknown".to_owned(),
    };

    println!("cargo:rustc-env=GIT_HASH={git_hash}");

    // also capture the full hash for reference
    let output_full = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output();

    let git_hash_full = match output_full {
        Ok(output) if output.status.success() => {
            String::from_utf8(output.stdout)
                .unwrap_or_else(|_| "unknown".to_owned())
                .trim()
                .to_owned()
        }
        _ => "unknown".to_owned(),
    };

    println!("cargo:rustc-env=GIT_HASH_FULL={git_hash_full}");

    // capture build timestamp in ISO 8601 format (UTC)
    let build_timestamp = {
        use time::format_description::well_known::Rfc3339;
        use time::OffsetDateTime;

        OffsetDateTime::now_utc()
            .format(&Rfc3339)
            .unwrap_or_else(|_| "unknown".to_owned())
    };

    println!("cargo:rustc-env=BUILD_TIMESTAMP={build_timestamp}");

    // rerun build script if git HEAD changes
    println!("cargo:rerun-if-changed=.git/HEAD");
}
