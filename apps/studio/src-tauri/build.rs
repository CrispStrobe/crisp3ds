fn main() {
    // `local_engine`: the shell starts the engine as a child process. True on desktop
    // targets when the `local-engine` feature is on (the default); never on phones, and
    // not in the sandboxed Mac App Store variant, which is built without the feature.
    println!("cargo::rustc-check-cfg=cfg(local_engine)");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if std::env::var_os("CARGO_FEATURE_LOCAL_ENGINE").is_some() && os != "android" && os != "ios" {
        println!("cargo::rustc-cfg=local_engine");
    }
    tauri_build::build()
}
