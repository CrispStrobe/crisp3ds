fn main() {
    // `native_engine`: the reconstruction (crates/dense) is linked into the app.
    // `local_engine`: the app can also start the Python engine as a child process. Desktop
    // targets only, and never in the sandboxed Mac App Store variant.
    println!("cargo::rustc-check-cfg=cfg(native_engine)");
    println!("cargo::rustc-check-cfg=cfg(local_engine)");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if std::env::var_os("CARGO_FEATURE_NATIVE_ENGINE").is_some() {
        println!("cargo::rustc-cfg=native_engine");
    }
    if std::env::var_os("CARGO_FEATURE_LOCAL_ENGINE").is_some() && os != "android" && os != "ios" {
        println!("cargo::rustc-cfg=local_engine");
    }
    println!("cargo::rerun-if-changed=settings-schema.json");
    tauri_build::build()
}
