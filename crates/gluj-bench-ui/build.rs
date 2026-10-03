fn main() {
    println!("cargo:rerun-if-changed=ui/icons/gluj-bench.ico");
    println!("cargo:rerun-if-changed=app.rc");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let definitions = [
            format!(
                "APP_VERSION_MAJOR={}",
                std::env::var("CARGO_PKG_VERSION_MAJOR").unwrap()
            ),
            format!(
                "APP_VERSION_MINOR={}",
                std::env::var("CARGO_PKG_VERSION_MINOR").unwrap()
            ),
            format!(
                "APP_VERSION_PATCH={}",
                std::env::var("CARGO_PKG_VERSION_PATCH").unwrap()
            ),
            format!(
                "APP_VERSION_STRING=\"{}\"",
                std::env::var("CARGO_PKG_VERSION").unwrap()
            ),
        ];
        let definitions: Vec<&str> = definitions.iter().map(String::as_str).collect();
        embed_resource::compile("app.rc", definitions.as_slice())
            .manifest_required()
            .expect("Windows application icon resource must compile");
    }
    slint_build::compile("ui/app-window.slint").expect("Slint UI must compile");
}
