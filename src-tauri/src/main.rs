fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--realsense-helper") {
        if let Err(error) = smart_agriculture_tomato_twin_lib::run_realsense_helper(&args[2..]) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }

    smart_agriculture_tomato_twin_lib::run();
}
