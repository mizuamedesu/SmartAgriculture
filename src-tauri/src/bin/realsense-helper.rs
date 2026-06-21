fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(error) = smart_agriculture_tomato_twin_lib::run_realsense_helper(&args) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
