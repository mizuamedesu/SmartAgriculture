fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--rebuild-assets") {
        let Some(session_root) = args.get(2) else {
            eprintln!("missing session root");
            std::process::exit(2);
        };
        match smart_agriculture_tomato_twin_lib::rebuild_scan_assets_cli(session_root) {
            Ok(result) => println!("{result}"),
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
        return;
    }
    if args.get(1).map(String::as_str) == Some("--export-mcap-samples") {
        let (Some(recording_path), Some(output_root)) = (args.get(2), args.get(3)) else {
            eprintln!("missing recording path or output folder");
            std::process::exit(2);
        };
        match smart_agriculture_tomato_twin_lib::export_mcap_samples_cli(
            recording_path,
            output_root,
        ) {
            Ok(result) => println!("{result}"),
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
        return;
    }
    if args.get(1).map(String::as_str) == Some("--realsense-helper") {
        if let Err(error) = smart_agriculture_tomato_twin_lib::run_realsense_helper(&args[2..]) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }

    smart_agriculture_tomato_twin_lib::run();
}
