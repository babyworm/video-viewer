use clap::Parser;

#[derive(Parser)]
#[command(name = "video_viewer", about = "YUV/Raw Video Viewer")]
struct Cli {
    /// Input file path
    input: Option<String>,

    #[arg(short = 'W', long)]
    width: Option<u32>,

    #[arg(short = 'H', long)]
    height: Option<u32>,

    #[arg(short, long)]
    format: Option<String>,

    #[arg(long = "vi")]
    input_format: Option<String>,

    #[arg(long = "vo")]
    output_format: Option<String>,

    #[arg(short, long)]
    output: Option<String>,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    if let Some(ref output_path) = cli.output {
        // Headless conversion mode
        let input_path = cli.input.as_deref().unwrap_or_else(|| {
            eprintln!("Error: input file required for conversion");
            std::process::exit(1);
        });
        let width = cli.width.unwrap_or_else(|| {
            eprintln!("Error: --width required for conversion");
            std::process::exit(1);
        });
        let height = cli.height.unwrap_or_else(|| {
            eprintln!("Error: --height required for conversion");
            std::process::exit(1);
        });
        let input_fmt = cli.input_format.as_deref().unwrap_or_else(|| {
            eprintln!("Error: --vi (input format) required for conversion");
            std::process::exit(1);
        });
        let output_fmt = cli.output_format.as_deref().unwrap_or_else(|| {
            eprintln!("Error: --vo (output format) required for conversion");
            std::process::exit(1);
        });

        let converter = video_viewer::conversion::converter::VideoConverter::new();
        let progress = |current: usize, total: usize| -> bool {
            eprint!("\rConverting frame {}/{}...", current + 1, total);
            true
        };
        match converter.convert(input_path, (width, height), input_fmt, output_path, output_fmt, Some(&progress)) {
            Ok((n, cancelled)) => {
                eprintln!();
                if cancelled {
                    eprintln!("Conversion cancelled after {} frames", n);
                    std::process::exit(1);
                }
                println!("Converted {} frames", n);
            }
            Err(e) => {
                eprintln!();
                eprintln!("Conversion error: {}", e);
                std::process::exit(1);
            }
        }
        return;
    }

    video_viewer::run_gui(cli.input, cli.width, cli.height, cli.format);
}
