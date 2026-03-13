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

    if cli.output.is_some() {
        eprintln!("Headless conversion not yet implemented");
        std::process::exit(1);
    }

    video_viewer::run_gui(cli.input, cli.width, cli.height, cli.format);
}
