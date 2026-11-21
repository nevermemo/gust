mod app;
mod bins;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    app::main()
}
