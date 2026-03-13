use video_viewer::core::reader::VideoReader;

#[test]
fn test_open_real_y4m() {
    let path = "/home/babyworm/sources/derf-y4m/FourPeople_1280x720_60.y4m";
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping: test file not found");
        return;
    }

    let mut reader = VideoReader::open(path, 0, 0, "", "BT.601").unwrap();
    eprintln!("Reader: {}x{} format={} frames={}",
        reader.width(), reader.height(), reader.format_name(), reader.total_frames());

    assert_eq!(reader.width(), 1280);
    assert_eq!(reader.height(), 720);
    assert!(reader.total_frames() > 0);

    // First frame
    let raw = reader.seek_frame(0).unwrap();
    let expected_raw = reader.width() as usize * reader.height() as usize * 3 / 2; // I420
    eprintln!("Frame 0: {} bytes (expected ~{})", raw.len(), expected_raw);

    let rgb = reader.convert_to_rgb(&raw).unwrap();
    let expected_rgb = reader.width() as usize * reader.height() as usize * 3;
    assert_eq!(rgb.len(), expected_rgb);

    // Verify not all zeros
    let nonzero = rgb.iter().filter(|&&b| b != 0).count();
    eprintln!("Non-zero bytes: {}/{}", nonzero, rgb.len());
    assert!(nonzero > rgb.len() / 2, "Image seems mostly black");

    // Channels
    let channels = reader.get_channels(&raw);
    eprintln!("Channels: {:?}", channels.keys().collect::<Vec<_>>());
    assert!(channels.contains_key("Y"));
    assert!(channels.contains_key("U"));
    assert!(channels.contains_key("V"));

    // Last frame
    let last = reader.total_frames() - 1;
    let raw_last = reader.seek_frame(last).unwrap();
    let rgb_last = reader.convert_to_rgb(&raw_last).unwrap();
    assert_eq!(rgb_last.len(), expected_rgb);
    eprintln!("Last frame ({}) OK", last);
}
