use std::io::{BufRead, Write};

/// Detect scene changes by comparing consecutive RGB frames.
///
/// Returns frame indices where a scene change is detected (i.e., where the
/// mean absolute difference to the previous frame exceeds `threshold`).
pub fn detect_scene_changes(
    frames: &[Vec<u8>],
    _width: u32,
    _height: u32,
    threshold: f64,
) -> Vec<usize> {
    let mut changes = Vec::new();
    for i in 1..frames.len() {
        let diff = mean_abs_diff(&frames[i - 1], &frames[i]);
        if diff > threshold {
            changes.push(i);
        }
    }
    changes
}

/// Mean absolute difference between two RGB buffers.
fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let sum: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    sum as f64 / a.len() as f64
}

/// Save scene change indices to a text file (one index per line).
pub fn save_scene_list(path: &str, indices: &[usize]) -> Result<(), String> {
    let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    for idx in indices {
        writeln!(file, "{}", idx).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Load scene change indices from a text file.
pub fn load_scene_list(path: &str) -> Result<Vec<usize>, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let reader = std::io::BufReader::new(file);
    let mut indices = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            let idx: usize = trimmed.parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
            indices.push(idx);
        }
    }
    Ok(indices)
}
