use std::fmt;
use std::io::{BufRead, Write};

/// Algorithm to use for scene change detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneAlgorithm {
    Mad,
    Histogram,
    Ssim,
}

impl fmt::Display for SceneAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SceneAlgorithm::Mad => write!(f, "MAD"),
            SceneAlgorithm::Histogram => write!(f, "Histogram"),
            SceneAlgorithm::Ssim => write!(f, "SSIM"),
        }
    }
}

impl SceneAlgorithm {
    /// Default detection threshold appropriate for this algorithm.
    pub fn default_threshold(&self) -> f64 {
        match self {
            SceneAlgorithm::Mad => 45.0,
            SceneAlgorithm::Histogram => 0.3,
            SceneAlgorithm::Ssim => 40.0,
        }
    }
}

/// Detect scene changes by comparing consecutive RGB frames using the MAD algorithm.
///
/// Returns frame indices where a scene change is detected (i.e., where the
/// mean absolute difference to the previous frame exceeds `threshold`).
pub fn detect_scene_changes(
    frames: &[Vec<u8>],
    width: u32,
    height: u32,
    threshold: f64,
) -> Vec<usize> {
    detect_scene_changes_with_algorithm(frames, width, height, threshold, SceneAlgorithm::Mad)
}

/// Detect scene changes using the specified algorithm.
///
/// Returns frame indices where a scene change is detected.
pub fn detect_scene_changes_with_algorithm(
    frames: &[Vec<u8>],
    width: u32,
    height: u32,
    threshold: f64,
    algorithm: SceneAlgorithm,
) -> Vec<usize> {
    let mut changes = Vec::new();
    for i in 1..frames.len() {
        let diff = match algorithm {
            SceneAlgorithm::Mad => mean_abs_diff(&frames[i - 1], &frames[i]),
            SceneAlgorithm::Histogram => histogram_diff(&frames[i - 1], &frames[i]),
            SceneAlgorithm::Ssim => ssim_diff(&frames[i - 1], &frames[i], width, height),
        };
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

/// Luma histogram difference using Pearson correlation.
///
/// Computes BT.709 luma histograms for both frames and returns
/// `1.0 - pearson_correlation`. Result range is 0..2; default threshold ~0.3.
pub fn histogram_diff(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() || a.len() < 3 {
        return 0.0;
    }

    let mut hist_a = [0u64; 256];
    let mut hist_b = [0u64; 256];

    for chunk in a.chunks_exact(3) {
        let luma = 0.2126 * chunk[0] as f64 + 0.7152 * chunk[1] as f64 + 0.0722 * chunk[2] as f64;
        hist_a[luma.round().clamp(0.0, 255.0) as usize] += 1;
    }
    for chunk in b.chunks_exact(3) {
        let luma = 0.2126 * chunk[0] as f64 + 0.7152 * chunk[1] as f64 + 0.0722 * chunk[2] as f64;
        hist_b[luma.round().clamp(0.0, 255.0) as usize] += 1;
    }

    let n = 256.0_f64;
    let sum_a: f64 = hist_a.iter().map(|&v| v as f64).sum();
    let sum_b: f64 = hist_b.iter().map(|&v| v as f64).sum();
    let mean_a = sum_a / n;
    let mean_b = sum_b / n;

    let mut cov = 0.0_f64;
    let mut var_a = 0.0_f64;
    let mut var_b = 0.0_f64;
    for i in 0..256 {
        let da = hist_a[i] as f64 - mean_a;
        let db = hist_b[i] as f64 - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-10 {
        // Identical (or degenerate) histograms — no scene change
        return 0.0;
    }

    let pearson = cov / denom;
    1.0 - pearson
}

/// SSIM-based scene difference using 8×8 blocks and BT.709 luma.
///
/// Returns `(1.0 - ssim) * 100.0` so higher values mean more different.
/// Default threshold ~40.0.
pub fn ssim_diff(a: &[u8], b: &[u8], width: u32, height: u32) -> f64 {
    let w = width as usize;
    let h = height as usize;

    if a.len() != b.len() || a.len() < 3 || w == 0 || h == 0 {
        return 0.0;
    }

    // Extract BT.709 luma plane
    let luma_a: Vec<f64> = a
        .chunks_exact(3)
        .map(|c| 0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64)
        .collect();
    let luma_b: Vec<f64> = b
        .chunks_exact(3)
        .map(|c| 0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64)
        .collect();

    if luma_a.len() != w * h || luma_b.len() != w * h {
        return 0.0;
    }

    // SSIM constants for 8-bit range (L=255)
    const C1: f64 = (0.01 * 255.0) * (0.01 * 255.0);
    const C2: f64 = (0.03 * 255.0) * (0.03 * 255.0);

    const BLOCK: usize = 8;
    let mut ssim_sum = 0.0_f64;
    let mut block_count = 0u64;

    let blocks_y = h / BLOCK;
    let blocks_x = w / BLOCK;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut sum_a = 0.0_f64;
            let mut sum_b = 0.0_f64;
            let n = (BLOCK * BLOCK) as f64;

            for dy in 0..BLOCK {
                for dx in 0..BLOCK {
                    let idx = (by * BLOCK + dy) * w + (bx * BLOCK + dx);
                    sum_a += luma_a[idx];
                    sum_b += luma_b[idx];
                }
            }

            let mu_a = sum_a / n;
            let mu_b = sum_b / n;

            let mut var_a = 0.0_f64;
            let mut var_b = 0.0_f64;
            let mut cov = 0.0_f64;

            for dy in 0..BLOCK {
                for dx in 0..BLOCK {
                    let idx = (by * BLOCK + dy) * w + (bx * BLOCK + dx);
                    let da = luma_a[idx] - mu_a;
                    let db = luma_b[idx] - mu_b;
                    var_a += da * da;
                    var_b += db * db;
                    cov += da * db;
                }
            }

            var_a /= n;
            var_b /= n;
            cov /= n;

            let numerator = (2.0 * mu_a * mu_b + C1) * (2.0 * cov + C2);
            let denominator = (mu_a * mu_a + mu_b * mu_b + C1) * (var_a + var_b + C2);

            if denominator > 1e-10 {
                ssim_sum += numerator / denominator;
                block_count += 1;
            }
        }
    }

    if block_count == 0 {
        return 0.0;
    }

    let ssim = ssim_sum / block_count as f64;
    (1.0 - ssim) * 100.0
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
