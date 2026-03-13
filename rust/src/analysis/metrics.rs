/// Calculate PSNR (Peak Signal-to-Noise Ratio) between two RGB images.
///
/// Uses MSE-based formula: PSNR = 10 * log10(255^2 / MSE).
/// Returns f64::INFINITY if images are identical.
pub fn calculate_psnr(img1: &[u8], img2: &[u8], w: u32, h: u32) -> f64 {
    let n = (w * h * 3) as usize;
    if img1.len() < n || img2.len() < n {
        log::warn!("Image buffers too small for PSNR: expected {}, got {} / {}", n, img1.len(), img2.len());
        return 0.0;
    }

    let mut mse_sum: f64 = 0.0;
    for i in 0..n {
        let diff = img1[i] as f64 - img2[i] as f64;
        mse_sum += diff * diff;
    }

    let mse = mse_sum / n as f64;

    if mse == 0.0 {
        return f64::INFINITY;
    }

    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

/// Calculate SSIM (Structural Similarity Index) between two RGB images.
///
/// Implementation follows Wang et al. 2004 with 11x11 sliding window.
/// C1 = (0.01 * 255)^2, C2 = (0.03 * 255)^2.
/// Operates on BT.709 luma.
pub fn calculate_ssim(img1: &[u8], img2: &[u8], w: u32, h: u32) -> f64 {
    let w = w as usize;
    let h = h as usize;
    let n = w * h * 3;
    if img1.len() < n || img2.len() < n {
        log::warn!("Image buffers too small for SSIM: expected {}, got {} / {}", n, img1.len(), img2.len());
        return 0.0;
    }

    // Convert to luma using BT.709
    let luma1 = rgb_to_luma(img1, w, h);
    let luma2 = rgb_to_luma(img2, w, h);

    let c1: f64 = (0.01 * 255.0) * (0.01 * 255.0); // 6.5025
    let c2: f64 = (0.03 * 255.0) * (0.03 * 255.0); // 58.5225

    let window_size = 11usize;
    let half_win = window_size / 2;

    if w < window_size || h < window_size {
        // Image too small for windowed SSIM, compute global SSIM
        return compute_ssim_global(&luma1, &luma2, c1, c2);
    }

    let mut ssim_sum = 0.0;
    let mut count = 0u64;

    for y in half_win..(h - half_win) {
        for x in half_win..(w - half_win) {
            let mut sum1 = 0.0_f64;
            let mut sum2 = 0.0_f64;
            let mut sum1_sq = 0.0_f64;
            let mut sum2_sq = 0.0_f64;
            let mut sum12 = 0.0_f64;
            let mut win_count = 0.0_f64;

            for wy in 0..window_size {
                for wx in 0..window_size {
                    let py = y + wy - half_win;
                    let px = x + wx - half_win;
                    let idx = py * w + px;

                    let v1 = luma1[idx];
                    let v2 = luma2[idx];

                    sum1 += v1;
                    sum2 += v2;
                    sum1_sq += v1 * v1;
                    sum2_sq += v2 * v2;
                    sum12 += v1 * v2;
                    win_count += 1.0;
                }
            }

            let mu1 = sum1 / win_count;
            let mu2 = sum2 / win_count;
            let sigma1_sq = sum1_sq / win_count - mu1 * mu1;
            let sigma2_sq = sum2_sq / win_count - mu2 * mu2;
            let sigma12 = sum12 / win_count - mu1 * mu2;

            let numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2);
            let denominator = (mu1 * mu1 + mu2 * mu2 + c1) * (sigma1_sq + sigma2_sq + c2);

            ssim_sum += numerator / denominator;
            count += 1;
        }
    }

    ssim_sum / count as f64
}

/// Calculate mean absolute difference between two images.
pub fn calculate_frame_difference(img1: &[u8], img2: &[u8]) -> f64 {
    assert_eq!(img1.len(), img2.len(), "Image buffers must be the same size");

    let n = img1.len();
    if n == 0 {
        return 0.0;
    }

    let mut total_diff: f64 = 0.0;
    for i in 0..n {
        total_diff += (img1[i] as f64 - img2[i] as f64).abs();
    }

    total_diff / n as f64
}

/// Calculate histogram correlation between two images.
///
/// Uses Pearson correlation coefficient on the luma histograms.
/// Returns value in range [-1.0, 1.0] where 1.0 means identical histograms.
pub fn calculate_histogram_difference(img1: &[u8], img2: &[u8]) -> f64 {
    assert_eq!(img1.len(), img2.len(), "Image buffers must be the same size");

    let n = img1.len() / 3;
    let hist1 = luma_histogram(img1, n);
    let hist2 = luma_histogram(img2, n);

    // Pearson correlation
    let mean1: f64 = hist1.iter().sum::<f64>() / 256.0;
    let mean2: f64 = hist2.iter().sum::<f64>() / 256.0;

    let mut cov = 0.0_f64;
    let mut var1 = 0.0_f64;
    let mut var2 = 0.0_f64;

    for i in 0..256 {
        let d1 = hist1[i] - mean1;
        let d2 = hist2[i] - mean2;
        cov += d1 * d2;
        var1 += d1 * d1;
        var2 += d2 * d2;
    }

    let denom = (var1 * var2).sqrt();
    if denom == 0.0 {
        return 1.0; // identical flat histograms
    }

    cov / denom
}

// --- Helper functions ---

fn rgb_to_luma(rgb: &[u8], w: usize, h: usize) -> Vec<f64> {
    let mut luma = Vec::with_capacity(w * h);
    for i in 0..(w * h) {
        let base = i * 3;
        let r = rgb[base] as f64;
        let g = rgb[base + 1] as f64;
        let b = rgb[base + 2] as f64;
        luma.push(0.2126 * r + 0.7152 * g + 0.0722 * b);
    }
    luma
}

fn compute_ssim_global(luma1: &[f64], luma2: &[f64], c1: f64, c2: f64) -> f64 {
    let n = luma1.len() as f64;
    let mu1: f64 = luma1.iter().sum::<f64>() / n;
    let mu2: f64 = luma2.iter().sum::<f64>() / n;

    let sigma1_sq: f64 = luma1.iter().map(|v| (v - mu1) * (v - mu1)).sum::<f64>() / n;
    let sigma2_sq: f64 = luma2.iter().map(|v| (v - mu2) * (v - mu2)).sum::<f64>() / n;
    let sigma12: f64 = luma1
        .iter()
        .zip(luma2.iter())
        .map(|(a, b)| (a - mu1) * (b - mu2))
        .sum::<f64>()
        / n;

    let numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2);
    let denominator = (mu1 * mu1 + mu2 * mu2 + c1) * (sigma1_sq + sigma2_sq + c2);

    numerator / denominator
}

fn luma_histogram(rgb: &[u8], pixel_count: usize) -> Vec<f64> {
    let mut hist = vec![0.0_f64; 256];
    for i in 0..pixel_count {
        let base = i * 3;
        let r = rgb[base] as f64;
        let g = rgb[base + 1] as f64;
        let b = rgb[base + 2] as f64;
        let y = (0.2126 * r + 0.7152 * g + 0.0722 * b).round() as u8;
        hist[y as usize] += 1.0;
    }
    hist
}
