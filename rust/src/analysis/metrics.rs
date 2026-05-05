/// Calculate PSNR (Peak Signal-to-Noise Ratio) between two RGB images.
///
/// Uses MSE-based formula: PSNR = 10 * log10(255^2 / MSE).
/// Returns f64::INFINITY if images are identical.
pub fn calculate_psnr(img1: &[u8], img2: &[u8], w: u32, h: u32) -> f64 {
    if w == 0 || h == 0 {
        return 0.0;
    }

    let n = (w * h * 3) as usize;
    if img1.len() < n || img2.len() < n {
        log::warn!(
            "Image buffers too small for PSNR: expected {}, got {} / {}",
            n,
            img1.len(),
            img2.len()
        );
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

/// Spatial metric variants used by the video-diff view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialMetricKind {
    /// Mean signed luma delta (`current - reference`) for each region.
    SignedDiff,
    /// Weighted multi-scale luma PSNR.
    MsPsnr,
    /// Multi-scale luma SSIM.
    MsSsim,
    /// Dependency-free VMAF-NEG-inspired proxy (not official libvmaf VMAF).
    VmafNegProxy,
}

impl SpatialMetricKind {
    pub fn display_name(self) -> &'static str {
        match self {
            SpatialMetricKind::SignedDiff => "Diff ΔY",
            SpatialMetricKind::MsPsnr => "MS-PSNR",
            SpatialMetricKind::MsSsim => "MS-SSIM",
            SpatialMetricKind::VmafNegProxy => "VMAF-NEG proxy",
        }
    }

    pub fn higher_is_better(self) -> bool {
        !matches!(self, SpatialMetricKind::SignedDiff)
    }
}

/// Per-region metric value. Coordinates are in source image pixels.
#[derive(Debug, Clone)]
pub struct MetricBlock {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub value: f64,
}

/// Full spatial metric result for one frame pair.
#[derive(Debug, Clone)]
pub struct SpatialMetricMap {
    pub kind: SpatialMetricKind,
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,
    pub overall: f64,
    pub blocks: Vec<MetricBlock>,
}

/// Calculate weighted multi-scale PSNR on BT.709 luma.
///
/// The first scale has the largest weight and each following scale is produced
/// by 2x2 averaging. This is intentionally dependency-free and fast enough for
/// interactive per-grid diagnostics.
pub fn calculate_ms_psnr(img1: &[u8], img2: &[u8], w: u32, h: u32) -> f64 {
    let w_usize = w as usize;
    let h_usize = h as usize;
    let n = w_usize * h_usize * 3;
    if img1.len() < n || img2.len() < n {
        log::warn!(
            "Image buffers too small for MS-PSNR: expected {}, got {} / {}",
            n,
            img1.len(),
            img2.len()
        );
        return 0.0;
    }

    let luma1 = rgb_to_luma(img1, w_usize, h_usize);
    let luma2 = rgb_to_luma(img2, w_usize, h_usize);
    calculate_ms_psnr_luma(luma1, luma2, w_usize, h_usize)
}

/// Calculate multi-scale SSIM on BT.709 luma.
///
/// Uses the standard five MS-SSIM weights when the image is large enough,
/// renormalizing the subset of weights when fewer scales are available.
pub fn calculate_ms_ssim(img1: &[u8], img2: &[u8], w: u32, h: u32) -> f64 {
    let w_usize = w as usize;
    let h_usize = h as usize;
    let n = w_usize * h_usize * 3;
    if img1.len() < n || img2.len() < n {
        log::warn!(
            "Image buffers too small for MS-SSIM: expected {}, got {} / {}",
            n,
            img1.len(),
            img2.len()
        );
        return 0.0;
    }

    let luma1 = rgb_to_luma(img1, w_usize, h_usize);
    let luma2 = rgb_to_luma(img2, w_usize, h_usize);
    calculate_ms_ssim_luma(luma1, luma2, w_usize, h_usize)
}

/// Dependency-free VMAF-NEG-inspired quality proxy, scaled to 0..100.
///
/// This is not the official Netflix VMAF model. It combines MS-SSIM with
/// MS-PSNR and applies a "no enhancement gain" edge penalty so sharpening the
/// distorted/current image above the reference is not rewarded. It exists to
/// support interactive spatial triage without linking libvmaf.
pub fn calculate_vmaf_neg_proxy(img1: &[u8], img2: &[u8], w: u32, h: u32) -> f64 {
    let w_usize = w as usize;
    let h_usize = h as usize;
    let n = w_usize * h_usize * 3;
    if img1.len() < n || img2.len() < n {
        log::warn!(
            "Image buffers too small for VMAF proxy: expected {}, got {} / {}",
            n,
            img1.len(),
            img2.len()
        );
        return 0.0;
    }

    let luma1 = rgb_to_luma(img1, w_usize, h_usize);
    let luma2 = rgb_to_luma(img2, w_usize, h_usize);
    calculate_vmaf_neg_proxy_luma(luma1, luma2, w_usize, h_usize)
}

/// Calculate one spatial metric value per grid/tile region.
///
/// `tile_size` is normally the main grid size selected in the toolbar. If it is
/// zero, a conservative 64-pixel analysis tile is used.
pub fn calculate_spatial_metric_map(
    reference_rgb: &[u8],
    current_rgb: &[u8],
    w: u32,
    h: u32,
    tile_size: u32,
    kind: SpatialMetricKind,
) -> SpatialMetricMap {
    let effective_tile = if tile_size == 0 { 64 } else { tile_size };
    let n = (w as usize) * (h as usize) * 3;
    if reference_rgb.len() < n || current_rgb.len() < n || w == 0 || h == 0 {
        return SpatialMetricMap {
            kind,
            width: w,
            height: h,
            tile_size: effective_tile,
            overall: 0.0,
            blocks: Vec::new(),
        };
    }

    let width = w as usize;
    let height = h as usize;
    let reference_luma = rgb_to_luma(reference_rgb, width, height);
    let current_luma = rgb_to_luma(current_rgb, width, height);
    let overall = match kind {
        SpatialMetricKind::SignedDiff => {
            mean_signed_luma_delta(&reference_luma, &current_luma, width, 0, 0, width, height)
        }
        SpatialMetricKind::MsPsnr => {
            calculate_ms_psnr_luma(reference_luma.clone(), current_luma.clone(), width, height)
        }
        SpatialMetricKind::MsSsim => {
            calculate_ms_ssim_luma(reference_luma.clone(), current_luma.clone(), width, height)
        }
        SpatialMetricKind::VmafNegProxy => calculate_vmaf_neg_proxy_luma(
            reference_luma.clone(),
            current_luma.clone(),
            width,
            height,
        ),
    };

    let mut blocks = Vec::new();
    let tile = effective_tile as usize;

    let mut y = 0usize;
    while y < height {
        let bh = tile.min(height - y);
        let mut x = 0usize;
        while x < width {
            let bw = tile.min(width - x);
            let value = match kind {
                SpatialMetricKind::SignedDiff => {
                    mean_signed_luma_delta(&reference_luma, &current_luma, width, x, y, bw, bh)
                }
                SpatialMetricKind::MsPsnr => {
                    let ref_crop = crop_luma(&reference_luma, width, x, y, bw, bh);
                    let cur_crop = crop_luma(&current_luma, width, x, y, bw, bh);
                    calculate_ms_psnr_luma(ref_crop, cur_crop, bw, bh)
                }
                SpatialMetricKind::MsSsim => {
                    let ref_crop = crop_luma(&reference_luma, width, x, y, bw, bh);
                    let cur_crop = crop_luma(&current_luma, width, x, y, bw, bh);
                    calculate_ms_ssim_luma(ref_crop, cur_crop, bw, bh)
                }
                SpatialMetricKind::VmafNegProxy => {
                    let ref_crop = crop_luma(&reference_luma, width, x, y, bw, bh);
                    let cur_crop = crop_luma(&current_luma, width, x, y, bw, bh);
                    calculate_vmaf_neg_proxy_luma(ref_crop, cur_crop, bw, bh)
                }
            };

            blocks.push(MetricBlock {
                x: x as u32,
                y: y as u32,
                w: bw as u32,
                h: bh as u32,
                value,
            });

            x += tile;
        }
        y += tile;
    }

    SpatialMetricMap {
        kind,
        width: w,
        height: h,
        tile_size: effective_tile,
        overall,
        blocks,
    }
}

/// Calculate SSIM (Structural Similarity Index) between two RGB images.
///
/// Implementation follows Wang et al. 2004 with 11x11 sliding window.
/// C1 = (0.01 * 255)^2, C2 = (0.03 * 255)^2.
/// Operates on BT.709 luma.
pub fn calculate_ssim(img1: &[u8], img2: &[u8], w: u32, h: u32) -> f64 {
    if w == 0 || h == 0 {
        return 0.0;
    }

    let w = w as usize;
    let h = h as usize;
    let n = w * h * 3;
    if img1.len() < n || img2.len() < n {
        log::warn!(
            "Image buffers too small for SSIM: expected {}, got {} / {}",
            n,
            img1.len(),
            img2.len()
        );
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
    assert_eq!(
        img1.len(),
        img2.len(),
        "Image buffers must be the same size"
    );

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
    assert_eq!(
        img1.len(),
        img2.len(),
        "Image buffers must be the same size"
    );

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

fn calculate_ms_psnr_luma(
    mut luma1: Vec<f64>,
    mut luma2: Vec<f64>,
    mut w: usize,
    mut h: usize,
) -> f64 {
    let weights = [0.4_f64, 0.3, 0.2, 0.1];
    let mut weighted_sum = 0.0;
    let mut used_weight = 0.0;
    let mut all_identical = true;

    for &weight in &weights {
        let psnr = psnr_from_luma(&luma1, &luma2);
        if psnr.is_finite() {
            all_identical = false;
        }
        let db = if psnr.is_infinite() { 100.0 } else { psnr };
        weighted_sum += db * weight;
        used_weight += weight;

        if w < 2 || h < 2 {
            break;
        }
        let (next1, nw, nh) = downsample_luma_2x(&luma1, w, h);
        let (next2, _, _) = downsample_luma_2x(&luma2, w, h);
        if nw == 0 || nh == 0 {
            break;
        }
        luma1 = next1;
        luma2 = next2;
        w = nw;
        h = nh;
    }

    if all_identical {
        f64::INFINITY
    } else if used_weight > 0.0 {
        weighted_sum / used_weight
    } else {
        0.0
    }
}

fn calculate_ms_ssim_luma(
    mut luma1: Vec<f64>,
    mut luma2: Vec<f64>,
    mut w: usize,
    mut h: usize,
) -> f64 {
    let weights = [0.0448_f64, 0.2856, 0.3001, 0.2363, 0.1333];
    let mut components: Vec<(f64, f64)> = Vec::new(); // (ssim, contrast-structure)

    for _ in 0..weights.len() {
        components.push(compute_ssim_components(&luma1, &luma2, w, h));
        if w < 2 || h < 2 {
            break;
        }
        let (next1, nw, nh) = downsample_luma_2x(&luma1, w, h);
        let (next2, _, _) = downsample_luma_2x(&luma2, w, h);
        if nw == 0 || nh == 0 {
            break;
        }
        luma1 = next1;
        luma2 = next2;
        w = nw;
        h = nh;
    }

    if components.is_empty() {
        return 0.0;
    }

    let used_weight_sum: f64 = weights.iter().take(components.len()).sum();
    let last = components.len() - 1;
    let mut score = 1.0_f64;
    for (i, &(ssim, cs)) in components.iter().enumerate() {
        let weight = weights[i] / used_weight_sum;
        let component = if i == last { ssim } else { cs };
        score *= component.max(0.0).powf(weight);
    }
    score.clamp(0.0, 1.0)
}

fn calculate_vmaf_neg_proxy_luma(luma1: Vec<f64>, luma2: Vec<f64>, w: usize, h: usize) -> f64 {
    let ms_ssim = calculate_ms_ssim_luma(luma1.clone(), luma2.clone(), w, h);
    let ms_psnr = calculate_ms_psnr_luma(luma1.clone(), luma2.clone(), w, h);
    let psnr_norm = if ms_psnr.is_infinite() {
        1.0
    } else {
        ((ms_psnr - 20.0) / 30.0).clamp(0.0, 1.0)
    };

    let ref_edge = luma_edge_energy(&luma1, w, h);
    let cur_edge = luma_edge_energy(&luma2, w, h);
    let edge_excess = ((cur_edge - ref_edge) / ref_edge.max(1.0)).max(0.0);

    (100.0 * (0.72 * ms_ssim + 0.28 * psnr_norm) - 20.0 * edge_excess).clamp(0.0, 100.0)
}

fn psnr_from_luma(luma1: &[f64], luma2: &[f64]) -> f64 {
    if luma1.is_empty() || luma1.len() != luma2.len() {
        return 0.0;
    }
    let mse = luma1
        .iter()
        .zip(luma2.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f64>()
        / luma1.len() as f64;

    if mse <= f64::EPSILON {
        f64::INFINITY
    } else {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    }
}

fn downsample_luma_2x(src: &[f64], w: usize, h: usize) -> (Vec<f64>, usize, usize) {
    let nw = w / 2;
    let nh = h / 2;
    if nw == 0 || nh == 0 {
        return (Vec::new(), 0, 0);
    }

    let mut out = vec![0.0_f64; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let sx = x * 2;
            let sy = y * 2;
            let a = src[sy * w + sx];
            let b = src[sy * w + sx + 1];
            let c = src[(sy + 1) * w + sx];
            let d = src[(sy + 1) * w + sx + 1];
            out[y * nw + x] = (a + b + c + d) * 0.25;
        }
    }
    (out, nw, nh)
}

fn compute_ssim_components(luma1: &[f64], luma2: &[f64], w: usize, h: usize) -> (f64, f64) {
    let c1: f64 = (0.01 * 255.0) * (0.01 * 255.0);
    let c2: f64 = (0.03 * 255.0) * (0.03 * 255.0);

    if luma1.len() != luma2.len() || luma1.len() != w * h || luma1.is_empty() {
        return (0.0, 0.0);
    }

    // Small regions (typical grid tiles) use global statistics. This keeps
    // interactive per-block SSIM stable and fast while preserving the 11x11
    // window behavior for larger frame-level maps.
    if w < 11 || h < 11 || w * h <= 4096 {
        return compute_ssim_global_components(luma1, luma2, c1, c2);
    }

    let i1 = integral_image(luma1, w, h);
    let i2 = integral_image(luma2, w, h);
    let luma1_sq: Vec<f64> = luma1.iter().map(|v| v * v).collect();
    let luma2_sq: Vec<f64> = luma2.iter().map(|v| v * v).collect();
    let luma12: Vec<f64> = luma1.iter().zip(luma2.iter()).map(|(a, b)| a * b).collect();
    let i1_sq = integral_image(&luma1_sq, w, h);
    let i2_sq = integral_image(&luma2_sq, w, h);
    let i12 = integral_image(&luma12, w, h);

    let window_size = 11usize;
    let half_win = window_size / 2;
    let win_count = (window_size * window_size) as f64;
    let mut ssim_sum = 0.0;
    let mut cs_sum = 0.0;
    let mut count = 0u64;

    for y in half_win..(h - half_win) {
        for x in half_win..(w - half_win) {
            let rx = x - half_win;
            let ry = y - half_win;
            let sum1 = sum_integral(&i1, w, rx, ry, window_size, window_size);
            let sum2 = sum_integral(&i2, w, rx, ry, window_size, window_size);
            let sum1_sq = sum_integral(&i1_sq, w, rx, ry, window_size, window_size);
            let sum2_sq = sum_integral(&i2_sq, w, rx, ry, window_size, window_size);
            let sum12 = sum_integral(&i12, w, rx, ry, window_size, window_size);

            let mu1 = sum1 / win_count;
            let mu2 = sum2 / win_count;
            let sigma1_sq = sum1_sq / win_count - mu1 * mu1;
            let sigma2_sq = sum2_sq / win_count - mu2 * mu2;
            let sigma12 = sum12 / win_count - mu1 * mu2;

            let cs = (2.0 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2);
            let ssim = ((2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2))
                / ((mu1 * mu1 + mu2 * mu2 + c1) * (sigma1_sq + sigma2_sq + c2));

            ssim_sum += ssim;
            cs_sum += cs;
            count += 1;
        }
    }

    if count == 0 {
        compute_ssim_global_components(luma1, luma2, c1, c2)
    } else {
        (
            (ssim_sum / count as f64).clamp(-1.0, 1.0),
            (cs_sum / count as f64).clamp(-1.0, 1.0),
        )
    }
}

fn compute_ssim_global_components(luma1: &[f64], luma2: &[f64], c1: f64, c2: f64) -> (f64, f64) {
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

    let cs = (2.0 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2);
    let ssim = ((2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2))
        / ((mu1 * mu1 + mu2 * mu2 + c1) * (sigma1_sq + sigma2_sq + c2));
    (ssim.clamp(-1.0, 1.0), cs.clamp(-1.0, 1.0))
}

fn integral_image(values: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row_sum = 0.0;
        for x in 0..w {
            row_sum += values[y * w + x];
            let idx = (y + 1) * (w + 1) + (x + 1);
            out[idx] = out[y * (w + 1) + (x + 1)] + row_sum;
        }
    }
    out
}

fn sum_integral(integral: &[f64], w: usize, x: usize, y: usize, bw: usize, bh: usize) -> f64 {
    let stride = w + 1;
    let x2 = x + bw;
    let y2 = y + bh;
    integral[y2 * stride + x2] - integral[y * stride + x2] - integral[y2 * stride + x]
        + integral[y * stride + x]
}

fn luma_edge_energy(luma: &[f64], w: usize, h: usize) -> f64 {
    if w < 2 || h < 2 || luma.len() < w * h {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut count = 0usize;
    for y in 0..(h - 1) {
        for x in 0..(w - 1) {
            let idx = y * w + x;
            let gx = (luma[idx + 1] - luma[idx]).abs();
            let gy = (luma[idx + w] - luma[idx]).abs();
            sum += gx + gy;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn mean_signed_luma_delta(
    reference_luma: &[f64],
    current_luma: &[f64],
    full_w: usize,
    x: usize,
    y: usize,
    bw: usize,
    bh: usize,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for row in y..(y + bh) {
        for col in x..(x + bw) {
            let idx = row * full_w + col;
            if idx >= reference_luma.len() || idx >= current_luma.len() {
                continue;
            }
            sum += current_luma[idx] - reference_luma[idx];
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn crop_luma(src: &[f64], full_w: usize, x: usize, y: usize, bw: usize, bh: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; bw * bh];
    for row in 0..bh {
        let src_start = (y + row) * full_w + x;
        let src_end = src_start + bw;
        let dst_start = row * bw;
        let dst_end = dst_start + bw;
        if src_end <= src.len() {
            out[dst_start..dst_end].copy_from_slice(&src[src_start..src_end]);
        }
    }
    out
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
