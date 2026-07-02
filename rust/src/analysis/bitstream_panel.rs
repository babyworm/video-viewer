//! Sidebar mini panel for bitstream telemetry (`.catb`) — UX spec §9.
//!
//! Stateless `show(&mut ui, &mut ctx) -> Option<BitstreamAction>` following
//! the `isp_sideband.rs` panel convention. The panel offers load/unload, a
//! codec/frames/resolution summary, the R4 frame-offset spinner, an
//! open/focus-window button, the current frame readout, and the fixed §7
//! error strings (shown here in red so failures are visible even when the
//! analysis window is closed).

use eframe::egui;

use crate::analysis::bitstream_stats::{viewer_to_catb_display, BitstreamFile};

/// Actions the panel requests from the app (§9).
pub enum BitstreamAction {
    LoadRequested,
    Unload,
    OpenWindow,
    SetOffset(i32),
    /// M4: toggle the main-canvas overlay (mirrors the window's ViewConfig).
    SetOverlayOnCanvas(bool),
    /// M5: kill the in-flight external decoder run.
    CancelDecode,
}

/// Read-only + mutable context handed to the panel each frame.
pub struct BitstreamPanelContext<'a> {
    pub bitstream: Option<&'a BitstreamFile>,
    pub bitstream_path: Option<&'a str>,
    /// Last load error (§7 fixed wording comes from the loader).
    pub error: Option<&'a str>,
    /// R4 manual frame offset (shared with the window's status strip).
    pub offset: i32,
    pub current_frame_idx: usize,
    /// Viewer total frames (0 when no video loaded).
    pub viewer_total_frames: usize,
    /// Viewer resolution, if a video is loaded.
    pub viewer_size: Option<(u32, u32)>,
    /// Whether the analysis window is currently open ("Focus Window" label).
    pub window_open: bool,
    /// M4: main-canvas overlay toggle state (app-owned, session only).
    pub overlay_on_canvas: bool,
    /// M5: seconds since the external decoder run started
    /// (None = no run in flight).
    pub decoding_elapsed: Option<f64>,
}

pub struct BitstreamPanel;

impl Default for BitstreamPanel {
    fn default() -> Self {
        Self
    }
}

impl BitstreamPanel {
    pub fn new() -> Self {
        Self
    }

    /// Render the panel; returns an action when the user requested one.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &mut BitstreamPanelContext,
    ) -> Option<BitstreamAction> {
        let mut action = None;

        ui.heading("Bitstream (.catb)");
        ui.separator();

        // M5: external decoder-run progress — shown regardless of whether a
        // telemetry file is already loaded (the run may replace it).
        if let Some(secs) = ctx.decoding_elapsed {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(format!("Decoding… {}s", secs.max(0.0) as u64));
                if ui.button("Cancel").clicked() {
                    action = Some(BitstreamAction::CancelDecode);
                }
            });
            ui.separator();
        }

        let Some(bs) = ctx.bitstream else {
            if ui.button("Load .catb…").clicked() {
                action = Some(BitstreamAction::LoadRequested);
            }
            ui.label("No telemetry loaded");
            if let Some(err) = ctx.error {
                ui.colored_label(egui::Color32::RED, err);
            }
            return action;
        };

        // File name + Unload.
        ui.horizontal(|ui| {
            let name = ctx
                .bitstream_path
                .map(|p| p.rsplit('/').next().unwrap_or(p))
                .unwrap_or("(unnamed)");
            ui.label(name)
                .on_hover_text(ctx.bitstream_path.unwrap_or(""));
            if ui.button("Unload").clicked() {
                action = Some(BitstreamAction::Unload);
            }
        });

        // codec · frames · resolution summary.
        let codec = if bs.catb.meta.codec.is_empty() {
            "?".to_string()
        } else {
            bs.catb.meta.codec.to_uppercase()
        };
        ui.label(format!(
            "codec {} · {} frames · {}×{}",
            codec,
            bs.frame_count(),
            bs.width,
            bs.height,
        ));

        // M-A §D: capture level + decoder contract badge (empty → "n/a").
        let level = &bs.catb.meta.capture_level;
        let level = if level.is_empty() { "n/a" } else { level.as_str() };
        let contract = &bs.catb.meta.contract;
        let contract = if contract.is_empty() {
            "n/a"
        } else {
            contract.as_str()
        };
        ui.weak(format!("level: {level} · contract: {contract}"))
            .on_hover_text(
                "Telemetry capture level recorded by the decoder \
                 (block/syntax/cabac/full; n/a = not recorded) and the \
                 decoder-contract version.",
            );

        // R4 frame-offset spinner (shared state with the window strip).
        let mut offset = ctx.offset;
        ui.horizontal(|ui| {
            ui.label("frame offset:");
            ui.add(egui::DragValue::new(&mut offset).speed(0.05))
                .on_hover_text(
                    "catb display frame = viewer frame + offset \
                     (compensates leading trim in the viewer YUV)",
                );
        });
        if offset != ctx.offset {
            action = Some(BitstreamAction::SetOffset(offset));
        }

        // Open / focus the analysis window.
        let label = if ctx.window_open {
            "Focus Window"
        } else {
            "Open Analysis Window"
        };
        if ui.button(label).clicked() {
            action = Some(BitstreamAction::OpenWindow);
        }

        // M4: mirror the window's layer stack onto the main canvas. The
        // window's ViewConfig is the single source of truth — there is no
        // separate overlay configuration.
        let mut overlay = ctx.overlay_on_canvas;
        if ui
            .checkbox(&mut overlay, "Overlay on main canvas")
            .on_hover_text(
                "Render the Bitstream window's current fill/Grid/MV/Part/\
                 Intra layers on the main viewer canvas",
            )
            .changed()
        {
            action = Some(BitstreamAction::SetOverlayOnCanvas(overlay));
        }

        // Current-frame summary (M0 scope: visible without the window).
        match viewer_to_catb_display(ctx.current_frame_idx, ctx.offset)
            .and_then(|d| bs.frame_summary(d))
        {
            Some(s) => {
                let bits = s
                    .slice_bits
                    .map(format_bits)
                    .unwrap_or_else(|| "-".to_string());
                ui.label(format!("POC {} · {} · {} bits", s.poc, s.frame_type, bits));
            }
            None => {
                ui.label(format!(
                    "viewer#{} ↔ (no catb frame)",
                    ctx.current_frame_idx
                ));
            }
        }

        // §7 warnings that must be visible even with the window closed.
        if ctx.viewer_total_frames > 0 && bs.frame_count() != ctx.viewer_total_frames {
            ui.colored_label(
                egui::Color32::YELLOW,
                format!(
                    "⚠ catb {} frames ≠ viewer {} — check offset",
                    bs.frame_count(),
                    ctx.viewer_total_frames,
                ),
            );
        }
        if let Some((vw, vh)) = ctx.viewer_size {
            if bs.width > 0 && bs.height > 0 && (bs.width != vw || bs.height != vh) {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    format!(
                        "⚠ Stream {}×{} ≠ viewer {}×{} — overlay scaled, values unreliable",
                        bs.width, bs.height, vw, vh,
                    ),
                );
            }
        }
        if let Some(err) = ctx.error {
            ui.colored_label(egui::Color32::RED, err);
        }

        action
    }
}

/// Group digits with thin separators: 24310 → "24,310".
pub fn format_bits(bits: i64) -> String {
    let neg = bits < 0;
    let digits = bits.unsigned_abs().to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3 + 1);
    let offset = digits.len() % 3;
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (i + 3 - offset).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    if neg {
        format!("-{out}")
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::format_bits;

    #[test]
    fn bits_formatting_groups_thousands() {
        assert_eq!(format_bits(0), "0");
        assert_eq!(format_bits(184), "184");
        assert_eq!(format_bits(7000), "7,000");
        assert_eq!(format_bits(24310), "24,310");
        assert_eq!(format_bits(1234567), "1,234,567");
        assert_eq!(format_bits(-7000), "-7,000");
    }
}
