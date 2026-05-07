use eframe::egui;

/// Direct-selectable main grid sizes.
pub const GRID_SIZES: &[u32] = &[0, 128, 64, 32, 16];
/// Direct-selectable sub-grid sizes (Off, then 4 doubling up to 64).
/// The visible subset depends on the main grid size: only entries with
/// `size <= max_sub_grid(main)` are enabled.
pub const SUB_GRID_SIZES: &[u32] = &[0, 4, 8, 16, 32, 64];

/// Largest sub-grid value selectable for a given main grid size.
/// Sub-grid is forbidden when main grid is Off, otherwise capped to half of main.
pub fn max_sub_grid(main_grid_size: u32) -> u32 {
    if main_grid_size == 0 {
        0
    } else {
        main_grid_size / 2
    }
}

pub struct Toolbar {
    /// Component index: 0=Full, 1=Y/R, 2=U/G, 3=V/B, 4=Split
    pub current_component: u8,
    pub grid_size: u32,
    pub sub_grid_size: u32,
    pub grid_idx: usize,
    pub sub_grid_idx: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ToolbarAction {
    SetComponent(u8),
    GridChanged,
    ToggleSubGrid,
    FitToView,
    ToggleAutoFit,
    Refresh,
    ZoomIn,
    ZoomOut,
    Zoom1to1,
    Zoom2to1,
    CenterImage,
}

impl Toolbar {
    pub fn new() -> Self {
        Self {
            current_component: 0,
            grid_size: 0,
            sub_grid_size: 0,
            grid_idx: 0,
            sub_grid_idx: 0,
        }
    }

    pub fn set_grid_size(&mut self, size: u32) -> bool {
        let Some(idx) = GRID_SIZES.iter().position(|&candidate| candidate == size) else {
            return false;
        };
        self.grid_size = size;
        self.grid_idx = idx;
        self.enforce_sub_grid_constraint();
        true
    }

    pub fn cycle_grid_size(&mut self) {
        self.grid_idx = (self.grid_idx + 1) % GRID_SIZES.len();
        self.grid_size = GRID_SIZES[self.grid_idx];
        self.enforce_sub_grid_constraint();
    }

    pub fn set_sub_grid_size(&mut self, size: u32) -> bool {
        let Some(idx) = SUB_GRID_SIZES.iter().position(|&candidate| candidate == size) else {
            return false;
        };
        if size > max_sub_grid(self.grid_size) {
            return false;
        }
        self.sub_grid_size = size;
        self.sub_grid_idx = idx;
        true
    }

    /// Clamp sub-grid to the constraint (size <= main / 2). Resets to Off when violated.
    /// Should be called whenever the main grid size changes.
    pub fn enforce_sub_grid_constraint(&mut self) {
        if self.sub_grid_size > max_sub_grid(self.grid_size) {
            self.sub_grid_size = 0;
            self.sub_grid_idx = 0;
        }
    }

    /// Show the toolbar. Returns an action if one was triggered.
    pub fn show(&mut self, ui: &mut egui::Ui, is_yuv: bool, auto_fit: bool) -> Option<ToolbarAction> {
        let mut action = None;

        ui.horizontal(|ui| {
            // --- Component selection ---
            ui.label("Component:");

            let components: &[(&str, u8)] = if is_yuv {
                &[
                    ("Full", 0),
                    ("Y", 1),
                    ("U", 2),
                    ("V", 3),
                    ("Split", 4),
                ]
            } else {
                &[
                    ("Full", 0),
                    ("R", 1),
                    ("G", 2),
                    ("B", 3),
                    ("Split", 4),
                ]
            };

            for &(label, idx) in components {
                let selected = self.current_component == idx;
                let btn = egui::Button::new(label).selected(selected);
                if ui.add(btn).clicked() {
                    self.current_component = idx;
                    action = Some(ToolbarAction::SetComponent(idx));
                }
            }

            ui.separator();

            // --- Grid selector ---
            ui.label("Grid:");
            let before_grid_size = self.grid_size;
            egui::ComboBox::from_id_salt("main_grid_size_combo")
                .selected_text(grid_size_label(self.grid_size))
                .width(86.0)
                .show_ui(ui, |ui| {
                    for (idx, &size) in GRID_SIZES.iter().enumerate() {
                        if ui
                            .selectable_value(&mut self.grid_size, size, grid_size_label(size))
                            .clicked()
                        {
                            self.grid_idx = idx;
                        }
                    }
                });
            if self.grid_size != before_grid_size {
                self.set_grid_size(self.grid_size);
                action = Some(ToolbarAction::GridChanged);
            }

            // --- Sub-grid selector ---
            ui.label("Sub:");
            let max_sub = max_sub_grid(self.grid_size);
            let before_sub_size = self.sub_grid_size;
            egui::ComboBox::from_id_salt("sub_grid_size_combo")
                .selected_text(grid_size_label(self.sub_grid_size))
                .width(86.0)
                .show_ui(ui, |ui| {
                    for (idx, &size) in SUB_GRID_SIZES.iter().enumerate() {
                        // Off is always selectable; non-zero entries gated by main grid.
                        let allowed = size == 0 || size <= max_sub;
                        ui.add_enabled_ui(allowed, |ui| {
                            if ui
                                .selectable_value(
                                    &mut self.sub_grid_size,
                                    size,
                                    grid_size_label(size),
                                )
                                .clicked()
                            {
                                self.sub_grid_idx = idx;
                            }
                        });
                    }
                });
            if self.sub_grid_size != before_sub_size {
                // Sanity-check: never let a click bypass the constraint.
                if self.sub_grid_size > max_sub {
                    self.sub_grid_size = before_sub_size;
                } else {
                    action = Some(ToolbarAction::ToggleSubGrid);
                }
            }

            ui.separator();

            // --- Zoom controls ---
            if ui.button("Fit").clicked() {
                action = Some(ToolbarAction::FitToView);
            }
            if ui.button("Center").on_hover_text("Center image (C)").clicked() {
                action = Some(ToolbarAction::CenterImage);
            }
            let auto_btn = egui::Button::new("Auto").selected(auto_fit);
            if ui.add(auto_btn).clicked() {
                action = Some(ToolbarAction::ToggleAutoFit);
            }
            if ui.button("↻").clicked() {
                action = Some(ToolbarAction::Refresh);
            }
            if ui.button("1:1").clicked() {
                action = Some(ToolbarAction::Zoom1to1);
            }
            if ui.button("2:1").clicked() {
                action = Some(ToolbarAction::Zoom2to1);
            }
            if ui.button("+").clicked() {
                action = Some(ToolbarAction::ZoomIn);
            }
            if ui.button("-").clicked() {
                action = Some(ToolbarAction::ZoomOut);
            }
        });

        action
    }
}

fn grid_size_label(size: u32) -> String {
    if size == 0 {
        "Off".to_string()
    } else {
        format!("{}×{}", size, size)
    }
}

impl Default for Toolbar {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert single-channel grayscale data to false-color RGB.
///
/// # Arguments
/// - `gray`: flat slice of `width * height` bytes (single channel)
/// - `width`, `height`: image dimensions
/// - `name`: channel name — one of "Y", "U", "V", "R", "G", "B"
///
/// # Returns
/// Vec<u8> of size `width * height * 3` in RGB order.
pub fn colorize_channel(gray: &[u8], width: u32, height: u32, name: &str) -> Vec<u8> {
    let n = (width * height) as usize;
    let mut out = vec![0u8; n * 3];
    match name {
        "Y" => {
            // Grayscale (luminance) — show as plain gray
            for i in 0..n {
                let v = gray[i];
                out[i * 3] = v;
                out[i * 3 + 1] = v;
                out[i * 3 + 2] = v;
            }
        }
        "U" => {
            // Cb — blue channel
            for i in 0..n {
                out[i * 3 + 2] = gray[i]; // B
            }
        }
        "V" => {
            // Cr — red channel
            for i in 0..n {
                out[i * 3] = gray[i]; // R
            }
        }
        "R" => {
            for i in 0..n {
                out[i * 3] = gray[i];
            }
        }
        "G" => {
            for i in 0..n {
                out[i * 3 + 1] = gray[i];
            }
        }
        "B" => {
            for i in 0..n {
                out[i * 3 + 2] = gray[i];
            }
        }
        _ => {
            // Unknown channel — treat as grayscale
            for i in 0..n {
                let v = gray[i];
                out[i * 3] = v;
                out[i * 3 + 1] = v;
                out[i * 3 + 2] = v;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_sizes_are_in_direct_selector_order() {
        assert_eq!(GRID_SIZES, &[0, 128, 64, 32, 16]);
    }

    #[test]
    fn toolbar_defaults_to_grid_off() {
        let toolbar = Toolbar::new();

        assert_eq!(toolbar.grid_size, 0);
        assert_eq!(toolbar.grid_idx, 0);
    }

    #[test]
    fn set_grid_size_updates_index_for_supported_sizes() {
        for (idx, &size) in GRID_SIZES.iter().enumerate() {
            let mut toolbar = Toolbar::new();

            assert!(toolbar.set_grid_size(size));
            assert_eq!(toolbar.grid_size, size);
            assert_eq!(toolbar.grid_idx, idx);
        }
    }

    #[test]
    fn set_grid_size_rejects_unsupported_sizes_without_mutating() {
        let mut toolbar = Toolbar::new();
        assert!(toolbar.set_grid_size(64));

        assert!(!toolbar.set_grid_size(48));
        assert_eq!(toolbar.grid_size, 64);
        assert_eq!(toolbar.grid_idx, 2);
    }

    #[test]
    fn cycle_grid_size_uses_direct_selector_order() {
        let mut toolbar = Toolbar::new();
        let mut seen = vec![toolbar.grid_size];

        for _ in 1..GRID_SIZES.len() {
            toolbar.cycle_grid_size();
            seen.push(toolbar.grid_size);
        }

        assert_eq!(seen, GRID_SIZES);
        toolbar.cycle_grid_size();
        assert_eq!(toolbar.grid_size, 0);
        assert_eq!(toolbar.grid_idx, 0);
    }

    #[test]
    fn sub_grid_sizes_are_doubling_from_4_to_64() {
        assert_eq!(SUB_GRID_SIZES, &[0, 4, 8, 16, 32, 64]);
    }

    #[test]
    fn max_sub_grid_is_half_of_main_or_zero() {
        assert_eq!(max_sub_grid(0), 0);
        assert_eq!(max_sub_grid(16), 8);
        assert_eq!(max_sub_grid(32), 16);
        assert_eq!(max_sub_grid(64), 32);
        assert_eq!(max_sub_grid(128), 64);
    }

    #[test]
    fn set_sub_grid_size_respects_main_grid_cap() {
        let mut toolbar = Toolbar::new();
        toolbar.set_grid_size(64);
        assert!(toolbar.set_sub_grid_size(32));
        assert_eq!(toolbar.sub_grid_size, 32);
        // 64 disallowed when main is 64 (would equal main, not <= main/2).
        assert!(!toolbar.set_sub_grid_size(64));
        assert_eq!(toolbar.sub_grid_size, 32);
        // Off is always allowed.
        assert!(toolbar.set_sub_grid_size(0));
    }

    #[test]
    fn shrinking_main_grid_clamps_oversized_sub_grid_to_off() {
        let mut toolbar = Toolbar::new();
        toolbar.set_grid_size(128);
        assert!(toolbar.set_sub_grid_size(64));
        toolbar.set_grid_size(32); // max_sub becomes 16, current sub is 64 → reset.
        assert_eq!(toolbar.sub_grid_size, 0);
        assert_eq!(toolbar.sub_grid_idx, 0);
    }

    #[test]
    fn main_grid_off_forbids_any_sub_grid() {
        let mut toolbar = Toolbar::new();
        // Off is always allowed.
        assert!(toolbar.set_sub_grid_size(0));
        // Non-zero rejected when main is Off.
        assert!(!toolbar.set_sub_grid_size(4));
        assert!(!toolbar.set_sub_grid_size(8));
    }
}
