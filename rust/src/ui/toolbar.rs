use eframe::egui;

/// Grid size cycle for main grid
const GRID_SIZES: &[u32] = &[0, 16, 32, 64, 128];
/// Grid size cycle for sub-grid
const SUB_GRID_SIZES: &[u32] = &[0, 4, 8, 16];

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
    ToggleGrid,
    ToggleSubGrid,
    FitToView,
    ToggleAutoFit,
    Refresh,
    ZoomIn,
    ZoomOut,
    Zoom1to1,
    Zoom2to1,
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

            // --- Grid toggle ---
            let grid_label = if self.grid_size > 0 {
                format!("Grid({})", self.grid_size)
            } else {
                "Grid".to_string()
            };
            if ui.button(grid_label).clicked() {
                self.grid_idx = (self.grid_idx + 1) % GRID_SIZES.len();
                self.grid_size = GRID_SIZES[self.grid_idx];
                action = Some(ToolbarAction::ToggleGrid);
            }

            // --- Sub-grid toggle ---
            let sub_label = if self.sub_grid_size > 0 {
                format!("Sub({})", self.sub_grid_size)
            } else {
                "Sub-Grid".to_string()
            };
            if ui.button(sub_label).clicked() {
                self.sub_grid_idx = (self.sub_grid_idx + 1) % SUB_GRID_SIZES.len();
                self.sub_grid_size = SUB_GRID_SIZES[self.sub_grid_idx];
                action = Some(ToolbarAction::ToggleSubGrid);
            }

            ui.separator();

            // --- Zoom controls ---
            if ui.button("Fit").clicked() {
                action = Some(ToolbarAction::FitToView);
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
