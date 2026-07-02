use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSettings {
    pub max_memory_mb: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplaySettings {
    pub zoom_min: f32,
    pub zoom_max: f32,
    pub dark_theme: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultSettings {
    pub fps: u32,
    pub color_matrix: String,
    pub width: u32,
    pub height: u32,
    pub format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralSettings {
    /// How many entries to keep in File → Recent Files. Configurable via
    /// Edit → Preferences. Cap of 20 by default; bumping this rewrites
    /// the existing list on the next save.
    pub max_recent_files: usize,
}

impl Default for GeneralSettings {
    fn default() -> Self {
        Self {
            max_recent_files: 20,
        }
    }
}

/// Persisted view configuration for the Bitstream Analysis window (§2:
/// toggle states survive window/app restarts). New optional section — old
/// settings.toml files without `[bitstream]` parse via `serde(default)`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BitstreamViewSettings {
    /// Fill layer name: "None" / "QP" / "bpp" / "Mode" / "MV-heat" /
    /// "Opportunity".
    pub fill: String,
    pub layer_mv: bool,
    pub layer_part: bool,
    /// M-B TU outline layer (`serde(default)` for pre-0.18 files).
    #[serde(default)]
    pub layer_tu: bool,
    /// M4 Intra direction layer. `serde(default)` keeps pre-0.13 settings
    /// files loading (field absent → false).
    #[serde(default)]
    pub layer_intra: bool,
    /// M-E deblocking loop-filter edge layer (`serde(default)` for
    /// pre-0.21 files).
    #[serde(default)]
    pub layer_lf: bool,
    pub layer_label: bool,
    pub layer_grid: bool,
    pub layer_sel: bool,
    /// Fill opacity 0.0–1.0 (default 0.6 per §2).
    pub opacity: f32,
    pub show_loupe: bool,
    pub inspector_collapsed: bool,
    /// Which vector the MV layer draws: "MV" / "MVP" / "MVD" (M4;
    /// `serde(default)` for pre-0.13 files).
    #[serde(default = "default_mv_source")]
    pub mv_source: String,
}

fn default_mv_source() -> String {
    "MV".to_string()
}

impl Default for BitstreamViewSettings {
    fn default() -> Self {
        // Mirrors the QP Map preset (§3) so a fresh install opens with the
        // preset combo showing "QP Map", not "Custom".
        Self {
            fill: "QP".to_string(),
            layer_mv: false,
            layer_part: false,
            layer_tu: false,
            layer_intra: false,
            layer_lf: false,
            layer_label: true,
            layer_grid: true,
            layer_sel: true,
            opacity: 0.6,
            show_loupe: false,
            inspector_collapsed: false,
            mv_source: default_mv_source(),
        }
    }
}

/// M5 external decoder-run launcher (arm's-length: the viewer only executes
/// a user-provided command template and consumes its output files — no
/// decoder is bundled or linked). Separate from [`BitstreamViewSettings`]
/// because that struct is rebuilt wholesale from the analysis window's
/// ViewConfig on every view change, which would wipe fields it doesn't know.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecoderSettings {
    /// Shell command template. Placeholders: `{input}` (bitstream path),
    /// `{workdir}` (per-bitstream work dir), `{telemetry}`
    /// (`{workdir}/telemetry.catb`), `{yuv}` (`{workdir}/decoded.yuv`).
    /// Empty = use the auto-detected codec-analyzer install when one is
    /// found (0.16.0, `decoder_run::detect_decoder_command`); otherwise the
    /// feature is disabled and Open Bitstream… only shows guidance.
    #[serde(default)]
    pub run_command: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Settings {
    pub cache: CacheSettings,
    pub display: DisplaySettings,
    pub defaults: DefaultSettings,
    #[serde(default)]
    pub general: GeneralSettings,
    #[serde(default)]
    pub bitstream: BitstreamViewSettings,
    #[serde(default)]
    pub decoder: DecoderSettings,
    #[serde(default)]
    pub recent_files: Vec<String>,
}

impl Default for CacheSettings {
    fn default() -> Self {
        Self { max_memory_mb: 512 }
    }
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            zoom_min: 0.1,
            zoom_max: 50.0,
            dark_theme: true,
        }
    }
}

impl Default for DefaultSettings {
    fn default() -> Self {
        Self {
            fps: 30,
            color_matrix: "BT.601".to_string(),
            width: 1920,
            height: 1080,
            format: "I420".to_string(),
        }
    }
}

impl Settings {
    fn config_path() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home)
            .join(".config")
            .join("video-viewer")
            .join("settings.toml")
    }

    pub fn load() -> Self {
        let path = Self::config_path();
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(content) => toml::from_str(&content).unwrap_or_default(),
                Err(_) => Self::default(),
            }
        } else {
            Self::default()
        }
    }

    pub fn save(&self) {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(content) = toml::to_string_pretty(self) {
            let _ = std::fs::write(&path, content);
        }
    }

    pub fn add_recent_file(&mut self, path: &str) {
        self.recent_files.retain(|p| p != path);
        self.recent_files.insert(0, path.to_string());
        let cap = self.general.max_recent_files.max(1);
        self.recent_files.truncate(cap);
    }
}
