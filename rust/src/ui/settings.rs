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
pub struct Settings {
    pub cache: CacheSettings,
    pub display: DisplaySettings,
    pub defaults: DefaultSettings,
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

impl Default for Settings {
    fn default() -> Self {
        Self {
            cache: CacheSettings::default(),
            display: DisplaySettings::default(),
            defaults: DefaultSettings::default(),
            recent_files: Vec::new(),
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
        self.recent_files.truncate(10);
    }
}
