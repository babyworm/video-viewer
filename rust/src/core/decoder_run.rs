//! External decoder-run launcher (M5) — arm's-length integration.
//!
//! The viewer never bundles or links a decoder. The user registers a shell
//! command *template* in Settings (e.g. `python -m codec_analyzer decoder-run
//! {input} --decoder-workdir {workdir} --telemetry-level block --yuv-output
//! {yuv}`); this module expands the placeholders, runs the command through
//! the platform shell in a background thread, and the app then consumes the
//! produced files (`telemetry.catb`, optionally `decoded.yuv`). No template
//! configured → the feature explains itself and does nothing.
//!
//! Placeholders (simple string substitution, values shell-quoted):
//! - `{input}`     — the bitstream path the user picked
//! - `{workdir}`   — per-bitstream work directory (`<name>.catb-run/`)
//! - `{telemetry}` — `{workdir}/telemetry.catb`
//! - `{yuv}`       — `{workdir}/decoded.yuv`

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// Fixed telemetry file name inside the work directory.
pub const TELEMETRY_FILE: &str = "telemetry.catb";
/// Fixed decoded-YUV file name inside the work directory.
pub const YUV_FILE: &str = "decoded.yuv";
/// Captured decoder stdout (workdir-relative).
pub const STDOUT_LOG: &str = "decoder-stdout.log";
/// Captured decoder stderr (workdir-relative).
pub const STDERR_LOG: &str = "decoder-stderr.log";

/// A command template with all placeholders substituted, plus the derived
/// output paths the completion handler needs.
#[derive(Debug, Clone)]
pub struct ExpandedCommand {
    /// The shell command line to execute (`sh -c` / `cmd /C`).
    pub command: String,
    pub workdir: PathBuf,
    pub telemetry: PathBuf,
    pub yuv: PathBuf,
    /// Whether the template referenced `{yuv}` at all — only then does the
    /// completion handler consider auto-opening the decoded YUV.
    pub wants_yuv: bool,
}

/// POSIX single-quote escaping: wrap in `'…'`, embedded `'` becomes `'\''`.
pub fn shell_escape_unix(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

/// `cmd /C` quoting: wrap in double quotes, double any embedded quotes.
pub fn shell_quote_windows(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

/// Platform path quoting used for placeholder values.
fn quote_path(p: &Path) -> String {
    let s = p.to_string_lossy();
    #[cfg(unix)]
    {
        shell_escape_unix(&s)
    }
    #[cfg(windows)]
    {
        shell_quote_windows(&s)
    }
}

/// Work directory for a bitstream: a sibling `<file name>.catb-run/`
/// (full file name, so `foo.h264` and `foo.h265` never collide).
pub fn derive_workdir(bitstream: &Path) -> PathBuf {
    let name = bitstream
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "bitstream".to_string());
    bitstream.with_file_name(format!("{name}.catb-run"))
}

/// Substitute the four placeholders into `template`. Paths are shell-quoted
/// so spaces/quotes in file names survive `sh -c` / `cmd /C`.
pub fn expand_command(template: &str, input: &Path, workdir: &Path) -> ExpandedCommand {
    let telemetry = workdir.join(TELEMETRY_FILE);
    let yuv = workdir.join(YUV_FILE);
    let wants_yuv = template.contains("{yuv}");
    let command = template
        .replace("{input}", &quote_path(input))
        .replace("{workdir}", &quote_path(workdir))
        .replace("{telemetry}", &quote_path(&telemetry))
        .replace("{yuv}", &quote_path(&yuv));
    ExpandedCommand {
        command,
        workdir: workdir.to_path_buf(),
        telemetry,
        yuv,
        wants_yuv,
    }
}

/// Last non-empty stderr line, for the one-line failure label.
pub fn last_stderr_line(stderr: &str) -> Option<String> {
    stderr
        .lines()
        .rev()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .map(str::to_string)
}

/// Decide whether the decoded YUV can be auto-opened as an I420 video of the
/// telemetry's resolution. Returns the frame count on success, or a
/// user-facing reason string when the file should not be opened.
///
/// `yuv_size` is `None` when the file was not produced.
pub fn plan_yuv_open(yuv_size: Option<u64>, width: u32, height: u32) -> Result<usize, String> {
    let size = yuv_size.ok_or_else(|| "decoder did not produce the YUV output".to_string())?;
    if width == 0 || height == 0 {
        return Err("stream resolution unknown — decoded YUV not auto-opened".to_string());
    }
    let luma = width as u64 * height as u64;
    let chroma = (width as u64).div_ceil(2) * (height as u64).div_ceil(2);
    let frame = luma + 2 * chroma;
    if size == 0 || !size.is_multiple_of(frame) {
        return Err(format!(
            "decoded YUV size {size} is not a whole number of {width}x{height} I420 frames — not auto-opened"
        ));
    }
    Ok((size / frame) as usize)
}

/// Build the platform shell invocation for a command line.
fn shell_command(command: &str) -> Command {
    #[cfg(unix)]
    {
        let mut c = Command::new("sh");
        c.arg("-c").arg(command);
        c
    }
    #[cfg(windows)]
    {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(command);
        c
    }
}

/// Run `command` through the platform shell, blocking until it exits or the
/// job is superseded/cancelled (scene-detect job-id gate).
///
/// - stdout/stderr are redirected to `stdout_log` / `stderr_log` files (no
///   pipe-buffer deadlock, and the logs stay inspectable in the workdir).
/// - The spawned [`Child`] is parked in `child_slot` so the UI thread can
///   `kill()` it (Cancel button). A `None` slot mid-run means it was killed.
/// - When `active_job` no longer equals `job_id`, the child is killed and
///   reaped, and the run reports `Err("cancelled")` (which the publisher
///   gate discards anyway).
pub fn run_shell_command(
    command: &str,
    stdout_log: &Path,
    stderr_log: &Path,
    job_id: usize,
    active_job: &AtomicUsize,
    child_slot: &Mutex<Option<Child>>,
) -> Result<(), String> {
    let stdout = std::fs::File::create(stdout_log)
        .map_err(|e| format!("cannot create {}: {e}", stdout_log.display()))?;
    let stderr = std::fs::File::create(stderr_log)
        .map_err(|e| format!("cannot create {}: {e}", stderr_log.display()))?;
    let mut cmd = shell_command(command);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    let mut child = cmd.spawn().map_err(|e| format!("failed to spawn shell: {e}"))?;
    {
        let mut slot = child_slot.lock().unwrap();
        if active_job.load(Ordering::Acquire) != job_id {
            // Cancelled before the child was even parked.
            let _ = child.kill();
            let _ = child.wait();
            return Err("cancelled".to_string());
        }
        *slot = Some(child);
    }
    let status = loop {
        if active_job.load(Ordering::Acquire) != job_id {
            if let Some(mut c) = child_slot.lock().unwrap().take() {
                let _ = c.kill();
                let _ = c.wait(); // reap — no zombie
            }
            return Err("cancelled".to_string());
        }
        let polled = {
            let mut slot = child_slot.lock().unwrap();
            match slot.as_mut() {
                // The UI thread killed and took the child.
                None => return Err("cancelled".to_string()),
                Some(c) => c
                    .try_wait()
                    .map_err(|e| format!("wait on decoder failed: {e}"))?,
            }
        };
        match polled {
            Some(st) => {
                // Exited — drop the handle so Cancel can't kill a reused PID.
                child_slot.lock().unwrap().take();
                break st;
            }
            None => std::thread::sleep(Duration::from_millis(120)),
        }
    };
    if status.success() {
        Ok(())
    } else {
        let text = std::fs::read_to_string(stderr_log).unwrap_or_default();
        let code = status
            .code()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "killed by signal".to_string());
        Err(match last_stderr_line(&text) {
            Some(line) => format!("decoder exited with status {code}: {line}"),
            None => format!("decoder exited with status {code}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_unix_plain_and_spaces() {
        assert_eq!(shell_escape_unix("abc"), "'abc'");
        assert_eq!(shell_escape_unix("/tmp/a b/c.h265"), "'/tmp/a b/c.h265'");
    }

    #[test]
    fn escape_unix_embedded_single_quote() {
        assert_eq!(shell_escape_unix("it's"), r#"'it'\''s'"#);
    }

    #[test]
    fn quote_windows_doubles_quotes() {
        assert_eq!(shell_quote_windows(r#"C:\a b\c.h265"#), r#""C:\a b\c.h265""#);
        assert_eq!(shell_quote_windows(r#"we"ird"#), r#""we""ird""#);
    }

    #[test]
    fn workdir_is_sibling_with_full_name() {
        assert_eq!(
            derive_workdir(Path::new("/data/streams/foo.h265")),
            PathBuf::from("/data/streams/foo.h265.catb-run")
        );
        // foo.h264 and foo.h265 must not collide.
        assert_ne!(
            derive_workdir(Path::new("/d/foo.h264")),
            derive_workdir(Path::new("/d/foo.h265"))
        );
    }

    #[test]
    fn expand_substitutes_all_placeholders() {
        let exp = expand_command(
            "dec {input} -w {workdir} -t {telemetry} -y {yuv}",
            Path::new("/tmp/in dir/a.h265"),
            Path::new("/tmp/in dir/a.h265.catb-run"),
        );
        assert!(exp.wants_yuv);
        assert_eq!(exp.telemetry, PathBuf::from("/tmp/in dir/a.h265.catb-run/telemetry.catb"));
        assert_eq!(exp.yuv, PathBuf::from("/tmp/in dir/a.h265.catb-run/decoded.yuv"));
        #[cfg(unix)]
        assert_eq!(
            exp.command,
            "dec '/tmp/in dir/a.h265' -w '/tmp/in dir/a.h265.catb-run' \
             -t '/tmp/in dir/a.h265.catb-run/telemetry.catb' \
             -y '/tmp/in dir/a.h265.catb-run/decoded.yuv'"
        );
    }

    #[test]
    fn expand_without_yuv_placeholder() {
        let exp = expand_command(
            "dec {input} --telemetry {telemetry}",
            Path::new("/a/b.h264"),
            Path::new("/a/b.h264.catb-run"),
        );
        assert!(!exp.wants_yuv);
        assert!(!exp.command.contains("{telemetry}"));
        assert!(!exp.command.contains("{input}"));
    }

    #[cfg(unix)]
    #[test]
    fn expand_escapes_quote_in_path() {
        let exp = expand_command(
            "dec {input}",
            Path::new("/tmp/it's.h265"),
            Path::new("/tmp/it's.h265.catb-run"),
        );
        assert_eq!(exp.command, r#"dec '/tmp/it'\''s.h265'"#);
    }

    #[test]
    fn last_stderr_line_skips_trailing_blanks() {
        assert_eq!(
            last_stderr_line("warn: x\nERROR: boom\n\n  \n"),
            Some("ERROR: boom".to_string())
        );
        assert_eq!(last_stderr_line("\n \n"), None);
        assert_eq!(last_stderr_line(""), None);
    }

    #[test]
    fn yuv_plan_accepts_exact_multiple() {
        // 64x64 I420 = 6144 bytes/frame.
        assert_eq!(plan_yuv_open(Some(6144 * 3), 64, 64), Ok(3));
        // Odd dimensions round chroma up: 3x3 → 9 + 2*4 = 17.
        assert_eq!(plan_yuv_open(Some(17), 3, 3), Ok(1));
    }

    #[test]
    fn yuv_plan_rejects_missing_empty_or_partial() {
        assert!(plan_yuv_open(None, 64, 64).is_err());
        assert!(plan_yuv_open(Some(0), 64, 64).is_err());
        assert!(plan_yuv_open(Some(6144 + 1), 64, 64).is_err());
        assert!(plan_yuv_open(Some(6144), 0, 0).is_err());
    }
}
