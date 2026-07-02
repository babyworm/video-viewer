//! Integration tests for the M5 decoder-run launcher (`core::decoder_run`).
//!
//! No external decoder is required: a `cp`-based *fake decoder* (always
//! available on unix) exercises the real thread → shell → completion → load
//! path. An optional env-gated test (`E2E_DECODER_CMD`) additionally runs a
//! real decoder command template against the checked-in HEVC fixture.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use video_viewer::analysis::bitstream_stats::BitstreamFile;
use video_viewer::core::decoder_run::{
    derive_workdir, expand_command, plan_yuv_open, run_shell_command, shell_escape_unix,
    STDERR_LOG, STDOUT_LOG,
};

// Shared synthetic .catb v4 fixture writer — tests/common/mod.rs.
mod common;
use common::{block_record, build_catb, encode_strings, frame_record, BlockSpec};

/// Minimal valid .catb: 1 IDR frame, one intra 64×64 block (Appendix A shape).
fn fixture_catb_bytes() -> Vec<u8> {
    let strings = encode_strings(&["", "64x64", "Intra", "IDR"]);
    let meta = br#"{"schema_version":1,"decoder":{"codec":"hevc","name":"fake","contract":"0.8.3","capture_level":""},"parameter_sets":[],"frames_meta":[{"slice_headers":[],"dpb":[],"stage_images":{},"exactness_missing":[""],"block_dropped_rows":[0]}]}"#;
    let frame = frame_record(0, 0, 3, 1, 0, 0, 0, 100, 20, 80, 0, 1, 0, 28);
    let block = block_record(&BlockSpec {
        w: 64,
        h: 64,
        qp: 32,
        partition_id: 1,
        prediction_mode_id: 2,
        bits: 50,
        bit_offset: 20,
        ..Default::default()
    });
    let aux = br#"{"loop_filters":[],"sao":[]}"#;
    build_catb(1, &strings, meta, &frame, &block, &[], aux)
}

/// Run `template` for `input` exactly like the app does: derive the workdir,
/// expand placeholders, then execute through `run_shell_command` with a live
/// job gate. Returns (result, expanded telemetry/yuv paths).
#[allow(clippy::type_complexity)]
fn run_template(
    template: &str,
    input: &Path,
) -> (Result<(), String>, PathBuf, PathBuf, bool) {
    let workdir = derive_workdir(input);
    std::fs::create_dir_all(&workdir).unwrap();
    let exp = expand_command(template, input, &workdir);
    let active = AtomicUsize::new(7);
    let slot = Mutex::new(None);
    let result = run_shell_command(
        &exp.command,
        &workdir.join(STDOUT_LOG),
        &workdir.join(STDERR_LOG),
        7,
        &active,
        &slot,
    );
    (result, exp.telemetry, exp.yuv, exp.wants_yuv)
}

#[test]
fn fake_decoder_cp_produces_loadable_telemetry() {
    let dir = tempfile::tempdir().unwrap();
    // Path with a space — must survive shell quoting.
    let sub = dir.path().join("in put");
    std::fs::create_dir_all(&sub).unwrap();
    let input = sub.join("stream.h265");
    std::fs::write(&input, b"not a real bitstream").unwrap();
    let fixture = dir.path().join("fixture.catb");
    std::fs::write(&fixture, fixture_catb_bytes()).unwrap();

    // Fake decoder: copies the fixture to {telemetry}; reads {input} so a
    // broken input quoting would fail the command.
    let template = format!(
        "cat {{input}} > /dev/null && cp {} {{telemetry}}",
        shell_escape_unix(&fixture.to_string_lossy())
    );
    let (result, telemetry, _yuv, wants_yuv) = run_template(&template, &input);
    assert_eq!(result, Ok(()));
    assert!(!wants_yuv, "template has no {{yuv}}");
    assert!(telemetry.is_file(), "telemetry.catb must exist");
    // Workdir is a sibling <name>.catb-run/ of the bitstream.
    assert_eq!(
        telemetry,
        sub.join("stream.h265.catb-run").join("telemetry.catb")
    );

    // The produced file goes through the normal load path.
    let bs = BitstreamFile::open(&telemetry).unwrap();
    assert_eq!((bs.width, bs.height), (64, 64));
    assert_eq!(bs.frame_count(), 1);
}

#[test]
fn fake_decoder_with_yuv_output_enables_auto_open_plan() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("clip.h264");
    std::fs::write(&input, b"x").unwrap();
    let fixture = dir.path().join("fixture.catb");
    std::fs::write(&fixture, fixture_catb_bytes()).unwrap();

    // Fake decoder also produces exactly two 64x64 I420 frames of YUV.
    let template = format!(
        "cp {} {{telemetry}} && head -c 12288 /dev/zero > {{yuv}}",
        shell_escape_unix(&fixture.to_string_lossy())
    );
    let (result, telemetry, yuv, wants_yuv) = run_template(&template, &input);
    assert_eq!(result, Ok(()));
    assert!(wants_yuv);

    // Completion-handler decisions (success branch, no video loaded):
    let bs = BitstreamFile::open(&telemetry).unwrap();
    let size = std::fs::metadata(&yuv).ok().map(|m| m.len());
    assert_eq!(plan_yuv_open(size, bs.width, bs.height), Ok(2));
}

#[test]
fn failure_reports_exit_code_and_last_stderr_line() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("bad.h265");
    std::fs::write(&input, b"x").unwrap();

    let template = "echo first warning >&2; echo boom: cannot parse {input} >&2; exit 3";
    let (result, telemetry, _, _) = run_template(template, &input);
    let err = result.unwrap_err();
    assert!(err.contains("status 3"), "got: {err}");
    assert!(err.contains("boom: cannot parse"), "got: {err}");
    assert!(!telemetry.exists());
}

#[test]
fn completion_handler_failure_branch_missing_telemetry() {
    // Exit 0 but no telemetry produced → the completion handler must detect
    // the missing file (mirrors finish_decoder_run's first check).
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("noop.h265");
    std::fs::write(&input, b"x").unwrap();
    let (result, telemetry, _, _) = run_template("true", &input);
    assert_eq!(result, Ok(()));
    assert!(!telemetry.is_file(), "success + no telemetry = handler error path");
}

#[test]
fn cancel_kills_long_running_child() {
    let dir = tempfile::tempdir().unwrap();
    let workdir = dir.path().join("w");
    std::fs::create_dir_all(&workdir).unwrap();
    let active = Arc::new(AtomicUsize::new(1));
    let slot: Arc<Mutex<Option<std::process::Child>>> = Arc::new(Mutex::new(None));

    let active2 = Arc::clone(&active);
    let slot2 = Arc::clone(&slot);
    let out_log = workdir.join(STDOUT_LOG);
    let err_log = workdir.join(STDERR_LOG);
    let start = std::time::Instant::now();
    let handle = std::thread::spawn(move || {
        run_shell_command("sleep 30", &out_log, &err_log, 1, &active2, &slot2)
    });
    // Wait until the child is parked, then cancel the way the UI does:
    // invalidate the job id and kill the parked child.
    for _ in 0..100 {
        if slot.lock().unwrap().is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    active.store(0, Ordering::Release);
    if let Some(child) = slot.lock().unwrap().as_mut() {
        let _ = child.kill();
    }
    let result = handle.join().unwrap();
    assert_eq!(result, Err("cancelled".to_string()));
    assert!(
        start.elapsed() < std::time::Duration::from_secs(10),
        "cancel must not wait for the child's natural exit"
    );
    assert!(slot.lock().unwrap().is_none(), "child must be reaped");
}

#[test]
fn thread_publish_gate_discards_stale_job() {
    // Full worker-thread publish path (scene-detect pattern): a superseded
    // job must not publish its result.
    let dir = tempfile::tempdir().unwrap();
    let workdir = dir.path().join("w");
    std::fs::create_dir_all(&workdir).unwrap();
    type Output = Arc<Mutex<Option<(usize, Result<(), String>)>>>;
    let output: Output = Arc::new(Mutex::new(None));
    let active = Arc::new(AtomicUsize::new(5));
    let slot: Arc<Mutex<Option<std::process::Child>>> = Arc::new(Mutex::new(None));

    let run = |job_id: usize, output: Output, active: Arc<AtomicUsize>, slot: Arc<Mutex<Option<std::process::Child>>>, workdir: PathBuf| {
        std::thread::spawn(move || {
            let result = run_shell_command(
                "true",
                &workdir.join(STDOUT_LOG),
                &workdir.join(STDERR_LOG),
                job_id,
                &active,
                &slot,
            );
            let mut s = output.lock().unwrap();
            if active.load(Ordering::Acquire) == job_id {
                *s = Some((job_id, result));
            }
        })
    };
    // Stale job (id 4 while active is 5): must not publish.
    run(4, Arc::clone(&output), Arc::clone(&active), Arc::clone(&slot), workdir.clone())
        .join()
        .unwrap();
    assert!(output.lock().unwrap().is_none());
    // Active job publishes.
    run(5, Arc::clone(&output), Arc::clone(&active), Arc::clone(&slot), workdir.clone())
        .join()
        .unwrap();
    assert_eq!(*output.lock().unwrap(), Some((5, Ok(()))));
}

#[test]
fn rerun_with_fresh_child_slot_does_not_kill_new_job() {
    // Regression: re-running while a job is in flight. The app gives every
    // job its *own* child slot (start_decoder_run replaces the Arc), so the
    // superseded worker — waking from its ≤120 ms poll sleep after the UI
    // thread killed its child — can only take/kill from its own slot, never
    // the replacement job's child. With a shared slot this reproducibly
    // killed job 2 ("Decoder run failed: cancelled").
    let dir = tempfile::tempdir().unwrap();
    let workdir = dir.path().join("w");
    std::fs::create_dir_all(&workdir).unwrap();
    let marker = dir.path().join("job2-done");
    let active = Arc::new(AtomicUsize::new(1));

    // Job 1: long-running, parked in its own slot.
    let slot1: Arc<Mutex<Option<std::process::Child>>> = Arc::new(Mutex::new(None));
    let (a1, s1) = (Arc::clone(&active), Arc::clone(&slot1));
    let (out1, err1) = (workdir.join("1.out"), workdir.join("1.err"));
    let h1 = std::thread::spawn(move || {
        run_shell_command("sleep 30", &out1, &err1, 1, &a1, &s1)
    });
    for _ in 0..100 {
        if slot1.lock().unwrap().is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    // Re-run, exactly like start_decoder_run: invalidate, kill the old
    // child via the *old* slot, then start job 2 with a *fresh* slot.
    active.store(0, Ordering::Release);
    if let Some(child) = slot1.lock().unwrap().as_mut() {
        let _ = child.kill();
    }
    active.store(2, Ordering::Release);
    let slot2: Arc<Mutex<Option<std::process::Child>>> = Arc::new(Mutex::new(None));
    let (a2, s2) = (Arc::clone(&active), Arc::clone(&slot2));
    let (out2, err2) = (workdir.join("2.out"), workdir.join("2.err"));
    // Job 2 outlives job 1's poll sleep so the old worker wakes mid-run.
    let cmd2 = format!("sleep 1 && touch {}", shell_escape_unix(&marker.to_string_lossy()));
    let h2 = std::thread::spawn(move || {
        run_shell_command(&cmd2, &out2, &err2, 2, &a2, &s2)
    });

    assert_eq!(h1.join().unwrap(), Err("cancelled".to_string()));
    assert_eq!(h2.join().unwrap(), Ok(()), "old worker must not kill the new job");
    assert!(marker.is_file(), "job 2's child must run to completion");
}

#[test]
fn yuv_plan_rejects_partial_file_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let yuv = dir.path().join("decoded.yuv");
    // 1.5 frames of 64x64 I420 (6144 bytes/frame).
    std::fs::write(&yuv, vec![0u8; 9216]).unwrap();
    let size = std::fs::metadata(&yuv).ok().map(|m| m.len());
    assert!(plan_yuv_open(size, 64, 64).is_err());
    // Missing file → None → error, matching the "not produced" branch.
    let missing = dir.path().join("nope.yuv");
    let size = std::fs::metadata(&missing).ok().map(|m| m.len());
    assert!(plan_yuv_open(size, 64, 64).is_err());
}

/// Optional: run a *real* decoder command template (e.g. the codec-analyzer
/// one from the Settings example) against the checked-in HEVC fixture.
/// Skipped unless `E2E_DECODER_CMD` is set.
#[test]
fn e2e_real_decoder_command() {
    let Ok(template) = std::env::var("E2E_DECODER_CMD") else {
        eprintln!("E2E_DECODER_CMD not set — skipping real-decoder E2E test");
        return;
    };
    let stream = PathBuf::from("../test_data/bitstream/hevc_bslice/hevc_bslice.h265");
    if !stream.is_file() {
        eprintln!("fixture bitstream missing — skipping");
        return;
    }
    // Run in a tempdir copy so the checked-in tree stays clean.
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("hevc_bslice.h265");
    std::fs::copy(&stream, &input).unwrap();
    let (result, telemetry, _, _) = run_template(&template, &input);
    assert_eq!(result, Ok(()));
    let bs = BitstreamFile::open(&telemetry).unwrap();
    assert!(bs.frame_count() > 0);
}
