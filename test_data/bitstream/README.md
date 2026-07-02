# .catb Reader Test Fixtures

Fixtures for validating the video-viewer `.catb` (CATB0001, telemetry v4) reader.
Generated 2026-07-02 with `codec-analyzer` + its instrumented FFmpeg decoder.

**License note: 이 디렉토리의 생성물(비트스트림/.catb/JSON/YUV)은 GPL 프로그램(계측 FFmpeg)의 출력물로 GPL 비대상이다.**

## Environment

```bash
cd /home/babyworm/work/codec-analyzer
export CODEC_ANALYZER_FFMPEG=/home/babyworm/work/codec-analyzer/.local/bin/ffmpeg
CA=".venv/bin/python -m codec_analyzer"
```

## .catb container

- Magic: bytes 0-7 = `CATB0001`
- Bytes 8-11: telemetry version (LE u32) = `4`
- Bytes 12-15: frame count (LE u32)
- Telemetry level: `block` (no loop_filters/cabac/coeff rows; intra pred mode syntax retained)

## Fixtures

| Fixture | Codec / Profile | Resolution | Frames (decode order: POC/type) | Notes |
|---|---|---|---|---|
| `hevc_intra/hevc_intra.h265` | HEVC Rext (profile_4)* | 64x64 yuv420p | 2: IDR(0), IDR(0) | intra-only, 8 CTUs total |
| `hevc_inter/hevc_inter.h265` | HEVC Main | 64x64 yuv420p | 3: IDR(0), P(1), P(2) | L0 references |
| `hevc_bslice/hevc_bslice.h265` | HEVC Main | 64x64 yuv420p | 4: IDR(0), P(3), B(1), B(2) | **decode != display order**; L0+L1 |
| `hevc_multi_idr/hevc_multi_idr.h265` | HEVC Main | 64x64 yuv420p | 6: IDR(0), P(1), P(2), IDR(0), P(1), P(2) | POC resets at decode_order 3 |
| `avc_mb/avc_cavlc.h264` | AVC High, CAVLC | 80x64 yuv420p | 2: IDR(0), IDR(0) | 4x4+8x8 intra transforms, macroblock rows |

*hevc_intra: codec-analyzer reports `profile_status=unsupported` (Rext/profile_4), but decoder telemetry is
100% ready and the .catb is complete — usable as a reader fixture; do not use it for profile-gate tests.

Per-fixture files: `<name>.h265|.h264` (bitstream), `<name>.catb` (binary telemetry),
`<name>_oracle_blocks.json`, `<name>_oracle_frames.json` (oracle dumps: per-frame
frame_index/decode_order/poc/frame_type/output/slice_bits/... and per-block rows).
`hevc_bslice/hevc_bslice_64x64_i420.yuv` = decoded output in **output (display) order**, 4 frames I420.

## Generation commands (full regeneration)

```bash
B=/path/to/video-viewer/test_data/bitstream

# 1) hevc_intra
$CA sample-bitstream --decoder-fixture --decoder-fixture-kind intra -o $B/hevc_intra/hevc_intra.h265
# 2) hevc_inter (IDR/P/P, L0)
$CA sample-bitstream --decoder-fixture --decoder-fixture-kind inter -o $B/hevc_inter/hevc_inter.h265
# 3) hevc_bslice (L0+L1, reorder)
$CA sample-bitstream --decoder-fixture --decoder-fixture-kind b-slice -o $B/hevc_bslice/hevc_bslice.h265
# 4) hevc_multi_idr: binary concat of two Annex-B inter streams (POC reset verified via `timeline`)
cat $B/hevc_inter/hevc_inter.h265 $B/hevc_inter/hevc_inter.h265 > $B/hevc_multi_idr/hevc_multi_idr.h265
# 5) avc_mb (CAVLC)
$CA sample-bitstream --codec avc --decoder-fixture --decoder-fixture-kind cavlc -o $B/avc_mb/avc_cavlc.h264

# .catb per fixture (b-slice also dumps YUV; adjust input/workdir per fixture)
$CA decoder-run $B/hevc_intra/hevc_intra.h265     --decoder-workdir $B/hevc_intra/telemetry     --telemetry-level block
$CA decoder-run $B/hevc_inter/hevc_inter.h265     --decoder-workdir $B/hevc_inter/telemetry     --telemetry-level block
$CA decoder-run $B/hevc_bslice/hevc_bslice.h265   --decoder-workdir $B/hevc_bslice/telemetry    --telemetry-level block \
    --yuv-output $B/hevc_bslice/hevc_bslice_64x64_i420.yuv
$CA decoder-run $B/hevc_multi_idr/hevc_multi_idr.h265 --decoder-workdir $B/hevc_multi_idr/telemetry --telemetry-level block
$CA decoder-run $B/avc_mb/avc_cavlc.h264          --decoder-workdir $B/avc_mb/telemetry         --telemetry-level block

# The .catb is written as <workdir>/codec-analyzer-telemetry.catb; it was moved to
# <fixture>/<name>.catb and the telemetry workdir (JSON + stage BMPs) deleted to keep the repo small.

# Oracle dumps (per fixture; JSON goes to stdout — `-o` is xlsx-only)
$CA blocks $B/<dir>/<name>.h265 --decoder-telemetry $B/<dir>/<name>.catb --format json > $B/<dir>/<name>_oracle_blocks.json
$CA frames $B/<dir>/<name>.h265 --decoder-telemetry $B/<dir>/<name>.catb --format json > $B/<dir>/<name>_oracle_frames.json
```

## SHA256

```
b1d4696df4e85d83067ce0945479ccd068c8e971d579ea7bfe4981afd52c883b  avc_mb/avc_cavlc.h264
dcf1cccff47cdd4021fe88ea05f9da246c8a496d74ca200235d9bbfefe90baf0  avc_mb/avc_cavlc.catb
d117d1159a31428a329280c2e4c9ab9f0e5807c4e2411e25c995e6a1bb2e6a12  hevc_bslice/hevc_bslice.h265
29d09023b8754f8c5a8caf588e7121fcea5ef684ffe11ff9f8afe1bd887bac94  hevc_bslice/hevc_bslice.catb
3bff733ed1a4bfd3c542a1ab3cbe1dd682dacb7362445c628bc3257fdecd15ff  hevc_bslice/hevc_bslice_64x64_i420.yuv
d1e9768e4f87a63776c60c3b6d54132c90dd4d3fb8d307d9be0f6a5ba15dda5d  hevc_inter/hevc_inter.h265
c90ca55de596712355bd924078f561a4fb765f2341b1a85e7d8ccd8697ffa9ab  hevc_inter/hevc_inter.catb
00385af88a5c4d9151edfe19113f4c09931f326c7540a9ebc48a44481f24af45  hevc_intra/hevc_intra.h265
77529a567ef0a4afe2a13a06b2087b053e8133388de75465bc790086bebf9a68  hevc_intra/hevc_intra.catb
2bf7ec2019567fb2c57d1477f984193be8684ab5b6dc5a0429d42bfc02c79745  hevc_multi_idr/hevc_multi_idr.h265
38f03bc445d75dbb304481d262ba7d5a3d1fe2361fdc9119cba80895a30852fc  hevc_multi_idr/hevc_multi_idr.catb
```

(Oracle JSON sha256 omitted — regenerate deterministically from the .catb with the commands above.)
