#!/usr/bin/env bash
set -euo pipefail

target=x86_64-pc-windows-gnu
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
crate_dir="$repo_root/rust"
out_dir="$repo_root/dist"
out="$out_dir/video-viewer-windows-x86_64.exe"
build_user="$(id -un)"

if [[ "$(id -u)" -eq 0 ]]; then
  sudo_cmd=()
  if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
    build_user="$SUDO_USER"
  fi
elif command -v sudo >/dev/null 2>&1; then
  sudo_cmd=(sudo)
else
  echo "sudo is required to install MinGW packages" >&2
  exit 1
fi

build_home="$(getent passwd "$build_user" | cut -d: -f6)"
build_path="$build_home/.cargo/bin:$PATH"

as_build_user() {
  if [[ "$(id -u)" -eq 0 && "$build_user" != "root" ]]; then
    runuser -u "$build_user" -- env \
      HOME="$build_home" \
      USER="$build_user" \
      LOGNAME="$build_user" \
      PATH="$build_path" \
      "$@"
  else
    PATH="$build_path" "$@"
  fi
}

as_build_user rustup target add "$target"

"${sudo_cmd[@]}" apt-get update
"${sudo_cmd[@]}" apt-get install -y gcc-mingw-w64-x86-64 binutils-mingw-w64-x86-64

as_build_user cargo build --manifest-path "$crate_dir/Cargo.toml" --release --target "$target"

as_build_user mkdir -p "$out_dir"
as_build_user cp "$crate_dir/target/$target/release/video-viewer.exe" "$out"
file "$out"
echo "$out"
