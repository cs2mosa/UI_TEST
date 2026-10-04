#!/bin/bash
# =============================================================================
# Docker entrypoint — starts Xvfb (virtual display) then dispatches commands
# =============================================================================
set -e

# Start virtual display for PySide6/Qt
Xvfb :99 -screen 0 1920x1080x24 -ac &
XVFB_PID=$!
export DISPLAY=:99

# Wait until Xvfb is actually accepting connections (up to 10 s)
for i in $(seq 1 20); do
    xdpyinfo -display :99 >/dev/null 2>&1 && break
    sleep 0.5
done
if ! xdpyinfo -display :99 >/dev/null 2>&1; then
    echo "[entrypoint] ERROR: Xvfb failed to start on :99" >&2
    exit 1
fi

# Graceful cleanup on exit
cleanup() {
    echo "[entrypoint] Shutting down Xvfb..."
    kill "$XVFB_PID" 2>/dev/null || true
}
trap cleanup EXIT

# -------------------------------------------------------------------------
# Dispatch based on first argument
# -------------------------------------------------------------------------
CMD="$1"
shift || true   # shift off the command; remaining args passed through

case "$CMD" in

    # Run the benchmark comparison
    benchmark)
        echo "[entrypoint] Running compare_latency.py $*"
        exec python /workspace/compare_latency.py "$@"
        ;;

    # Launch the HERMES desktop UI
    ui-hermes)
        echo "[entrypoint] Launching HERMES UI $*"
        exec python /workspace/vlm_test_ui/main.py --hermes "$@"
        ;;

    # Launch the InfiniPot-V desktop UI
    ui-infinipot)
        echo "[entrypoint] Launching InfiniPot-V UI $*"
        exec python /workspace/temper/vlm_test_ui/main.py --qwen "$@"
        ;;

    # Launch either UI with Mock VLM (no model download needed)
    ui-mock)
        echo "[entrypoint] Launching HERMES UI with Mock VLM"
        exec python /workspace/vlm_test_ui/main.py "$@"
        ;;

    # Interactive shell
    bash|sh)
        exec bash
        ;;

    # Help / unknown
    --help|help|"")
        cat <<'EOF'
URCA_PROJECTS Docker Container
================================
Usage: docker run [docker-opts] urca-vlm <COMMAND> [args...]

Commands:
  benchmark     Run the latency benchmark (compare_latency.py)
                Example: benchmark --video /data/video.mp4 --num-chunks 5

  ui-hermes     Launch the HERMES desktop UI (requires X11 or Xvfb)
                Example: ui-hermes --model-path /workspace/.hf_cache/hub/...

  ui-infinipot  Launch the InfiniPot-V desktop UI
                Example: ui-infinipot --model-path Qwen/Qwen3-VL-4B-Instruct

  ui-mock       Launch the UI with Mock VLM — no GPU or model needed
                Example: ui-mock

  bash          Open an interactive shell inside the container

Environment variables you can override with -e:
  HF_HOME            HuggingFace cache dir     (default: /workspace/.hf_cache)
  DISPLAY            X11 display               (default: :99 / Xvfb)
  HF_TOKEN           HuggingFace access token  (for gated models)

Mount points:
  /data              Place your .mp4 video files here
  /workspace/.hf_cache   Persist downloaded model weights here (recommended)
  /results           Benchmark output JSON files are written here
EOF
        ;;

    # Pass-through: run any arbitrary command
    *)
        exec "$CMD" "$@"
        ;;
esac
