#!/usr/bin/env bash
# Sample whole-card GPU telemetry while a workflow runs.
#
# Rank telemetry in forward_*.json only sees one process's torch allocator: it
# cannot see the other rank, the NCCL buffers, the CUDA context or the cuBLAS
# workspace.  Deciding "how much room is left on this card" therefore requires
# nvidia-smi.  The 1 s granularity misses shorter spikes, so treat the reported
# peak as a lower bound and keep real headroom above it.
#
# Usage:
#   scripts/sample_gpu_during_run.sh <case-name> -- <command...>
#   scripts/sample_gpu_during_run.sh <case-name> --pid <pid>
#
# Examples:
#   scripts/sample_gpu_during_run.sh 720p10s -- \
#       .venv/bin/python scripts/submit_workflow.py wf.json --wait
#   scripts/sample_gpu_during_run.sh idle --seconds 30

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${H3_TP_RESULTS_DIR:-$REPO_ROOT/results}"
INTERVAL="${H3_SMI_INTERVAL:-1}"

usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    exit "${1:-0}"
}

[[ $# -ge 1 ]] || usage 2
case "$1" in
    -h|--help) usage 0 ;;
esac

CASE_NAME="$1"
shift

WAIT_PID=""
WAIT_SECONDS=""
COMMAND=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pid)     WAIT_PID="${2:?--pid needs a value}"; shift 2 ;;
        --seconds) WAIT_SECONDS="${2:?--seconds needs a value}"; shift 2 ;;
        --)        shift; COMMAND=("$@"); break ;;
        *)         echo "Unexpected argument: $1" >&2; usage 2 ;;
    esac
done

if [[ -z "$WAIT_PID" && -z "$WAIT_SECONDS" && ${#COMMAND[@]} -eq 0 ]]; then
    echo "Nothing to observe: give -- <command>, --pid, or --seconds" >&2
    exit 2
fi

command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is required" >&2
    exit 1
}

mkdir -p "$RESULTS_DIR"
CSV="$RESULTS_DIR/e2e_smi_${CASE_NAME}.csv"

# Header is written separately: --format=csv,noheader keeps the stream clean for
# the summary below, but a bare CSV with no header is hostile to later readers.
printf 'index,timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.current.sm,power.draw\n' > "$CSV"

nvidia-smi \
    --query-gpu=index,timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.current.sm,power.draw \
    --format=csv,noheader,nounits -l "$INTERVAL" >> "$CSV" 2>/dev/null &
SMI_PID=$!

cleanup() {
    if kill -0 "$SMI_PID" 2>/dev/null; then
        kill "$SMI_PID" 2>/dev/null || true
        wait "$SMI_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

STATUS=0
if [[ ${#COMMAND[@]} -gt 0 ]]; then
    echo "[smi] sampling into $CSV while running: ${COMMAND[*]}" >&2
    "${COMMAND[@]}" || STATUS=$?
elif [[ -n "$WAIT_PID" ]]; then
    echo "[smi] sampling into $CSV while pid $WAIT_PID is alive" >&2
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep "$INTERVAL"; done
else
    echo "[smi] sampling into $CSV for ${WAIT_SECONDS}s" >&2
    sleep "$WAIT_SECONDS"
fi

cleanup
trap - EXIT INT TERM

echo "[smi] wrote $CSV" >&2

# Peak per card.  max(memory.used) is the number that decides whether a layout
# fits; the mean is reported only to show how long the card sat near its peak.
awk -F', *' '
    NR > 1 && $1 ~ /^[0-9]+$/ {
        used[$1] = $3 + 0
        if (used[$1] > peak[$1]) peak[$1] = used[$1]
        total[$1] = $4 + 0
        sum[$1] += $3 + 0
        n[$1] += 1
        if ($6 + 0 > temp[$1]) temp[$1] = $6 + 0
        if ($7 + 0 > 0 && ($7 + 0 < clkmin[$1] || clkmin[$1] == 0)) clkmin[$1] = $7 + 0
        if ($7 + 0 > clkmax[$1]) clkmax[$1] = $7 + 0
    }
    END {
        printf "\n%-7s %11s %11s %11s %9s %14s\n", \
               "device", "peak_MiB", "free_MiB", "mean_MiB", "max_degC", "sm_clock_MHz"
        for (i in peak) {
            printf "cuda:%-2s %11.0f %11.0f %11.0f %9.0f %6.0f-%-7.0f\n", \
                   i, peak[i], total[i] - peak[i], sum[i] / n[i], temp[i], clkmin[i], clkmax[i]
        }
        printf "\nsamples: %d per card at %s s interval\n", n[0], INTERVAL
    }
' INTERVAL="$INTERVAL" "$CSV" >&2

exit "$STATUS"
