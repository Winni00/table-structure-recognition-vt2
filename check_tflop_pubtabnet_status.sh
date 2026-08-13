#!/usr/bin/env bash

# Show the current status of the detached TFLOP PubTabNet subset rerun.
#
# This helper reads the detached tmux session, reports whether the Python
# process is still alive, and prints the latest visible progress lines from the
# tmux pane. It is meant to be safe to run repeatedly while the benchmark is
# running in the background on the cluster.

set -euo pipefail

SESSION_NAME="tflop_pubtabnet_subset"
RUN_DIR="/cluster/home/trinhwin/vt2/docling/results/tflop_pubtabnet_official_repro/subset_1008"
RUN_LOG="${RUN_DIR}/run.log"

if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session not found: ${SESSION_NAME}"
  exit 1
fi

echo "Session: ${SESSION_NAME}"

PY_PID="$(pgrep -f "save_dir ${RUN_DIR}" | tail -n 1 || true)"
if [[ -n "${PY_PID}" ]]; then
  echo
  echo "Process"
  ps -p "${PY_PID}" -o pid,etime,time,%cpu,%mem,stat,cmd
else
  echo
  echo "Process: no active Python process found for ${RUN_DIR}"
fi

echo
echo "Latest tmux output"
tmux capture-pane -p -t "${SESSION_NAME}" | tail -n 20

echo
echo "Run log"
if [[ -f "${RUN_LOG}" ]]; then
  if [[ -s "${RUN_LOG}" ]]; then
    tail -n 20 "${RUN_LOG}"
  else
    echo "Log file exists but is currently empty: ${RUN_LOG}"
  fi
else
  echo "Log file not found yet: ${RUN_LOG}"
fi
