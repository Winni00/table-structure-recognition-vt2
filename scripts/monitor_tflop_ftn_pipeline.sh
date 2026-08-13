#!/bin/bash
set -uo pipefail

cd /cluster/home/trinhwin/vt2/docling

JOB_IDS="${1:-26409,26410,26411,26412,26413,26414}"
INTERVAL="${2:-300}"
OUT_DIR=/cluster/home/trinhwin/vt2/docling/results/tflop_fintabnet_pipeline_monitor
mkdir -p "$OUT_DIR"
STATE_FILE="$OUT_DIR/notified_events.txt"
touch "$STATE_FILE"

notify() {
  local key="$1"
  local title="$2"
  local body="$3"

  if rg -Fx "$key" "$STATE_FILE" >/dev/null 2>&1; then
    return 0
  fi
  echo "$key" >> "$STATE_FILE"

  {
    echo "[$(date --iso-8601=seconds)] $title"
    echo "$body"
    echo
  } >> "$OUT_DIR/notifications.log"

  # Optional phone push via ntfy. Put only the topic name in:
  # results/tflop_fintabnet_pipeline_monitor/ntfy_topic.txt
  if [ -s "$OUT_DIR/ntfy_topic.txt" ] && command -v curl >/dev/null 2>&1; then
    local topic
    topic=$(tr -d '[:space:]' < "$OUT_DIR/ntfy_topic.txt")
    if [ -n "$topic" ]; then
      curl -fsS \
        -H "Title: $title" \
        -H "Priority: high" \
        -d "$body" \
        "https://ntfy.sh/$topic" >/dev/null 2>&1 || true
    fi
  fi

  # Optional email notification. Put the email address in:
  # results/tflop_fintabnet_pipeline_monitor/email.txt
  if [ -s "$OUT_DIR/email.txt" ] && command -v mail >/dev/null 2>&1; then
    local email
    email=$(tr -d '[:space:]' < "$OUT_DIR/email.txt")
    if [ -n "$email" ]; then
      printf '%s\n' "$body" | mail -s "$title" "$email" >/dev/null 2>&1 || true
    fi
  fi
}

while true; do
  ts=$(date --iso-8601=seconds)
  status_tmp="$OUT_DIR/status.tmp"
  {
    echo "timestamp: $ts"
    echo
    echo "## squeue"
    squeue -u "$USER" -o '%.18i %.12P %.25j %.8u %.10a %.2t %.10M %.10l %R' || true
    echo
    echo "## sacct"
    sacct -j "$JOB_IDS" --format=JobID,JobName%28,Partition,State,Elapsed,ExitCode,NodeList%20 -P 2>/dev/null || true
    echo
    echo "## recent errors"
    for log in logs/*{ftn10k,ftn50k,full100k}*.err; do
      [ -f "$log" ] || continue
      if rg -i "traceback|error|failed|outofmemory|out of memory|cuda|oom|dependencyneversatisfied" "$log" >/dev/null 2>&1; then
        echo "--- $log"
        tail -n 80 "$log"
      fi
    done
  } > "$status_tmp"
  mv "$status_tmp" "$OUT_DIR/latest_status.txt"
  cp "$OUT_DIR/latest_status.txt" "$OUT_DIR/status_${ts//:/-}.txt"

  sacct_data=$(sacct -j "$JOB_IDS" --format=JobID,JobName%28,State,ExitCode -P 2>/dev/null || true)
  while IFS='|' read -r job_id job_name state exit_code; do
    [ "$job_id" = "JobID" ] && continue
    [ -z "${job_id:-}" ] && continue
    case "$state" in
      FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*)
        notify "bad-${job_id}-${state}-${exit_code}" \
          "TFLOP pipeline problem: ${job_name}" \
          "Job ${job_id} (${job_name}) is ${state}, exit=${exit_code}. Check $OUT_DIR/latest_status.txt"
        ;;
      COMPLETED)
        case "$job_name" in
          ftn10k_teds_merge|ftn50k_teds_merge|tflop_ftn_full100k)
            notify "done-${job_id}-${job_name}" \
              "TFLOP pipeline step completed: ${job_name}" \
              "Job ${job_id} (${job_name}) completed. Check $OUT_DIR/latest_status.txt"
            ;;
        esac
        ;;
    esac
  done <<< "$sacct_data"

  sleep "$INTERVAL"
done
