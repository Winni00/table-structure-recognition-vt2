#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/cluster/home/trinhwin/vt2/docling}"
DATA_DIR="${DATA_DIR:-$ROOT/data/target_domain_train}"
STATUS_DIR="${STATUS_DIR:-$ROOT/results/target_domain_train_upload_status}"
MIN_FILES="${MIN_FILES:-80000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
STABLE_POLLS_REQUIRED="${STABLE_POLLS_REQUIRED:-6}"

mkdir -p "$STATUS_DIR"
status_file="$STATUS_DIR/status.tsv"
ready_file="$STATUS_DIR/READY"
previous_count=-1
stable_polls=0

printf 'timestamp\tfiles\tbytes\tstable_polls\n' > "$status_file"
rm -f "$ready_file"

while true; do
    file_count=$(find "$DATA_DIR" -type f | wc -l)
    byte_count=$(du -sk "$DATA_DIR" | awk '{print $1 * 1024}')

    if [[ "$file_count" -ge "$MIN_FILES" && "$file_count" -eq "$previous_count" ]]; then
        stable_polls=$((stable_polls + 1))
    else
        stable_polls=0
    fi

    printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$file_count" "$byte_count" "$stable_polls" >> "$status_file"

    if [[ "$stable_polls" -ge "$STABLE_POLLS_REQUIRED" ]]; then
        printf 'ready_at=%s\nfiles=%s\nbytes=%s\n' \
            "$(date --iso-8601=seconds)" "$file_count" "$byte_count" > "$ready_file"
        "$ROOT/.venv/bin/python" "$ROOT/scripts/inventory_split_target_domain_train.py" \
            --data-dir "$DATA_DIR" \
            --output-dir "$ROOT/results/target_domain_train_inventory_split" \
            > "$STATUS_DIR/inventory.log" 2>&1
        touch "$STATUS_DIR/INVENTORY_COMPLETE"
        exit 0
    fi

    previous_count="$file_count"
    sleep "$POLL_SECONDS"
done
