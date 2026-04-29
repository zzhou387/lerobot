#!/usr/bin/env bash
#
# fetch_from_s3.sh
#
# Download files from an S3 path to a local directory.
# Parallel multipart transfers for speed on large files.
#
# Usage:
#   ./fetch_from_s3.sh <s3_path> <local_dir>
#   ./fetch_from_s3.sh --dry-run <s3_path> <local_dir>
#
# Arguments:
#   s3_path    – Source prefix on S3 (e.g. "manav/compressed" or
#                "manav/compressed/M_001").
#                Reads from  s3://$S3_BUCKET/<s3_path>/...
#   local_dir  – Local directory to save files into.
#
# Examples:
#   ./fetch_from_s3.sh manav/compressed          ~/unix_runs/from_s3
#   ./fetch_from_s3.sh --dry-run luca/raw/M_042  /tmp/m42

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.conf"

LOG_FILE="$SCRIPT_DIR/fetch_from_s3.log"
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"; }
die() { log "FATAL: $*"; exit 1; }

cleanup() {
    local code=$?
    if [[ $code -ne 0 ]]; then
        log "Download interrupted or failed (exit code $code)."
        log "Re-run the same command to resume – aws s3 sync is incremental."
    fi
}
trap cleanup EXIT

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dryrun"
    shift
fi

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 [--dry-run] <s3_path> <local_dir>"
    echo "  s3_path    – source prefix on S3 (e.g. manav/compressed)"
    echo "  local_dir  – local directory to save into"
    exit 1
fi

S3_PATH="${1%/}"
LOCAL_DIR="$2"

command -v aws >/dev/null 2>&1 || die "aws CLI not found. Install with: pip install awscli"
aws sts get-caller-identity >/dev/null 2>&1 || die "AWS credentials not configured. Run: aws configure"

aws s3api head-bucket --bucket "$S3_BUCKET" --region "$S3_REGION" 2>/dev/null \
    || die "Cannot access bucket s3://$S3_BUCKET – check permissions."

SRC="s3://${S3_BUCKET}/${S3_PATH}/"

OBJECT_COUNT=$(aws s3 ls "$SRC" --region "$S3_REGION" --recursive --summarize 2>/dev/null \
    | grep "Total Objects:" | awk '{print $3}' || echo "?")
TOTAL_SIZE=$(aws s3 ls "$SRC" --region "$S3_REGION" --recursive --summarize 2>/dev/null \
    | grep "Total Size:" | awk '{print $3}' || echo "?")

if [[ "$OBJECT_COUNT" == "0" ]]; then
    die "No objects found at $SRC – nothing to download."
fi

TOTAL_SIZE_HUMAN="$TOTAL_SIZE bytes"
if [[ "$TOTAL_SIZE" =~ ^[0-9]+$ ]]; then
    TOTAL_SIZE_HUMAN=$(numfmt --to=iec "$TOTAL_SIZE" 2>/dev/null || echo "$TOTAL_SIZE bytes")
fi

AVAIL_SPACE=$(df --output=avail -B1 "$(dirname "$LOCAL_DIR")" 2>/dev/null | tail -1 | tr -d ' ' || echo "")
if [[ -n "$AVAIL_SPACE" && "$TOTAL_SIZE" =~ ^[0-9]+$ ]]; then
    MARGIN=$((TOTAL_SIZE + TOTAL_SIZE / 10))
    if [[ "$AVAIL_SPACE" -lt "$MARGIN" ]]; then
        AVAIL_HUMAN=$(numfmt --to=iec "$AVAIL_SPACE" 2>/dev/null || echo "$AVAIL_SPACE bytes")
        die "Insufficient disk space. Need ~$TOTAL_SIZE_HUMAN but only $AVAIL_HUMAN available at $(dirname "$LOCAL_DIR")."
    fi
fi

mkdir -p "$LOCAL_DIR"

log "=== S3 Download ==="
log "  Source : $SRC  ($OBJECT_COUNT objects, $TOTAL_SIZE_HUMAN)"
log "  Dest   : $LOCAL_DIR"
log "  Region : $S3_REGION"
[[ -n "$DRY_RUN" ]] && log "  Mode   : DRY-RUN (no files will be downloaded)"

aws s3 sync "$SRC" "$LOCAL_DIR" \
    --region "$S3_REGION" \
    --exact-timestamps \
    --only-show-errors \
    --cli-connect-timeout 30 \
    --cli-read-timeout 60 \
    $DRY_RUN \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [[ $EXIT_CODE -eq 0 ]]; then
    log "Download complete."
else
    die "aws s3 sync exited with code $EXIT_CODE."
fi

log "=== Done ==="