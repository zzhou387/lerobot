#!/usr/bin/env bash
#
# push_to_s3.sh
#
# Upload files from a local directory to an S3 path.
# Parallel multipart transfers for speed on large files.
# Mirror of fetch_from_s3.sh.
#
# Usage:
#   ./push_to_s3.sh <local_dir> <s3_path>
#   ./push_to_s3.sh --dry-run <local_dir> <s3_path>
#   ./push_to_s3.sh --delete <local_dir> <s3_path>      # also remove remote
#                                                        # files not present locally
#
# Arguments:
#   local_dir  – Local source directory.
#   s3_path    – Destination prefix on S3 (e.g. "manav/outputs/dp_edge_slide").
#                Writes to  s3://$S3_BUCKET/<s3_path>/...
#
# Examples:
#   ./push_to_s3.sh /data/lerobot/outputs/eval/dp_pilot100_step30000  manav/eval/edge_slide_step30000
#   ./push_to_s3.sh --dry-run ~/checkpoints/dp_v1                     manav/checkpoints/dp_v1
#
# Safety notes:
#   - Default sync is ADD/UPDATE: existing remote files with the same
#     content stay; remote files NOT present locally are kept untouched.
#     Pass --delete to remove remote-only files (one-way mirror).
#   - Uploads require s3:PutObject permission on the bucket. The script
#     calls "aws s3api head-bucket" first to fail fast on missing access.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.conf"

LOG_FILE="$SCRIPT_DIR/push_to_s3.log"
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"; }
die() { log "FATAL: $*"; exit 1; }

cleanup() {
    local code=$?
    if [[ $code -ne 0 ]]; then
        log "Upload interrupted or failed (exit code $code)."
        log "Re-run the same command to resume – aws s3 sync is incremental."
    fi
}
trap cleanup EXIT

DRY_RUN=""
DELETE_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN="--dryrun"; shift ;;
        --delete)  DELETE_FLAG="--delete"; shift ;;
        --) shift; break ;;
        -*) die "Unknown flag: $1" ;;
        *) break ;;
    esac
done

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 [--dry-run] [--delete] <local_dir> <s3_path>"
    echo "  local_dir  – local source directory"
    echo "  s3_path    – destination prefix on S3 (e.g. manav/eval/edge_slide_step30000)"
    exit 1
fi

LOCAL_DIR="${1%/}"
S3_PATH="${2%/}"

[[ -d "$LOCAL_DIR" ]] || die "Local dir does not exist: $LOCAL_DIR"

command -v aws >/dev/null 2>&1 || die "aws CLI not found. Install with: pip install awscli"
aws sts get-caller-identity >/dev/null 2>&1 || die "AWS credentials not configured. Run: aws configure"

aws s3api head-bucket --bucket "$S3_BUCKET" --region "$S3_REGION" 2>/dev/null \
    || die "Cannot access bucket s3://$S3_BUCKET – check permissions."

DST="s3://${S3_BUCKET}/${S3_PATH}/"

# Local stats
FILE_COUNT=$(find "$LOCAL_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
TOTAL_SIZE=$(du -sb "$LOCAL_DIR" 2>/dev/null | awk '{print $1}' || echo "?")
TOTAL_SIZE_HUMAN="$TOTAL_SIZE bytes"
if [[ "$TOTAL_SIZE" =~ ^[0-9]+$ ]]; then
    TOTAL_SIZE_HUMAN=$(numfmt --to=iec "$TOTAL_SIZE" 2>/dev/null || echo "$TOTAL_SIZE bytes")
fi

if [[ "$FILE_COUNT" == "0" ]]; then
    die "No files found in $LOCAL_DIR – nothing to upload."
fi

log "=== S3 Upload ==="
log "  Source : $LOCAL_DIR  ($FILE_COUNT files, $TOTAL_SIZE_HUMAN)"
log "  Dest   : $DST"
log "  Region : $S3_REGION"
[[ -n "$DRY_RUN" ]] && log "  Mode   : DRY-RUN (no files will be uploaded)"
[[ -n "$DELETE_FLAG" ]] && log "  Mirror : --delete enabled (remote-only files will be DELETED)"

aws s3 sync "$LOCAL_DIR" "$DST" \
    --region "$S3_REGION" \
    --only-show-errors \
    --cli-connect-timeout 30 \
    --cli-read-timeout 60 \
    $DRY_RUN \
    $DELETE_FLAG \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [[ $EXIT_CODE -eq 0 ]]; then
    log "Upload complete."
else
    die "aws s3 sync exited with code $EXIT_CODE."
fi

log "=== Done ==="
