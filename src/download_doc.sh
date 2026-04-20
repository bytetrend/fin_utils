#!/usr/bin/env bash
set -euo pipefail

# To download all documents from a URL (Usually pointing to a directory listing), you can use wget with the appropriate options.
# Usage: ./download_docs.sh URL [DEST_DIR] [EXTENSIONS]
# Example: ./download_docs.sh 'https://example.com/path/' '/tmp/downloads' 'pdf,docx,doc,xls,xlsx,txt,csv,ppt,pptx'

if ! command -v wget >/dev/null 2>&1; then
  echo "Error: wget is required but not installed." >&2
  exit 1
fi

if [ $# -lt 1 ]; then
  echo "Usage: $0 URL [DEST_DIR] [EXTENSIONS]" >&2
  exit 1
fi

URL="$1"
DEST="${2:-./downloads}"
EXTENSIONS="${3:-pdf,doc,docx,xls,xlsx,txt,csv,ppt,pptx}"

# Normalize destination and ensure it exists
mkdir -p "$DEST"

# Ensure URL ends with a slash so wget mirrors the path correctly
case "$URL" in
  */) ;;
  *) URL="${URL}/" ;;
esac

echo "Downloading files matching: $EXTENSIONS"
echo "From: $URL"
echo "To: $DEST"

# wget options:
#  -r, --recursive        : recursive download
#  -l 0                   : infinite recursion depth (0 = infinite)
#  -np, --no-parent       : do not ascend to parent directories
#  -nH, --no-host-directories : do not create host folder
#  -P, --directory-prefix : save under destination root
#  --accept               : comma-separated list of accepted extensions
#  --restrict-file-names=unix : normalize filenames for UNIX
wget --recursive --level=0 --no-parent --no-host-directories \
     --directory-prefix="$DEST" --accept="$EXTENSIONS" \
     --restrict-file-names=unix "$URL"