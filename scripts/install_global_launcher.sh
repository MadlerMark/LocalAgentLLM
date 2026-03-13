#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$HOME/.local/bin"
TARGET="$TARGET_DIR/localagent"

mkdir -p "$TARGET_DIR"
cat > "$TARGET" <<EOF
#!/usr/bin/env bash
exec "$SCRIPT_DIR/localagent.sh" "\$@"
EOF
chmod +x "$TARGET"

echo "Installed: $TARGET"
echo "Ensure $TARGET_DIR is in PATH, then run: localagent help"
