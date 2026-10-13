#!/bin/sh
# Stands between PlantUML and Graphviz dot. render.py points GRAPHVIZ_DOT at this script.
#
# PlantUML 1.2025.9 does not read dot's exit code. When dot ends without writing a byte (it died
# or was killed), PlantUML calls it once more in another layout mode and, if that call is empty
# too, draws a "For some reason, dot/GraphViz has crashed" picture in place of the diagram, in the
# diagram's own scale and colours, with exit code 0 and nothing on stderr.
#
# This script hands every call to the real dot and passes its output on unchanged. When the real
# dot returns no output it prints one error line; PlantUML then draws its generic crash picture,
# which render.py recognises.
#
# RENDER_REAL_DOT: path of the real dot (set by render.py).
real="${RENDER_REAL_DOT:?RENDER_REAL_DOT is not set}"
for arg in "$@"; do
  [ "$arg" = "-V" ] && exec "$real" "$@"   # the version query answers on stderr only
done
out=$(mktemp "${TMPDIR:-/tmp}/dot_guard.XXXXXX") || exit 1
trap 'rm -f "$out"' EXIT
"$real" "$@" > "$out"
code=$?
[ -s "$out" ] || echo "Error: dot ended without output (exit code $code)"
cat "$out"
exit $code
