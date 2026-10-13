#!/bin/sh
# Test stand-in for Graphviz dot: the real dot, except that for a graph of at least
# DYING_DOT_MIN_BYTES bytes of dot text (default 1000) it ends without output and with exit code
# 139, as a dot killed by SIGSEGV does. With DYING_DOT_DEATHS=n only the first n such calls die;
# unset, every such call dies. The version query and smaller graphs (PlantUML's -testdot, a
# two-class diagram) go to the real dot.
# DYING_DOT_REAL: the real dot. DYING_DOT_STATE: a directory for the graph text and the count.
real="${DYING_DOT_REAL:?DYING_DOT_REAL is not set}"
for arg in "$@"; do
  [ "$arg" = "-V" ] && exec "$real" "$@"
done
state="${DYING_DOT_STATE:?DYING_DOT_STATE is not set}"
mkdir -p "$state"
graph=$(mktemp "$state/graph.XXXXXX") || exit 1
cat > "$graph"
size=$(wc -c < "$graph" | tr -d ' ')
if [ "$size" -ge "${DYING_DOT_MIN_BYTES:-1000}" ]; then
  deaths=$(ls "$state" | grep -c '^death\.')
  if [ -z "$DYING_DOT_DEATHS" ] || [ "$deaths" -lt "$DYING_DOT_DEATHS" ]; then
    : > "$state/death.$$"
    rm -f "$graph"
    exit 139
  fi
fi
"$real" "$@" < "$graph"
code=$?
rm -f "$graph"
exit $code
