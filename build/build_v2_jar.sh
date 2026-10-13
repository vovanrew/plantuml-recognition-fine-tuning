#!/bin/sh
# Assemble the scorer-v2 extractor JAR without Gradle: the v1 JAR with the two classes changed on
# fork branch stats-extractor-graph recompiled and replaced in place. Every other entry of the
# archive is carried over untouched from the v1 JAR. The two entries get a fixed time and file mode,
# so the same sources and the same javac give the same JAR, byte for byte.
#   build_v2_jar.sh <fork checkout> <v1 jar> <out jar>
# Needs a JDK whose javac accepts --release 8 (class-file version 52, as the v1 JAR) and `zip`.
set -eu
FORK=$1; V1=$2
OUT=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
SRC=$FORK/src/main/java/net/sourceforge/plantuml
A=net/sourceforge/plantuml/stats/DiagramStatsExtractor.class
B=net/sourceforge/plantuml/klimt/creole/CreoleHorizontalLine.class
WORK=$(mktemp -d)
javac --release 8 -g -encoding UTF-8 -Xlint:-options -cp "$V1" -d "$WORK" \
  "$SRC/stats/DiagramStatsExtractor.java" "$SRC/klimt/creole/CreoleHorizontalLine.java"
# zip stores an entry's time and file mode: fix both, so the result does not depend on the clock or the umask
TZ=UTC touch -t 202610040000.00 "$WORK/$A" "$WORK/$B"
chmod 644 "$WORK/$A" "$WORK/$B"
cp "$V1" "$OUT"
(cd "$WORK" && TZ=UTC zip -q -X "$OUT" "$A" "$B")
rm -rf "$WORK"
shasum -a 256 "$OUT"
