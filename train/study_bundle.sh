#!/bin/bash
# The study's commit as a git bundle for the card, with the study's tags: a scored run's manifest
# checks the serving files against the tag FREEZE_TAG of infer/manifest.py (`frozen_files`), so the
# card's checkout must hold that tag (methodology/compute-and-serving.md §8, rule 2).
#
#   study_bundle.sh create <repository> <bundle>   refuse unless FREEZE_TAG exists in <repository>;
#                                                   bundle HEAD and every tag
#   study_bundle.sh check <checkout>               refuse unless FREEZE_TAG exists in <checkout>
#
# train/card_session.sh `pack` runs `create` on the Mac; train/pod_setup.sh runs `check` on the
# card after the checkout. Tests: train/tests/test_study_bundle.py
set -euo pipefail
step=${1:?create or check}; repo=${2:?repository}

freeze_tag() {
    local tag
    tag=$(sed -n 's/^FREEZE_TAG = "\(.*\)"/\1/p' "$repo/infer/manifest.py")
    [ -n "$tag" ] || { echo "no FREEZE_TAG in $repo/infer/manifest.py" >&2; exit 1; }
    git -C "$repo" rev-parse -q --verify "refs/tags/$tag" >/dev/null \
        || { echo "tag $tag is not in $repo; infer/manifest.py's frozen_files reads it" >&2; exit 1; }
    echo "$tag"
}

case $step in
    create)
        bundle=${3:?bundle}
        bundle=$(cd "$(dirname "$bundle")" && pwd)/$(basename "$bundle")   # git -C would resolve it in <repository>
        tag=$(freeze_tag)
        git -C "$repo" bundle create "$bundle" HEAD --tags 2>/dev/null
        echo "bundle: HEAD $(git -C "$repo" rev-parse HEAD) and $(git bundle list-heads "$bundle" | grep -c refs/tags/) tags, $tag among them" ;;
    check)
        tag=$(freeze_tag)
        echo "tag $tag present" ;;
    *)  echo "unknown step $step"; exit 2 ;;
esac
