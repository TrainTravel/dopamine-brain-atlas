#!/usr/bin/env bash
# Downloads the four Neurosynth v7 files into ./neurosynth (about 15 MB).
set -euo pipefail
mkdir -p neurosynth
base=https://raw.githubusercontent.com/neurosynth/neurosynth-data/master
for f in data-neurosynth_version-7_coordinates.tsv.gz data-neurosynth_version-7_metadata.tsv.gz \
         data-neurosynth_version-7_vocab-terms_source-abstract_type-tfidf_features.npz \
         data-neurosynth_version-7_vocab-terms_vocabulary.txt; do
  [ -s "neurosynth/$f" ] || curl -fsSL -o "neurosynth/$f" "$base/$f"
done
echo "Neurosynth files ready in ./neurosynth"
