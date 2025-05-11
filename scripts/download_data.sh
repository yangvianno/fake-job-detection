#!/usr/bin/env bash
set -e

mkdir -p data/raw

kaggle datasets download shivamb/real-or-fake-fake-jobposting-prediction \
  -p data/raw \
  --unzip

echo "✅ Dataset downloaded & extracted to data/raw/"