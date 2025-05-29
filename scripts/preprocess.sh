#!/usr/bin/env bash

set -euo pipefail

echo "🔄 Running data prep…"
python -m src.data.make_dataset
echo "✅ Data ready in data/processed/"