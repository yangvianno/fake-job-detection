#!/usr/bin/env bash

set -euo pipefail

echo "🔄 Running data prep…"
python -c "from src.data.make_dataset import make_dataset; make_dataset()"
echo "✅ Data ready in data/processed/"