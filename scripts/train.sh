#!/usr/bin/env bash
set -euo pipefail

echo "🏋️‍♀️ Starting training…"
python src/train.py \
  --model_type mlp \
  --batch_size 32 \
  --epochs 5 \
  --dropout 0.5 \
  --min_recall 0.85 \
  --decision_threshold 0.5
echo "✅ Training complete, model in models/production/"