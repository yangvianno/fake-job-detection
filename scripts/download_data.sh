#!/usr/bin/env bash
pip install kaggle

kaggle datasets download -d shivamb/real-or-fake-fake-jobposting-prediction \
  -p data/raw --unzip
