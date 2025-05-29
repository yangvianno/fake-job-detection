.PHONY: download preprocess train serve clean

download:
	scripts/download_data.sh

preprocess:
	scripts/preprocess.sh

train:
	scripts/train.sh

serve:
	uvicorn src.api.app:app --reload

clean:
	rm -rf data/processed models/production
