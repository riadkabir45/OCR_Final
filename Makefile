SHELL := bash

genData:
	make clean
	python3 scripts/generate_damaged_dataset.py \
		--input-dir dataset/raw --output-dir dataset/damaged --seed 45 \
		--remove-overlapping-text --reduce-damage-boxes \
		--min-damage-pixels 20 --diff-threshold 150
	python3 scripts/preprocess_damaged.py \
		--input-dir dataset/damaged --output-dir dataset/processed
	python3 scripts/organize_dataset.py \
		--input-dir dataset/processed --output-dir dataset/textDataset --seed 45
	python3 scripts/organize_dataset.py \
		--input-dir dataset/damaged --output-dir dataset/boxDataset --seed 45
	

clean:
	export GLOBIGNORE="dataset/raw" && rm -rf dataset/* && unset GLOBIGNORE