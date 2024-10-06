#!/bin/bash

# Path to directory containing speakers: their audio and text files
base_dataset_dir="/media/newhddd/SpeechDatasets/TTS/syspin/files_16000"

# Path to store alignments, checkpoints, tokens, logs
base_data_dir="runs/aligner"

# Path to the config file
config_file="utils/aligner/aligner_train_config.yaml"

# Iterate over all subdirectories in base_dataset_dir (speakers)
for speaker in "$base_dataset_dir"/*; do
  # Extract the speaker name (basename of the directory)
  speaker_name=$(basename "$speaker")

  # Modify dataset_dir and data_dir in the YAML config
  sed -i "s|dataset_dir:.*|dataset_dir: ${base_dataset_dir}/${speaker_name}|g" "$config_file"
  sed -i "s|data_dir:.*|data_dir: ${base_data_dir}/${speaker_name}|g" "$config_file"

  Extract mels and tokens
  python utils/aligner/character_preprocess.py --config "$config_file"

  Train aligner
  python utils/aligner/train.py --config "$config_file"

  # extract durations
  python utils/aligner/extract_durations.py --config "$config_file"

done
