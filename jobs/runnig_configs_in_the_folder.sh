#!/bin/bash

# Ensure the script is run from the project root (one level up from jobs)
PROJECT_ROOT="$(dirname "$(dirname "$0")")"

# Set the base directory for configs (passed as a command-line argument)
CONFIG_DIR="$1"

# Check if a directory is passed, otherwise exit
if [ -z "$CONFIG_DIR" ]; then
  echo "Usage: $0 <config_directory>"
  exit 1
fi

# Find all files in the given directory and its subdirectories
find "$PROJECT_ROOT/$CONFIG_DIR" -type f | while read config_file; do
  # Check if the file has a .yaml extension
  if [[ $config_file == *.yaml ]]; then
    echo "Running experiment with config: $config_file"

    # Run the Python script from the project root with the found YAML config
    (cd "$PROJECT_ROOT" && python scripts/generic_train.py --config "$config_file")

    echo "Finished experiment with config: $config_file"
  else
    echo "Skipping non-YAML file: $config_file"
  fi
done
