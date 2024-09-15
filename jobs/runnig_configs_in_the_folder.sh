#!/bin/bash

# Set the base directory for configs (passed as a command-line argument)
CONFIG_DIR=$1

# Ensure the script is run from the project root
PROJECT_ROOT="$(pwd)"

# Check if a directory is passed, otherwise exit
if [ -z "$CONFIG_DIR" ]; then
  echo "Usage: $0 <config_directory>"
  exit 1
fi

# Find all YAML files in the given directory and its subdirectories
find "$CONFIG_DIR" -type f -name "*.yaml" | while read config_file; do
  echo "Running experiment with config: $config_file"

  # Ensure the script is run from the root
  (cd "$PROJECT_ROOT" && python scripts/generic_train.py --config "$config_file")

  echo "Finished experiment with config: $config_file"
done
