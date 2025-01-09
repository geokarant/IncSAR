#!/bin/bash

JSON_FOLDER="./exps/basic_experiments/mstar_b4inc1"

for json_file in "$JSON_FOLDER"/*.json; do
    echo "Processing file: $json_file" 
    python main.py --config="$json_file"
done