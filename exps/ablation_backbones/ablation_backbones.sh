#!/bin/bash

JSON_FOLDER="./exps/ablation_backbones"

for json_file in "$JSON_FOLDER"/*.json; do
    echo "Processing file: $json_file" 
    python main.py --config="$json_file"
done
