#!/bin/bash

JSON_FOLDER="./exps/basic_experiments/mstar_b2inc2"

for json_file in "$JSON_FOLDER"/*.json; do
    echo "Processing file: $json_file" 
    python main.py --config="$json_file"
done
