#!/bin/bash

JSON_FOLDER="./exps/cross_domain"

for json_file in "$JSON_FOLDER"/*.json; do
    echo "Processing file: $json_file" 
    python main.py --config="$json_file"
done
