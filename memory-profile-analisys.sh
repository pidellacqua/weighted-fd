#!/usr/bin/env bash
RESULTS_PATH="../ignore/outputs/memory-evaluation"

for dir_path in "$RESULTS_PATH"/*; do
    if [ -d "$dir_path" ]; then
        dirname=$(basename "$dir_path")

        # Split tokens con underscore
        IFS='_' read -ra tokens <<< "$dirname"

        algorithm="${tokens[0]}"
        dataset="${tokens[1]}"
        dataset_type="${tokens[-2]}"
        seed="${tokens[-1]}"

        for file in "$dir_path"/*.dat; do
            if [ -f "$file" ]; then
                echo "Plotting memory profile from: $file"
                
                # qui puoi aggiungere il comando che fa il plot, es:
                mprof plot "$file" -o "${algorithm}_${dataset}" -title "${algorithm} on ${dataset}"
            fi
        done
    fi
done
