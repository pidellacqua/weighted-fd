#!/usr/bin/env bash
RESULTS_PATH="./ignore/outputs/memory-evaluation"
OUTPUT_PATH="./plots"

usage() {
    echo "Usage: $0 [-r results_path] [-o output_path] [-h]"
    echo "  -r    Path to results directory (default: $RESULTS_PATH)"
    echo "  -o    Path to output directory (default: $OUTPUT_PATH)"
    echo "  -h    Show this help message"
    exit 0
}

while getopts "r:o:h" opt; do
    case $opt in
        r) RESULTS_PATH="$OPTARG" ;;
        o) OUTPUT_PATH="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# RESULTS_PATH="./ignore/outputs/memory-evaluation"
# OUTPUT_PATH="./plots"


# Funzione per tradurre dataset
translate_dataset() {
    case "$1" in
        "cifar10") echo "CIFAR10" ;;
        "mnist") echo "MNIST" ;;
        "fashion-mnist") echo "FashionMNIST" ;;
        *)
            echo "Unsupported dataset: $1"
            exit 1
            ;;
    esac
}

# Funzione per tradurre metodo/algoritmo
translate_method() {
    case "$1" in
        "selective-fd") echo "Selective-FD" ;;
        "fed-md") echo "FedMD" ;;
        "weighted-fd") echo "Weighted-FD" ;;
        *)
            echo "Unsupported method: $1"
            exit 1
            ;;
    esac
}

if [ ! -d "$OUTPUT_PATH" ]; then
    mkdir -p "$OUTPUT_PATH"
fi

for dir_path in "$RESULTS_PATH"/*; do
    if [ -d "$dir_path" ]; then
        dirname=$(basename "$dir_path")

        # Split tokens con underscore
        IFS='_' read -ra tokens <<< "$dirname"

        algorithm=$(translate_method "${tokens[0]}")
        dataset=$(translate_dataset "${tokens[1]}")
        dataset_type="${tokens[-2]}"
        seed="${tokens[-1]}"

        for file in "$dir_path"/*.dat; do
            if [ -f "$file" ]; then
                echo "Plotting memory profile from: $file"
                
                # qui puoi aggiungere il comando che fa il plot, es:
                mprof plot "$file" -f -o "${OUTPUT_PATH}/${algorithm}_${dataset}" -t "${algorithm} on ${dataset}"
            fi
        done
    fi
done
