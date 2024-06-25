#!/bin/bash

for MODEL in "bart-base" "flan-t5-base"; do
    python3 ./code/analyze_annotations.py \
        --ann1 "annika" --ann2 "sofia" --ann3 "mitja" \
        --model $MODEL
    done
