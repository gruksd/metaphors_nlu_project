#!/bin/bash

for DATAFILE in $(ls ./preds/metaphor_paraphrase/* | cat); do
    python3 ./code/process_preds.py $DATAFILE
done
