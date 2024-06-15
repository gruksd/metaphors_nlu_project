#!/bin/bash

for PREDFILE in preds/metaphor_paraphrase/*
do
	python3 code/annotation_data.py $PREDFILE
done
