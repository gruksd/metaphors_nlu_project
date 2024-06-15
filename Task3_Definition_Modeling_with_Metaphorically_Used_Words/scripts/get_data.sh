#!/bin/bash

# Download CoDWoE
wget https://codwoe.atilf.fr/data/full_dataset.zip
unzip full_dataset
mv full_dataset/en.complete.csv data/codwoe/
rm --recursive full_dataset*

# Download metaphor-paraphrase-dataset
wget https://raw.githubusercontent.com/xiaoyuisrain/metaphor-paraphrase-dataset/master/dataset.json
mv dataset.json data/metaphor-paraphrase/

# Download NAACL
wget http://www.tkl.iis.u-tokyo.ac.jp/~ishiwatari/naacl_data.zip
unzip naacl_data.zip
rm naacl_data.zip
