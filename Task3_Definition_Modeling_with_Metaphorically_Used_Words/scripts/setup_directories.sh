#!/bin/bash

mkdir --parents \
	annotation/{data,keys} \
	code \
	data/{codwoe,metaphor-paraphrase} \
	err \
	models/{bart-base,flan-t5-base} \
	out \
	preds/{metaphor_paraphrase,oxford_test,wordnet_test} \
	scores/{metaphor_paraphrase,oxford_test,wordnet_test} \
	scripts
