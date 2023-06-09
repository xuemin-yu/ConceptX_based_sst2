#!/bin/bash

scriptDir=scripts
inputPath=data # path to a sentence file
dataset=sst2

# model name or path to a finetuned model
model="glue-cased-models"

# maximum sentence length
sentence_length=300
# analyze latent concepts of layer 12
layer=12

outputDir=layer${layer} #do not change this
mkdir ${outputDir}

working_file=$input.tok.sent_len #do not change this

#5. Extract layer-wise activations
source activate neurox_pip
python -m neurox.data.extraction.transformers_extractor --decompose_layers --filter_layers ${layer} --output_type json ${model} ${working_file} ${outputDir}/${working_file}.activations.json --include_special_tokens

#6. Create a dataset file with word and sentence indexes
python ${scriptDir}/create_data_single_layer.py --text-file ${working_file}.modified --activation-file ${outputDir}/${working_file}.activations-layer${layer}.json --output-prefix ${outputDir}/${working_file}-layer${layer}

#7. Filter number of tokens to fit in the memory for clustering. Input file will be from step 4. minfreq sets the minimum frequency. If a word type appears is coming less than minfreq, it will be dropped. if a word comes
minfreq=5
maxfreq=20
delfreq=20
python ${scriptDir}/frequency_filter_data.py --input-file ${outputDir}/${working_file}-layer${layer}-dataset.json --frequency-file ${working_file}.words_freq --sentence-file ${outputDir}/${working_file}-layer${layer}-sentences.json --minimum-frequency $minfreq --maximum-frequency $maxfreq --delete-frequency ${delfreq} --output-file ${outputDir}/${working_file}-layer${layer}

#8. Run clustering
conda activate clustering

mkdir ${outputDir}/results
DATASETPATH=${outputDir}/${working_file}-layer${layer}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json
VOCABFILE=${outputDir}/processed-vocab.npy
POINTFILE=${outputDir}/processed-point.npy
RESULTPATH=${outputDir}/results
CLUSTERS=5,5,5  #Comma separated for multiple values or three values to define a range
# first number is number of clusters to start with, second is number of clusters to stop at and third one is the increment from the first value
# 600 1000 200 means [600,800,1000] number of clusters

#echo "Extracting Data!"
python -u ${scriptDir}/extract_data.py --input-file $DATASETPATH --output-path $outputDir

echo "Creating Clusters!"
python -u ${scriptDir}/get_agglomerative_clusters.py --vocab-file $VOCABFILE --point-file $POINTFILE --output-path $RESULTPATH  --cluster $CLUSTERS --range 1
echo "DONE!"

#9. Extract the prediction for CLS token
mkdir ${outputDir}/CLS_tokens
source activate explanation
python ${scriptDir}/extract_cls_prediction.py --dataset-name-or-path ${inputPath}/$dataset --model-name ${model} --tokenizer-name ${model} --save-dir ${outputDir}/CLS_tokens/ --layer ${layer}