# ConceptX_based_sst2
Analyzing Latent Concepts in Pre-trained Transformer Models on the SST2 dataset by using the ConceptX and NeuroX toolkits.

## Setting the environment
The step for embedding extraction needs to set up the NeuroX environment using env_neurox.yml.
```
conda env create --file=env_neurox.yml
```

The step for clustering needs to set up a clustering environment using env_clustering.yml.
```
conda env create --file=env_clustering.yml
```

## Tokenization, Sentence Filtering, and Vocabulary Size Calculation
**Run 'base_code.sh' for tokenization, sentence length filtering, and calculating vocabulary size**
* Step1: Tokenize text with moses tokenizer
* Step2: Do sentence length filtering and keep sentences max length of 300
* Step3: Modify the input file to be compatible with the model
* Step3: Calculate vocabulary size

## Embedding Extraction and Concept Clustering
**Run 'layer_code.sh' for embedding extraction and clustering**
* Step4: Extract layer-wise activations
* Step5: Create a dataset file with word and sentence indexes for a single layer
* Step6: Filter the number of tokens to fit in the memory for clustering.
* Step7: Run clustering
