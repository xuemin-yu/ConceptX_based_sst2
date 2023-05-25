# ConceptX_based_sst2
Analyzing Latent Concepts in Pre-trained Transformer Models on the SST2 dataset by using the ConceptX and NeuroX toolkits.

## Setting the environment
### The step for embedding extraction needs to set up the NeuroX environment:
1. Manual Installation for NeuroX
``` 
git clone https://github.com/fdalvi/NeuroX.git
```

2. Create a conda environment with python 3.8 for latest version of NeuroX
```
conda create -n neurox_pip python=3.8
conda activate neurox_pip
```

3. Install the dependencies required to run the NeuroX toolkit
``` 
cd Neurox
pip install -e .
```

4. Set the directory path back to the main directory and deactivate the environment
```
cd ..
conda deactivate
```

### The step for clustering needs to set up a clustering environment using env_clustering.yml.
```
conda env create --file=env_clustering.yml
```

### The step for CLS prediction needs to set up a explanation environment using env_explanation.yml.
```
conda env create --file=env_explanation.yml
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


