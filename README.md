# ConceptX_based_sst2

## Clustering Steps
**Run 'base_code.sh' for tokenization, sentence length filtering, and calculate vocabulary size**
* Step1: Tokenize text with moses tokenizer
* Step2: Do sentence length filtering and keep sentences max length of 300
* Step3: Modify the input file to be compatible with the model
* Step3: Calculate vocabulary size

**Run 'layer_code.sh' for embedding extraction and clustering**
* Step4: Extract layer-wise activations
* Step5: Create a dataset file with word and sentence indexes for a single layer
* Step6: Filter number of tokens to fit in the memory for clustering.
* Step7: Run clustering
