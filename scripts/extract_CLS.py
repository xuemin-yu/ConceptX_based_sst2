""" Extract the [CLS] token representation from the BERT model for the SST2 dataset.

This script is used to extract the [CLS] token representation from the BERT model for the SST2 dataset.
The dataset is loaded from the datasets library.
The model is saved in the save_dir directory.

Example:
    $ python extract_CLS.py --dataset_name_or_path sst2 --model_name bert-base-cased --tokenizer_name bert-base-cased --save_dir ./save_dir --layer 0

"""

import argparse
from datasets import load_from_disk
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json
import numpy as np
import torch.nn.functional as F
import psutil
import os


def get_dataset(args):
    """
    Load the dataset from the datasets library or local directory.

    Returns
    -------
    dataset: datasets.arrow_dataset.Dataset
    """
    dataset = load_from_disk(args.dataset_name_or_path)
    # If the dataset is loaded from the datasets library, using the following code
    # from datasets import load_dataset
    # dataset = load_dataset('glue', 'sst2')

    dataset = dataset['train'][:100]

    return dataset


def get_hidden_states_inputs(model, tokenizer, dataset):
    """
    Input the sentences into the model to get the hidden states and predicted label of the model.

    Parameters
    ----------
    model: transformers.modeling_utils.PreTrainedModel
        The pre-trained model to be used.

    tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase
        The tokenizer to be used.

    dataset: datasets.arrow_dataset.Dataset
        The dataset to be tokenized.
    """
    input_text = tokenizer(dataset['sentence'], padding=True, truncation=True, return_tensors="pt")

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    with torch.no_grad():
        outputs = model(**input_text, output_hidden_states=True)

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024 * 1024):.2f} GB")

    predictions = F.softmax(outputs[0], dim=-1)
    labels = torch.argmax(predictions, axis=1)  # 0: Negative, 1: Positive

    return input_text, outputs.hidden_states, labels


def save_tokens(ids, tags, layer_hidden_states, labels, layer, save_path):
    """
    Save the [CLS] token representation and info into a json file.

    Parameters
    ----------
    ids: torch.Tensor
        The token ids of the sentences.

    tags: list
        The tags of the sentences.

    layer_hidden_states: list
        The hidden states of the sentences.

    labels: torch.Tensor
        The predicted labels of the sentences.

    layer: int
        The layer to be extracted.

    save_path: str
        The path to save the json file.

    Returns
    -------
    layer_cls_info: list
        The [CLS] token representation and info.
    """
    layer_cls_info = []
    predictions = []

    for j in range(len(ids)):  # len(ids) => # of sentences

        # get the data representation of a layer for a sentence
        layer_content = layer_hidden_states[layer][j]
        predicted_result = labels[j].item()

        predictions.append(str(predicted_result) + " " + str(-1) + " " + str(j))

        cls_info = []

        # token+senten_id | | token_id | | senten_id | | token_rep | | prediction
        # cls_info.append("[CLS_" + str(tags[j]) + "]" + str(j) + "|||" + str(ids[j][0].item()) + "|||" + str(j))

        # token ||| position_id ||| sentence_id
        cls_info.append("[CLS]" + str(j) + "|||" + str(-1) + "|||" + str(j))
        cls_info.append(layer_content[0].tolist())
        cls_info.append(predicted_result)

        layer_cls_info.append(cls_info)

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024 * 1024):.2f} GB")

    # check the directory exists or not
    if not os.path.exists(save_path + 'info'):
        os.makedirs(save_path + 'info')

    path = save_path + 'info/CLS_token_info_layer_' + str(layer) + '.json'
    # save the [CLS] representation and info
    with open(path, "w") as json_file:
        json.dump(layer_cls_info, json_file, indent=4)

    # check the directory exists or not
    if not os.path.exists(save_path + 'explanations'):
        os.makedirs(save_path + 'explanations')

    path = save_path + 'explanations/explanation_layer_' + str(layer) + '.txt'
    # save the predicted results in a txt file
    with open(path, "w") as txt_file:
        for line in predictions:
            txt_file.write(line + "\n")

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024 * 1024):.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name_or_path',
                        type=str,
                        default='./saved_sst2',
                        help='The name or path of the dataset to be loaded.')

    parser.add_argument('--model_name',
                        type=str,
                        default='bert-base-cased',
                        help='The name or path of the pre-trained model to be used.')

    parser.add_argument('--tokenizer_name',
                        type=str,
                        default='bert-base-cased',
                        help='The name or path of the tokenizer to be used.')

    parser.add_argument('--save_dir',
                        type=str,
                        default='CLS_tokens/',
                        help='The directory to save the extracted [CLS] token representation and info.')

    parser.add_argument('--layer',
                        type=str,
                        default=12,
                        help='The layer to be extracted.')

    args = parser.parse_args()

    dataset = get_dataset(args)

    model = BertForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)

    inputs, hidden_states, labels = get_hidden_states_inputs(model, tokenizer, dataset)

    ids = inputs['input_ids']

    layer = int(args.layer)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    save_tokens(ids, dataset['label'], hidden_states, labels, layer, args.save_dir)


if __name__ == '__main__':
    main()