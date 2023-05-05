import argparse

import sys
sys.path.append('NeuroX')


import NeuroX.neurox.data.extraction.transformers_extractor as transformers_extractor


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_desc', type=str, default="bert-base-cased")
    parser.add_argument("--decompose_layers", action="store_true",
                        help="Save activations from each layer in a separate file")
    parser.add_argument("--include_special_tokens", action="store_true",
                        help="Include special tokens like [CLS] and [SEP] in the extracted representations")
    parser.add_argument("--filter_layers", default=None, type=str,
                        help="Comma separated list of layers to save activations for. The layers will be saved in the order specified in this argument.", )
    parser.add_argument('--input_corpus', type=str, default="/glue_ver/data/sst2_train.json")
    parser.add_argument('--output_file', type=str, default="tok.sent_len")
    parser.add_argument('--output_type', type=str, default="json")

    args = parser.parse_args()

    positions = transformers_extractor.extract_representations(model_desc=args.model_desc,
                                                       input_corpus=args.input_corpus,
                                                       output_file=args.output_file,
                                                       output_type=args.output_type,
                                                       decompose_layers=args.decompose_layers,
                                                       filter_layers=args.filter_layers,
                                                       include_special_tokens=args.include_special_tokens)


if __name__ == "__main__":
    main()

