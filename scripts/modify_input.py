import argparse


# read the input.in.tok.sent_len file
def get_dataset():
    with open(args.text_file) as f:
        dataset = f.readlines()
    return dataset


# add [CLS] at the beginning of each sentence and [SEP] at the end of each sentence
def add_special_tokens(dataset):
    for i in range(len(dataset)):
        dataset[i] = "[CLS] " + dataset[i][:-1] + " [SEP]"
    return dataset


# save the modified dataset to a file
# each line is a sentence
def save_dataset(dataset):
    with open(args.output_file, "w") as f:
        f.write("\n".join(dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--text-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)

    args = parser.parse_args()

    dataset = get_dataset()
    dataset = add_special_tokens(dataset)
    save_dataset(dataset)
