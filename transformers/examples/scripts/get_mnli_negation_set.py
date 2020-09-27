from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
import json
import pickle

import stanza
from stanza.utils.conll import CoNLL
import jsonlines

# from parallel import parallelized

def process_line(line, nlp):
    try:
        doc = nlp(line)
        conll = CoNLL.convert_dict(doc.to_dict())
        return conll[0]

    except Exception as e:
    #     # print(e)
        return []




NEGATION_WORDS = ['not', 'Not', 'No', 'no', "n't", "N't"]


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=Path, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    assert args.data_path.exists(), "data does not exist"
    # assert args.generated_data_path.is_dir(), "export path does not exist"
    total_lines = 0
    lines_with_neg_words = 0

    with args.data_path.open() as f, open(args.output, 'w') as writer:
        writer.write("index\tpromptID\tpairID\tgenre\ts1bp\ts2bp\ts1p\ts2p\tsentence1\tsentence2\tlabel1\tgold_label\n")
        for line in tqdm(f, desc="doing god's work", unit=" lines"):

            splitted = line.strip().split("\t")
            sentence_b = splitted[9]
            sentence_a = splitted[8]
            total_lines += 1
            if any((neg_word in sentence_a.split() or neg_word in sentence_b.split())for neg_word in NEGATION_WORDS):
                writer.write(line)
                lines_with_neg_words += 1

    print("Total: ", total_lines)
    print("Lines with neg words: ", lines_with_neg_words)
    print("ratio: ", (float(lines_with_neg_words) / float(total_lines)) * 100)


if __name__ == '__main__':
    main()
