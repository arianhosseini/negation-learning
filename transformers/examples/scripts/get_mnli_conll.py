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







def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=Path, required=True)
    parser.add_argument('--conll_out', type=str, required=True)
    parser.add_argument('--json_out', type=str, required=True)
    parser.add_argument("--max_lines", default=30000, type=int)
    parser.add_argument("--pos_batch_size", default=3000, type=int)
    parser.add_argument("--tokenize_batch_size", default=128, type=int)

    args = parser.parse_args()

    assert args.data_path.exists(), "data does not exist"
    # assert args.generated_data_path.is_dir(), "export path does not exist"
    stanza.download('en')
    nlp = stanza.Pipeline(processors='tokenize,pos,mwt,lemma,depparse',
                          pos_batch_size=args.pos_batch_size,
                          tokenize_batch_size=args.tokenize_batch_size,
                          mwt_batch_size=100,
                          lemma_batch_size=100,
                          depparse_batch_size=6000)

    line_count = 0
    lines = []
    guids = []
    ind2guid = []
    all_docs = []
    num = 0
    with args.data_path.open() as f, open(args.conll_out, 'w') as writer, jsonlines.open(args.json_out, 'w') as json_out:
        for line in tqdm(f, desc="doing god's work", unit=" lines"):

            if num > args.max_lines:
                break
            line = line.strip().split("\t")
            guid = line[2]
            sentence_b = line[9]
            gold_label = line[-1]
            sentence_a = line[8]
            pid = line[1]
            ind = line[0]

            if len(sentence_b.split()) > 7 and len(sentence_b.split()) <= 20:
                doc = nlp(sentence_b)
                conll = CoNLL.convert_dict(doc.to_dict())
                # print(conll[0])
                for conll_words in conll[0]:
                    writer.write("\t".join(conll_words) + '\n')
                writer.write("\n")
                json_doc = {'sentence1': sentence_a,
                            'sentence2': '',
                            'promptID': pid,
                            'pairID': guid,
                            'gold_label': gold_label,
                            'index': ind
                            }
                json_out.write(json_doc)
                num += 1



if __name__ == '__main__':
    main()
