from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
import json
import pickle
import os

import stanza
from stanza.utils.conll import CoNLL
import jsonlines

from parallel import parallelized

def process_lines(lines, nlp):
    try:
        doc = nlp(lines)
        conll = CoNLL.convert_dict(doc.to_dict())
        return conll

    except Exception as e:
    #     # print(e)
        return []







def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=Path, required=True)
    parser.add_argument('--generated_data_path', type=str, required=True)
    parser.add_argument("--max_num_proc", default=20, type=int)
    parser.add_argument("--pos_batch_size", default=15000, type=int)
    parser.add_argument("--tokenize_batch_size", default=15000, type=int)

    args = parser.parse_args()

    assert args.data_path.exists(), "data does not exist"
    # assert args.generated_data_path.is_dir(), "export path does not exist"


    stanza.download('en')
    nlp = stanza.Pipeline(processors='tokenize,mwt,pos,lemma,depparse',
                          pos_batch_size=args.pos_batch_size,
                          tokenize_batch_size=args.tokenize_batch_size,
                          tokenize_pretokenized=False,
                          mwt_batch_size=15000,
                          lemma_batch_size=15000,
                          depparse_batch_size=15000)

    line_count = 0
    lines = []
    with args.data_path.open() as f:
        for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
            line = line.strip()
            if len(line.split()) > 7 and len(line.split()) <= 20:
                lines.append(line)


    # file_count = 0
    print(len(lines))
    writer = open(args.generated_data_path, 'w')


    for batch in tqdm(range(0, len(lines), 15000)):

        data = lines[batch:batch + 15000]
        data = '\n'.join(data)
        conlls = process_lines(data, nlp)
        # conlls = parallelized(process_line, job_kwargs, max_workers=args.max_num_proc, progress=False)
        for conll_line in conlls:
            if len(conll_line) > 0:
                for conll_words in conll_line:
                    writer.write("\t".join(conll_words) + '\n')
                writer.write("\n")

    writer.close()
    print("done")

if __name__ == '__main__':
    main()
