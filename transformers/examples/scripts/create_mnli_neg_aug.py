from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
import json
import pickle
import jsonlines

def main():
    parser = ArgumentParser()
    parser.add_argument('--neg_path', type=Path, required=True)
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--tsv_path', type=str, required=True)
    parser.add_argument('--separate', action='store_true')


    args = parser.parse_args()
    nli_label_map = {'neutral':'neutral', 'contradiction':'entailment', 'entailment': 'contradiction'}
    assert args.neg_path.exists(), "data does not exist"
    # assert args.generated_data_path.is_dir(), "export path does not exist"
    neg_lines = open(args.neg_path, 'r')
    json_docs = open(args.json_path, 'r').readlines()
    i = 0
    if not args.separate:
        tsv_out = open(args.tsv_path, 'w')
        tsv_out.write("index\tpromptID\tpairID\tgenre\ts1bp\ts2bp\ts1p\ts2p\tsentence1\tsentence2\tlabel1\tgold_label\n")
    else:
        neutral_tsv_out = open(args.tsv_path+"_neutral.tsv", 'w')
        neutral_tsv_out.write("index\tpromptID\tpairID\tgenre\ts1bp\ts2bp\ts1p\ts2p\tsentence1\tsentence2\tlabel1\tgold_label\n")
        cont_tsv_out = open(args.tsv_path+"_cont.tsv", 'w')
        cont_tsv_out.write("index\tpromptID\tpairID\tgenre\ts1bp\ts2bp\ts1p\ts2p\tsentence1\tsentence2\tlabel1\tgold_label\n")
        entail_tsv_out = open(args.tsv_path+"_entail.tsv", 'w')
        entail_tsv_out.write("index\tpromptID\tpairID\tgenre\ts1bp\ts2bp\ts1p\ts2p\tsentence1\tsentence2\tlabel1\tgold_label\n")
        writer_dict = {"entailment": entail_tsv_out, "neutral": neutral_tsv_out, "contradiction": cont_tsv_out}


    for line in neg_lines:
        print(line)
        print(i)
        negated_sentence_b = line.split('\t')[2]
        matched_rule = line.split('\t')[1]

        if matched_rule != "N/A":
            print(negated_sentence_b)
            doc = json.loads(json_docs[i])
            print(doc)
            if not args.separate:
                tsv_out.write("{}\t{}\t{}\tgenre\ts1bp\ts2bp\ts1p\ts2p\t{}\t{}\tlabel1\t{}\n".format(doc['index'],
                                                                                                     doc['promptID'],
                                                                                                     doc['pairID'],
                                                                                                     doc['sentence1'],
                                                                                                     negated_sentence_b,
                                                                                                     nli_label_map[doc['gold_label']]
                                                                                                    ))
            else:

                writer_dict[nli_label_map[doc['gold_label']]].write("{}\t{}\t{}\tgenre\ts1bp\ts2bp\ts1p\ts2p\t{}\t{}\tlabel1\t{}\n".format(doc['index'],
                                                                                                     doc['promptID'],
                                                                                                     doc['pairID'],
                                                                                                     doc['sentence1'],
                                                                                                     negated_sentence_b,
                                                                                                     nli_label_map[doc['gold_label']]
                                                                                                    ))

        i += 1


if __name__ == '__main__':
    main()
