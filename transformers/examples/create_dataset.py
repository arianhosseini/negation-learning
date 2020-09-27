import os
import glob
import jsonlines
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
import pprint
import random



def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=Path, required=True)
    parser.add_argument('--generated_data_path', type=Path, required=True)
    parser.add_argument('--mix', action="store_true")
    parser.add_argument('--likelihood', action="store_true")
    parser.add_argument('--no_ref', action="store_true")
    parser.add_argument('--model', type=str, choices=["bert", "roberta"])

    rules_stats = {}



    args = parser.parse_args()
    assert args.data_path.exists(), "data does not exist"

    MASK_TOKEN = "[MASK]" if args.model == "bert" else "<mask>"

    uuid = 0
    with open(args.data_path) as data_file, jsonlines.open(args.generated_data_path, 'w') as generated_file:
        for line in data_file.readlines():
            columns = line.strip().split('\t')
            if (len(columns) == 4) and (columns[1] != "N/A"):
                random_word = columns[0].split()[random.randint(0, len(columns[0].split())-1)]
                random_prob = random.random() if args.mix else 0
                if random_prob < 0.5:
                    doc = {
                        "uuid": uuid,
                        "matched_rule": columns[1],
                        "masked_sentences": [((columns[0] + " ") if not args.no_ref else "") + ( (columns[2] if not args.likelihood else columns[0]).replace( random_word if (args.likelihood and columns[3] not in columns[0]) else columns[3] , MASK_TOKEN))],
                        "obj_label": random_word if (args.likelihood and columns[3] not in columns[0]) else columns[3]
                    }
                else:
                    if args.likelihood:
                        doc = {
                            "uuid": uuid,
                            "matched_rule": columns[1],
                            "masked_sentences": [((columns[2] + " ") if not args.no_ref else "") + (columns[2]).replace(columns[3] , MASK_TOKEN)],
                            "obj_label": columns[3]
                        }
                    else:
                        doc = {
                            "uuid": uuid,
                            "matched_rule": columns[1],
                            "masked_sentences": [((columns[2] + " ") if not args.no_ref else "") + ( (columns[0]).replace( random_word if (columns[3] not in columns[0]) else columns[3] , MASK_TOKEN))],
                            "obj_label": random_word if (columns[3] not in columns[0]) else columns[3]
                        }


                generated_file.write(doc)
                uuid += 1

                if columns[1] in rules_stats:
                    rules_stats[columns[1]] += 1
                else:
                    rules_stats[columns[1]] = 1
    print("Sample: ")
    print(doc)
    print("Done")
    rules_stats['total'] = sum(rules_stats.values())
    pprint.pprint(rules_stats)


    # assert args.generated_data_path.is_dir(), "export path does not exist"

if __name__ == '__main__':
    main()
