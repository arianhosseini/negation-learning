from argparse import ArgumentParser
from pathlib import Path
import os
from os import getenv
import torch
import logging
import json
import random
import numpy as np
from collections import namedtuple
from tempfile import TemporaryDirectory

from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch import nn

from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers.modeling_bert import BertForPreTraining, BertConfig, BertForNegPreTraining, BertForNegKLPreTraining
from transformers.tokenization_bert import BertTokenizer
from transformers.optimization import AdamW, WarmupLinearSchedule
import dist_comms


InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids")

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]



def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    # is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array)
    return features

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BertLayerNorm') != -1:
        m.eval().half()

class FuckWrapper(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        model = BertForNegPreTraining.from_pretrained('bert-base-cased')
        model = model.to(device)

        self.core = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        # self.core = nn.DataParallel(BertForNegPreTraining.from_pretrained('bert-base-cased'))
    def forward(self, **input):
        return self.core(**input)
    def save_pretrained(self, output_dir):
        self.core.module.save_pretrained(output_dir)

class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"epoch_{self.data_epoch}.json"
        metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            # is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)
        logging.info(f"Loading training examples for epoch {epoch}")
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                # is_nexts[i] = features.is_next
        assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        # self.is_nexts = is_nexts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                # torch.tensor(self.is_nexts[item].astype(np.int64))
                )


def main():
    parser = ArgumentParser()
    parser.add_argument('--pregenerated_neg_data', type=Path, required=True)
    parser.add_argument('--pregenerated_pos_data', type=Path, required=True)
    parser.add_argument('--pregenerated_genericskb_neg', type=Path, required=False)
    parser.add_argument('--pregenerated_genericskb_pos', type=Path, required=False)
    parser.add_argument('--pregenerated_data', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True, help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")

    parser.add_argument("--max_seq_len", default=512, type=int)

    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--port_idx",
                        type=int)

    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--kr_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--kr_freq",
                        default=0.98,
                        type=float)
    parser.add_argument("--mlm_freq",
                        default=0,
                        type=float)
    parser.add_argument("--gkb_freq",
                        default=0.9,
                        type=float)
    parser.add_argument('--no_mlm',
                        action='store_true',
                        help="don't do any MLM training")
    parser.add_argument('--no_ul',
                        action='store_true',
                        help="don't do any UL training")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()





    assert args.pregenerated_data.is_dir(), \
        "--pregenerated_data should point to the folder of files made by pregenerate_training_data.py!"

    if args.pregenerated_genericskb_neg is not None:
        assert args.pregenerated_genericskb_neg.is_dir(), \
            "--pregenerated_genericskb_neg should point to the folder of files"
    if args.pregenerated_genericskb_pos is not None:
        assert args.pregenerated_genericskb_pos.is_dir(), \
            "--pregenerated_genericskb_pos should point to the folder of files"





    samples_per_epoch = []
    for i in range(args.epochs):
        epoch_file = args.pregenerated_data / f"epoch_{i}.json"
        metrics_file = args.pregenerated_data / f"epoch_{i}_metrics.json"
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                exit("No training data was found!")
            print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break
    else:
        num_data_epochs = args.epochs



    if args.local_rank == -1 or args.no_cuda:
        print(torch.cuda.is_available())
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
        print(n_gpu)
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        print("GPU Device: ", device)
        n_gpu = 1
        dist_comms.init_distributed_training(args.local_rank, args.port_idx)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    logging.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


    pt_output = Path(getenv('PT_OUTPUT_DIR', ''))
    args.output_dir = Path(os.path.join(pt_output, args.output_dir))

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logging.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    total_train_examples = 0
    for i in range(args.epochs):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    config = BertConfig.from_pretrained(args.bert_model)

    # core_model = BertForNegPreTraining.from_pretrained('bert-base-cased')
    core_model = BertForNegKLPreTraining.from_pretrained('bert-base-cased')
    core_model = core_model.to(device)

    # Prepare optimizer
    param_optimizer = list(core_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        core_model, optimizer = amp.initialize(core_model, optimizer, opt_level=args.fp16_opt_level)

    model = torch.nn.parallel.DistributedDataParallel(core_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    global_step = 0
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {total_train_examples}")
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)
    model.train()

    if args.local_rank == 0 or args.local_rank == -1:
        before_train_path = Path(os.path.join(args.output_dir,"before_training"))
        print("Before training path: ", before_train_path)
        before_train_path.mkdir(parents=True, exist_ok=True)
        model.module.save_pretrained(os.path.join(args.output_dir,"before_training"))

        tokenizer.save_pretrained(os.path.join(args.output_dir, "before_training"))

    neg_epoch_dataset = PregeneratedDataset(epoch=0, training_path=args.pregenerated_neg_data, tokenizer=tokenizer,
                                        num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory)

    pos_epoch_dataset = PregeneratedDataset(epoch=0, training_path=args.pregenerated_pos_data, tokenizer=tokenizer,
                                        num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory)

    if args.pregenerated_genericskb_neg is not None:
        gkb_neg_epoch_dataset = PregeneratedDataset(epoch=0, training_path=args.pregenerated_genericskb_neg, tokenizer=tokenizer,
                                            num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory)
    if args.pregenerated_genericskb_pos is not None:
        gkb_pos_epoch_dataset = PregeneratedDataset(epoch=0, training_path=args.pregenerated_genericskb_pos, tokenizer=tokenizer,
                                            num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory)


    if args.local_rank == -1:
        neg_train_sampler = RandomSampler(neg_epoch_dataset)
        pos_train_sampler = RandomSampler(pos_epoch_dataset)
        if args.pregenerated_genericskb_neg is not None:
            gkb_neg_train_sampler = RandomSampler(gkb_neg_epoch_dataset)
        if args.pregenerated_genericskb_pos is not None:
            gkb_pos_train_sampler = RandomSampler(gkb_pos_epoch_dataset)
    else:
        neg_train_sampler = DistributedSampler(neg_epoch_dataset)
        pos_train_sampler = DistributedSampler(pos_epoch_dataset)
        if args.pregenerated_genericskb_neg is not None:
            gkb_neg_train_sampler = DistributedSampler(gkb_neg_epoch_dataset)
        if args.pregenerated_genericskb_pos is not None:
            gkb_pos_train_sampler = DistributedSampler(gkb_pos_epoch_dataset)

    neg_train_dataloader = DataLoader(neg_epoch_dataset, sampler=neg_train_sampler, batch_size=args.train_batch_size)
    pos_train_dataloader = DataLoader(pos_epoch_dataset, sampler=pos_train_sampler, batch_size=args.train_batch_size)
    if args.pregenerated_genericskb_neg is not None:
        gkb_neg_train_dataloader = DataLoader(gkb_neg_epoch_dataset, sampler=gkb_neg_train_sampler, batch_size=args.train_batch_size)
    if args.pregenerated_genericskb_pos is not None:
        gkb_pos_train_dataloader = DataLoader(gkb_pos_epoch_dataset, sampler=gkb_pos_train_sampler, batch_size=args.train_batch_size)

    def inf_train_gen():
        while True:
            for kr_step, kr_batch in enumerate(neg_train_dataloader):
                yield kr_step, kr_batch

    kr_gen = inf_train_gen()

    def pos_inf_train_gen():
        while True:
            for kr_step, kr_batch in enumerate(pos_train_dataloader):
                yield kr_step, kr_batch

    pos_kr_gen = pos_inf_train_gen()

    if args.pregenerated_genericskb_neg is not None:
        def gkb_neg_inf_train_gen():
            while True:
                for kr_step, kr_batch in enumerate(gkb_neg_train_dataloader):
                    yield kr_step, kr_batch

        gkb_neg_gen = gkb_neg_inf_train_gen()
    if args.pregenerated_genericskb_pos is not None:
        def gkb_pos_inf_train_gen():
            while True:
                for kr_step, kr_batch in enumerate(gkb_pos_train_dataloader):
                    yield kr_step, kr_batch

        gkb_pos_gen = gkb_pos_inf_train_gen()



    for epoch in range(args.epochs):
        epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                            num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory)
        if args.local_rank == -1:
            train_sampler = RandomSampler(epoch_dataset)
        else:
            train_sampler = DistributedSampler(epoch_dataset)

        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        if  n_gpu > 1 and args.local_rank == -1  or (n_gpu <=1 and args.local_rank == 0):
            logging.info("** ** * Saving fine-tuned model ** ** * ")
            model.module.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            for step, batch in enumerate(train_dataloader):
                if not args.no_mlm and (random.random() > args.mlm_freq):
                    model.train()

                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, lm_label_ids = batch

                    outputs = model(input_ids=input_ids,
                                    attention_mask=input_mask,
                                    token_type_ids=segment_ids,
                                    masked_lm_labels=lm_label_ids,
                                    negated=False)
                    loss = outputs[0]
                    if n_gpu > 1:
                        loss = loss.mean() # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)

                    if args.local_rank == 0 or args.local_rank == -1:
                        nb_tr_steps += 1
                        # pbar.update(1)
                        mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                        pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        scheduler.step()  # Update learning rate schedule
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1
                pbar.update(1)

                if random.random() > args.kr_freq:
                    if random.random() > 0.5 and not args.no_ul:
                        kr_step, kr_batch = next(kr_gen)
                        kr_batch = tuple(t.to(device) for t in kr_batch)
                        input_ids, input_mask, segment_ids, lm_label_ids = kr_batch

                        outputs = model(input_ids=input_ids,
                                        attention_mask=input_mask,
                                        token_type_ids=segment_ids,
                                        masked_lm_labels=lm_label_ids,
                                        negated=True)
                        loss = outputs[0]
                    else:
                        kr_step, kr_batch = next(pos_kr_gen)
                        kr_batch = tuple(t.to(device) for t in kr_batch)
                        input_ids, input_mask, segment_ids, lm_label_ids = kr_batch

                        outputs = model(input_ids=input_ids,
                                        attention_mask=input_mask,
                                        token_type_ids=segment_ids,
                                        masked_lm_labels=lm_label_ids,
                                        negated=False)
                        loss = outputs[0]

                    if n_gpu > 1:
                        loss = loss.mean() # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    if args.local_rank == 0 or args.local_rank == -1:
                        nb_tr_steps += 1
                        mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                        pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        scheduler.step()  # Update learning rate schedule
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1
                if (args.pregenerated_genericskb_pos is not None and args.pregenerated_genericskb_neg is not None) and random.random() > args.gkb_freq:
                    if random.random() > 0.5 and not args.no_ul:
                        kr_step, kr_batch = next(gkb_neg_gen)
                        kr_batch = tuple(t.to(device) for t in kr_batch)
                        input_ids, input_mask, segment_ids, lm_label_ids = kr_batch

                        outputs = model(input_ids=input_ids,
                                        attention_mask=input_mask,
                                        token_type_ids=segment_ids,
                                        masked_lm_labels=lm_label_ids,
                                        negated=True)
                        loss = outputs[0]
                    else:
                        kr_step, kr_batch = next(gkb_pos_gen)
                        kr_batch = tuple(t.to(device) for t in kr_batch)
                        input_ids, input_mask, segment_ids, lm_label_ids = kr_batch

                        outputs = model(input_ids=input_ids,
                                        attention_mask=input_mask,
                                        token_type_ids=segment_ids,
                                        masked_lm_labels=lm_label_ids,
                                        negated=False)
                        loss = outputs[0]

                    if n_gpu > 1:
                        loss = loss.mean() # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    if args.local_rank == 0 or args.local_rank == -1:
                        nb_tr_steps += 1
                        mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                        pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        scheduler.step()  # Update learning rate schedule
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1




    # Save a trained model
    if  n_gpu > 1 and args.local_rank == -1  or (n_gpu <=1 and args.local_rank == 0):
        logging.info("** ** * Saving fine-tuned model ** ** * ")
        model.module.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
