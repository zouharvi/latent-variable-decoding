import json
import torch
import copy
import pickle
import os
import tqdm
import logging
from os.path import join

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils import get_flant5_tokenizer

logger = logging.getLogger(__name__)


class QTDataProcessor:
    DEFAULT_CONFIG = {
        "dataset": "tsmh",
        "plm_tokenizer_name": "flant5_aug",
        "data_dir": "computed/"
    }

    def __init__(self, config=DEFAULT_CONFIG, language='english'):
        self.config = config
        self.language = language

        self.data_dir = config['data_dir']
        self.dataset = config['dataset']

        # Get tensorized samples
        cache_path = self.get_cache_path()
        if os.path.exists(cache_path):
            # load cached tensors if exists
            self.tensor_samples, self.stored_info = pickle.load(
                open(cache_path, 'rb'))
            logger.info(f'Loaded tensorized examples from cache: {cache_path}')
        else:
            # generate tensorized samples
            self.tensor_samples = {}
            tensorizer = QGTensorizer(self.config)
            suffix = f'{self.config["plm_tokenizer_name"]}.jsonl'

            if self.dataset == "tsmh":
                paths = {
                    'test': join(self.data_dir, f'qg_masked.{suffix}')
                }

            for split, path in paths.items():
                logger.info(
                    f'Tensorizing examples from {path}; results will be cached in {cache_path}')
                is_training = (split == 'trn')

                with open(path, 'r') as f:
                    samples = [json.loads(x) for x in f]
                tensor_samples = [
                    tensorizer.tensorize_example(sample, is_training)
                    for sample in tqdm.tqdm(samples)
                ]
                print(split, len(tensor_samples))

                self.tensor_samples[split] = QGDataset(
                    sorted(
                        [(doc_key, tensor)
                         for doc_key, tensor in tensor_samples],
                        key=lambda x: -x[1]['input_ids'].size(0)
                    )
                )
            self.stored_info = tensorizer.stored_info

            # TODO: this is dangerous to use pickle with tensor because the ABI is not stable
            # cache tensorized samples
            with open(cache_path, 'wb') as f:
                pickle.dump((self.tensor_samples, self.stored_info), f)

    def get_tensor_examples(self):
        # For each split, return list of tensorized samples to allow variable length input (batch size = 1)
        return self.tensor_samples['test']

    def get_stored_info(self):
        return self.stored_info

    def get_cache_path(self):
        cache_path = join(
            self.data_dir, f'cached.tensors.{self.config["plm_tokenizer_name"]}.bin'
        )
        return cache_path


class QGTensorizer:
    DEFAULT_CONFIG = {"plm_tokenizer_name": "flant5_aug"}

    def __init__(self, config=DEFAULT_CONFIG):
        self.config = copy.deepcopy(config)
        if config["plm_tokenizer_name"] == "flant5_aug":
            self.tokenizer = get_flant5_tokenizer()
        else:
            raise Exception("Unknown tokenizer name " +
                            config["plm_tokenizer_name"])

        # used in evaluation
        self.stored_info = {
            'example': {},
        }

    def safe_tokenizer_convert_to_ids(self, sequence):
        # -5 is beginning of sentence
        sequence = [self.tokenizer.convert_tokens_to_ids(
            [x])[0] if x is not None else -5 for x in sequence]
        return sequence

    def tensorize_example(
        self, example, is_training=False
    ):
        # Keep info to store
        doc_id = example['doc_id']
        self.stored_info['example'][doc_id] = example

        is_training = torch.tensor(is_training, dtype=torch.bool)

        # sentences/segments
        input_sentence = copy.deepcopy(example['input_sentence'])
        target_sentence = copy.deepcopy(example["target_sentence"])

        # copy everything, including prefix
        input_term_mask = copy.deepcopy(example['input_term_mask'])
        target_term_mask = copy.deepcopy(example['target_term_mask'])

        input_ids = torch.tensor(self.safe_tokenizer_convert_to_ids(
            input_sentence), dtype=torch.long)
        target_ids = torch.tensor(self.safe_tokenizer_convert_to_ids(
            target_sentence), dtype=torch.long)
        input_len, target_len = input_ids.size(0), target_ids.size(0)

        input_mask = torch.tensor([1] * input_len, dtype=torch.long)
        target_mask = torch.tensor([1] * target_len, dtype=torch.long)

        input_term_mask = torch.tensor(input_term_mask, dtype=torch.long)
        target_term_mask = torch.tensor(target_term_mask, dtype=torch.long)

        if not is_training:
            # convert constraints to a two dimensional tensor, contraints is a nested list of various length
            constraint_ids = []
            for clause in example['target_constraints']:
                tmp_ids = [self.safe_tokenizer_convert_to_ids(
                    mention) for mention in clause]
                constraint_ids.append(tmp_ids)

            # constraint_ids = util.func.nested_list_to_tensor(constraint_ids, padding_value=self.tz.pad_token_id)
            self.stored_info['example'][doc_id]['constraint_ids'] = constraint_ids

        # One segment per example
        example_tensor = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'target_ids': target_ids,
            'target_mask': target_mask,
            'input_term_mask': input_term_mask,
            'target_term_mask': target_term_mask,
            'is_training': is_training,
        }

        return doc_id, example_tensor


def mt_collate_fn(batch, pad_token_id=0):
    # TODO: this is copied  from the MT data loader
    """
        Collate function for the MT dataloader.
        Default "pad_token_id": 0, as in the `t5-base` tokenizer
    """
    doc_keys, batch = zip(*batch)
    batch = {k: [example[k] for example in batch] for k in batch[0]}
    batch_size = len(batch["input_ids"])

    max_input_len = max([example.size(0) for example in batch["input_ids"]])
    max_target_len = max([example.size(0) for example in batch["target_ids"]])

    for k in ["input_ids", "input_mask", "input_term_mask"]:
        # (batch_size, max_target_len)
        if k not in batch:
            continue
        batch[k] = torch.stack([
            F.pad(x, (0, max_input_len - x.size(0)), value=pad_token_id) for x in batch[k]
        ], dim=0)
    for k in ["target_ids", "target_mask", "target_term_mask"]:
        if k not in batch:
            continue
        # (batch_size, max_target_len)
        batch[k] = torch.stack([
            F.pad(x, (0, max_target_len - x.size(0)), value=pad_token_id) for x in batch[k]
        ], dim=0)
    for k in ["is_training"]:
        if k not in batch:
            continue
        batch["is_training"] = torch.tensor(
            batch["is_training"], dtype=torch.bool)

    return doc_keys, batch


class QGDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
