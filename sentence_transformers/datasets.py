"""
This files contains various pytorch dataset classes, that provide
data to the Transformer model
"""
import sys
from functools import partial
from itertools import tee
from multiprocessing.pool import Pool
from pathlib import Path

import tables
from torch.utils.data import Dataset
from typing import List, Iterable
from torch import Tensor
import bisect
import torch
import logging
import numpy as np
from tqdm import tqdm
from . import SentenceTransformer
from .readers.InputExample import InputExample


class SentencesDataset(Dataset):
    """
    Dataset for smart batching, that is each batch is only padded to its longest sequence instead of padding all
    sequences to the max length.
    The SentenceBertEncoder.smart_batching_collate is required for this to work.
    SmartBatchingDataset does *not* work without it.
    """
    tokens: np.ndarray
    labels: Tensor

    _label_type_mapping = {
        int: torch.long,
        np.int64.__class__: torch.long,
        np.uint8.__class__: torch.long,
        float: torch.float,
    }

    def __init__(self, examples: Iterable[InputExample], model: SentenceTransformer,
                 dataset_cache_id=None, show_progress_bar: bool = None):
        """
        Create a new SentencesDataset with the tokenized texts and the labels as Tensor
        """
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.convert_input_examples(examples, model, dataset_cache_id)

    def convert_input_examples(self, examples: Iterable[InputExample], model: SentenceTransformer,
                               dataset_cache_id: str, cache_dir=Path('.cache')):
        """
        Converts input examples to a SmartBatchingDataset usable to train the model with
        SentenceTransformer.smart_batching_collate as the collate_fn for the DataLoader

        smart_batching_collate as collate_fn is required because it transforms the tokenized texts to the tensors.

        :param examples:
            the input examples for the training
        :param model
            the Sentence BERT model for the conversion
        :param cache_dir:
            directory where tokenized dataset is cached
        :param dataset_cache_id:
            dataset identifier as cache file name
        :return: a SmartBatchingDataset usable to train the model with SentenceTransformer.smart_batching_collate as the collate_fn
            for the DataLoader
        """

        dataset_h5_cache = cache_dir / dataset_cache_id / 'cache.h5'
        tokens_array = 'tokens'
        labels_array = 'labels'

        if dataset_h5_cache.exists():
            self.read_h5_dataset(dataset_h5_cache, labels_array, tokens_array)
            return

        examples_iter1, examples_iter2 = tee(examples)

        if self.show_progress_bar:
            examples_iter1 = tqdm(examples_iter1, desc='Convert dataset')
            examples_iter2 = tqdm(examples_iter2, desc='Convert labels')

        with Pool() as pool:
            tokens_iter1, tokens_iter2 = tee(
                pool.imap(
                    func=partial(self._convert_example, model=model),
                    iterable=examples_iter1,
                    chunksize=128
                )
            )

            if self.show_progress_bar:
                tokens_iter1 = tqdm(tokens_iter1, desc='Longest sequence searching')
                tokens_iter2 = tqdm(tokens_iter2, desc='Pad sequences')

            max_seq_len = self._get_max_seq_len(tokens_iter1, model)

            padded_tokens = pool.imap(
                func=partial(self._pad_sequences, pad_size=max_seq_len),
                iterable=tokens_iter2,
                chunksize=128
            )

            labels = (example.label for example in examples_iter2)

            dataset_h5_cache.parent.mkdir(parents=True, exist_ok=True)

            with tables.open_file(dataset_h5_cache, mode='w') as hf:
                tokens_storage = hf.create_earray(hf.root, tokens_array,
                                                  atom=tables.UInt16Atom(),
                                                  shape=[0, 3, max_seq_len])

                labels_storage = hf.create_earray(hf.root, labels_array,
                                                  atom=tables.UInt8Atom(),
                                                  shape=[0])

                for padded_seq, label in tqdm(zip(padded_tokens, labels), desc='Writing'):
                    tokens_storage.append([padded_seq])
                    labels_storage.append([label])

        self.read_h5_dataset(dataset_h5_cache, labels_array, tokens_array)

    def read_h5_dataset(self, dataset_h5_cache, labels_array, tokens_array):
        h5_file = tables.open_file(dataset_h5_cache, mode='r')

        self.tokens = getattr(h5_file.root, tokens_array)
        self.labels = getattr(h5_file.root, labels_array)

    @staticmethod
    def _convert_example(example: InputExample, model: SentenceTransformer):
        return [model.tokenize(text) for text in example.texts]

    @staticmethod
    def _get_max_seq_len(tokens: Iterable[List], model: SentenceTransformer):
        max_seq_len = max(max(map(len, seqs)) for seqs in tokens)
        model_max_seq_len = getattr(model, 'max_seq_length', sys.maxsize)
        return min(max_seq_len, model_max_seq_len)

    @staticmethod
    def _pad_sequences(sequences, pad_size: int):
        return [SentencesDataset._pad_array(seq, pad_size) for seq in sequences]

    @staticmethod
    def _pad_array(array: List, pad_size: int):
        array = array[:pad_size]
        return np.pad(array, pad_width=(0, pad_size - len(array)), mode='constant').tolist()

    def __getitem__(self, item):
        label = self.labels[item]

        if type(label) != torch.Tensor:
            label = torch.tensor(label, dtype=self._label_type_mapping.get(type(label)))

        return self.tokens[item].tolist(), label

    def __len__(self):
        return self.tokens.shape[0]


class SentenceLabelDataset(Dataset):
    """
    Dataset for training with triplet loss.
    This dataset takes a list of sentences grouped by their label and uses this grouping to dynamically select a
    positive example from the same group and a negative example from the other sentences for a selected anchor sentence.

    This dataset should be used in combination with dataset_reader.LabelSentenceReader

    One iteration over this dataset selects every sentence as anchor once.

    This also uses smart batching like SentenceDataset.
    """
    tokens: List[List[str]]
    labels: Tensor
    num_labels: int
    labels_right_border: List[int]

    def __init__(self, examples: List[InputExample], model: SentenceTransformer, provide_positive: bool = True,
                 provide_negative: bool = True):
        """
        Converts input examples to a SentenceLabelDataset usable to train the model with
        SentenceTransformer.smart_batching_collate as the collate_fn for the DataLoader

        Assumes only one sentence per InputExample and labels as integers from 0 to max_num_labels
        and should be used in combination with dataset_reader.LabelSentenceReader.

        Labels with only one example are ignored.

        smart_batching_collate as collate_fn is required because it transforms the tokenized texts to the tensors.

        :param examples:
            the input examples for the training
        :param model
            the Sentence BERT model for the conversion
        :param provide_positive:
            set this to False, if you don't need a positive example (e.g. for BATCH_HARD_TRIPLET_LOSS).
        :param provide_negative:
            set this to False, if you don't need a negative example (e.g. for BATCH_HARD_TRIPLET_LOSS
            or MULTIPLE_NEGATIVES_RANKING_LOSS).
        """
        self.convert_input_examples(examples, model)
        self.idxs = np.arange(len(self.tokens))
        self.positive = provide_positive
        self.negative = provide_negative

    def convert_input_examples(self, examples: List[InputExample], model: SentenceTransformer):
        """
        Converts input examples to a SentenceLabelDataset.

        Assumes only one sentence per InputExample and labels as integers from 0 to max_num_labels
        and should be used in combination with dataset_reader.LabelSentenceReader.

        Labels with only one example are ignored.

        :param examples:
            the input examples for the training
        :param model
            the Sentence Transformer model for the conversion
        """
        self.labels_right_border = []
        self.num_labels = 0
        inputs = []
        labels = []

        label_sent_mapping = {}
        too_long = 0
        label_type = None
        for ex_index, example in enumerate(tqdm(examples, desc="Convert dataset")):
            if label_type is None:
                if isinstance(example.label, int):
                    label_type = torch.long
                elif isinstance(example.label, float):
                    label_type = torch.float
            tokenized_text = model.tokenize(example.texts[0])

            if hasattr(model, 'max_seq_length') and model.max_seq_length is not None and model.max_seq_length > 0 and len(tokenized_text) >= model.max_seq_length:
                too_long += 1
            if example.label in label_sent_mapping:
                label_sent_mapping[example.label].append(ex_index)
            else:
                label_sent_mapping[example.label] = [ex_index]
            labels.append(example.label)
            inputs.append(tokenized_text)

        grouped_inputs = []
        for i in range(len(label_sent_mapping)):
            if len(label_sent_mapping[i]) >= 2:
                grouped_inputs.extend([inputs[j] for j in label_sent_mapping[i]])
                self.labels_right_border.append(len(grouped_inputs))
                self.num_labels += 1

        tensor_labels = torch.tensor(labels, dtype=label_type)

        logging.info("Num sentences: %d" % (len(grouped_inputs)))
        logging.info("Sentences longer than max_seqence_length: {}".format(too_long))
        logging.info("Number of labels with >1 examples: {}".format(self.num_labels))
        self.tokens = grouped_inputs
        self.labels = tensor_labels

    def __getitem__(self, item):
        if not self.positive and not self.negative:
            return [self.tokens[item]], self.labels[item]

        label = bisect.bisect_right(self.labels_right_border, item)
        left_border = 0 if label == 0 else self.labels_right_border[label-1]
        right_border = self.labels_right_border[label]
        positive_item = np.random.choice(np.concatenate([self.idxs[left_border:item], self.idxs[item+1:right_border]]))
        negative_item = np.random.choice(np.concatenate([self.idxs[0:left_border], self.idxs[right_border:]]))

        if self.positive:
            positive = [self.tokens[positive_item]]
        else:
            positive = []
        if self.negative:
            negative = [self.tokens[negative_item]]
        else:
            negative = []

        return [self.tokens[item]]+positive+negative, self.labels[item]

    def __len__(self):
        return len(self.tokens)
