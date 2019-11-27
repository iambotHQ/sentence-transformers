"""
This examples loads a pre-trained model and evaluates it on the STSbenchmark dataset
"""
import logging
from multiprocessing import set_start_method

import numpy as np
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSDataReader

try:
    set_start_method('spawn')
except RuntimeError:
    pass

np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def main():
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    sts_reader = STSDataReader('datasets/stsbenchmark')

    test_data = SentencesDataset(examples=sts_reader.get_examples('sts-test.csv'),
                                 model=model,
                                 dataset_cache_id='sts-eval')
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=16)
    evaluator = EmbeddingSimilarityEvaluator(test_dataloader)

    model.evaluate(evaluator)


if __name__ == '__main__':
    main()
