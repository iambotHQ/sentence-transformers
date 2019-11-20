"""
This script trains sentence transformers with a triplet loss function.

As corpus, we use the wikipedia sections dataset that was describd by Dor et al., 2018, Learning Thematic Similarity Metric Using Triplet Networks.

See docs/pretrained-models/wikipedia-sections-modesl.md for further details.

You can get the dataset by running examples/datasets/get_data.py
"""
import torch

from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, \
    models
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method
from sentence_transformers.readers import TripletReader
from sentence_transformers.evaluation import TripletEvaluator
from datetime import datetime

import csv
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

try:
    set_start_method('spawn')
except RuntimeError:
    pass

if __name__ == '__main__':
    # Create a torch.DataLoader that passes training batch instances to our model
    cuda_batch_size_multiplier = max(torch.cuda.device_count(), 1)
    train_batch_size = 16 * cuda_batch_size_multiplier
    triplet_reader = TripletReader('examples/datasets/iambot-wikipedia-sections-triplets',
                                   s1_col_idx=0, s2_col_idx=1, s3_col_idx=2, delimiter=',',
                                   quoting=csv.QUOTE_MINIMAL, has_header=True)
    output_path = "output/bert-base-wikipedia-sections-mean-tokens-" + datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S")
    num_epochs = 1

    # Configure sentence transformers for training and train on the provided dataset
    # Use BERT for mapping tokens to embeddings
    word_embedding_model = models.BERT('bert-base-uncased')

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    logging.info("Read Triplet train dataset")
    train_data = SentencesDataset(examples=triplet_reader.get_examples('train.csv'), model=model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size,
                                  num_workers=cuda_batch_size_multiplier)
    train_loss = losses.TripletLoss(model=model)

    logging.info("Read Wikipedia Triplet dev dataset")
    dev_data = SentencesDataset(examples=triplet_reader.get_examples('validation.csv', 1000),
                                model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
    evaluator = TripletEvaluator(dev_dataloader)

    warmup_steps = int(len(train_data) * num_epochs / train_batch_size * 0.1)  # 10% of train data

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=output_path)

    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################

    model = SentenceTransformer(output_path)
    test_data = SentencesDataset(examples=triplet_reader.get_examples('test.csv'), model=model)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=train_batch_size)
    evaluator = TripletEvaluator(test_dataloader)

    model.evaluate(evaluator)
