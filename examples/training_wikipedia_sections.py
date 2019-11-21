import argparse
import csv
import logging
from datetime import datetime

import torch
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader, DistributedSampler

import sentence_transformers as st
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.readers import TripletReader

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[st.LoggingHandler()])


def get_triplet_dataset(dataset_dir: str, csv_file: str, model: st.SentenceTransformer,
                        max_examples=0) -> st.SentencesDataset:
    triplet_reader = TripletReader(dataset_dir,
                                   s1_col_idx=0, s2_col_idx=1, s3_col_idx=2, delimiter=',',
                                   quoting=csv.QUOTE_MINIMAL, has_header=True)

    return st.SentencesDataset(examples=triplet_reader.get_examples(csv_file, max_examples),
                               model=model)


def get_data_loader(dataset: st.SentencesDataset, shuffle: bool,
                    batch_size: int, distributed=False) -> DataLoader:
    return DataLoader(dataset,
                      shuffle=shuffle if not distributed else False,
                      batch_size=batch_size,
                      sampler=DistributedSampler(dataset) if distributed else None)


def get_model(local_rank=0) -> st.SentenceTransformer:
    word_embedding_model: st.BERT = st.models.BERT('bert-base-uncased')
    pooling_model = st.models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                      pooling_mode_mean_tokens=True,
                                      pooling_mode_cls_token=False,
                                      pooling_mode_max_tokens=False)
    return st.SentenceTransformer(modules=[word_embedding_model, pooling_model],
                                  local_rank=local_rank)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = 'examples/datasets/iambot-wikipedia-sections-triplets'

    output_path = 'output/bert-base-wikipedia-sections-mean-tokens-' + \
                  datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    batch_size = 16
    num_epochs = 1

    is_distributed = torch.cuda.device_count() > 1 and args.local_rank >= 0

    if is_distributed:
        torch.distributed.init_process_group(backend='nccl')

    model = get_model(local_rank=args.local_rank)

    logging.info('Read Triplet train dataset')
    train_data = get_triplet_dataset(dataset_path, 'train.csv', model)
    train_dataloader = get_data_loader(
        dataset=train_data,
        shuffle=True,
        batch_size=batch_size,
        distributed=is_distributed)

    logging.info('Read Wikipedia Triplet dev dataset')
    dev_dataloader = get_data_loader(
        dataset=get_triplet_dataset(dataset_path, 'validation.csv', model, 1000),
        shuffle=False,
        batch_size=batch_size,
        distributed=is_distributed
    )
    evaluator = TripletEvaluator(dev_dataloader)

    warmup_steps = int(len(train_data) * num_epochs / batch_size * 0.1)

    loss = st.losses.TripletLoss(model=model)

    model.fit(train_objectives=[(train_dataloader, loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=output_path,
              local_rank=args.local_rank)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.local_rank == 0 or not is_distributed:
        model = st.SentenceTransformer(output_path)
        test_data = get_triplet_dataset(dataset_path, 'test.csv', model)
        test_dataloader = get_data_loader(test_data, shuffle=False, batch_size=batch_size)
        evaluator = TripletEvaluator(test_dataloader)

        model.evaluate(evaluator)


if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    main()
