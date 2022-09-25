# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import os

from datasets import load_dataset
from src.utils import target_offset
import numpy as np
import datasets

def load_data(args):
    if args.dataset == "dbpedia14":
        dataset = load_dataset("csv", column_names=["label", "title", "sentence"],
                                data_files={"train": os.path.join(args.data_folder, "dbpedia_csv/train.csv"),
                                            "validation": os.path.join(args.data_folder, "dbpedia_csv/test.csv")})
        dataset = dataset.map(target_offset, batched=True)
        num_labels = 14
    elif args.dataset == "ag_news":
        dataset = load_dataset("ag_news")
        num_labels = 4
    elif args.dataset == "imdb":
        dataset = load_dataset("imdb", ignore_verifications=True, cache_dir = '/mnt/cloud/bairu/repos/dataset_cache/imdb/')
        num_labels = 2
    elif args.dataset == "yelp":
        dataset = load_dataset("yelp_polarity")
        num_labels = 2
    elif args.dataset == "mnli":
        dataset = load_dataset("glue", "mnli")
        num_labels = 3
    elif args.dataset == 'sst':
        data_files = {'train': "train.tsv", 'test': "test.tsv"}
        dataset = load_dataset("sst-2", data_files = data_files, column_names = ['text', 'label'])
        num_labels = 2
    dataset = dataset.shuffle(seed=0)
    
    

    return dataset, num_labels


class IMDBDataset():
    def __init__(self,):
        imdb_dataset = datasets.load_dataset("imdb", cache_dir = '/mnt/cloud/bairu/dataset_cache/imdb/')
        imdb_dataset.pop("unsupervised")
        imdb_dataset = imdb_dataset.map(self.preprocess_imdb, batched=True,)
        # imdb_dataset = imdb_dataset.map(self.tokenize_corpus, batched=True,)

        train_dataset = imdb_dataset['train']
        test_dataset = imdb_dataset['test']
        
        num_train = len(train_dataset['sentence'])
        num_test = len(test_dataset['sentence'])
        np.random.seed(10)
        rand_train_order = np.random.permutation(num_train)
        rand_test_order = np.random.permutation(num_test)
        train_idxs = rand_train_order[:20000]
        valid_idxs = rand_train_order[20000:]
        test_idxs = rand_test_order
        self.train_dataset = train_dataset.select(train_idxs)
        self.valid_dataset = train_dataset.select(valid_idxs)
        self.test_dataset = test_dataset.select(test_idxs)



    def preprocess_imdb(self, examples):
        orig_texts = examples['text']
        filtered_texts = []
        for text in orig_texts:
            filtered_text = text.replace("<br />", "")
            filtered_texts.append(filtered_text)
        return {"sentence": filtered_texts}


