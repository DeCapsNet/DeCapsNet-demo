import os.path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


class MyDataset(Dataset):
    def __init__(self, tokenizer, posts, mappings, tags, max_len):
        self.tokenizer = tokenizer
        self.posts = posts  # all posts
        self.mappings = mappings  # for each user
        self.tags = tags
        self.max_len = max_len
        self.data = []
        for i in range(len(self.mappings)):
            sample = {}
            posts = self.posts[self.mappings[i][0]:self.mappings[i][-1] + 1]
            tokenized = tokenizer(posts, truncation=True, padding="max_length", max_length=max_len)
            for k, v in tokenized.items():
                sample[k] = v
            self.data.append(sample)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.tags[index]


class SelectedDataset(Dataset):
    def __init__(self, input_dir, tokenizer, max_len, split="train"):
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        self.tags = []
        input_dir1 = os.path.join(input_dir, split)
        for fname in os.listdir(input_dir1):
            label = float(fname[-5])
            sample = {}
            posts = []
            # sample["text"] = open(os.path.join(input_dir1, fname), encoding="utf-8").read()
            # tokenized = tokenizer(sample["text"], truncation=True, padding="max_length", max_length=max_len)
            with open(os.path.join(input_dir1, fname), encoding="utf-8") as f:
                temp = f.read()
                temp1 = temp.split("\n")
            for post in temp1:
                temp_post = post.split("@^_^@@^_^@@^_^@")[0]
                posts.append(temp_post)
            tokenized = tokenizer(posts, truncation=True, padding="max_length", max_length=max_len)
            for k, v in tokenized.items():
                sample[k] = v
            self.data.append(sample)  # [{"text", "input_ids"...},{}]
            self.tags.append(label)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.tags[index]


def my_collate(data):
    labels = []
    processed_batch = []
    for item, label in data:
        user_feats = {}
        for k, v in item.items():
            user_feats[k] = torch.LongTensor(v)
        processed_batch.append(user_feats)
        labels.append(label)
    labels = torch.LongTensor(np.array(labels))
    labels = labels.unsqueeze(-1)
    return processed_batch, labels
