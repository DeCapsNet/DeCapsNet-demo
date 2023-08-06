import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
import xml.dom.minidom
import string
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import random

def subset(alist, idxs):
    sub_list = []
    for idx in idxs:
        sub_list.append(alist[idx])

    return sub_list


def split_list(alist, group_num=4, shuffle=True, retain_left=False):
    index = list(range(len(alist)))

    if shuffle:
        random.shuffle(index)

    elem_num = len(alist) // group_num
    sub_lists = {}

    for idx in range(group_num):
        start, end = idx * elem_num, (idx + 1) * elem_num
        sub_lists['set' + str(idx)] = subset(alist, index[start:end])

    if retain_left and group_num * elem_num != len(index):
        sub_lists['set' + str(idx + 1)] = subset(alist, index[end:])

    return sub_lists


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

with open(f"../../data/Topic-Restricted/split_filter_All-Mental-Health/train.pkl", "rb") as f:
    train_posts, train_labels = pickle.load(f)  # 114477 32

new_train_posts = []
new_train_labels = []
new_depress = 0
new_control = 0

for posts, label in tqdm(zip(train_posts, train_labels), total=len(train_labels)):
    posts = [x.replace("<NEGATE_FLAG>", "").replace("[ removed ]", "").strip() for x in posts]
    posts = [x for x in posts if len(x) > 0]
    if label == 1:
        print(posts)
        break
        new_train_labels.append(1)
        new_train_posts.append(posts)
        new_depress += 1
    elif label == 0:
        new_control += 1
        new_train_labels.append(0)
        new_train_posts.append(posts)
print(new_control)
print(new_depress)

with open(f"../../data/Topic-Restricted/split_filter_All-Mental-Health/train.pkl", "rb") as f:
    train_posts, train_labels = pickle.load(f)  # 114477 32

new_train_posts = []
new_train_labels = []
new_depress = 0
new_control = 0
depress = sum(train_labels)
control = len(train_labels) - depress

k = 0
for posts, label in tqdm(zip(train_posts, train_labels), total=len(train_labels)):
    k += 1
    # if k >= 2000:
    #     break
    posts = [x.replace("<NEGATE_FLAG>", "").replace("[ removed ]", "").strip() for x in posts]
    posts = [x for x in posts if len(x) > 0]
    if label == 1:
        new_train_labels.append(1)
        new_train_posts.append(posts)
        new_depress += 1
    elif label == 0:
        if len(new_train_labels):
            if len(posts) > 1000:
                tmp_len = len(posts)
                nn = int(len(posts) / 300)
                new_control += nn
                split = split_list(range(nn), group_num=nn, retain_left=False)
                for i in range(nn):
                    now_posts = []
                    for idxx in sorted(split["set" + str(i)]):
                        now_posts.append(posts[idxx])
                    new_train_posts.append(now_posts)
                    new_train_labels.append(0)
            else:
                new_control += 1
                new_train_labels.append(0)
                new_train_posts.append(posts)
print(new_control)
print(new_depress)
print(control)
print(depress)

with open(f"../../data/Topic-Restricted/split_filter_All-Mental-Health/val.pkl", "rb") as f:
    train_posts, train_labels = pickle.load(f)  # 114477 32

depress = sum(train_labels)
control = len(train_labels) - depress

k = 0
for posts, label in tqdm(zip(train_posts, train_labels), total=len(train_labels)):
    k += 1
    # if k >= 2000:
    #     break
    posts = [x.replace("<NEGATE_FLAG>", "").replace("[ removed ]", "").strip() for x in posts]
    posts = [x for x in posts if len(x) > 0]
    if label == 1:
        new_train_labels.append(1)
        new_train_posts.append(posts)
        new_depress += 1
    elif label == 0:
        if len(new_train_labels):
            if len(posts) > 1000:
                tmp_len = len(posts)
                nn = int(len(posts) / 300)
                new_control += nn
                split = split_list(range(nn), group_num=nn, retain_left=False)
                for i in range(nn):
                    now_posts = []
                    for idxx in sorted(split["set" + str(i)]):
                        now_posts.append(posts[idxx])
                    new_train_posts.append(now_posts)
                    new_train_labels.append(0)
            else:
                new_control += 1
                new_train_labels.append(0)
                new_train_posts.append(posts)
print(new_control)
print(new_depress)
print(control)
print(depress)

with open(f"../../data/Topic-Restricted/split_filter_All-Mental-Health/test.pkl", "rb") as f:
    train_posts, train_labels = pickle.load(f)  # 114477 32

depress = sum(train_labels)
control = len(train_labels) - depress

k = 0
for posts, label in tqdm(zip(train_posts, train_labels), total=len(train_labels)):
    k += 1
    # if k >= 2000:
    #     break
    posts = [x.replace("<NEGATE_FLAG>", "").replace("[ removed ]", "").strip() for x in posts]
    posts = [x for x in posts if len(x) > 0]
    if label == 1:
        new_train_labels.append(1)
        new_train_posts.append(posts)
        new_depress += 1
    elif label == 0:
        if len(new_train_labels):
            if len(posts) > 1000:
                tmp_len = len(posts)
                nn = int(len(posts) / 300)
                new_control += nn
                split = split_list(range(nn), group_num=nn, retain_left=False)
                for i in range(nn):
                    now_posts = []
                    for idxx in sorted(split["set" + str(i)]):
                        now_posts.append(posts[idxx])
                    new_train_posts.append(now_posts)
                    new_train_labels.append(0)
            else:
                new_control += 1
                new_train_labels.append(0)
                new_train_posts.append(posts)
print(new_control)
print(new_depress)
print(control)
print(depress)

print(len(new_train_labels))
print(len(new_train_posts))

index = list(range(len(new_train_labels)))
random_state = random.randint(0, 10000)
train_index, test_index, _, _ = train_test_split(index, index, test_size=0.2, random_state=random_state, shuffle=True)
val_index, test_index, _, _ = train_test_split(test_index, test_index, test_size=0.5, random_state=random_state,
                                               shuffle=True)
final_train_posts = []
final_train_labels = []
with open("../../data/Topic-Restricted/split_filter_All-Mental-Health/train_new.pkl", "wb") as f:
    for i in train_index:
        final_train_labels.append(new_train_labels[i])
        final_train_posts.append(new_train_posts[i])
    pickle.dump([final_train_posts, final_train_labels], f)

final_val_posts = []
final_val_labels = []
with open("../../data/Topic-Restricted/split_filter_All-Mental-Health/val_new.pkl", "wb") as f:
    for i in val_index:
        final_val_labels.append(new_train_labels[i])
        final_val_posts.append(new_train_posts[i])
    pickle.dump([final_val_posts, final_val_labels], f)

final_test_posts = []
final_test_labels = []
with open("../../data/Topic-Restricted/split_filter_All-Mental-Health/test_new.pkl", "wb") as f:
    for i in test_index:
        final_test_labels.append(new_train_labels[i])
        final_test_posts.append(new_train_posts[i])
    pickle.dump([final_test_posts, final_test_labels], f)
