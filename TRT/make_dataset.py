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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

templates = ["I have little interest or pleasure in dong things",
             "I always feel down, depressed and hopeless.",
             "Sometimes I have trouble falling asleep, sometimes I sleep too much.",
             "I feel tired and have little energy.",
             "Sometimes my appetite is poor, sometimes I cannot stop overeating.",
             "I feel bad about myself, think myself a failure. And I have let other people down.",
             "I have trouble concentrating on things.",
             "I move and speak much slower than before, but sometimes I have been moving around a lot more than usual.",
             "I think that I would be better off dead, and I have thoughts of hurting myself."]

sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')

templates_embs = sbert.encode(templates)

with open(f"../../data/Topic-Restricted/split_filter_All-Mental-Health/train.pkl", "rb") as f:
    train_posts, train_labels = pickle.load(f)
train_ids = [f"{i:05}" for i in range(len(train_labels))]

with open(f"../../data/Topic-Restricted/split_filter_All-Mental-Health/val.pkl", "rb") as f:
    val_posts, val_labels = pickle.load(f)
val_ids = [f"{i:05}" for i in range(len(val_labels))]

with open(f"../../data/Topic-Restricted/split_filter_All-Mental-Health/test.pkl", "rb") as f:
    test_posts, test_labels = pickle.load(f)
test_ids = [f"{i:05}" for i in range(len(test_labels))]

topK = 32

os.makedirs(f"../../data/Topic-Restricted/processed/phq9_temp2_top{topK}", exist_ok=True)
os.makedirs(f"../../data/Topic-Restricted/processed/phq9_temp2_top{topK}/train", exist_ok=True)
os.makedirs(f"../../data/Topic-Restricted/processed/phq9_temp2_top{topK}/val", exist_ok=True)
os.makedirs(f"../../data/Topic-Restricted/processed/phq9_temp2_top{topK}/test", exist_ok=True)

print("train")
for id0, posts, label in tqdm(zip(train_ids, train_posts, train_labels), total=len(train_ids)):
    posts = [x.replace("<NEGATE_FLAG>", "").replace("[ removed ]", "").strip() for x in posts]
    posts = [x for x in posts if len(x) > 0]
    embs = sbert.encode(posts)  # 当前一个用户
    posts_templates_sim = cosine_similarity(embs, templates_embs)
    sim_socres = np.sum(posts_templates_sim, axis=1)

    top_ids = sim_socres.argsort()[-topK:]
    top_ids = np.sort(top_ids)
    sel_posts = [posts[ii] for ii in top_ids]
    with open(f"../../data/Topic-Restricted/processed/phq9_temp2_top{topK}/train/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))

print("val")
for id0, posts, label in tqdm(zip(val_ids, val_posts, val_labels), total=len(val_ids)):
    posts = [x.replace("<NEGATE_FLAG>", "").replace("[ removed ]", "").strip() for x in posts]
    posts = [x for x in posts if len(x) > 0]
    embs = sbert.encode(posts)  # 当前一个用户
    posts_templates_sim = cosine_similarity(embs, templates_embs)
    sim_socres = np.sum(posts_templates_sim, axis=1)

    top_ids = sim_socres.argsort()[-topK:]
    top_ids = np.sort(top_ids)
    sel_posts = [posts[ii] for ii in top_ids]
    with open(f"../../data/Topic-Restricted/processed/phq9_temp2_top{topK}/val/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))

print("test")
for id0, posts, label in tqdm(zip(test_ids, test_posts, test_labels), total=len(test_ids)):
    posts = [x.replace("<NEGATE_FLAG>", "").replace("[ removed ]", "").strip() for x in posts]
    posts = [x for x in posts if len(x) > 0]
    embs = sbert.encode(posts)
    posts_templates_sim = cosine_similarity(embs, templates_embs)
    sim_socres = np.sum(posts_templates_sim, axis=1)

    top_ids = sim_socres.argsort()[-topK:]
    top_ids = np.sort(top_ids)
    sel_posts = [posts[ii] for ii in top_ids]
    with open(f"../../data/Topic-Restricted/processed/phq9_temp2_top{topK}/test/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))
