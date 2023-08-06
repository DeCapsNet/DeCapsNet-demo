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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

with open("../../data/RSDD/processed/training.pkl", "rb") as f:
    user_ids_train, user_posts_train, user_labels_train = pickle.load(f)

with open("../../data/RSDD/processed/validation.pkl", "rb") as f:
    user_ids_val, user_posts_val, user_labels_val = pickle.load(f)

with open("../../data/RSDD/processed/testing.pkl", "rb") as f:
    user_ids_test, user_posts_test, user_labels_test = pickle.load(f)

topK = 32

os.makedirs(f"../../data/RSDD/processed/phq9_temp2_top{topK}", exist_ok=True)
os.makedirs(f"../../data/RSDD/processed/phq9_temp2_top{topK}/train", exist_ok=True)
os.makedirs(f"../../data/RSDD/processed/phq9_temp2_top{topK}/val", exist_ok=True)
os.makedirs(f"../../data/RSDD/processed/phq9_temp2_top{topK}/test", exist_ok=True)

print("train")
for id0, posts, label in tqdm(zip(user_ids_train, user_posts_train, user_labels_train), total=len(user_ids_train)):
    embs = sbert.encode(posts)
    pair_sim = cosine_similarity(embs, templates_embs)  # [posts, templates]
    sim_scores = np.sum(pair_sim, axis=1)  # [posts, 1]
    top_ids = sim_scores.argsort()[-topK:]
    top_ids = np.sort(top_ids)  # sort in time order
    sel_posts = [posts[ii] for ii in top_ids]
    with open(f"../../data/RSDD/processed/phq9_temp2_top{topK}/train/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))

print("validation")
# val
for id0, posts, label in tqdm(zip(user_ids_val, user_posts_val, user_labels_val), total=len(user_ids_val)):
    embs = sbert.encode(posts)
    pair_sim = cosine_similarity(embs, templates_embs)  # [posts, templates]
    sim_scores = np.sum(pair_sim, axis=1)  # [posts, 1]
    top_ids = sim_scores.argsort()[-topK:]
    top_ids = np.sort(top_ids)  # sort in time order
    sel_posts = [posts[ii] for ii in top_ids]
    with open(f"../../data/RSDD/processed/phq9_temp2_top{topK}/val/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))

# test
for id0, posts, label in tqdm(zip(user_ids_test, user_posts_test, user_labels_test), total=len(user_ids_test)):
    embs = sbert.encode(posts)
    pair_sim = cosine_similarity(embs, templates_embs)  # [posts, templates]
    sim_scores = np.sum(pair_sim, axis=1)  # [posts, 1]
    top_ids = sim_scores.argsort()[-topK:]
    top_ids = np.sort(top_ids)  # sort in time order
    sel_posts = [posts[ii] for ii in top_ids]
    with open(f"../../data/RSDD/processed/phq9_temp2_top{topK}/test/{id0}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))
