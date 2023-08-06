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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# templates
templates = ["I have little interest or pleasure in dong things",
             "I always feel down, depressed and hopeless.",
             "Sometimes I have trouble falling asleep, sometimes I sleep too much.",
             "I feel tired and have little energy.",
             "Sometimes my appetite is poor, sometimes I cannot stop overeating.",
             "I feel bad about myself, think myself a failure. And I have let other people down.",
             "I have trouble concentrating on things.",
             "I move and speak much slower than before, but sometimes I have been moving around a lot more than usual.",
             "I think that I would be better off dead, and I have thoughts of hurting myself."]

print("1----------------1")
with open("../../data/eRisk2018/eRisk2018.pkl", "rb") as f:
    data = pickle.load(f)

sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')

train_posts = data["train_posts"]
train_mappings = data["train_mappings"]
train_tags = data["train_labels"]
train_embs = data["train_embs"]
val_posts = data["val_posts"]
val_mappings = data["val_mappings"]
val_tags = data["val_labels"]
val_embs = data["val_embs"]
test_posts = data["test_posts"]
test_mappings = data["test_mappings"]
test_tags = data["test_labels"]
test_embs = data["test_embs"]

templates_embs = sbert.encode(templates)

print("2----------------2")
# take care, require ~100G RAM
train_posts = np.array(train_posts)
val_posts = np.array(val_posts)
test_posts = np.array(test_posts)

print("3----------------3")
phq9_sim = cosine_similarity(train_embs, templates_embs)
phq9_sim = np.sum(phq9_sim, axis=1)
# phq9_sim = np.sort(phq9_sim, axis=1)
# phq9_sim = np.sum(phq9_sim[:, -3:], axis=1)
phq9_sim_val = cosine_similarity(val_embs, templates_embs)
phq9_sim_val = np.sum(phq9_sim_val, axis=1)
phq9_sim_test = cosine_similarity(test_embs, templates_embs)
phq9_sim_test = np.sum(phq9_sim_test, axis=1)
# phq9_sim_test = np.sort(phq9_sim_test, axis=1)
# phq9_sim_test = np.sum(phq9_sim_test[:, -3:], axis=1)
print(phq9_sim.shape)
print(phq9_sim_val.shape)
print(phq9_sim_test.shape)

print('4---------------------------4')
topK = 16
os.makedirs(f"../../data/eRisk2018/processed/phq9_temp2__maxsim{topK}", exist_ok=True)
os.makedirs(f"../../data/eRisk2018/processed/phq9_temp2__maxsim{topK}/train", exist_ok=True)
os.makedirs(f"../../data/eRisk2018/processed/phq9_temp2__maxsim{topK}/val", exist_ok=True)
os.makedirs(f"../../data/eRisk2018/processed/phq9_temp2__maxsim{topK}/test", exist_ok=True)

for i, (mapping, label) in enumerate(zip(train_mappings, train_tags)):
    # print(mapping)
    posts = train_posts[mapping]
    sim_scores = phq9_sim[mapping]
    top_ids = sim_scores.argsort()[-topK:]
    top_ids = np.sort(top_ids)  # sort in time order
    sel_posts = posts[top_ids]
    with open(f"../../data/eRisk2018/processed/phq9_temp2__maxsim{topK}/train/{i:06}_{label}.txt",
              "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))

print('5---------------------------5')
for i, (mapping, label) in enumerate(zip(val_mappings, val_tags)):
    # print(mapping)
    posts = val_posts[mapping]
    sim_scores = phq9_sim_val[mapping]
    top_ids = sim_scores.argsort()[-topK:]
    top_ids = np.sort(top_ids)  # sort in time order
    sel_posts = posts[top_ids]
    with open(f"../../data/eRisk2018/processed/phq9_temp2__maxsim{topK}/val/{i:06}_{label}.txt",
              "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))

print('6---------------------------6')
for i, (mapping, label) in enumerate(zip(test_mappings, test_tags)):
    posts = test_posts[mapping]
    sim_scores = phq9_sim_test[mapping]
    top_ids = sim_scores.argsort()[-topK:]
    top_ids = np.sort(top_ids)  # sort in time order
    sel_posts = posts[top_ids]
    with open(f"../../data/eRisk2018/processed/phq9_temp2__maxsim{topK}/test/{i:06}_{label}.txt",
              "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))
