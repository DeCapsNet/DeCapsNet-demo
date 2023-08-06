import random
import re
import os
import pickle
import xml.dom.minidom
import emoji
from tqdm import tqdm
from TextPreProcessor import text_processor
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def get_input_data(path):
    post_num = 0
    dom = xml.dom.minidom.parse(path)
    collection = dom.documentElement
    title = collection.getElementsByTagName('TITLE')
    text = collection.getElementsByTagName('TEXT')
    tim = collection.getElementsByTagName('DATE')
    posts = []
    for i in range(len(title)):
        post = title[i].firstChild.data + " " + text[i].firstChild.data
        post = re.sub('\n', ' ', post)
        if len(post) > 0:
            posts.append(post.strip())
            post_num = post_num + 1
    tempop = []
    for post in posts:
        temp_post = " ".join(text_processor.pre_process_doc(post))
        temp_post = emoji.demojize(temp_post)
        tempop.append(temp_post)

    # 去重并按照时间顺序排序
    list2 = list(set(tempop))
    list2.sort(key=tempop.index)
    posts = list(reversed(list2))
    post_num = len(posts)
    return posts, post_num

'''8/1/1'''
train_posts = []
train_mappings = []
train_embs = []
train_names = []
train_labels = []
val_posts = []
val_mappings = []
val_embs = []
val_names = []
val_labels = []
test_posts = []
test_mappings = []
test_embs = []
test_names = []
test_labels = []

path = "../../data/eRisk2018/combine/"
filenames = sorted(os.listdir(path))
# split
index = list(range(len(filenames)))
random_state = random.randint(0, 10000)
train_index, test_index, _, _ = train_test_split(index, index, test_size=0.2, random_state=random_state, shuffle=True)
val_index, test_index, _, _ = train_test_split(test_index, test_index, test_size=0.5, random_state=random_state,
                                               shuffle=True)

print("1-------------------1")
for idx, fname in tqdm(enumerate(filenames), total=len(filenames)):
    posts, post_num = get_input_data(path + fname)
    if idx in train_index:
        train_mappings.append(list(range(len(train_posts), len(train_posts) + post_num)))
        train_posts.extend(posts)
        train_labels.append(int(fname[-5]))
        train_names.append(str(fname[:-6]))
    elif idx in test_index:
        test_mappings.append(list(range(len(test_posts), len(test_posts) + post_num)))
        test_posts.extend(posts)
        test_labels.append(int(fname[-5]))
        test_names.append(str(fname[:-6]))
    elif idx in val_index:
        val_mappings.append(list(range(len(val_posts), len(val_posts) + post_num)))
        val_posts.extend(posts)
        val_labels.append(int(fname[-5]))
        val_names.append(str(fname[:-6]))

print(len(train_posts))
print(len(train_labels))
print(len(val_posts))
print(len(val_labels))
print(len(test_posts))
print(len(test_labels))

print("2-------------------2")
train_embs = sbert.encode(train_posts, convert_to_tensor=False)
print(train_embs.shape)
val_embs = sbert.encode(val_posts, convert_to_tensor=False)
print(val_embs.shape)
test_embs = sbert.encode(test_posts, convert_to_tensor=False)
print(test_embs.shape)

print("3-------------------3")
with open("../../data/eRisk2018/eRisk2018" + ".pkl", "wb") as f:
    pickle.dump({
        "train_posts": train_posts,
        "train_mappings": train_mappings,
        "train_labels": train_labels,
        "train_embs": train_embs,
        "train_names": train_names,
        "val_posts": val_posts,
        "val_mappings": val_mappings,
        "val_labels": val_labels,
        "val_embs": val_embs,
        "val_names": val_names,
        "test_posts": test_posts,
        "test_mappings": test_mappings,
        "test_labels": test_labels,
        "test_embs": test_embs,
        "test_names": test_names}, f)

