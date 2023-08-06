import pickle

with open("../data/eRisk2018/eRisk2018.pkl", "rb") as f:
    data = pickle.load(f)

train_posts = data["train_posts"]
train_mappings = data["train_mappings"]
train_tags = data["train_labels"]
val_posts = data["val_posts"]
val_mappings = data["val_mappings"]
val_tags = data["val_labels"]
test_posts = data["test_posts"]
test_mappings = data["test_mappings"]
test_tags = data["test_labels"]

all = len(train_tags) + len(val_tags) + len(test_tags)
depression = sum(train_tags) + sum(val_tags) + sum(test_tags)
control = all - depression
print("all:", all)
print("depression:", depression)
print("control:", control)

all_mp = 0
for mp in train_mappings:
    all_mp += len(mp)
for mp in val_mappings:
    all_mp += len(mp)
for mp in test_mappings:
    all_mp += len(mp)
avg_mp = all_mp / all
print("avg_posts:", avg_mp)

all_length = 0
for post in train_posts:
    all_length += len(post.split())
for post in val_posts:
    all_length += len(post.split())
for post in test_posts:
    all_length += len(post.split())
avg_length = all_length / (len(train_posts) + len(val_posts) + len(test_posts))
print("avg_posts_len:", avg_length)
