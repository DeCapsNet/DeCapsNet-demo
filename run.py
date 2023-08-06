import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.optimization import AdamW
from dataset import my_collate, SelectedDataset
from model import MyNetwork
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score
from parser_utils import get_parser
import matplotlib.pyplot as plt
import logging
import pickle
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(args, tokenizer, mode="train"):
    if mode == "train":
        print("-----------------Data Processing-----------------")
        train_set = SelectedDataset(input_dir=args.input_dir, tokenizer=tokenizer, max_len=args.max_len, split="train")
        val_set = SelectedDataset(input_dir=args.input_dir, tokenizer=tokenizer, max_len=args.max_len, split="val")
        train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, collate_fn=my_collate)
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, collate_fn=my_collate)
        return train_loader, val_loader
    elif mode == "test":
        test_set = SelectedDataset(input_dir=args.input_dir, tokenizer=tokenizer, max_len=args.max_len, split="test")
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, collate_fn=my_collate)
        return test_loader
    elif mode == "templates":
        print("-----------------Templates Processing-----------------")
        templates = ["I have little interest or pleasure in dong things",
                     "I always feel down, depressed and hopeless.",
                     "Sometimes I have trouble falling asleep, sometimes I sleep too much.",
                     "I feel tired and have little energy.",
                     "Sometimes my appetite is poor, sometimes I cannot stop overeating.",
                     "I feel bad about myself, think myself a failure. And I have let other people down.",
                     "I have trouble concentrating on things.",
                     "I move and speak much slower than before, but sometimes I have been moving around a lot more than usual.",
                     "I think that I would be better off dead, and I have thoughts of hurting myself."]
        templates_en = tokenizer(templates, truncation=True, padding="max_length", max_length=args.max_len_templates)
        templates_torch = []
        sample = {}
        for k, v in templates_en.items():
            sample[k] = torch.LongTensor(v)
        templates_torch.append(sample)
        return templates_torch


def train(args, model, optimizer, train_loader, val_loader, templates_torch, logger):
    val_f1 = 0
    Train_Loss = []
    Val_Loss = []
    Val_F1 = []

    # train
    model.train()
    print("-----------------Begin Training-----------------")
    for epoch in range(args.epochs):
        print("-----------------training epoch:{}-----------------".format(epoch))
        logger.info("epoch:" + str(epoch))
        batch_loss = 0
        all_loss = 0
        for batchi, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            train_data, train_tags = batch
            logits, ls = model(train_data, train_tags, templates_torch[0])
            all_loss += ls / len(train_tags)
            batch_loss += ls
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            batch_loss = 0
        all_loss /= len(train_loader)
        Train_Loss.append(all_loss)
        print("-----------------epoch{}: loss:{} -----------------".format(epoch, all_loss))

        all_val_loss = 0
        # evaluate
        model.eval()

        print("-----------------Begin Validating-----------------")
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                val_data, val_tags = batch
                logits, ls = model(val_data, val_tags, templates_torch[0])
                all_val_loss += ls / len(val_tags)
                if idx == 0:
                    total_pred = logits
                    total_target = val_tags
                else:
                    total_pred = torch.cat((total_pred, logits), dim=0)
                    total_target = torch.cat((total_target, val_tags), dim=0)
            all_val_loss /= len(val_loader)
        Val_Loss.append(all_val_loss)
        print(total_target.shape)  # [user_size, 1]
        target = total_target.cpu().numpy()
        total_pred = total_pred.cpu().numpy()
        pred = np.array(total_pred[:, 1] > total_pred[:, 0], dtype=float)
        score = total_pred[:, 1]
        F1 = f1_score(y_true=target, y_pred=pred)
        AUC = roc_auc_score(y_true=target, y_score=score)
        print("-----------------F1:{} , AUC:{}-----------------".format(F1, AUC))
        # print(classification_report(y_true=target, y_pred=pred))
        print("-----------------Confusion_matrix:-----------------")
        print(confusion_matrix(y_true=target, y_pred=pred))

        logger.info("seed:" + str(args.seed) + " val_F1:" + str(F1) + "  val_AUC:" + str(AUC))

        Val_F1.append(F1)
        if val_f1 < F1:
            val_f1 = F1
            print("save model")
            torch.save(model.state_dict(), args.save_path)
        print("val_F1:{}".format(val_f1))

    checkpoint = {
        "train_loss": torch.tensor(Train_Loss),
        "val_loss": torch.tensor(Val_Loss),
        "val_f1": torch.tensor(Val_F1)
    }
    return checkpoint


def test(args, model, test_loader, templates_torch, logger):
    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    print("-----------------Begin Testing-----------------")
    data = {}
    all_user = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            test_data, test_tags = batch
            if args.save_embeddings:
                logits, loss, batch_user, templates_encoder, capsule_B, capsule_R = model(test_data, test_tags,
                                                                                          templates_torch[0])
                all_user = all_user + batch_user
            else:
                logits, loss = model(test_data, test_tags, templates_torch[0])
            if idx == 0:
                total_pred = logits
                total_target = test_tags
            else:
                total_pred = torch.cat((total_pred, logits), dim=0)
                total_target = torch.cat((total_target, test_tags), dim=0)
    if args.save_embeddings:
        data["all_user"] = all_user
        data["template_encoder"] = templates_encoder
        data["coefficient_b"] = capsule_B
        data["coefficient_c"] = capsule_R
        pickle.dump(data, open(args.save_embedding_path, "wb"))
    print(total_target.shape)  # [user_size, 1]
    print(total_pred.shape)  # [user_size, 2]
    target = total_target.cpu().numpy()
    total_pred = total_pred.cpu().numpy()
    pred = np.array(total_pred[:, 1] > total_pred[:, 0], dtype=float)
    # print(total_pred)
    score = total_pred[:, 1]
    F1 = f1_score(y_true=target, y_pred=pred)
    AUC = roc_auc_score(y_true=target, y_score=score)
    # ACC = accuracy_score(y_true=target, y_pred=pred)
    print("-----------------F1:{} , AUC:{}-----------------".format(F1, AUC))
    # print(classification_report(y_true=target, y_pred=pred))
    print("-----------------Confusion_matrix:-----------------")
    print(confusion_matrix(y_true=target, y_pred=pred))

    logger.info("bsz:" + str(args.train_batch_size)  + "  l1:" + str(
        args.alpha) + "  l2:" + str(args.beta) + "  l3:" + str(args.gamma) + "  seed:" + str(args.seed) + " lr:" + str(args.learning_rate) + " dropout:" + str(
        args.dropout) + " F1:" + str(F1) + "  AUC:" + str(AUC))
    return F1, AUC


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def plot_loss(args):
    checkpoint = torch.load(args.save_ls_path)
    train_loss = checkpoint["train_loss"]
    val_loss = checkpoint["val_loss"]
    val_f1 = checkpoint["val_f1"]
    epoch = np.arange(len(train_loss))
    plt.figure()
    plt.plot(epoch, train_loss, label="train_loss")
    plt.plot(epoch, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    plt.plot(epoch, val_f1, label="val_f1")
    plt.xlabel("epoch")
    plt.ylabel("val_f1")
    plt.legend()
    plt.show()


def main():
    args = get_parser().parse_args()
    setup_seed(args.seed)
    logger = get_logger(args.save_log_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    # model
    model = MyNetwork(args, freeze=False).cuda()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    # test
    if args.only_test:
        # load data
        test_loader = load_data(args, tokenizer, mode="test")
        templates_torch = load_data(args, tokenizer, mode="templates")

        # test
        F1, AUC = test(args, model, test_loader, templates_torch, logger)
    # train
    else:
        train_loader, val_loader = load_data(args, tokenizer, mode="train")
        templates_torch = load_data(args, tokenizer, mode="templates")

        # 训练
        checkpoint = train(args, model, optimizer, train_loader, val_loader, templates_torch, logger)

        # 测试
        test_loader = load_data(args, tokenizer, mode="test")
        F1, AUC = test(args, model, test_loader, templates_torch, logger)
        if args.save_loss:
            checkpoint["test_F1"] = torch.tensor(F1)
            checkpoint["test_AUC"] = torch.tensor(AUC)
            torch.save(checkpoint, args.save_ls_path)


if __name__ == '__main__':
    main()


