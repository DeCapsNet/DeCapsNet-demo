import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class MyNetwork(nn.Module):
    def __init__(self, args, freeze=True):
        super(MyNetwork, self).__init__()
        self.num_symptoms = args.num_symptoms
        self.bert_dim = args.bert_dim
        self.alpha = args.alpha
        self.beta = args.beta
        self.temperature1 = args.temperature1
        self.gamma = args.gamma
        self.temperature2 = args.temperature2
        self.encoder = AutoModel.from_pretrained(args.model_type)
        self.save_embeddings = args.save_embeddings
        if freeze:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
        else:
            unfreeze = ["pooler", "encoder.layer.11", "encoder.layer.10", "encoder.layer.9", "encoder.layer.8"]
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
                for unname in unfreeze:
                    if unname in name:
                        param.requires_grad = True
        self.capsule = CapsuleNetwork(args)

        self.template_projection = nn.Parameter(torch.rand((768, 768))).cuda()
        self.posts_projection = nn.Parameter(torch.rand((768, 768))).cuda()

        self.init_weights()

    def forward(self, batch, labels, templates):
        templates_outputs = self.encoder(templates["input_ids"].cuda(), templates["attention_mask"].cuda(),
                                         templates["token_type_ids"].cuda())
        templates_encoder = torch.mean(templates_outputs.last_hidden_state, dim=1)  # [9, 768]
        templates_encoder = torch.matmul(templates_encoder, self.template_projection)
        bsz = len(batch)
        batch_xx = torch.zeros([bsz, self.num_symptoms, self.bert_dim], dtype=torch.float).cuda()
        all_posts_label = []
        attention_loss = 0.0
        batch_user = []
        for batchi, user_feats in enumerate(batch):
            post_outputs = self.encoder(user_feats["input_ids"].cuda(), user_feats["attention_mask"].cuda(),
                                        user_feats["token_type_ids"].cuda())
            x = torch.mean(post_outputs.last_hidden_state, dim=1)  # [16, 768]
            x = torch.matmul(x, self.posts_projection)

            d = x.shape[-1]
            if batchi == 0:
                all_posts = x
            else:
                all_posts = torch.cat([all_posts, x], dim=0)
            all_posts_label += [labels[batchi]] * x.shape[0]

            similarity = torch.matmul(x, templates_encoder.transpose(0, 1)) / math.sqrt(d)
            similarity1 = similarity

            # softmax
            similarity = F.softmax(similarity, dim=0)  # [16, 9]
            batch_xx[batchi] = torch.matmul(similarity.transpose(0, 1), x)
            attention_loss += self.attention_loss(similarity)

            # 保存pkl
            if self.save_embeddings:
                sample = {}
                sample["posts_emb"] = x.to("cpu")
                sample["user_emb"] = batch_xx[batchi].to("cpu")
                sample["similarity"] = similarity1.to("cpu")
                sample["attention"] = similarity.to("cpu")
                sample["label"] = labels[batchi].to("cpu")
                batch_user.append(sample)
        attention_loss /= bsz
        batch_xx_flatten = batch_xx.reshape(bsz, -1)  # user representations
        all_posts_label = torch.LongTensor(all_posts_label).cuda()
        # print(batch_xx.shape)
        # print(batch_xx_flatten.shape)
        # print(all_posts.shape)
        # print(all_posts_label.shape)
        contrastive_user_loss = self.contrastive_loss(batch_xx_flatten, labels, self.temperature1)
        contrastive_post_loss = self.contrastive_loss(all_posts, all_posts_label, self.temperature2)
        logits, margin_loss, capsule_B, capsule_R = self.capsule(batch_xx, labels)
        loss = 100 * margin_loss + self.alpha * attention_loss + self.beta * contrastive_user_loss + self.gamma * contrastive_post_loss
        if self.save_embeddings:
            return logits, loss, batch_user, templates_encoder.to("cpu"), capsule_B.to("cpu"), capsule_R.to("cpu")
        else:
            return logits, loss

    def attention_loss(self, attention):
        # attention [16, 9]
        I = torch.eye(attention.shape[1], attention.shape[1]).cuda()
        return torch.norm(torch.matmul(attention.T, attention) - I) ** 2

    def contrastive_loss(self, features, labels, temperature=0.1):
        # features: bsz, dim
        features = F.normalize(features, p=2, dim=1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().cuda()

        # compute logits, anchor_dot_contrast: (bsz, bsz), x_i_j: (z_i*z_j)/t
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        if torch.any(torch.isnan(log_prob)):
            raise ValueError("Log_prob has nan!")

        # compute mean of log-likelihood over positive
        mask_sum = mask.sum(1)
        mask_sum[mask_sum == 0] += 1
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum)

        if torch.any(torch.isnan(mean_log_prob_pos)):
            raise ValueError("mean_log_prob_pos has nan!")

        loss = - mean_log_prob_pos
        if torch.any(torch.isnan(loss)):
            raise ValueError("loss has nan!")
        loss = loss.mean()
        return loss

    def init_weights(self):
        nn.init.xavier_uniform_(self.template_projection)
        nn.init.xavier_uniform_(self.posts_projection)

        self.template_projection.requires_grad_(True)
        self.posts_projection.requires_grad_(True)


class CapsuleNetwork(nn.Module):
    def __init__(self, args):
        super(CapsuleNetwork, self).__init__()
        self.R = args.num_symptoms
        self.D = args.bert_dim
        self.output_dim = args.output_dim
        self.output_atoms = args.output_atoms
        self.num_routing = args.num_routing
        self.save_embeddings = args.save_embeddings

        # [R, D, output_dim*output_atoms]
        self.capsule_W = nn.Parameter(torch.rand((self.R, self.D, self.output_dim * self.output_atoms))).cuda()
        self.init_weights()

    def forward(self, input_tensor, labels):
        # [B, R, D, output_dim*output_atoms]
        input_tiled = torch.unsqueeze(input_tensor, -1).repeat(1, 1, 1, self.output_dim * self.output_atoms)
        # print(input_tiled.shape)
        # [B, R, output_dim*output_atoms]
        capsule_P = torch.sum(input_tiled * self.capsule_W, dim=2)
        # [B, R, output_dim, output_atoms]
        capsule_P_reshaped = torch.reshape(capsule_P, [-1, self.R, self.output_dim, self.output_atoms])
        # [B, R, output_dim]
        capsule_B_shape = np.stack([input_tensor.shape[0], self.R, self.output_dim])
        self.capsule_V, self.capsule_B, self.capsule_R = self.routing(capsule_P_reshaped, capsule_B_shape, num_dims=4)

        self.logits = self.get_logits()  # [post_num, 1]
        ls = self.margin_loss(labels.cuda(), self.logits, max_margin=0.95, min_margin=0.05,
                              downweight=0.5)  # [post_num]

        self.loss = torch.mean(ls)
        # print(self.logits)
        return self.logits, self.loss, self.capsule_B, self.capsule_R

    def routing(self, capsule_P, capsule_B_shape, num_dims):
        # print(capsule_P)
        tran_shape = [3, 0, 1, 2]
        for i in range(num_dims - 4):
            tran_shape += [i + 4]
        return_shape = [1, 2, 3, 0]
        for i in range(num_dims - 4):
            return_shape += [i + 4]

        # capsule_P [B, R, output_dim, output_atoms]
        # capsule_P_trans [output_atoms, B, R, output_dim]
        capsule_P_trans = capsule_P.permute(tran_shape)
        # capsule_B [B, R, output_dim]
        capsule_B = torch.zeros(capsule_B_shape[0], capsule_B_shape[1], capsule_B_shape[2]).cuda()
        capsule_Vs = []

        for iter in range(self.num_routing):
            # capsule_R [B, R, output_dim]
            capsule_R = F.softmax(capsule_B, dim=2)
            # capsule_S_unrolled [output_atoms, B, R, output_dim]
            capsule_S_unrolled = capsule_R * capsule_P_trans
            # capsule_S [B, R, output_dim, output_atoms]
            capsule_S_trans = capsule_S_unrolled.permute(return_shape)
            # capsule_S [B, output_dim, output_atoms]
            capsule_S = torch.sum(capsule_S_trans, dim=1)
            # print(capsule_S)
            # capsule_V [B, output_dim, output_atoms]
            capsule_V = self.squash_(capsule_S)
            capsule_Vs.append(capsule_V)

            tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
            tile_shape[1] = self.R  # [1, R, 1, 1]
            capsule_V_rep = capsule_V.unsqueeze(1).repeat(tile_shape)
            capsule_B = capsule_B + torch.sum(capsule_P * capsule_V_rep, dim=3)
        return capsule_Vs[self.num_routing - 1], capsule_B, capsule_R

    def squash_(self, input_tensor):
        norm = torch.norm(input_tensor, dim=2, keepdim=True)
        norm_squared = norm * norm
        return (input_tensor / norm) * (norm_squared / (1 + norm_squared))

    def get_logits(self):
        logits = torch.norm(self.capsule_V, dim=-1)
        return logits

    def init_weights(self):
        nn.init.xavier_uniform_(self.capsule_W)
        self.capsule_W.requires_grad_(True)

    def margin_loss(self, labels, raw_logits, max_margin=0.95, min_margin=0.05, downweight=0.5):
        labels_onehot = F.one_hot(labels, 2).squeeze()
        positive_cost = labels_onehot * (raw_logits < max_margin).float() * ((max_margin - raw_logits) ** 2)
        negative_cost = (1 - labels_onehot) * (raw_logits > min_margin).float() * ((raw_logits - min_margin) ** 2)
        cost = torch.sum(0.5 * positive_cost + downweight * 0.5 * negative_cost, dim=1)  # [post_num]
        return cost
