'''
MF base model , used for SML and MF-based baselines
class MFbasemode : the model we used
'''
import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
from torch.autograd import Variable
import os
try:
    from tqdm import tqdm
except:
    pass
import  torch.nn.functional as F
import numpy as np

class MFbasemode(nn.Module):
    def __init__(self, num_user=0, num_item=0, laten_factor=10):
        super(MFbasemode, self).__init__()
        self.user_bais = nn.Embedding(num_user, 1)
        self.item_bais = nn.Embedding(num_item, 1)
        self.user_laten = nn.Embedding(num_user, laten_factor)
        self.item_laten = nn.Embedding(num_item, laten_factor)
        self.user_num = num_user
        self.item_num = num_item
        self.hidden_dim = laten_factor
    def reset_parameters(self):
        self.user_bais.reset_parameters()
        self.user_laten.reset_parameters()
        self.item_bais.reset_parameters()
        self.item_laten.reset_parameters()

    def forward(self, user, item,norm=False):
        user_bais = self.user_bais(user).squeeze(-1)
        item_bais = self.item_bais(item).squeeze(-1)
        userembedding = self.user_laten(user)
        itemembedding = self.item_laten(item)
        interaction = torch.mul(userembedding,itemembedding).sum(dim=-1)
        if norm:
            interaction = interaction / (userembedding**2).sum(dim=-1).sqrt()
        result = interaction
        return userembedding,itemembedding,result

    def test(self,inputs_data,topK=20):
        user = inputs_data[:, 0]
        pos_negs_items  = inputs_data[:, 1:] #all item
        user_embedding = self.user_laten(user)
        user_bias = self.user_bais(user)
        pos_negs_items_embedding = self.item_laten(pos_negs_items) # batch * 999 * item_laten
        pos_negs_bias = self.item_bais(pos_negs_items)

        user_embedding_ = user_embedding.unsqueeze(1) # add a din to user embedding
        pos_negs_interaction = torch.mul(user_embedding_,pos_negs_items_embedding).sum(-1)
        #pos_negs_interaction = torch.chain_matmul(user_embedding,pos_negs_items_embedding)  # we can use this code to compute the interaction

        pos_negs_scores =  pos_negs_interaction
        # the we have compute all score for pos and neg interactions ,each row has scorces of one pos inter and neg_num(99)  neg inter
        _, rank = torch.topk(pos_negs_scores, topK)
        pos_rank_idx = (rank < 1 ).nonzero()  # where hit 0, o is our target item
        if torch.any((rank<1).sum(-1)>1):
            print("user embedding:",user_embedding[0])
            print("item embedding:",pos_negs_items[0])
            print("score:",pos_negs_interaction[0])
            print("rank",rank)
            print("pos_rank:", pos_rank_idx)
            np.save("pos_negs.npy",pos_negs_interaction.data.cpu().numpy())
            raise RuntimeError("compute rank error")
        have_hit_num = pos_rank_idx.shape[0]
        have_hit_rank = pos_rank_idx[:,1].float()
        # DCG_ = 1.0 / torch.log(have_hit_rank+2)
        # NDCG = DCG_ * torch.log(torch.Tensor([2.0]).cuda())
        if have_hit_num>0:
            batch_NDCG = 1/torch.log2(have_hit_rank+2)   #NDCG is equal to this format
            batch_NDCG = batch_NDCG.sum()
        else:
            batch_NDCG = 0
        Batch_hit = have_hit_num * 1.0

        return Batch_hit,batch_NDCG,pos_rank_idx[:,0]

    def test2(self,inputs_data,topK=20):
        user = inputs_data[:,0]
        pos_negs_items  = inputs_data[:,1:] #all item
        user_embedding = self.user_laten(user)
        user_bias = self.user_bais(user)
        pos_negs_items_embedding = self.item_laten(pos_negs_items) # batch * 999 * item_laten
        pos_negs_bias = self.item_bais(pos_negs_items)

        user_embedding = user_embedding.unsqueeze(1) # add a din to user embedding
        pos_negs_interaction = torch.mul(user_embedding,pos_negs_items_embedding).sum(-1)
        #pos_negs_interaction = torch.chain_matmul(user_embedding,pos_negs_items_embedding)  # we can use this code to compute the interaction

        pos_negs_scores =  pos_negs_interaction
        # the we have compute all score for pos and neg interactions ,each row has scorces of one pos inter and neg_num(99)  neg inter
        _,rank = torch.topk(pos_negs_scores, topK)
        pos_rank_idx = (rank < 1 ).nonzero()  # where hit 0, o is our target item
        have_hit_num = pos_rank_idx.shape[0]
        have_hit_rank = pos_rank_idx[:,1].float()
        # DCG_ = 1.0 / torch.log(have_hit_rank+2)
        # NDCG = DCG_ * torch.log(torch.Tensor([2.0]).cuda())
        if have_hit_num>0:
            batch_NDCG = 1/torch.log2(have_hit_rank+2)   #NDCG is equal to this format
            batch_NDCG = batch_NDCG.sum()
        else:
            batch_NDCG = 0
        Batch_hit = have_hit_num * 1.0
        return pos_rank_idx[:,0],rank,Batch_hit, batch_NDCG

    def set_parameters(self,user_weight,item_weight):
        self.user_laten.weight.data.copy_(user_weight[:,0:-1])
        self.user_bais.weight.data.copy_(user_weight[:,-1].unsqueeze(-1))
        self.item_laten.weight.data.copy_(item_weight[:,0:-1])
        self.item_bais.weight.data.copy_(item_weight[:,-1].unsqueeze(-1))



class MF2(nn.Module):
    def __init__(self, num_user=0, num_item=0, laten_factor=10):
        super(MF2, self).__init__()
        self.user_bais = nn.Embedding(num_user, 1)
        self.item_bais = nn.Embedding(num_item, 1)
        self.user_laten = nn.Embedding(num_user, laten_factor)
        self.item_laten = nn.Embedding(num_item, laten_factor)
        self.user_num = num_user
        self.item_num = num_item
        self.hidden_dim = laten_factor

    def forward(self, user, item, neg_item=None):
        if neg_item is not None:  # used for train
            user_bais = self.user_bais(user).squeeze(-1)
            item_bais = self.item_bais(item).squeeze(-1)
            neg_item_bais = self.item_bais(neg_item).squeeze(-1)

            userembedding = self.user_laten(user)
            itemembedding = self.item_laten(item)
            neg_itemembedding = self.item_laten(neg_item)

            interaction = torch.mul(userembedding,itemembedding).sum(dim=-1)
            neg_interaction = torch.mul(userembedding,neg_itemembedding).sum(dim=-1)
            result_pos = user_bais + item_bais + interaction
            result_neg = user_bais + neg_item_bais + neg_interaction
            score = result_pos -result_neg
            bpr_loss = -torch.sum(F.logsigmoid(score))

            l2loss = torch.norm(userembedding, dim=-1).sum() + torch.norm(itemembedding,dim=-1).sum() + torch.norm(neg_itemembedding).sum()
            return bpr_loss,l2loss

        else:         # used for test
            user_bais = self.user_bais(user).squeeze(-1)
            item_bais = self.item_bais(item).squeeze(-1)
            userembedding = self.user_laten(user)
            itemembedding = self.item_laten(item)
            interaction = torch.mul(userembedding, itemembedding).sum(dim=-1)
            result = user_bais + item_bais + interaction
            return userembedding, itemembedding, result



