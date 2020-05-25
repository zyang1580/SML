import torch
from evalution.evalution_function import  get_MAP,get_MRR,get_NDCG,get_Precision,get_Recall,get_Rec_NDCG
try:
    from tqdm import tqdm
except:
    pass
import numpy as np
def test_model(model,test_set,old_user=None,old_item=None,topK=10,need_pbar=False):
    model.eval()
    num_test = 0
    recall_all = []
    ndcg_all = []
    #print("topK",topK)
    gpu_id = model.user_laten.weight.device.index
    device = model.user_laten.weight.device
    for batch_idx,datas in enumerate(test_set):
        datas = datas.long().to(device)
        batch_hit,batch_ndcg,_ = model.test(datas,topK=topK)
        recall_all.append(batch_hit)
        ndcg_all.append(batch_ndcg)
        num_test += datas.shape[0]
    # print(data.shape)
    # print("recall:", np.array(recall_all).mean(), "ndcg : ", np.array(ndcg_all).mean())
    recall_all = np.array(recall_all)
    ndcg_all = np.array(ndcg_all)
    return recall_all.sum()/num_test,ndcg_all.sum()/num_test

def test_model_pre(model,test_set,old_user=None,new_user=None,old_item=None,new_item=None,topK=10,need_pbar=False):
    model.eval()
    num_test = 0
    recall_all = []
    ndcg_all = []
    ouser_oitem = 0
    ouser_nitem = 0
    nuser_oitem = 0
    nuser_nitem = 0

    for batch_idx, datas in enumerate(test_set):
        datas = datas.long().cuda()
        idx,rank,batch_hit,batch_ndcg = model.test2(datas)
        hit_inter = datas[idx][:, 0:2]
        hit_inter = hit_inter.cpu().numpy()
        for itr in hit_inter:
            u = itr[0]
            i = itr[1]
            if u in old_user:
                if i in old_item:
                    ouser_oitem += 1
                else:
                    ouser_nitem += 1
            else:
                if i in old_item:
                    nuser_oitem += 1
                else:
                    nuser_nitem += 1
        recall_all.append(batch_hit)
        ndcg_all.append(batch_ndcg.data.cpu().numpy())
        num_test += datas.shape[0]
    # print(data.shape)
    # print("recall:", np.array(recall_all).mean(), "ndcg : ", np.array(ndcg_all).mean())
    recall_all = np.array(recall_all)
    ndcg_all = np.array(ndcg_all)
    all_hit = nuser_nitem+nuser_oitem + ouser_nitem + ouser_oitem
    print("old user old item:",ouser_oitem*1.0/all_hit,ouser_oitem*1.0/num_test)
    print("old user new item", ouser_nitem * 1.0 / all_hit,ouser_nitem*1.0/num_test)
    print("new user old item", nuser_oitem * 1.0 / all_hit,nuser_oitem*1.0/num_test)
    print("new user new item", nuser_nitem * 1.0 / all_hit,nuser_nitem*1.0/num_test)
    print(recall_all.sum()-all_hit)
    print("num test:",num_test)
    return recall_all.sum()/num_test,ndcg_all.sum()/num_test





