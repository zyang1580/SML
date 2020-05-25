import torch
from evalution.evalution_function import  get_MAP,get_MRR,get_NDCG,get_Precision,get_Recall,get_Rec_NDCG
try:
    from tqdm import tqdm
except:
    pass
def test_model(model,test_set,topK=10,need_pbar=False):
    model.eval()
    user = test_set[0]
    item = test_set[1]
    neg_item = test_set[2]
    Pres ,Recs,MAPs,NDCGs,MRRs= [],[],[],[],[]
    if need_pbar:
        pbar = tqdm(range(len(user)), total=len(user), desc='({0:^3})'.format(" testing "))
    else:
        pbar = range(len(user))
    for i in pbar:
        (Pre,Rec,MAP,NDCG,MRR) = evalution_for_user(model,user[i],item[i],neg_item[i],topK)
        Pres.append(Pre)
        Recs.append(Rec)
        MAPs.append(MAP)
        NDCGs.append(NDCG)
        MRRs.append(MRR)
    Pres = torch.tensor(Pres).mean()
    Recs = torch.tensor(Recs).mean()
    MAPs = torch.tensor(MAPs).mean()
    NDCGs = torch.tensor(NDCGs).mean()
    MRRs = torch.tensor(MRRs).mean()

    return (Pres,Recs,MAPs,NDCGs,MRRs)



def evalution_for_user(model,u,item_list,neg_list,topK):
    item_list = item_list.copy()
    item_len = len(item_list)
    target_list = torch.arange(0,item_len)  # target item
    item_list.extend(neg_list)
    all_item = item_list
    all_item_num = len(all_item)
    user_full = all_item_num*[u]

    input_list = torch.LongTensor([user_full,all_item]).transpose(1,0).cuda()
    _,_,pre = model(input_list[:,0],input_list[:,1])
    (_,ranklist) = torch.topk(pre,topK)
    target_list = target_list.to(ranklist.device)

    Pre = torch.tensor([0.0]) #get_Precision(ranklist, target_list, topK)
    Rec,dcg = get_Rec_NDCG(ranklist,target_list)
    Ap = torch.tensor([0.0]) #get_MAP(ranklist, target_list)

    rr = torch.tensor([0.0]) # get_MRR(ranklist, target_list)

    # Pre = get_Precision(ranklist,target_list,topK)
    # Rec = get_Recall(ranklist,target_list,topK=topK)
    # Ap = get_MAP(ranklist,target_list)
    # dcg = get_NDCG(ranklist,target_list)
    # rr = get_MRR(ranklist,target_list)

    return (Pre,Rec,Ap,dcg,rr)









