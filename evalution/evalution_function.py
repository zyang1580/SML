import torch

"""

you need use one tensor input=[target item , negative items] as the model inout, then the idx of input represent each item
then rank the output of model with their idx, the rank list is the ranked idx, then target is the unranked target items idx!!  

ranklist: rank list, tensor, idx 
target items: taget items number,tensor, idx

"""
def hit(ranklist,target_items):
    '''
    the countes of traget items in ranklist
    :param ranklist:
    :param target_items:
    :return:
    '''
    count = 0
    ones_array = torch.ones_like(ranklist)
    zeros_array = torch.zeros_like(ranklist)
    match_array = torch.where(ranklist < (target_items[-1]+1) ,ones_array,zeros_array)
    pos_of_target = torch.nonzero(match_array)[:,0]  # position of target item in the ranklist
    return pos_of_target.shape[0]     #return count of hits target items

def get_Rec_NDCG(ranklist,target_items):
    idcg = IDCG(target_items.shape[0])
    ones_array = torch.ones_like(ranklist)
    zeros_array = torch.zeros_like(ranklist)
    match_array = torch.where(ranklist < (target_items[-1] + 1), ones_array, zeros_array)
    rank_of_target = torch.nonzero(match_array)[:, 0]  # position of target item in the ranklist
    hits = rank_of_target.shape[0]
    if rank_of_target.shape[0] > 0:  # if hits
        dcg = 1.0 / torch.log2(rank_of_target.float() + 2)  # must add 1,for rank is start from 1
        dcg = dcg.sum()
        dcg = dcg /idcg
    else:
        dcg=0
    return hits/target_items.shape[0],dcg

def get_Precision(ranklist,target_items,topK):
    """
    Precision
    :param ranklist:
    :param target_items:
    :param topK:
    :return:
    """
    return hit(ranklist,target_items)/topK

def get_Recall(ranklist,target_items,topK=10):
    '''
    Recall
    :param ranklist:
    :param target_items:
    :param topK:
    :return:
    '''
    return  hit(ranklist,target_items)/target_items.shape[0]

def get_NDCG(ranklist,target_items):
    """
    NDCG
    :param ranklist:
    :param target_items:
    :return:
    """
    idcg = IDCG(target_items.shape[0])

    '''
    Now, find the rank of target items in the ranklist
    '''
    ones_array = torch.ones_like(ranklist)
    zeros_array = torch.zeros_like(ranklist)
    match_array = torch.where(ranklist < (target_items[-1] + 1), ones_array, zeros_array)
    rank_of_target = torch.nonzero(match_array)[:, 0]  # position of target item in the ranklist
    if rank_of_target.shape[0] > 0: #if hits
        dcg = 1.0 / torch.log2( rank_of_target.float()+2)   #  must add 1,for rank is start from 1
        dcg = dcg.sum()
        return dcg / idcg
    else:
        return 0



def IDCG(n):
    """
    IDCG,
    :param n: target items number
    :return: IDCG
    """
    idcg = 0
    arr = torch.arange(n).float()+2
    arr_log = 1.0 / torch.log2(arr)
    return arr_log.sum()

def get_MRR(ranklist,target_items):
    """
    the first rank of the target_items in the ranklist
    :param ranklist: rank list
    :param target_items: target items
    :return: MRR
    """
    ones_array = torch.ones_like(ranklist)
    zeros_array = torch.zeros_like(ranklist)
    match_array = torch.where(ranklist < (target_items[-1] + 1), ones_array, zeros_array)
    rank_of_target = torch.nonzero(match_array)[:, 0]   # position of target item in the ranklist
    if rank_of_target.shape[0] > 0: # if have hits
        first_rank_of_target = rank_of_target[0] + 1  # the best rank of target items, add 1 for rank start from 1
        return 1.0/ first_rank_of_target.float()
    else:
        return 0

def get_MAP(ranklist,target_items):
    '''
    get AP
    :param ranklist:
    :param target_item:
    :return:
    '''
    ones_array = torch.ones_like(ranklist).to(ranklist.device)
    zeros_array = torch.zeros_like(ranklist).to(ranklist.device)
    match_array = torch.where(ranklist < (target_items[-1] + 1), ones_array, zeros_array) # is target 1, else 0
    rank_of_target = torch.nonzero(match_array).float()[:, 0]  # position of target item in the ranklist

    if rank_of_target.shape[0] > 0:  # if hits > 0,have hit
        rank_of_target = rank_of_target.float() + 1
        hits = torch.arange(rank_of_target.shape[0]).float().to(ranklist.device) + 1  # hits is also from 1
        precs = hits / rank_of_target
        min_len = min(ranklist.shape[0], target_items.shape[0])
        return torch.sum(precs) / (min_len*1.0)
    else:
        return 0








