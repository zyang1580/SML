'''
MF-based baselines, include spmf, full-retrain and fine-tune
this code also include codes for pretrain MFbase model for these baselines
'''
import numpy as np
import torch
try:
    from model import MF
except:
    import MF
import torch.utils
import argparse
import time
import torch.nn.functional as F
import os
from torch.utils.data import Dataset

def test_hit_new(data,have_idx,new_user,new_item):
    hit_itr = data[have_idx][:,0:2]
    hit_user = hit_itr[:,0]
    hit_item = hit_itr[:,1]
    N = hit_user.shape[0]
    hit_new_user = 0
    hit_new_item = 0
    for i in range(N):
        if hit_user[i] in new_user:
            hit_new_user += 1
        if hit_item[i] in new_item:
            hit_new_item += 1
    return hit_new_user, hit_new_item

class offlineDataset_withsample(Dataset):
    """
    data set for offline train ,and  prepare for dataloader
    """
    def __init__(self,dataset,neg_num=1):
        super(offlineDataset_withsample, self).__init__()
        self.user = dataset[:,0]
        self.item = dataset[:,1]
        self.neg_num =neg_num
        print("user max:",self.user.max())
        print("user max:", self.item.max())
        user_list = {}
        self.item_all = np.unique(self.item)
        for n in range(self.user.shape[0]):
            u = self.user[n]
            i = self.item[n]
            try:
                user_list[u].append(i)
            except:
                user_list[u] = [i]
        self.user_list = user_list

    def __len__(self):
        return  self.user.shape[0]
    def __getitem__(self, idx):
        user = self.user[idx]
        item = self.item[idx]
        neg_items = []
        #neg_item = self.item_all[0]
        for i in range(self.neg_num):
            neg_item = np.random.choice(self.item_all,1)[0]
            while neg_item in self.user_list[user]:
                neg_item = np.random.choice(self.item_all,1)[0]
            neg_items.append(neg_item)
        return (user,item,neg_item)

class Reservious(object):
    def __init__(self,length):
        super(Reservious, self).__init__()
        self.t = 0
        self.len = length
        self.pool = np.zeros((length,2),dtype=np.long)
        print("pool size:",self.pool.shape)
        self.pool_have = 0

    def updata(self,new_data):
        if self.t <= self.len:
            new_num = new_data.shape[0]
            max_id = min(self.len,self.pool_have+new_num)
            self.pool[self.pool_have:max_id] = new_data[:max_id-self.pool_have]
            if max_id != self.len:
                new_data = new_data[max_id-self.pool_have:]
            self.pool_have = self.pool_have + max_id
            self.t = max_id
        new_num = new_data.shape[0]
        p = self.len*1.0 / (self.t+np.arange(new_num)+1)
        m = np.random.rand(new_num)
        select_data = new_data[np.where(m<p)]
        for i in range(select_data.shape[0]):
            idx = np.random.randint(0,self.len,1)
            self.pool[idx] = select_data[i]
        self.t += new_num
    def init_pool(self,new_data):
        num = new_data.shape[0]
        rand_idx = np.random.randint(0,num,self.len)
        #self.pool[:] = new_data[rand_idx]
        self.pool[:] = new_data[-self.len:]
        self.pool_have = self.len
        self.t = num

class SPMF(object):
    '''
    MF-base baselines, SPMF , full-retrain MF, fine-tune MF
    '''
    def __init__(self,args,datasets,user_num,item_num,laten_dim):
        super(SPMF, self).__init__()
        self.MFbase = MF.MFbasemode(num_user=user_num,num_item=item_num,laten_factor=laten_dim).cuda()
        print("args lr:",args.lr)
        self.lr = args.lr
        self.optimizer = torch.optim.Adam(self.MFbase.parameters(),lr=args.lr,weight_decay=0)
        #self.optimizer = torch.optim.SGD(self.MFbase.parameters(),lr=args.lr)
        print("optimizer state:",self.optimizer.state_dict())
        self.pool_size = args.pool_size
        self.Reservious = Reservious(self.pool_size)
        print("self.pool_size:",self.pool_size)
        self.all_item = np.ones(0,dtype=np.long)
        self.dataset = datasets

        self.new_user = torch.from_numpy(datasets.test_new_user).long().cuda()
        self.new_item = torch.from_numpy(datasets.test_new_item).long().cuda()

        self.neg_num = args.neg_num
        self.batch_size = args.batch_size
        self.lambda_u = args.l2_u
        self.lambda_i = args.l2_i
        self.recall = []
        self.ndcg = []
        self.hit_new_user = []
        self.hit_new_item = []
        self.epochs = args.epochs
        self.run_stage = 0
        self.test_num = []
        self.user_hit = None
        self.pool_init_type = args.pool_init_type

    def get_next_data(self,stage_id,types="only_new"):
        '''
        get model next data. For more information ,please read the method of corresponding dataset class
        :param stage_id: stage idx
        :return: set_t,set_tt,now_test
        '''
        # types = "onliy_new"
        # if self.run_stage == 0:
        #     types = "not_only_new"
        set_t,now_test = self.dataset.get_next(stage_id,types=types)
        return set_t,now_test

    def base_train_not_train(self, stage_id):
        set_t, now_test = self.get_next_data(stage_id, types="not_only_new")
        if self.pool_init_type == 1: # directly put the nearest data into Reservious
            self.Reservious.init_pool(set_t)
        F_recall, F_ndcg,_,_ = self.test(now_test)
        print("before train test---", "recall(5,10,20):", F_recall, "ndcg (5,10,20):", F_ndcg)
        if self.pool_init_type == 0:  # init by updating
            self.updata_reservious(set_t)

    def init_pool(self):
        pass

    def base_train(self,stage_id,epochs,l2_u,l2_i):
        '''
        pretraining a MFbase model for baselines
        :param stage_id: period idx
        :param epochs: max epoch
        :param l2_u: l2 for user
        :param l2_i: l2 for item  , you need set l2_i = l2_i for fair comparsion
        :return:
        '''
        print("********base train: (l2_u,l2_i): ({},{})*****".format(l2_u,l2_i))
        set_t,now_test = self.get_next_data(stage_id,types="not_only_new")
        train = offlineDataset_withsample(set_t,neg_num=1)
        train = torch.utils.data.DataLoader(train,batch_size=self.batch_size,shuffle=True,num_workers=4)
        max_recall20 = 0
        max_REC = None
        max_Ndcg = None
        max_epoch = 0
        not_change_num = 0
        for epoch in range(epochs):
            self.MFbase.train()
            loss_all = 0
            s_time = time.time()
            for (bat_num,(bat_user,bat_item,bat_neg)) in enumerate(train):
                self.MFbase.zero_grad()
                bat_user = bat_user.long().cuda()
                bat_item = bat_item.long().cuda()
                bat_neg = bat_neg.long().cuda()
                user_embedding, item_embedding, pog_score = self.MFbase(bat_user, bat_item)
                _, neg_embedding, neg_score = self.MFbase(bat_user, bat_neg)
                # pog_score = F.relu6(pog_score) - F.relu(-pog_score)
                # neg_score = F.relu(neg_score) - F.relu6(-neg_score)
                # user_ = (user_embedding ** 2).sum(dim=-1).sqrt().detach()
                # pog_score = pog_score / user_
                # neg_score = neg_score / user_
                #bce_ = torch.sum(F.logsigmoid(pog_score - neg_score))
                bce_ = torch.mean(torch.log(torch.sigmoid(pog_score)+1e-15)) + torch.mean(torch.log(1.0 - torch.sigmoid(neg_score)+1e-15))  # sum
                l2_loss = l2_u * 0.5 * torch.sum((user_embedding ** 2)) + l2_i * 0.5 *(torch.sum((item_embedding ** 2)) + torch.sum(neg_embedding**2))
                loss = - bce_ + l2_loss
                loss_all += loss.data
                loss.backward()
                self.optimizer.step()
            loss_all /= (bat_num+1)*self.batch_size
            print("epoch:{}, time:{:.1f}, loss:{:.4f}".format(epoch,time.time()-s_time,loss_all.item()))
            if (epoch % 2) == 0:
                F_recall, F_ndcg,_,_ = self.test(now_test)
                not_change_num += 1
                if F_recall[-1] > max_recall20:
                    max_recall20 = F_recall[-1]
                    max_REC = F_recall
                    max_Ndcg = F_ndcg
                    max_epoch = epoch
                    not_change_num = 0
                    torch.save(self.MFbase.state_dict(),"/home/wangpenghui/zhangyang/yelp_0113/save_model/spmf3/best-mean-start29-spmf-"+"-"+str(l2_u)+"-"+str(self.lr)+"lr.pt")
                print("test---", "recall(5,10,20):", F_recall, "ndcg (5,10,20):", F_ndcg , "max reccall",max_REC,"max epoch:",max_epoch)
                if not_change_num > 50: # >=25
                    print("max not change up to 20 epochs, stop ......")
                    break
            if epoch %50 ==0:#30
                torch.save(self.MFbase.state_dict(),"/home/wangpenghui/zhangyang/yelp_0113/save_model/spmf3/mean-start29-spmf-"+str(epoch)+"-"+str(l2_u)+"-"+str(self.lr)+"lr.pt")

        F_recall, F_ndcg,_,_ = self.test(now_test)
        print("FInal test---", "recall(5,10,20):", F_recall, "ndcg (5,10,20):", F_ndcg)
        print("max: epoch",max_epoch,"max_recall:",max_REC,"max_Ndcg",max_Ndcg)



    def run_one_stage(self,stage_id):
        '''
        SPMF run one stage(period)!
        :param stage_id: period idx
        :return: true or false
        '''
        set_t, now_test = self.get_next_data(stage_id)
        if set_t is None:
            return False
        self.test_num.append(now_test.shape[0])
        self.all_item = np.union1d(self.all_item, set_t[:, 1])
        if self.Reservious.pool_have >0:
            train_data = np.concatenate([self.Reservious.pool[0:self.Reservious.pool_have], set_t],axis=0)
        else:
            train_data = set_t
            #del set_t
        self.user_hit_num_in_W_R(train_data)

        itr = round(train_data.shape[0]/self.batch_size)

        p = self.compute_R_W_P(train_data)
        print("start train...")
        F_recall, F_ndcg = self.test(now_test)
        print("before train test---", "recall(5,10,20):", F_recall, "ndcg (5,10,20):", F_ndcg)
        max_recall20 = 0
        max_recall = None
        max_Ndcg = None
        not_chang = 0
        for epoch in range(self.epochs):
            self.MFbase.train()
            s_time = time.time()
            loss_all = 0
            for bat_num in range(itr):
                self.MFbase.zero_grad()
                bat_user,bat_item,bat_neg = self.sample_batch(train_data,self.batch_size,p,self.neg_num)
                bat_user = torch.from_numpy(bat_user)
                bat_item = torch.from_numpy(bat_item)
                bat_neg = torch.from_numpy(bat_neg)

                bat_user = bat_user.cuda().long()
                bat_item = bat_item.cuda().long()
                bat_neg = bat_neg.cuda().long()
                user_embedding, item_embedding, pog_score = self.MFbase(bat_user,bat_item)
                _, neg_embedding, neg_score = self.MFbase(bat_user,bat_neg)
                #pog_score = F.relu6(pog_score) - F.relu(-pog_score)
                 # user_ = (user_embedding ** 2).sum(dim=-1).  ().detach()
                # pog_score = pog_score / user_
                # neg_score = neg_score / user_
                bce_ = torch.mean(torch.log(torch.sigmoid(pog_score)+1e-15)) + torch.mean(torch.log(1.0-torch.sigmoid(neg_score)+1e-15)) # sum
                l2_loss = self.lambda_u*0.5 * torch.sum((user_embedding**2)) + self.lambda_i*0.5 * (torch.sum((item_embedding**2))+ torch.sum((neg_embedding**2)))
                loss = - bce_ + l2_loss
                loss_all += loss.data
                loss.backward()
                self.optimizer.step()

            loss_all = loss_all/(bat_num+1)
            print("epoch: {} ,time:{:.1f}, loss:{:.4f}".format(epoch, time.time() - s_time, loss_all.item()))
            if epoch%1==0:
                not_chang += 1
                F_recall, F_ndcg,_,_ = self.test(now_test)
                print("        epoch test---", "recall(5,10,20):", F_recall, "ndcg (5,10,20):", F_ndcg)
                if max_recall20 < F_recall[-1]:
                    max_recall20 = F_recall[-1]
                    max_recall = F_recall
                    max_Ndcg = F_ndcg
                    not_chang = 0
                if not_chang >=5:
                    if self.pool_init_type == 1:  # for news, you can use early-stop or not, if not, please don't break
                        break
                    pass


        self.updata_reservious(set_t)
        F_recall, F_ndcg,hit_new_user,hit_new_item = self.test(now_test)
        print("FInal test---","recall(5,10,20):",F_recall,"ndcg (5,10,20):",F_ndcg,"hit new user:",hit_new_user,"hit new item:",hit_new_item)
        self.recall.append(F_recall)
        self.ndcg.append(F_ndcg)
        return True

    def run_one_stage2(self,stage_id,read_data_type='only_new'):
        '''
        full-rtrain and fine-tune run one stage(period). Need set pool_size=0
        :param stage_id: period idx
        :param read_data_type: "only_new" fine-tune MF, "not_only_new" full-retrain MF
        :return: True or False
        '''
        set_t, now_test = self.get_next_data(stage_id,types=read_data_type)
        if set_t is None:
            return False
        self.test_num.append(now_test.shape[0])
        self.all_item = np.union1d(self.all_item, set_t[:, 1])
        if self.Reservious.pool_have >0:
            train_data = np.concatenate([self.Reservious.pool[0:self.Reservious.pool_have], set_t],axis=0)
            print("pool having.....")
        else:
            train_data = set_t
            #del set_t
        self.user_hit_num_in_W_R(train_data)
        train = offlineDataset_withsample(train_data)
        train = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=True, num_workers=4)

        itr = round(train_data.shape[0]/self.batch_size)

        #p = self.compute_R_W_P(train_data)
        print("start train...")
        F_recall, F_ndcg ,_ ,_ = self.test(now_test)
        print("before train test---", "recall(5,10,20):", F_recall, "ndcg (5,10,20):", F_ndcg)
        max_recall20 = 0
        max_recall = None
        max_Ndcg = None
        not_chang = 0
        for epoch in range(self.epochs):
            self.MFbase.train()
            s_time = time.time()
            loss_all = 0
            for (bat_num, (bat_user, bat_item, bat_neg)) in enumerate(train):
                # print(bat_num)
                self.MFbase.zero_grad()
                bat_user = bat_user.long().cuda()
                bat_item = bat_item.long().cuda()
                bat_neg = bat_neg.long().cuda()
                user_embedding, item_embedding, pog_score = self.MFbase(bat_user, bat_item)
                _, neg_embedding, neg_score = self.MFbase(bat_user, bat_neg)
                # pog_score = F.relu6(pog_score) - F.relu(-pog_score)
                # neg_score = F.relu(neg_score) - F.relu6(-neg_score)
                # user_ = (user_embedding ** 2).sum(dim=-1).sqrt().detach()
                # pog_score = pog_score / user_
                # neg_score = neg_score / user_
                #bce_ = torch.sum(F.logsigmoid(pog_score-neg_score))
                bce_ = torch.mean(torch.log(torch.sigmoid(pog_score) + 1e-15)) + torch.mean(torch.log(1.0 - torch.sigmoid(neg_score) + 1e-15))# mean
                l2_loss = self.lambda_u * 0.5 * torch.sum((user_embedding ** 2)) + self.lambda_i*0.5 * (torch.sum((item_embedding**2))+ torch.sum((neg_embedding**2)))
                loss = - bce_ + l2_loss
                loss_all += loss.data
                loss.backward()
                self.optimizer.step()
            loss_all = loss_all/(bat_num+1)
            print("epoch: {} ,time:{:.1f}, loss:{:.4f}".format(epoch, time.time() - s_time, loss_all.item()))
            not_chang += 1
            if epoch%5==0:
                F_recall, F_ndcg,_,_ = self.test(now_test)
                print("        epoch test---", "recall(5,10,20):", F_recall, "ndcg (5,10,20):", F_ndcg)
                if max_recall20 < F_recall[-1]:
                    max_recall20 = F_recall[-1]
                    max_recall = F_recall
                    max_Ndcg = F_ndcg
                    not_chang = 0
                if not_chang >5:
                    if self.pool_init_type == 1:  # for news, you can use early stop or not, if not, please don't break
                        break
                    pass

        #self.updata_reservious(set_t)
        F_recall, F_ndcg,hit_newu,hit_newi = self.test(now_test,stage_idx=stage_id)
        print("max result ",max_recall,max_Ndcg)
        print("FInal test---","recall(5,10,20):",F_recall,"ndcg (5,10,20):",F_ndcg,"hit user:",hit_newu,"hit item:",hit_newi)
        self.recall.append(F_recall)
        self.ndcg.append(F_ndcg)
        self.hit_new_user.append(hit_newu)
        self.hit_new_item.append(hit_newi)
        return True

    def test(self,test_data,topk = [5,10,20],stage_idx=None):
        self.MFbase.eval()
        test_num = test_data.shape[0]
        test_batch = 1024
        recall_all = []
        ndcg_all = []
        hit_new_user = []
        hit_new_item = []
        hit_itr_20 = []
        hit_itr_10 =[]
        hit_itr_5 =[]


        for i in range(0,test_num,test_batch):
            end_idx = min(i+test_batch,test_num)
            bat_test = test_data[i:end_idx]
            bat_test = torch.from_numpy(bat_test).cuda().long()
            b_hit = []
            b_ndcg = []
            b_hit_user = []
            b_hit_item = []
            b_hit_itr = []
            for k in topk:
                batch_hit, batch_ndcg, hit_idx = self.MFbase.test(bat_test, topK=k)
                try:
                    batch_ndcg = batch_ndcg.cpu().numpy()
                except:
                    pass
                b_hit.append(batch_hit)
                b_ndcg.append(batch_ndcg)

                hit_u, hit_i = test_hit_new(bat_test, hit_idx, self.new_user, self.new_item)
                b_hit_user.append(hit_u)
                b_hit_item.append(hit_i)
                b_hit_itr.append(bat_test[hit_idx][:,0:2])

            recall_all.append(b_hit)
            ndcg_all.append(b_ndcg)
            hit_new_user.append(hit_u)
            hit_new_item.append(hit_i)
            hit_itr_20.append(b_hit_itr[-1])
            hit_itr_10.append(b_hit_itr[1])
            hit_itr_5.append(b_hit_itr[0])

        recall = np.array(recall_all)
        ndcg = np.array(ndcg_all)
        hit_user = np.array(hit_new_user) * 1.0
        hit_item = np.array(hit_new_item) * 1.0
        if stage_idx is not None:
            hit_itr_20 = torch.cat(hit_itr_20,dim=0).cpu().numpy()
            hit_itr_10 = torch.cat(hit_itr_10,dim=0).cpu().numpy()
            hit_itr_5 = torch.cat(hit_itr_5,dim=0).cpu().numpy()
            #np.save("/home/wangpenghui/zhangyang/yelp_0113/save_model/"+str(self.epochs)+"-retrain"+str(self.lr)+str(self.lambda_u)+str(stage_idx)+"only-20-hit-itr.npy",hit_itr_20)
            #np.save("/home/wangpenghui/zhangyang/yelp_0113/save_model/"+str(self.epochs)+"-retrain"+str(self.lr)+str(self.lambda_u) + str(stage_idx)+"only-10-hit-itr.npy", hit_itr_10)
            #np.save("/home/wangpenghui/zhangyang/yelp_0113/save_model/"+str(self.epochs)+"-retrain"+str(self.lr)+str(self.lambda_u) + str(stage_idx)+ "only-5-hit-itr.npy", hit_itr_5)
        return recall.sum(axis=0)/test_num, ndcg.sum(axis=0)/test_num, hit_user.sum(axis=0)/test_num, hit_item.sum(axis=0)/test_num

    def updata_reservious(self,train_data):
        self.Reservious.updata(train_data)

    def compute_R_W_P(self,R_TR_data):
        """
        according rank compute P Probabilistic
        :param R_TR_data:
        :return:
        """
        self.MFbase.eval()
        batch_size = 10240
        N = R_TR_data.shape[0]
        score = []
        for i in range(0,N,batch_size):
            end_idx = min(N,i+batch_size)
            user = R_TR_data[i:end_idx,0]
            item = R_TR_data[i:end_idx,1]
            user = torch.from_numpy(user).cuda()
            item = torch.from_numpy(item).cuda()
            _, _, bat_score = self.MFbase(user,item)
            score.append(bat_score)
        score = torch.cat(score,dim=0)
        score =score.squeeze(-1)
        rank_idx = torch.argsort(score,descending=True)
        num = rank_idx.shape[0]
        rank_ = torch.arange(num)+1
        rank_ = rank_.to(score.device)
        p = torch.zeros_like(score)
        p[rank_idx] = rank_.float()
        w = torch.exp(p*1.0/num)
        p = w/w.sum()
        return p.cpu().numpy()

    def user_hit_num_in_W_R(self,data):
        if self.user_hit is None:
            self.user_hit = {}
        for k in range(data.shape[0]):
            u = data[k,0]
            i = data[k,1]
            try:
                self.user_hit[u].add(i)
            except:
                self.user_hit[u] = set([i])

    def sample_batch(self,data,batch_size,p,neg_num):
        idx = np.arange(data.shape[0])
        bat_idx = np.random.choice(idx,batch_size,p=p)
        bat_data = data[bat_idx]
        bat_user = bat_data[:,0]
        bat_item = bat_data[:,1]
        bat_neg = []
        for i in range(bat_data.shape[0]):
            m = np.random.choice(self.all_item,neg_num)
            u = bat_user[i]
            while m[0] in self.user_hit[u]:
                m = np.random.choice(self.all_item,neg_num)
            bat_neg.append(m)
        bat_neg = np.array(bat_neg)
        return bat_user.reshape(-1,1),bat_item.reshape(-1,1),bat_neg

    def run(self,start_stage,method='full'):
        self.run_stage = 0
        stage_id = start_stage
        while (1):
            print("#################################runing stage:{}########################".format(stage_id))
            if method=='spmf':  # spmf
                run_flag = self.run_one_stage(stage_id)
            else:
                if method=='full': # full-retrain
                    run_flag = self.run_one_stage2(stage_id,read_data_type='not_only_new')
                else:              # fine-tune
                    run_flag = self.run_one_stage2(stage_id, read_data_type='only_new')
            if run_flag:
                stage_id += 1
                self.run_stage += 1
            else:
                test_num = np.array(self.test_num).reshape(-1,1)
                recall = np.array(self.recall)
                ndcg = np.array(self.ndcg)

                print("average recall:",recall.mean(axis=0))
                print("average recall:", ndcg.mean(axis=0))

                print(test_num)
                print(recall)
                print(ndcg)

                print("hit new user:",self.hit_new_user)
                print("hit new item:",self.hit_new_item)

                N = test_num.shape[0]
                N3 = round(N*1.0/3)
                pre3_num = test_num[0:N3]
                pre3_recall = recall[0:N3]
                pre3_ndcg = ndcg[0:N3]

                rate3 = pre3_num /pre3_num.sum()
                recall3 = pre3_recall * rate3
                ndcg3 = pre3_ndcg * rate3
                print("pre 3 (val) reslut,recall,ndcg:", recall3.sum(axis=0), ndcg3.sum(axis=0))
                rate_7 = test_num[N3:] /test_num[N3:].sum()
                recall7 = recall[N3:] *rate_7
                ndcg7 = ndcg[N3:]*rate_7
                print("last 7 (test) results,recall ,ndcg:", recall7.sum(axis=0), ndcg7.sum(axis=0))


                rate = test_num /test_num.sum()
                recall = recall * rate
                ndcg = ndcg *rate
                print("weight average recall@20:", recall.sum(axis=0))
                print("weight average ndcg@20:", ndcg.sum(axis=0))
                break

class StreamingData(object):
    def __init__(self,file_pathe):
        super(StreamingData, self).__init__()
        information = np.load(file_pathe+"information.npy")
        self.user_num = information[1]
        self.item_num = information[2]
        self.itr_num = information[0]
        self.path = file_pathe
        self.test_new_user = np.load(file_pathe + "test_new_user.npy").astype(np.long)
        self.test_new_item = np.load(file_pathe + "test_new_item.npy").astype(np.long)

    def get_next(self,stage_id,types="not_only_new"):
        try:
            if types == "not_only_new":
                train_data = []
                for i in range(0,stage_id):
                    train_data.append(np.load(self.path+"train/"+str(i)+".npy").astype(np.long))
                train_data = np.concatenate(train_data,axis=0)
            else:
                train_data = np.load(self.path+"train/"+str(stage_id-1)+".npy").astype(np.long)
        except:
            print("read train data roung , may be there is no new data,finished")
            return None, None
        try:
            test_data = np.load(self.path+"test/"+str(stage_id)+".npy").astype(np.long)
        except:
            print("read test data roung , may be there is no new data,finished")
            return None, None
        print("NOTICED: will train: {} , will test:{} ".format(stage_id-1,stage_id))
        return train_data, test_data




def get_parse():
    parser = argparse.ArgumentParser(description='MF and TR parameters.')
    parser.add_argument('--lr', type=float, default=0.01, # 0.01
                        help='Learning rate.')
    parser.add_argument('--l2_u', type=float, default=1e-5,
                        help='user l2. should be same to l2_i')
    parser.add_argument('--l2_i', type=float, default=1e-5,
                        help='item l2.should be same to l2_u ')
    parser.add_argument('--epochs', type=int, default=20,  #
                        help='Number of epochs to train of each stage.')
    parser.add_argument('--batch_size', type=int, default=256,  # full-retrain 256 , others 64
                        help='batch size of train.')
    parser.add_argument('--laten_dim', type=int, default=64,  # 64
                        help='batch size of train.')
    parser.add_argument('--neg_num', type=int, default=1,  # 1
                        help='neg num.')
    parser.add_argument('--pool_size', type=int, default=0,  #
                        help='batch size of train.')
    parser.add_argument('--laten', type=int, default=64,  # 64
                        help='dim of embedding.')
    parser.add_argument('--cuda', type=int, default=1,
                        help='which GPU be used?.default 1')
    parser.add_argument('--method', default='full',
                        help='full, fine, spmf')
    parser.add_argument('--pool_init_type', type=int, default=0, # only for spmf
                        help='Reservious of SPMF init methods, 0: update , 1: init, yelp=0, news (adressa) =1 ')
    parser.add_argument('--data_path', default='/home/wangpenghui/zhangyang/datasets/',
                        help='data path')
    parser.add_argument('--data_name', default='yelp',
                        help='dataset name')
    parser.add_argument('--pre_model', default="/home/wangpenghui/zhangyang/yelp_0113/save_model/best-mean-start29-spmf--1e-07-0.01lr.pt",
                        help='data path')
    parser.add_argument('--start_idx', type=int, default=30,
                        help='retraining from which period: yelp 30, news(adressa) 48')
    return parser

if __name__=="__main__":
    print("start")
    parser = get_parse()
    args = parser.parse_args()
    print("parameters:",args)
    data_path = args.data_path + args.data_name+"/" #'/home/wangpenghui/zhangyang/datasets/yelp/'

    if args.data_name=='yelp':
        args.pool_init_type = 0
    elif args.data_name == 'news': # adressa
        args.pool_init_type =1
    else:
        args.pool_init_type = 0

    dataset = StreamingData(data_path)
    user_num = dataset.user_num
    item_num = dataset.item_num
    laten_dim = args.laten_dim
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    args.l2_i = args.l2_u  # set l2_i = l2_u
    l2_u = [0]
    l2_i = [0]
    pool_size = [0]
    for i in l2_u:
        for j in pool_size:
            print("*******************(l2_u,pool size):({},{})********".format(i,j))
            print("*##**##*")
            # args.l2_u = i
            # args.lw_i = j
            #args.pool_size = j
            print(args)
            torch.manual_seed(2000)
            torch.cuda.manual_seed(2001)
            np.random.seed(2002)
            torch.backends.cudnn.deterministic = True
            model = SPMF(args,dataset,user_num, item_num, laten_dim)
            #model.MFbase.load_state_dict(torch.load("/home/wangpenghui/zhangyang/yelp_0113/save_model/best-mean-start29-spmf--1e-07-0.01lr.pt"))
            model.MFbase.load_state_dict(torch.load(args.pre_model))
            # if you want to train a pre-train model, run model.base_train(29)
            if args.method == 'spmf':
                model.base_train_not_train(start_idx-1)  # yelp 29, news 47
            model.run(args.start_idx,method=args.method)    # yelp 30 news 48
            print("\n *##**##* \n")

