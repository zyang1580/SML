import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import scipy.sparse as sp
try:
    from tqdm import tqdm
except:
    pass
from concurrent.futures import ThreadPoolExecutor
try:
    import networkx as nx
except:
    pass
import random

torch.manual_seed(2019)

class offlineDataset(Dataset):
    """
    data set for offline train ,and  prepare for dataloader
    """
    def __init__(self,dataset):
        super(offlineDataset, self).__init__()
        self.user = dataset[:,0]
        self.item = dataset[:,1]
        try:
            self.neg_item = dataset[:,2]                                                        #sampled negative items
        except:
            raise AssertionError("don't have neg_item!!!!")
    def __len__(self):
        return  self.user.shape[0]
    def __getitem__(self, idx):
        user = self.user[idx]
        item = self.item[idx]
        neg_item = self.neg_item[idx]
        return user,item,neg_item



class offlineDataset_withsample(Dataset):
    """
    data set for offline train ,and  prepare for dataloader
    """
    def __init__(self,dataset):
        super(offlineDataset_withsample, self).__init__()
        self.user = dataset[:,0]
        self.item = dataset[:,1]
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
        #neg_item = self.item_all[0]
        neg_item = np.random.choice(self.item_all,1)[0]
        while neg_item in self.user_list[user]:
            neg_item = np.random.choice(self.item_all,1)[0]
        return (user,item,neg_item)



class onlineDataset2(object):
    '''
    dataset fot online traing,can chnage train/test set according to time. it can't be used by deafult.
    ddont,select negative item for train, but for test
    '''
    def __init__(self,path='dataset/',datasetname='News',file_path_list=None,need_pbar=False,neg_evalution=1000):
        '''
        :param file_path_list:
        :param all_tslot: how many t slot to all the file??
        '''
        super(onlineDataset2, self).__init__()
        self.all_interactiones = []
        self.all_t_items = []
        self.all_tslot = len(file_path_list)  # time slot number
        self.now_t = 0
        self.file_list = file_path_list
        self.dataname = datasetname
        self.path =path
        self.need_pbar=need_pbar
        self.neg_evalution =neg_evalution
        information = np.load(self.path+self.dataname+'/'+"information.npy")
        self.user_number = information[1]
        self.inter_all = information[0]
        self.item_number = information[2]
        print("interaction number:{},user number:{},item number:{}".format(information[0],information[1],information[2]))


    def creat_dataset_offline(self,ration = [14/21,2/21,5/21],need_pbar=False):
        train = []
        validation = []
        test =[]
        train_num = round(len(self.file_list) * ration[0])
        validation_num = round(len(self.file_list) * ration[1])
        for i in range(train_num):
            train.append(np.load(self.path+self.dataname+'/'+self.file_list[i]+".npy"))
        for i in range(train_num,train_num+validation_num):
            validation.append(np.load(self.path+self.dataname+'/'+self.file_list[i]+".npy"))
        for i in range(train_num+validation_num,len(self.file_list)):
            test.append(np.load(self.path + self.dataname + '/' + self.file_list[i] + ".npy"))

        train = np.concatenate(train,axis=0)
        validation = np.concatenate(validation,axis=0)
        test = np.concatenate(test,axis=0)

        test = self.testset_for_evalution_fast(np.concatenate([train, validation], axis=0), test)
        validation = self.testset_for_evalution_fast(train,validation)
        return train,validation,test

    def creat_data_initonline(self,ration=[14/21,2/21,5/21]):
        train =[]
        train_num = round(len(self.file_list) * ration[0])
        validation_num = round(len(self.file_list) * ration[1])
        self.now_t = train_num
        for i in range(train_num):  #  firt train_num file used for train
            train.append(np.load(self.path+self.dataname+'/'+self.file_list[i]+".npy"))
        test = np.load(self.path+self.dataname+'/'+self.file_list[self.now_t]+".npy")
        train = np.concatenate(train,axis=0)
        self.now_t += 1

        test = self.testset_for_evalution_fast(train, test) # realy used for test
        return train,test

    def next_data(self):
        if self.now_t == self.all_tslot:
            return None,None
        train = []
        for i in range(self.now_t):  # data still now_t for train
            train.append(np.load(self.path + self.dataname + '/' + self.file_list[i] + ".npy"))
        test = np.load(self.path + self.dataname + '/' + self.file_list[self.now_t] + ".npy")
        self.now_t += 1
        train = np.concatenate(train, axis=0)
        test = self.testset_for_evalution_fast(train, test) # realy used for test
        return train,test




    def testset_for_evalution_fast(self,train_set,test_set):
        """
        we change the dataset from tensor [one user, one item] to  list [one user, all pos itemes,neg items]
        :param train_set:  used when selected  negative samples for test
        :param test_set:  test set when time t
        :param neg_evalution: how many negative samples for each user!!!  different from negative samples for train!!!
        :return: [one user, all item,]  if need neg item,[one,user,negitem]
        """
        user_dict = {}
        max_id = 0
        user = []
        item = []
        neg_item = []
        neg_evalution = self.neg_evalution

        for i in range(test_set.shape[0]):
            user_i = test_set[i, 0]                                                             #user_i = user_dict[test_set[i, 0]]
            item_i = test_set[i, 1]                                                             #item_i = user_dict[test_set[i, 1]]
            try:
                user_id = user_dict[user_i]
                item[user_id].append(item_i)
            except:
                user_dict[user_i] = max_id
                user.append(user_i)
                item.append([item_i])
                max_id += 1

        """
        then we sampled negative item for test
        """
        now_item = np.union1d(np.unique(train_set[:,1]),np.unique(test_set[:,1]))
        #sample_idx = torch.randint(0, now_item.shape[0], [len(user),neg_evalution+1000])
        if self.need_pbar:
            pbar = tqdm(range(len(user)), total=len(user), desc='({0:^3})'.format("neg sample"))
        else:
            pbar = range(len(user))
        for i in pbar:
            u = user[i]
            sample_idx_u = np.random.randint(0, now_item.shape[0], [neg_evalution+1000])            #list(sample_idx[i].numpy())
            sampled_item = now_item[sample_idx_u]
            have_hit = np.array(item[i])
            can_be_select = np.setdiff1d(sampled_item,have_hit)

            if can_be_select.shape[0] >= neg_evalution:
                np.random.shuffle(can_be_select )
                neg_i_item = list(can_be_select[0:neg_evalution])
            else:
                raise AssertionError("please select more!!")
            neg_item.append(neg_i_item)
        test_set=[user,item,neg_item]
        return test_set



class onlineDataset(object):
    '''
    dataset fot online traing,can chnage train/test set according to time. it can't be used by deafult.
    '''
    def __init__(self,path='dataset/',datasetname='News',file_path_list=None,all_tslot=100):
        '''
        :param file_path_list:
        :param all_tslot: how many t slot to all the file??
        '''
        super(onlineDataset, self).__init__()
        self.all_interactiones = []
        self.all_t_items = []
        self.now_train = None
        self.now_test = None
        self.all_tslot = len(file_path_list)  # time slot number
        self.now_t = 0
        for file_t in file_path_list:
            df_file_t = pd.read_csv(path+datasetname+'/'+file_t+".csv")
            new_item = pd.read_csv(path+datasetname+'/'+file_t+".newitem.csv") #new item when t
            df_file_t_values = torch.from_numpy(df_file_t.values)[:,1:3]
            t_new_item = torch.from_numpy(new_item.values[:,1])
            self.all_interactiones.append(df_file_t_values)
            self.all_t_items.append(t_new_item)

    def creat_init_train_test(self,ration=[0.7,0.1,0.2]):
        """
        used for creat init train/test to find adptive parameters!!
        :param ration: ration[0]: init train ; ration[1]: val (init test) ; ration[2]: for online
        :return:
        """
        train_t_slot = round(self.all_tslot * ration[0])
        test_t_slot = round(self.all_tslot * ration[1])
        self.now_train = train_t_slot
        self.now_test = train_t_slot+test_t_slot
        self.now_t = train_t_slot+test_t_slot
        # train_set = torch.cat(self.all_interactiones[0:self.now_train])
        # test_set = torch.cat(self.all_interactiones[self.now_train:self.now_test])

        # creat data set used for really training and evalution

        print("sample neagtive for each interaction")
        try:
            train_set_train = np.load("./trainset_right.npy")
            train_set_train = torch.from_numpy(train_set_train)
        except:
            print("new creating!!! init train set")
            train_set_train = self.gen_neg_train_fast(torch.cat(self.all_interactiones[0:self.now_train]),
                                                      torch.cat(self.all_interactiones[self.now_train:self.now_test]),
                                                      neg_number=10)
        # print("sample for test set starting!!!!")
        # test_set_evalution = self.testset_for_evalution_fast(torch.cat(self.all_interactiones[0:self.now_train]),
        #                                                      torch.cat(self.all_interactiones[self.now_train:self.now_test]),
        #                                                      neg_evalution=1000)
        # print("sample for test set finished!!!!")

        return train_set_train,0#test_set_evalution

    def creat_online_train_test(self,time_add=True):
        '''
        creat train/test set for online traning pipline
        :param time_add:
        :return:
        '''
        if time_add:
            self.now_train = self.now_t
            self.now_test = self.now_t+1
            self.now_t += 1
            train_set = torch.cat(self.all_interactiones[0:self.now_train],dim=0)
            test_set = self.all_interactiones[self.now_test]

            # creat data set used for really training and evalution
            test_set_evalution = self.testset_for_evalution(train_set,test_set,neg_evalution=1000)
            train_set_train = self.gen_neg_train(train_set,test_set,neg_number=10)

            return  train_set_train,test_set_evalution
        else:
            raise AssertionError("time not add!!!")

    def gen_neg_train(self,trainset,testset,neg_number=None):
        '''
        :param trainset:tensor [one user, one pos item]
        :param testset:
        :param neg_number:
        :return: trainset: tensor [one user, one pos item, one neg item] * negnumber for one
        input line[one user, one pos item] with the same user, pos item, but different neg item
        '''

        if neg_number is None:
            print("dont't select neg sample numbers!!")
            return None
        if trainset.shape[1]>2:
            print("if really used parwised loss?? we treat it as yes, and only user interaction actived or not!!")
        trainset = trainset[:,0:2]
        testset = testset[:,0:2]

        now_item = torch.cat(self.all_t_items[0:self.now_t],dim=0)
        all_interaction = torch.cat([trainset,testset],dim=0)[:,0:2] #only need user and item interation information

        #then reprent the interation information as sparse matrix
        pos = all_interaction.transpose()
        v = torch.ones(pos.shape[1])
        all_interaction_sparse = sp.csr_matrix(pos,v) # save interactive as sparse matrix

        neg_sampled = torch.zeros(all_interaction.shape[0],neg_number)
        for i in range(trainset.shape[0]):
            u = trainset[i,0]
            have_sampled = []
            for j in range(neg_number):
                sample_idx = torch.randint(0,now_item.shape[0],[1])[0]
                sample_item = now_item[sample_idx]

                while all_interaction_sparse[u,sample_item] >0 or sample_item in have_sampled:  # wwhich time select again
                    sample_idx =torch.randint(0,now_item.shape[0],[1])[0]
                    sample_item = now_item[sample_idx]
                neg_sampled[i,j] = sample_item
                have_sampled.append(sample_idx)

        neg_sampled = neg_sampled.reshape(-1,1)
        trainset = trainset.repeat(1,neg_number).reshape(-1,2) #  reshape
        trainset = torch.cat([trainset,neg_sampled],dim=1)
        return trainset

    def gen_neg_train_fast(self,trainset,testset,neg_number=None):
        '''
        :param trainset:tensor [one user, one pos item]
        :param testset:
        :param neg_number:
        :return: trainset: tensor [one user, one pos item, one neg item] * negnumber for one
        input line[one user, one pos item] with the same user, pos item, but different neg item
        '''

        if neg_number is None:
            print("dont't select neg sample numbers!!")
            return None
        if trainset.shape[1]>2:
            print("if really used parwised loss?? we treat it as yes, and only user interaction actived or not!!")
            trainset = trainset[:,0:2]
            testset = testset[:,0:2]

        now_item = torch.cat(self.all_t_items[0:self.now_t],dim=0).numpy()
        all_interaction = torch.cat([trainset,testset],dim=0)                     #only need user and item interation information

        #then reprent the interation information as sparse matrix
        pos = all_interaction.transpose(1,0)
        v = np.ones(pos.shape[1])
        all_interaction_sparse = sp.csr_matrix((v,(pos[0],pos[1])),shape=(pos[0].max()+1,pos[0].max()+1))
        graph = nx.from_scipy_sparse_matrix(all_interaction_sparse,create_using = nx.DiGraph)
        inter_list = nx.to_dict_of_lists(graph)
        del all_interaction_sparse
        del graph

        neg_sampled = np.zeros((trainset.shape[0],neg_number),dtype=np.longlong)
        pbar = tqdm(range(trainset.shape[0]), total=trainset.shape[0], desc='({0:^3})'.format("train neg sample"))
        for i in pbar:
            u = int(trainset[i,0])
            have_hit = inter_list[u]                                                    #np.nonzero(all_interaction_sparse[u,:]>0)[1]
            sample_item = np.random.choice(now_item, neg_number+len(have_hit)+1000)     #[0:neg_number+have_hit.shape[0]]'''+have_hit.shape[0]+1000'''
            sample_item = np.setdiff1d(sample_item, np.array(have_hit))
            np.random.shuffle(sample_item)
            neg_sampled[i,:] = sample_item[0:neg_number]
        neg_sampled = torch.from_numpy(neg_sampled.reshape(-1,1))                       # to torch
        trainset = trainset.repeat(1,neg_number).reshape(-1,2)                          #  reshape
        trainset = torch.cat([trainset,neg_sampled],dim=1)                              # one user ,one pos item, one neg item
        return trainset

    # def gen_neg_train_fastfast(self,trainset,testset,neg_number=None):
    #     '''
    #     :param trainset:tensor [one user, one pos item]
    #     :param testset:
    #     :param neg_number:
    #     :return: trainset: tensor [one user, one pos item, one neg item] * negnumber for one
    #     input line[one user, one pos item] with the same user, pos item, but different neg item
    #     '''
    #
    #     if neg_number is None:
    #         print("dont't select neg sample numbers!!")
    #         return None
    #     if trainset.shape[1]>2:
    #         print("if really used parwised loss?? we treat it as yes, and only user interaction actived or not!!")
    #         trainset = trainset[:,0:2]
    #         testset = testset[:,0:2]
    #
    #     now_item = torch.cat(self.all_t_items[0:self.now_t],dim=0).numpy()
    #     all_interaction = torch.cat([trainset,testset],dim=0)                                #only need user and item interation information
    #
    #     #then reprent the interation information as sparse matrix
    #     pos = all_interaction.transpose(1,0)
    #     v = np.ones(pos.shape[1])
    #     all_interaction_sparse = sp.csr_matrix((v,(pos[0],pos[1])), shape=(pos[0].max()+1,pos[0].max()+1))
    #
    #     neg_sampled = np.zeros((all_interaction.shape[0],neg_number), dtype=np.long)
    #     pbar = tqdm(range(trainset.shape[0]), total=trainset.shape[0], desc='({0:^3})'.format("train neg sample"))
    #     for i in pbar:
    #         u = int(trainset[i,0])
    #         have_hit = all_interaction_sparse[u,:].tocoo().col                              #np.nonzero(all_interaction_sparse[u,:]>0)[1]
    #         sample_item = np.random.choice(now_item, neg_number+len(have_hit)+1000)         #[0:neg_number+have_hit.shape[0]]'''+have_hit.shape[0]+1000'''
    #         sample_item = np.setdiff1d(sample_item, np.array(have_hit))[0:neg_number]
    #         neg_sampled[i,:] = sample_item
    #     neg_sampled = torch.from_numpy(neg_sampled.reshape(-1,1))                           # to torch
    #     trainset = trainset.repeat(1,neg_number).reshape(-1,2)                              #  reshape  one interaction have neg_number copy
    #     trainset = torch.cat([trainset,neg_sampled],dim=1)                                  # one user ,one pos item, one neg item
    #     return trainset


    def gen_neg_train_fast_multi(self,trainset,testset,neg_number=None):
        '''
        :param trainset:tensor [one user, one pos item]
        :param testset:
        :param neg_number:
        :return: trainset: tensor [one user, one pos item, one neg item] * negnumber for one
        input line[one user, one pos item] with the same user, pos item, but different neg item
        '''

        if neg_number is None:
            print("dont't select neg sample numbers!!")
            return None
        if trainset.shape[1]>2:
            print("if really used parwised loss?? we treat it as yes, and only user interaction actived or not!!")
            trainset = trainset[:,0:2]
            testset = testset[:,0:2]

        now_item = torch.cat(self.all_t_items[0:self.now_t],dim=0).numpy()
        all_interaction = torch.cat([trainset,testset],dim=0)[:,0:2]                        #only need user and item interation information

        #then reprent the interation information as sparse matrix
        pos = all_interaction.transpose(1,0)
        v = np.ones(pos.shape[1])
        all_interaction_sparse = sp.csr_matrix((v,(pos[0],pos[1])),shape=(pos[0].max()+1,pos[1].max()+1))
        #all_interaction_sparse = torch.sparse_coo_tensor(pos,v) # save interactive as sparse matrix

        neg_sampled = np.zeros((all_interaction.shape[0],neg_number),dtype=int)
        pbar = tqdm(range(trainset.shape[0]), total=trainset.shape[0], desc='({0:^3})'.format("train neg sample"))

        def selectfor_one_interaction( j ):
            u = trainset[j, 0]
            have_hit = np.nonzero(all_interaction_sparse[u, :] > 0)[1]
            sample_item = np.random.choice(now_item,neg_number + have_hit.shape[0] + 1000)  # [0:neg_number+have_hit.shape[0]]
            sample_item = np.setdiff1d(sample_item, have_hit)[0:neg_number]
            neg_sampled[j, :] = sample_item
            if j%10000==0:
                print("number:",j)

        with ThreadPoolExecutor(max_workers=20) as executor:
            executor.map(selectfor_one_interaction,pbar)

        neg_sampled = torch.tensor(neg_sampled.reshape(-1,1))                               # to torch
        trainset = trainset.repeat(1,neg_number).reshape(-1,2)                              #  reshape
        trainset = torch.cat([trainset,neg_sampled],dim=1)                                  # one user ,one pos item, one neg item
        return trainset

    def testset_for_evalution(self,train_set,test_set, neg_evalution=1000):
        """
        we change the dataset from tensor [one user, one item] to  list [one user, all pos itemes,neg items]
        :param train_set:  used when selected  negative samples for test
        :param test_set:  test set when time t
        :param neg_evalution: how many negative samples for each user!!!  different from negative samples for train!!!
        :return: [one user, all item,]  if need neg item,[one,user,negitem]
        """
        user_dict = {}
        max_id = 0
        user = []
        item = []
        neg_item = []
        for i in range(test_set.shape[0]):
            user_i = test_set[i, 0]                                                         #user_i = user_dict[test_set[i, 0]]
            item_i = test_set[i, 1]                                                         #item_i = user_dict[test_set[i, 1]]
            try:
                user_id = user_dict[user_i]
                item[user_id].append(item_i)
            except:
                user_dict[user_i] = max_id
                user.append(user_i)
                item.append([item_i])
                max_id += 1

        """
        then we sampled negative item for test
        """
        now_item = torch.cat(self.all_t_items[0:self.now_t],dim=0)
        pos = torch.cat([train_set[:,0:2],test_set[:,0:2]],dim=0).transpose(1,0)
        v = torch.ones(pos.shape[1])

        all_interaction = sp.csr_matrix((v,(pos[0],pos[1])),shape=(pos[0].max()+1,pos[1].max()+1))
        array_item_max = pos[1].max()+1
        pbar = tqdm(range(len(user)), total=len(user), desc='({0:^3})'.format(1))
        for i in pbar:
            u = user[i]
            neg_i_item = []
            for s in range(neg_evalution):
                sample_idx = torch.randint(0,now_item.shape[0],[1])[0]
                sampled_item = now_item[sample_idx]
                if sampled_item <= array_item_max:
                    while all_interaction[u,sampled_item] > 0 or sampled_item in neg_i_item:   # if the item have been sampled, don't select again
                        sample_idx = torch.randint(0, now_item.shape[0], [1])[0]
                        sampled_item = now_item[sample_idx]
                neg_i_item.append(sample_idx)
            neg_item.append(neg_i_item)
        test_set=[user,item,neg_item]
        return test_set

    def testset_for_evalution_fast(self,train_set,test_set, neg_evalution=1000):
        """
        we change the dataset from tensor [one user, one item] to  list [one user, all pos itemes,neg items]
        :param train_set:  used when selected  negative samples for test
        :param test_set:  test set when time t
        :param neg_evalution: how many negative samples for each user!!!  different from negative samples for train!!!
        :return: [one user, all item,]  if need neg item,[one,user,negitem]
        """
        user_dict = {}
        max_id = 0
        user = []
        item = []
        neg_item = []

        for i in range(test_set.shape[0]):
            user_i = test_set[i, 0]                                                             #user_i = user_dict[test_set[i, 0]]
            item_i = test_set[i, 1]                                                             #item_i = user_dict[test_set[i, 1]]
            try:
                user_id = user_dict[user_i]
                item[user_id].append(item_i)
            except:
                user_dict[user_i] = max_id
                user.append(user_i)
                item.append([item_i])
                max_id += 1

        """
        then we sampled negative item for test
        """
        now_item = torch.cat(self.all_t_items[0:self.now_t],dim=0)


        #sample_idx = torch.randint(0, now_item.shape[0], [len(user),neg_evalution+1000])
        pbar = tqdm(range(len(user)), total=len(user), desc='({0:^3})'.format("neg sample"))
        for i in pbar:
            u = user[i]
            sample_idx_u = torch.randint(0, now_item.shape[0], [neg_evalution+1000])            #list(sample_idx[i].numpy())
            sampled_item = now_item[sample_idx_u].numpy()
            have_hit = np.array(item[i])
            can_be_select = np.setdiff1d(sampled_item,have_hit)

            if can_be_select.shape[0] >= neg_evalution:
                np.random.shuffle(can_be_select )
                neg_i_item = list(can_be_select[0:neg_evalution])
            else:
                raise AssertionError("please select more!!")

            neg_item.append(neg_i_item)
        test_set=[user,item,neg_item]
        return test_set









