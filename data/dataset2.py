import numpy as np
import torch
try:
    from tqdm import tqdm
except:
    pass
from torch.utils.data import Dataset
import copy


class data_prefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device_ = device
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_user, self.next_item, self.next_neg_item = next(self.loader)
        except StopIteration:
            self.next_user = None
            self.next_item = None
            self.next_neg_item =None
            return
        with torch.cuda.stream(self.stream):
            self.next_user = self.next_user.cuda(device=self.device_, non_blocking=True)
            self.next_item = self.next_item.cuda(device=self.device_, non_blocking=True)
            self.next_neg_item = self.next_neg_item.cuda(device=self.device_, non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_user = self.next_user.long()
            self.next_item = self.next_item.long()
            self.next_neg_item = self.next_neg_item.long()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        user = self.next_user
        item = self.next_item
        neg_item = self.next_neg_item
        self.preload()
        return user, item, neg_item

class testDataset(Dataset):
    '''
    Dateset subclass for test , used when the test is too large will cause out of memory!
    '''
    def __init__(self,dataset):
        super(testDataset, self).__init__()
        self.data = dataset
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]


class online_data():
    def __init__(self,path="dataset/",datasetname="News",file_path_list=None,test_list=None,validation_list=None,partion=7/21,test_partion=16/21,need_pbar=False):
        self.path = path
        self.dataname = datasetname
        self.file_list = file_path_list
        self.test_list = test_list
        self.val_list = validation_list
        self.len = len(file_path_list)
        self.start_time = round(self.len * partion)  #start not use as train
        self.start_test_time = round(self.len * test_partion)                     #start as test
        self.test_count = 0
        self.train_count = 0
        information = np.load(self.path+self.dataname+"/"+"information.npy")
        self.user_number = information[1]
        self.inter_all = information[0]
        self.item_number = information[2]
        print(information)
        a = [np.load(self.path+self.dataname+"/train/"+file+".npy") for file in self.file_list]
        a = np.concatenate(a,axis=0)
        print(a.shape[0],np.unique(a[:,0]).shape[0], np.unique(a[:,1]).shape[0])
    def get_offline(self,do_test=True,do_val=True):
        '''
        :return: offline train set and list of test set
        '''

        test_set = None
        val_set = None
        if self.test_list is not None and do_test:
            test_set = [np.load(self.path+self.dataname+"/test/"+test_file+".npy") for test_file in self.test_list]
            print("the number of test file is:",len(test_set))
        if self.val_list is not None and do_val:
            val_set = [np.load(self.path + self.dataname + "/validation/" + val_file + ".npy") for val_file in
                        self.val_list]
        if do_val:
            train_set = []
            print("as train files num:", self.start_time - len(self.val_list))
            for i in range(self.start_time - len(self.val_list)):  #some file in real train used be validation, left as train
                train_set.append(np.load(self.path + self.dataname + "/train/" + self.file_list[i] + ".npy"))
            train_set = np.concatenate(train_set, axis=0)
        else:   # not do val, ie add val set ti train sets
            train_set = []
            print("as train files num:", self.start_time)
            for i in range(self.start_time):
                train_set.append(np.load(self.path + self.dataname + "/train/" + self.file_list[i] + ".npy"))
            train_set = np.concatenate(train_set, axis=0)
        return train_set, val_set, test_set

    def next_online(self):
        '''
        :return: get the next time train and test set, conlude the first time get the init sets
        '''
        now_time = self.start_time+self.train_count
        if now_time >= self.len:
            return None,None
        train_set = []
        for i in range(now_time):
            train_set.append(np.load(self.path + self.dataname + "/train/" + self.file_list[i] + ".npy"))
        train_set = np.concatenate(train_set, axis=0)
        now_test = np.load(self.path+self.dataname+"/test/"+self.test_list[self.test_count]+".npy")
        print("will train:",self.file_list[i-1],"will test:",self.test_list[self.test_count])
        self.train_count += 1
        self.test_count += 1
        return train_set,now_test
    def next_online_nopre(self):
        '''
        :return: get the next time train and test set, conlude the first time get the init sets
        '''
        now_time = self.start_time + self.train_count
        if now_time >= (self.len-1):
            return None,None
        train_set = np.load(self.path + self.dataname + "/train/" + self.file_list[now_time] + ".npy")
        if now_time >= (self.start_test_time-1):
            now_test = np.load(self.path+self.dataname+"/test/"+self.test_list[self.test_count]+".npy")
            print("will train:", self.file_list[now_time], "will test:", self.test_list[self.test_count])
            self.test_count += 1
        else:
            now_test = None
            print("will train:", self.file_list[now_time], "will test: None")
        self.train_count += 1
        return train_set, now_test

    def get_time_t_data(self,time):
        test_time = self.start_time + time
        if test_time >= self.len:
            print("out of time")
            return None,None
        print("get time:{} data".format(time))
        print("train data numbers:", test_time)
        print("test file:",self.test_list[time])
        train_set = []
        for i in range(test_time):
            train_set.append(np.load(self.path + self.dataname + "/train/" + self.file_list[i] + ".npy"))
        train_set = np.concatenate(train_set, axis=0)
        now_test = np.load(self.path+self.dataname+"/test/"+self.test_list[time]+".npy")
        return train_set,now_test


class testDataset(Dataset):
    '''
    Dateset subclass for test , used when the test is too large will cause out of memory!
    '''
    def __init__(self,dataset):
        super(testDataset, self).__init__()
        self.data = dataset
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]

class trainDataset_withPreSample(Dataset):
    '''
    this is Dataset type for train transfer, the input dataset has sampled enough
    neg item. the each epoch, will select on cloumn neg_item as neg item
    '''
    def __init__(self,input_dataset):
        super(trainDataset_withPreSample, self).__init__()
        self.all_data = copy.deepcopy(input_dataset)
        self.have_read = 0
        self.neg_flag = np.arange(1, self.all_data.shape[1])
        np.random.shuffle(self.neg_flag)
        self.neg_all = input_dataset.shape[1]-2
        self.used_neg_count = 0
        self.data_len = input_dataset.shape[0]

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        user = self.all_data[idx,0]
        item = self.all_data[idx,1]
        neg_item = self.all_data[idx,self.neg_flag[self.used_neg_count]]
        self.have_read += 1
        if self.have_read >= self.data_len:
            self.have_read = 0
            self.used_neg_count += 1
            if self.used_neg_count >= self.neg_all:
                np.random.shuffle(self.neg_flag)
                self.used_neg_count = 0
        return user,item,neg_item

class transfer_data():
    def __init__(self,args,path="dataset/",datasetname="News",online_train_time=21,file_path_list=None,test_list=None,validation_list=None,online_test_time=48):
        '''
        :param args:  parameters collection
        :param path:  data path
        :param datasetname: dataset name
        :param online_train_time: the stage start to training
        :param file_path_list: file path list :such as , for yelp-- [0,1, ..., 39] i.e stage ids
        :param test_list:  test path list : such as for yelp-- [30,31,...,39]
        :param validation_list: not used!!
        :param online_test_time: the stage id that start to online testing
        '''
        self.TR_sample_type = args.TR_sample_type # TR sample typr
        self.TR_stop_ = args.TR_stop_             # if stop train transfer when online test stages
        self.MF_sample = args.MF_sample
        self.current_as_set_tt = args.set_t_as_tt
        self.path = path
        self.dataname = datasetname
        self.file_list = file_path_list
        self.test_list = test_list
        self.val_list = validation_list
        self.len = len(file_path_list)
        self.online_trian_time = online_train_time
        self.online_test_time = online_test_time
        self.start_test_time = online_test_time  #from which time, online train
        self.test_count = 0                      #start as test
        information = np.load(self.path+self.dataname+"/"+"information.npy")
        self.user_number = information[1]
        self.inter_all = information[0]
        self.item_number = information[2]
        print(information)
        a = [np.load(self.path+self.dataname+"/train/"+file+".npy") for file in self.file_list]
        a = np.concatenate(a,axis=0)
        print(a.shape[0],np.unique(a[:,0]).shape[0], np.unique(a[:,1]).shape[0])
        del a
    def reinit(self):  # used for args.pass_num great than 1, reint the dataset
        self.test_count = 0
        self.start_test_time = copy.deepcopy(self.online_test_time)

    def next_test(self): # not used, please use next_train which contain next_test
        '''
        used for really test stage
        '''
        if self.start_test_time >= self.len:
            return None,None
        train_set = []
        for i in range(self.start_test_time):
            train_set.append(np.load(self.path + self.dataname + "/train/" + self.file_list[i] + ".npy"))
        train_set = np.concatenate(train_set, axis=0)
        now_test = np.load(self.path+self.dataname+"/test/"+self.test_list[self.test_count]+".npy")
        self.start_test_time += 1
        self.test_count += 1
        return train_set,now_test

    def next_train(self, d_time):
        """
        d_time: the diss tance between self.online_trian_time and now_time
        in online train stage we need D_t and D_(t+1) ,different from D_(t+1) in test, we use it to
        train transfer model,so loss should be compute.
        :return:
         set:      D_t
         set_tt:   D_(t+1)
         stop_: if ture ,stop train when online train stage
        """
        now_time = self.online_trian_time + d_time
        if (now_time+1) >= self.len:   # end!!
            return None,None,None,None
        print("now time:", now_time)
        print("will be test data:", now_time + 1)
        if (now_time+1) < self.start_test_time:  # stop online train，if stop, we will stop online train
            if self.MF_sample == "alone":
                set_t = np.load(self.path + self.dataname + "/train/" + self.file_list[now_time] + ".npy")
            elif self.MF_sample == 'all':
                set_t = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time] + ".npy")
            else:
                raise TypeError("now such type when read next train sets")

            val = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time+1] + ".npy")
            
            print("now time:", self.file_list[now_time])
            if self.TR_sample_type == 'alone':
                if self.current_as_set_tt:
                    set_tt = np.load(self.path + self.dataname + "/train/" + self.file_list[now_time] + ".npy")
                    print("set tt is:",self.path + self.dataname + "/train/" + self.file_list[now_time] + ".npy")
                else:
                    set_tt = np.load(self.path + self.dataname + "/train/" + self.file_list[now_time+1] + ".npy")
                    print("set_tt is:", self.path + self.dataname + "/train/" + self.file_list[now_time+1] + ".npy")
            elif self.TR_sample_type == 'all':
                if self.current_as_set_tt:
                    set_tt = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time] + ".npy")
                else:
                    set_tt = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time + 1] + ".npy")
                print("t+1 data stes",self.file_list[now_time + 1] )
            else:
                raise TypeError("no such TR sample type")

            return set_t,set_tt,None,val
        elif self.TR_stop_: # stop train transfer when online test satge
            if self.MF_sample == "alone":
                set_t = np.load(self.path + self.dataname + "/train/" + self.file_list[now_time] + ".npy")
            elif self.MF_sample == 'all':
                set_t = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time] + ".npy")
            else:
                raise TypeError("now such type when read next train sets")

            print("now time:", self.file_list[now_time])
            print("will be test data:", self.test_list[self.test_count])
            now_test = np.load(self.path+self.dataname+"/test/"+self.test_list[self.test_count]+".npy")
            val = now_test
            self.test_count += 1
            return set_t, None, now_test,val
        else:
            """
            if select this ,will still train in online stage,it should first do test, 
            then use the data to train transfer!! 
            """
            if self.MF_sample == "alone":
                set_t = np.load(self.path + self.dataname + "/train/" + self.file_list[now_time] + ".npy")
            elif self.MF_sample == 'all':
                set_t = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time] + ".npy")
            else:
                raise TypeError("now such type when read next train sets")

            val = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time+1] + ".npy")
            '''according TR sample ty to load set_tt'''
            if self.TR_sample_type == 'alone':
                if self.current_as_set_tt:
                    set_tt = np.load(self.path + self.dataname + "/train/" + self.file_list[now_time] + ".npy")
                    print("settt is:", self.path + self.dataname + "/train/" + self.file_list[now_time] + ".npy")
                else:
                    set_tt = np.load(self.path + self.dataname + "/train/" + self.file_list[now_time+1] + ".npy")
                    print("settt is:", self.path + self.dataname + "/train/" + self.file_list[now_time+1] + ".npy")
                #set_tt = np.load(self.path + self.dataname + "/train/" + self.file_list[now_time + 1] + ".npy")
            elif self.TR_sample_type == 'all':
                if self.current_as_set_tt:
                    set_tt = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time] + ".npy")
                    print("set_tt is: ",self.path + self.dataname + "/test/" + self.file_list[now_time] + ".npy")
                else:
                    set_tt = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time + 1] + ".npy")
                    print("set_tt is",self.file_list[now_time + 1] )
                #set_tt = np.load(self.path + self.dataname + "/test/" + self.file_list[now_time + 1] + ".npy")
                print("t+1 datasets,",self.file_list[now_time + 1])
            else:
                raise TypeError("no such TR sample type")

            now_test = np.load(self.path + self.dataname + "/test/" + self.test_list[self.test_count] + ".npy")
            print("real test:",self.test_list[self.test_count])
            self.test_count += 1
            return set_t, set_tt, now_test,val




def select_neg_forinteraction(path='dataset/',datasetname='News',file_path_list=None,leave_for_init_train = 0.7,neg_num=999):
    inter_all = []
    information_all = np.load(path+datasetname+"/information.npy")
    for one_file in file_path_list:
        inter_all.append(np.load(path+datasetname+"/"+one_file+".npy"))
    all_time = len(inter_all)
    not_select_timenum = round( all_time * leave_for_init_train ) #[0,1,...,not_select_timenum-1] not need select neg sample
    start_select_time = not_select_timenum
    user_item = {}
    now_all_inter = np.concatenate(inter_all[0:start_select_time],axis=0)
    for one_inter in now_all_inter:   # we change the interaction to user item  dict-list
        try:
            user_item[one_inter[0]].add(one_inter[1])
        except:
            user_item[one_inter[0]] = set([one_inter[1]])

    now_all_item = set(np.unique(now_all_inter[:,1]))
    now_time = start_select_time

    neg_inter_all = []
    for i in range(start_select_time,all_time):
        neg_inter = []
        inter_all_now = inter_all[i]
        pbar = tqdm(inter_all_now, total=inter_all_now.shape[0], desc='({0:^3})'.format("neg sample"))
        for one_inter in pbar:
            now_all_item.add(one_inter[1])
            try:
                user_item[one_inter[0]].add(one_inter[1])                   # add ner item list of user
                have_hit_item = np.array(list(user_item[one_inter[0]]))     # item list of user to np array
            except:                                                         # new user
                user_item[one_inter[0]] = set([one_inter[1]])
                have_hit_item = np.array([one_inter[1]])

            all_item = np.array(list(now_all_item))                         # all item till now

            # can_be_select = np.setdiff1d(all_item , have_hit_item)
            # np.random.shuffle(can_be_select)
            # select_ = can_be_select[0:neg_num]

            select_more = np.random.choice(all_item, neg_num+1000)
            select_ = np.unique(np.setdiff1d(select_more, have_hit_item))
            while select_.shape[0] < neg_num:                                # select still there is 999 non repeat item
                select_more = np.random.choice(all_item, neg_num + 1000)
                select_ = np.unique(np.setdiff1d(select_more, have_hit_item))
            np.random.shuffle(select_)                                        # shuffle the set we will select,else will in increase order!!
            select_ = select_[0:neg_num].reshape(1,-1)                        # add a dim , for concat
            neg_inter.append(select_)

        neg_inter = np.concatenate(neg_inter,axis=0)                          # (N * neg_num) * 1
        neg_inter_all.append(neg_inter)

    test_set = []
    j = 0
    for i in range(start_select_time, all_time):
        inter = inter_all[i]
        inter_new = np.concatenate([inter, neg_inter_all[j]], axis=1)
        j += 1
        test_set.append(inter_new)
        np.save(path+datasetname+"/test/"+str(i)+".npy", inter_new)

if __name__=="__main__":
    data_path = '../dataset/'  # /home/zyang/code/onlineMF_new
    data_name = 'news_3week_all'
    file_list = ["2017010" + str(i) if i < 10 else "201701" + str(i) for i in range(1, 22)]
    select_neg_forinteraction(path=data_path, datasetname=data_name, file_path_list=file_list )














