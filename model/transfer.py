'''
SML model !!
Notced: The transfers in this code are all not used, we use transfer in model.conv_transfer.
see " class meta_train "
'''
import model.MF as MF
import torch
import torch.nn as nn
from data.dataset2 import transfer_data
from data.dataset import offlineDataset_withsample as SampleDaset
from data.dataset2 import trainDataset_withPreSample as PreSampleDatast
from data.dataset2 import testDataset
import torch.nn.functional as F
import torch.utils.data
from model.train_model import train_one_epoch
from evalution.evaluation2 import test_model
from torch.utils.tensorboard import SummaryWriter
import copy
import numpy as np
import time
from model.conv_transfer import ConvTransfer,ConvTransfer_com,ConvTransfer_com2



class Gelu(nn.Module):
    def __init__(self):
        super(Gelu, self).__init__()
    def forward(self,x):
        return x*torch.sigmoid(1.702*x)

class MLP(nn.Module):
    def __init__(self,in_dim,out_dim,hid_dim,dropout):
        super(MLP, self).__init__()
        model_list = nn.ModuleList()
        input_dim = in_dim
        for i in range(len(hid_dim)):
            model_list.append(nn.Linear(input_dim,hid_dim[i]))
            model_list.append(nn.Tanh())
            input_dim = hid_dim[i]
        model_list.append(nn.Dropout(p=dropout))
        model_list.append(nn.Linear(input_dim,out_dim))
        self.model_ = nn.Sequential(*model_list)
    def forward(self, x):
        return self.model_(x)

"""
some other type transfer are following,  not used!!
"""
class MLP2(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP2, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class transfer(nn.Module):
    def __init__(self,laten_dim):
        super(transfer, self).__init__()
        self.user_mlp = MLP(laten_dim,laten_dim,[128],dropout=0.5)
        self.item_mlp = MLP(laten_dim, laten_dim, [128], dropout=0.5)
    def forward(self, wetght_pre_t_user, weight_t_user, weight_pre_t_item, wetght_t_item):
        delta_weight_user = weight_t_user - wetght_pre_t_user
        delta_weight_item = wetght_t_item - weight_pre_t_item
        _delta_weight_user = self.user_mlp(delta_weight_user)
        _delta_weight_item = self.item_mlp(delta_weight_item)

        weight_t_user_ = wetght_pre_t_user + _delta_weight_user
        weight_t_item_ = weight_pre_t_item + _delta_weight_item
        return weight_t_user_, weight_t_item_

    def run_MF(self,wetght_pre_t_user, weight_t_user, weight_pre_t_item,
              wetght_t_item,weight_pre_t_negitem, wetght_t_negitem):
        delta_weight_user = weight_t_user - wetght_pre_t_user
        delta_weight_item = wetght_t_item - weight_pre_t_item
        delta_weight_negitem = wetght_t_negitem - weight_pre_t_negitem
        _delta_weight_user = self.user_mlp(delta_weight_user)
        _delta_weight_item = self.item_mlp(delta_weight_item)
        _delta_weight_negitem = self.item_mlp(delta_weight_negitem)

        weight_t_user_ = wetght_pre_t_user + _delta_weight_user
        weight_t_item_ = weight_pre_t_item + _delta_weight_item
        weight_t_negitem_ = weight_pre_t_negitem + _delta_weight_negitem

        scoer_item = torch.mul(weight_t_user_,weight_t_item_).sum(dim=-1)
        scoer_negitem = torch.mul(weight_t_user_,weight_t_negitem_).sum(dim=-1 )
        bpr_loss = -torch.sum(F.logsigmoid(scoer_item-scoer_negitem))
        return bpr_loss

class transfer2(nn.Module):
    def __init__(self,input_dim,out_dim,MF_bais,need_BN=False):
        super(transfer2, self).__init__()
        self.MF_bais = MF_bais
        self.need_item_transfer = True
        self.need_LN = False
        self.user_transfer = nn.Linear(input_dim*2,out_dim,bias=False)
        if self.need_item_transfer:
            self.item_transfer = nn.Linear(input_dim*2,out_dim,bias=False)
        else:
            print("don't use item transfer!!")
        if self.need_LN:
            self.LN_in_1 = nn.LayerNorm(input_dim)
            self.LN_in_2 = nn.LayerNorm(input_dim)
            self.LN_out = nn.LayerNorm(out_dim)
        #self.set_parametsers()
    def set_parametsers(self):
        m = self.user_transfer.weight.data.shape[0]
        n = self.user_transfer.weight.data.shape[1]
        a = torch.zeros([m,round(n/2)])
        b = torch.eye(m,round(n/2))
        init_weight = torch.cat([a,b],dim=1)
        self.user_transfer.weight.data.copy_(init_weight)
        self.item_transfer.weight.data.copy_(init_weight)

    def forward(self,x_t,x_hat,input_type):
        if self.need_LN:
            #x_t = self.LN_in_1(x_t)
            x_hat = self.LN_in_2(x_hat)

        inputs = torch.cat([x_t, x_hat], dim=-1)
        if input_type == "user":
            x_tt = self.user_transfer(inputs)
        elif input_type=='item':
            if self.need_item_transfer:
                x_tt = self.item_transfer(inputs)
            else:
                x_tt = x_hat
        else:
            raise TypeError("No such type in transfer2 forward function")
        if self.need_LN:
            x_tt = self.LN_out(x_tt)
        return x_tt

    def run_MF(self,user_weight_last,user_weight_hat,item_weight_last,item_weight_hat,negitem_weight_last,negitem_weight_hat):
        user_weight_new = self.forward(user_weight_last,user_weight_hat,'user')
        if self.need_item_transfer:
            item_weight_new = self.forward(item_weight_last,item_weight_hat,'item')
            negitem_weight_new = self.forward(negitem_weight_last,negitem_weight_hat,'item')
        else:
            item_weight_new = item_weight_hat
            negitem_weight_new = negitem_weight_hat

        if self.MF_bais:
            item_score = torch.mul(user_weight_new[:,0:-1], item_weight_new[:,0:-1]).sum(dim=-1) + item_weight_new[:,-1]
            negitem_score = torch.mul(user_weight_new[:,0:-1], negitem_weight_new[:,0:-1]).sum(dim=-1) + negitem_weight_new[:,-1]
            bpr_loss = -torch.sum(F.logsigmoid(item_score - negitem_score))
            #raise RuntimeError("when with bias ,need to check the function again")
        else:
            item_score = torch.mul(user_weight_new,item_weight_new).sum(dim=-1)
            negitem_score = torch.mul(user_weight_new,negitem_weight_new).sum(dim=-1)
            bpr_loss = -torch.sum(F.logsigmoid(item_score-negitem_score))
        return bpr_loss

class transfer_oneGRU(nn.Module):
    def __init__(self, input_dim, out_dim, MF_bais, need_BN=False):
        super(transfer_oneGRU, self).__init__()
        self.MF_bais = MF_bais
        self.need_item_transfer = True
        self.user_transfer = nn.GRUCell(input_dim * 2, out_dim, bias=True)
        if self.need_item_transfer:
            self.item_transfer = nn.GRUCell(input_dim * 2, out_dim, bias=True)
        else:
            print("don't use item transfer!!")

    def forward(self, x_t, x_hat, input_type):
        inputs = torch.cat([x_t, x_hat], dim=-1)
        if input_type == "user":
            x_tt = self.user_transfer(inputs,x_t)
        elif input_type == 'item':
            if self.need_item_transfer:
                x_tt = self.item_transfer(inputs,x_t)
            else:
                x_tt = x_hat
        else:
            raise TypeError("No such type in transfer2 forward function")
        return x_tt

    def run_MF(self, user_weight_last, user_weight_hat, item_weight_last, item_weight_hat, negitem_weight_last,
               negitem_weight_hat):
        user_weight_new = self.forward(user_weight_last, user_weight_hat, 'user')
        if self.need_item_transfer:
            item_weight_new = self.forward(item_weight_last, item_weight_hat, 'item')
            negitem_weight_new = self.forward(negitem_weight_last, negitem_weight_hat, 'item')
        else:
            item_weight_new = item_weight_hat
            negitem_weight_new = negitem_weight_hat

        if self.MF_bais:
            item_score = torch.mul(user_weight_new[:, 0:-1], item_weight_new[:, 0:-1]).sum(dim=-1) + item_weight_new[:,
                                                                                                     -1]
            negitem_score = torch.mul(user_weight_new[:, 0:-1], negitem_weight_new[:, 0:-1]).sum(
                dim=-1) + negitem_weight_new[:, -1]
            bpr_loss = -torch.sum(F.logsigmoid(item_score - negitem_score))
            # raise RuntimeError("when with bias ,need to check the function again")
        else:
            item_score = torch.mul(user_weight_new, item_weight_new).sum(dim=-1)
            negitem_score = torch.mul(user_weight_new, negitem_weight_new).sum(dim=-1)
            bpr_loss = -torch.sum(F.logsigmoid(item_score - negitem_score))
        return bpr_loss

class transfer3(nn.Module):
    def __init__(self, input_dim, out_dim, MF_bais):
        super(transfer3, self).__init__()
        self.MF_bais = MF_bais
        self.need_item_transfer = True
        self.user_transfer = MLP(input_dim*2,out_dim*2,[128],0)
        if self.need_item_transfer:
            self.item_transfer = MLP(input_dim*2,out_dim*2,[128],0)
        else:
            print("don't use item transfer!!")

        # self.set_parametsers()

    def forward(self, x_t, x_hat, input_type):
        inputs = torch.cat([x_t, x_hat], dim=-1)
        if input_type == "user":
            alpha =6 * F.sigmoid(self.user_transfer(inputs)) -3
            y = torch.mul(alpha,inputs)
            y = y.view(-1,2,x_t.shape[1])
            x_tt = torch.sum(y,dim=-2)
        elif input_type == 'item':
            if self.need_item_transfer:
                alpha = F.sigmoid(self.item_transfer(inputs))
                y = torch.mul(alpha, inputs)
                y = y.view(-1, 2, x_t.shape[1])
                x_tt = torch.sum(y, dim=-2)
            else:
                x_tt = x_hat
        else:
            raise TypeError("No such type in transfer2 forward function")
        return x_tt

    def run_MF(self, user_weight_last, user_weight_hat, item_weight_last, item_weight_hat, negitem_weight_last,
               negitem_weight_hat):
        user_weight_new = self.forward(user_weight_last, user_weight_hat, 'user')
        if self.need_item_transfer:
            item_weight_new = self.forward(item_weight_last, item_weight_hat, 'item')
            negitem_weight_new = self.forward(negitem_weight_last, negitem_weight_hat, 'item')
        else:
            item_weight_new = item_weight_hat
            negitem_weight_new = negitem_weight_hat

        if self.MF_bais:
            item_score = torch.mul(user_weight_new[:, 0:-1], item_weight_new[:, 0:-1]).sum(dim=-1) + item_weight_new[:,
                                                                                                     -1]
            negitem_score = torch.mul(user_weight_new[:, 0:-1], negitem_weight_new[:, 0:-1]).sum(
                dim=-1) + negitem_weight_new[:, -1]
            bpr_loss = -torch.sum(F.logsigmoid(item_score - negitem_score))
            # raise RuntimeError("when with bias ,need to check the function again")
        else:
            item_score = torch.mul(user_weight_new, item_weight_new).sum(dim=-1)
            negitem_score = torch.mul(user_weight_new, negitem_weight_new).sum(dim=-1)
            bpr_loss = -torch.sum(F.logsigmoid(item_score - negitem_score))
        return bpr_loss

"""
The above are some other type transfers,  not used!!
"""



"""
SML model !!
"""
class meta_train():
    '''
    SML model !!!
    '''
    def __init__(self,args,datasets:transfer_data,user_num,item_num,laten_dim):
        '''
        :param args:
        :param datasets:
        :param user_num: user number
        :param item_num: item number
        :param laten_dim: embedding dim
        '''
        if args.data_name!='yelp':
            """for yelp we can also run this code. For result in our paper, we don't run this code, 
            but we have tried to add this code and keep the same parameters, we can get similar result   """
            self.MFbase = MF.MFbasemode(num_user=user_num,num_item=item_num,laten_factor=laten_dim).cuda()

        """MF model, we directly load pretrain MF-model"""
        # the save methods for yelp and news have some difference, we need different method to load them
        # if your model not in cuda ,try to put it into cuda!
        if args.data_name =='yelp':
            self.MFbase = torch.load(args.pre_model)
        else:
            self.MFbase = torch.load(args.pre_model, map_location='cpu').cuda()

        self.transfer_type = args.transfer_type
        self.with_MF_bias = args.TR_with_MF_bias
        self.test_in_TR_train = args.test_in_TR_Train
        self.TR_train_sampleTYpe = args.TR_sample_type
        print("with MF bias:",self.with_MF_bias)
        print("transfer type:", self.transfer_type)
        self.need_writer = args.need_writer
        self.MF_TrainDataset = None
        if args.MF_sample == 'alone':
            self.MF_TrainDataset = SampleDaset
        elif args.MF_sample == 'all':
            self.MF_TrainDataset = PreSampleDatast



        if args.need_writer:
            path = "m-num"+str(args.multi_num)+"-MF-lr"+str(args.MF_lr)+"-l2-"+str(args.l2)+"e-"+str(args.MF_epochs)+"--TR-lr"+str(args.TR_lr)+"-l2-"+str(args.TR_l2)+"-e-"+str(args.TR_epochs)+str(args.TR_sample_type)+"user-norm"+str(args.norm)
            self.writer = SummaryWriter(comment=path)


        if self.with_MF_bias: # input the MF bais into transfer!
            self.last_user_weight = torch.cat([self.MFbase.user_laten.weight.data, self.MFbase.user_bais.weight.data], dim=-1)
            self.last_item_weight = torch.cat([self.MFbase.item_laten.weight.data, self.MFbase.item_bais.weight.data], dim=-1)
            self.user_weight_hat = torch.cat([self.MFbase.user_laten.weight.data, self.MFbase.user_bais.weight.data], dim=-1)
            self.item_weight_hat = torch.cat([self.MFbase.item_laten.weight.data, self.MFbase.item_bais.weight.data], dim=-1)
            self.last_user_weight_hat = copy.deepcopy(self.user_weight_hat)
            self.last_item_weight_hat = copy.deepcopy(self.item_weight_hat)
            input_dim = laten_dim + 1
        else:                # don't input the MF bais into transfer!
            input_dim = laten_dim

            self.last_user_weight = torch.zeros_like(self.MFbase.user_laten.weight.data)
            self.last_item_weight = torch.zeros_like(self.MFbase.item_laten.weight.data)
            self.user_weight_hat = copy.deepcopy(self.MFbase.user_laten.weight.data)
            self.item_weight_hat = copy.deepcopy(self.MFbase.item_laten.weight.data)

            self.last_user_weight_hat = copy.deepcopy(self.user_weight_hat)
            self.last_item_weight_hat = copy.deepcopy(self.item_weight_hat)

        ''' creat transfer model'''
        if self.transfer_type == "transfer":
            self.transfer = transfer(input_dim).cuda()
        elif self.transfer_type == 'transfer2':
            self.transfer = transfer2(input_dim, input_dim, self.with_MF_bias).cuda()
        elif self.transfer_type == "GRU":
            self.transfer = transfer_oneGRU(input_dim,input_dim,self.with_MF_bias).cuda()
            self.transfer_type = 'transfer2' # the optertion of GRU is same to transfer2
        elif self.transfer_type == "transfer3":
            self.transfer = transfer3(input_dim, input_dim, self.with_MF_bias).cuda()
            self.transfer_type = "transfer2"
        elif self.transfer_type == "conv":
            self.transfer = ConvTransfer(input_dim, input_dim).cuda()
            self.transfer_type = "transfer2"
        elif self.transfer_type == "conv_com":
            self.transfer = ConvTransfer_com(input_dim, input_dim).cuda()
            self.transfer_type = "transfer2"
        elif self.transfer_type == "conv_com2":
            self.transfer = ConvTransfer_com2(input_dim, input_dim).cuda()
            self.transfer_type = "transfer2"
        else:
            raise TypeError("No such type transfer!!!")

        self.dataset = datasets

        '''optimizer for two parts  MFbase and transfer!'''
        self.MF_optimizer = torch.optim.Adam(self.MFbase.parameters(), lr=args.MF_lr, weight_decay=0)
        self.transfer_optimizer = torch.optim.Adam(self.transfer.parameters(), lr=args.TR_lr, weight_decay=args.TR_l2) #1

        '''save model performance'''
        self.recall = []
        self.ndcg = []
        self.test_num = []

        self.recall_5 = []
        self.ndcg_5 = []
        self.MF_itr = 0
        self.TR_itr = 0

        self.recall_10 = []
        self.ndcg_10 = []

    def get_next_data(self,stage_id):
        '''
        get model next data. For more information ,please read the method of corresponding dataset class
        :param stage_id: stage idx
        :return: set_t,set_tt,now_test
        '''
        set_t,set_tt,now_test,val = self.dataset.next_train(stage_id)
        return set_t,set_tt,now_test,val

    def MF_train_onestage(self,args,set_t,stage_id,val=None): # this right for main
        '''
        MFbase train model. the tranfer will be fixed this satge!!
        :param args: hyper-parameters
        :param set_t: time t set
        :param stage_id: satge idx
        :param val: not usful, only for analyze
        :return: None
        '''
        clip = args.clip_grad
        need_adaptive = args.need_adaptive
        self.transfer.eval()
        if val is not None:
            val = testDataset(val)
            val = torch.utils.data.DataLoader(val,
                                                   batch_size=1024,
                                                   num_workers=args.numworkers,
                                                   pin_memory=False
                                                   )

        print("******MF (inner) training ******")
        set_t_dataloader = self.MF_TrainDataset(set_t)
        set_t_dataloader = torch.utils.data.DataLoader(set_t_dataloader,
                                                       batch_size=args.MF_batch_size,
                                                       shuffle=True,
                                                       num_workers=args.numworkers,
                                                       pin_memory=False)
        if val is not None:
            recall, ndcg = test_model(self.MFbase, val, topK=args.topK)
            print("before train MF test:recall:{:.4f} ndcg:{:.4f}".format(recall, ndcg))
            if self.need_writer:
                self.writer.add_scalar("Acc/MF-recall"+str(args.topK), recall,self.MF_itr) #stage_id * epochs
                self.writer.add_scalar("Acc/MF-ndcg" + str(args.topK), ndcg, self.MF_itr)
                user_ = self.MFbase.user_laten.weight.data[:]
                user_ = (user_**2).sum(dim=-1)
                self.writer.add_scalar("norm/user-norm",user_.mean(),self.MF_itr)
                del user_
                self.MF_itr += 1



        for epoch in range(args.MF_epochs):
            self.MFbase.train()
            self.transfer.eval()

            loss_all = 0
            for batch_id,(user,item,neg_item) in enumerate(set_t_dataloader):
                self.MFbase.zero_grad()
                self.transfer.zero_grad()
                user = user.cuda().long()
                item = item.cuda().long()
                neg_item = neg_item.cuda().long()
                weight_user_last = self.last_user_weight[user]
                weight_item_last = self.last_item_weight[item]
                weight_negitem_last = self.last_item_weight[neg_item]
                weight_user_MF = self.MFbase.user_laten(user)
                weight_item_MF = self.MFbase.item_laten(item)
                weight_negitem_MF = self.MFbase.item_laten(neg_item)

                loss_batch = self.transfer.run_MF(weight_user_last,
                                                  weight_user_MF,
                                                  weight_item_last,
                                                  weight_item_MF,
                                                  weight_negitem_last,
                                                  weight_negitem_MF,
                                                  norm=args.norm)

                # l2loss = torch.norm(weight_user_MF, dim=-1).sum() + torch.norm(weight_item_MF, dim=-1).sum() + torch.norm(
                #     weight_negitem_MF).sum()
                l2loss = 0.5 * torch.sum(weight_user_MF**2 + weight_item_MF**2 + weight_negitem_MF**2)

                loss_batch = loss_batch + args.l2 * l2loss

                if need_adaptive:
                    beta = 0.1
                    count = user.bincount()
                    user_max = user.max()
                    user_unique = torch.unique(user)
                    user_laten = self.MFbase.user_laten(user_unique)
                    norm_user = (user_laten.detach()**2).sum(dim=-1).sqrt()
                    count_ = count[user_unique]
                    loss_adpative = torch.mul(beta * count_ / norm_user, (user_laten**2).sum(dim=-1)).sum()
                    loss_batch += loss_adpative

                loss_all += loss_batch.data
                loss_batch.backward()
                # gradient_norm = False
                # if gradient_norm:
                #     user_laten_norm = (self.MFbase.user_laten.weight.data**2).sum(dim=-1).sqrt()
                #     self.MFbase.user_laten.weight.grad /= user_laten_norm.unsqueeze(-1)
                if clip:
                    # torch.nn.utils.clip_grad_value_(self.transfer.parameters(),2)
                    max_norm = args.maxnorm_grad
                    torch.nn.utils.clip_grad_norm_(self.transfer.parameters(), max_norm, norm_type=2)
                self.MF_optimizer.step()
                #print(self.MFbase.user_laten.weight.grad)

            loss_all = loss_all/(batch_id + 1)
            loss_all = loss_all.item() / args.MF_batch_size

            if val is not None:
                recall, ndcg = test_model(self.MFbase, val, topK=args.topK)
                print("MF-stage:",stage_id,"epoch:",epoch,"loss:{:.5f}".format(loss_all),"recall:{:.4f}".format(recall),"ndcg:{:.4f}".format(ndcg))
                if self.need_writer:
                    self.writer.add_scalar("Acc/MF-recall" + str(args.topK), recall, self.MF_itr)
                    self.writer.add_scalar("Acc/MF-ndcg" + str(args.topK), ndcg, self.MF_itr)
                    self.writer.add_scalar("Loss/MF-loss", loss_all, self.MF_itr)
            else:
                print("MF-stage:",stage_id,"epoch:",epoch,"loss:",loss_all)
                if self.need_writer:
                    self.writer.add_scalar("Loss/MF-loss",self.MF_itr)

            if self.need_writer:
                user_ = self.MFbase.user_laten.weight.data[:]
                user_ = (user_ ** 2).sum(dim=-1)
                self.writer.add_scalar("norm/user-norm", user_.mean(), self.MF_itr)
                del user_
                self.MF_itr += 1

    def MF_train_onestage_nopass(self,args,set_t,stage_id,val=None):
        '''
        MFbase train model. the tranfer will be fixed this satge!!
        :param args: hyper-parameters
        :param set_t: time t set
        :param stage_id: satge idx
        :return: None
        '''
        clip = args.clip_grad
        need_adaptive = args.need_adaptive
        self.transfer.eval()
        if val is not None:
            val = testDataset(val)
            val = torch.utils.data.DataLoader(val,
                                                   batch_size=1024,
                                                   num_workers=args.numworkers,
                                                   pin_memory=False
                                                   )

        print("******MF (inner) training ******")
        set_t_dataloader = self.MF_TrainDataset(set_t)
        set_t_dataloader = torch.utils.data.DataLoader(set_t_dataloader,
                                                       batch_size=args.MF_batch_size,
                                                       shuffle=True,
                                                       num_workers=args.numworkers,
                                                       pin_memory=False)
        epochs = args.MF_epochs
        if val is not None:
            recall, ndcg = test_model(self.MFbase, val, topK=args.topK)
            print("before train MF test:recall:{:.4f} ndcg:{:.4f}".format(recall, ndcg))
            if self.need_writer:
                self.writer.add_scalar("Acc/MF-recall"+str(args.topK), recall,self.MF_itr) #stage_id * epochs
                self.writer.add_scalar("Acc/MF-ndcg" + str(args.topK), ndcg, self.MF_itr)
                user_ = self.MFbase.user_laten.weight.data[:]
                user_ = (user_**2).sum(dim=-1)
                self.writer.add_scalar("norm/user-norm",user_.mean(),self.MF_itr)
                del user_
                self.MF_itr += 1



        for epoch in range(args.MF_epochs):
            self.MFbase.train()
            self.transfer.eval()

            loss_all = 0
            for batch_id,(user,item,neg_item) in enumerate(set_t_dataloader):
                self.MFbase.zero_grad()
                self.transfer.zero_grad()
                user = user.cuda().long()
                item = item.cuda().long()
                neg_item = neg_item.cuda().long()
                user_embeding, itemembdding, item_score = model(user, item)
                _, neg_itemembdding, negitem_score = model(user, neg_item)
                pos_loss = -torch.mean(torch.log(torch.sigmoid(item_score) + 1e-15))
                neg_loss = -torch.mean(torch.log(1 - torch.sigmoid(negitem_score) + 1e-15))
                loss_batch = pos_loss + neg_loss

                l2loss = 0.5 * torch.sum(weight_user_MF**2 + weight_item_MF**2 + weight_negitem_MF**2)

                loss_batch = loss_batch + args.l2 * l2loss

                if need_adaptive:
                    beta = 0.1
                    count = user.bincount()
                    user_max = user.max()
                    user_unique = torch.unique(user)
                    user_laten = self.MFbase.user_laten(user_unique)
                    norm_user = (user_laten.detach()**2).sum(dim=-1).sqrt()
                    count_ = count[user_unique]
                    loss_adpative = torch.mul(beta * count_ / norm_user, (user_laten**2).sum(dim=-1)).sum()
                    loss_batch += loss_adpative

                loss_all += loss_batch.data
                loss_batch.backward()
                # gradient_norm = False
                # if gradient_norm:
                #     user_laten_norm = (self.MFbase.user_laten.weight.data**2).sum(dim=-1).sqrt()
                #     self.MFbase.user_laten.weight.grad /= user_laten_norm.unsqueeze(-1)
                if clip:
                    # torch.nn.utils.clip_grad_value_(self.transfer.parameters(),2)
                    max_norm = args.maxnorm_grad
                    torch.nn.utils.clip_grad_norm_(self.transfer.parameters(), max_norm, norm_type=2)
                self.MF_optimizer.step()
                #print(self.MFbase.user_laten.weight.grad)

            loss_all = loss_all/(batch_id + 1)
            loss_all = loss_all.item() / args.MF_batch_size

            if val is not None:
                recall, ndcg = test_model(self.MFbase, val, topK=args.topK)
                print("MF-stage:",stage_id,"epoch:",epoch,"loss:{:.5f}".format(loss_all),"recall:{:.4f}".format(recall),"ndcg:{:.4f}".format(ndcg))
                if self.need_writer:
                    self.writer.add_scalar("Acc/MF-recall" + str(args.topK), recall, self.MF_itr)
                    self.writer.add_scalar("Acc/MF-ndcg" + str(args.topK), ndcg, self.MF_itr)
                    self.writer.add_scalar("Loss/MF-loss", loss_all, self.MF_itr)
            else:
                print("MF-stage:",stage_id,"epoch:",epoch,"loss:",loss_all)
                if self.need_writer:
                    self.writer.add_scalar("Loss/MF-loss",self.MF_itr)

            if self.need_writer:
                user_ = self.MFbase.user_laten.weight.data[:]
                user_ = (user_ ** 2).sum(dim=-1)
                self.writer.add_scalar("norm/user-norm", user_.mean(), self.MF_itr)
                del user_
                self.MF_itr += 1

    def transfer_train_onestage(self,args,set_tt,stage_id,compute_performance=False,val=None):
        """
        for better stop , we can sample a set_tt_test beased on set_tt, similar to online test traing set!!!!
        noticed: (1) because only user/item in set_tt(including negative items) will supervised!!  so ,we only need
                 put these weight into transfer ,in training stage.
                 (2) when, get weight_t used for next, stage, that all weight need be computed ! how about the new
                 user/item weights??
        :param args: some hyper_paraments
        :param set_tt: next time set
        :param stage_id: equal to time_stage
        :return:
        """
        clip = args.clip_grad
        need_adpative = False
        print("********* this is Transfer model training stage ***********")
        self.MFbase.eval()  # don't compute MFbase gradient
        ''' what's sample type Dataset wused! '''
        if self.TR_train_sampleTYpe == "alone": # select neg item only from now satge get data.ie set_tt
            set_tt = SampleDaset(set_tt)
            compute_performance = False
            if val is not  None:
                now_test = testDataset(val)
                now_test = torch.utils.data.DataLoader(now_test,
                                                       batch_size=1024,
                                                       num_workers=args.numworkers,
                                                       pin_memory=False
                                                       )
                compute_performance = True


        elif self.TR_train_sampleTYpe == 'all': # select neg item form 'all' item like real test
            now_test = testDataset(set_tt)
            now_test = torch.utils.data.DataLoader(now_test,
                                                   batch_size=1024,
                                                   num_workers=args.numworkers,
                                                   pin_memory=False
                                                   )
            set_tt = PreSampleDatast(set_tt)
            compute_performance = True

        if compute_performance:
            recall, ndcg = test_model(self.MFbase, now_test, topK=args.topK)
            print("before train transfer test:recall:{:.4f} ndcg:{:.4f}".format(recall,ndcg))
            if self.need_writer:
                self.writer.add_scalar("Acc/tr-TR-recall@" + str(args.topK), recall, self.TR_itr)
                self.writer.add_scalar("Acc/tr-TR-ndcg@" + str(args.topK), ndcg, self.TR_itr)
                self.TR_itr += 1

        set_tt = torch.utils.data.DataLoader(set_tt,
                                             batch_size=args.TR_batch_size,
                                             shuffle=True,
                                             num_workers=args.numworkers,
                                             pin_memory=False)
        s_time = time.time()
        for epoch in range(args.TR_epochs):
            self.transfer.train()
            loss_all = 0
            for batch_id,(user,item,neg_item) in enumerate(set_tt):
                self.transfer.zero_grad()
                user =user.cuda().long()
                item = item.cuda().long()
                neg_item = neg_item.cuda().long()

                weight_user_last = self.last_user_weight[user]
                weight_item_last = self.last_item_weight[item]
                weight_negitem_last = self.last_item_weight[neg_item]
                weight_user_MF = self.user_weight_hat[user]
                weight_item_MF = self.item_weight_hat[item]
                weight_negitem_MF = self.item_weight_hat[neg_item]

                loss_batch = self.transfer.run_MF(weight_user_last,
                                                  weight_user_MF,
                                                  weight_item_last,
                                                  weight_item_MF,
                                                  weight_negitem_last,
                                                  weight_negitem_MF,
                                                  norm=False)

                loss_all += loss_batch.data
                loss_batch.backward()
                if clip:
                    #torch.nn.utils.clip_grad_value_(self.transfer.parameters(),2)
                    max_norm = args.maxnorm_grad
                    torch.nn.utils.clip_grad_norm_(self.transfer.parameters(), max_norm,norm_type=2)
                self.transfer_optimizer.step()
            loss_all = loss_all/(batch_id+1)

            print("one epcohs TR time cost:",time.time()-s_time)
            """
            if need test, we can add code in here to test the real performance in here
            we use now test transfer performance
            """
            if self.need_writer:
                self.writer.add_scalar("Loss/TR-loss",loss_all.item() / args.TR_batch_size,self.TR_itr)
            if compute_performance:
                self.updata()
                recall, ndcg = test_model(self.MFbase, now_test, topK=args.topK)
                print("stage:{}, epcoh：{}，loss:{:.4f},*****val result  reacll:{:.4f}  ndcg:{:.4f}".format(stage_id,epoch,loss_all.item() / args.TR_batch_size,recall, ndcg))
                #self.load_MFbase_weight(self.user_weight_hat,self.item_weight_hat)  # recover the weight to train transfer
                if self.need_writer:
                    itrs = stage_id * args.TR_epochs + epoch
                    self.writer.add_scalar("Acc/tr-TR-recall@"+str(args.topK), recall,self.TR_itr)
                    self.writer.add_scalar("Acc/tr-TR-ndcg@"+str(args.topK), ndcg, self.TR_itr)
            else:
                print("stage:", stage_id, "epoch:", epoch, "transfer train loss:", loss_all.item() / args.TR_batch_size)
        print("stage ",stage_id," transfer trained finished!!!!")



    def train_one_stage3(self,args,stage_id):
        """
        run SML for one stage (period) !
        :param args: hyper-parameters
        :param stage_id: stage idx
        :return: bool  True: sucess  False: without data available.
        """
        # self.last_user_weight = torch.zeros_like(self.MFbase.user_laten.weight.data)
        # self.last_user_weight.copy_(self.MFbase.user_laten.weight.data)
        # self.last_item_weight = torch.zeros_like(self.MFbase.item_laten.weight.data)
        # self.last_item_weight.copy_(self.MFbase.item_laten.weight.data)
        # self.MF_optimizer.load_state_dict()   # if used adam, the optimizer may influence the train satge!!
        ''' save MFbase  last weight'''
        self.save_MF_weight(save_as='last')               # save MF weight as last_weight

        #the following code ,can be used when W_hat is pretrained, and not connected with transfer.
        #self.load_MFbase_weight(self.last_user_weight_hat,self.last_item_weight_hat)

        set_t, set_tt, now_test,val = self.get_next_data(stage_id)
        if set_t is None:   #  no data avaliable, stop
            return False
        if now_test is None:  # online training
            for phase in range(args.multi_num): # outer loop
                self.MF_train_onestage(args, set_t, stage_id,val=val)
                self.MFbase.eval()
                ''' save MFbase weight'''
                self.save_MF_weight(save_as='hat') # save as weight_hat
                self.updata()

                if self.need_writer:
                    self.writer.add_scalars("Scale/user_weight",{"weight_hat":torch.norm(self.user_weight_hat),
                                                                 "weight_last":torch.norm(self.last_user_weight)},stage_id)
                    self.writer.add_scalars("Scale/item_weight",
                                            {"weight_hat": torch.norm(self.item_weight_hat).item(),
                                             "weight_last": torch.norm(self.last_item_weight).item()},stage_id)
                '''train transfer'''
                self.transfer_train_onestage(args,set_tt,stage_id,val=val)
                if args.Load_W_hat:
                    self.load_MFbase_weight(self.user_weight_hat,self.item_weight_hat)
            '''updata model'''
            self.updata()
            return True
        elif set_tt is None: # stop training transfer, you can use "stage_id > 32 for yelp to stop when test stage not val stages"
            s_time = time.time()
            print("stop train transfer while test###!!!!!")
            args.MF_epochs = 2    # when stop training , the epoch number for MF updating
            self.MF_train_onestage(args, set_t, stage_id, val=val)
            self.MFbase.eval()
            self.save_MF_weight(save_as='hat')
            '''real test model '''
            self.updata()
            self.test_num.append(now_test.shape[0])
            now_test = testDataset(now_test)
            now_test = torch.utils.data.DataLoader(now_test,
                                                   batch_size=1024,
                                                   num_workers=args.numworkers,
                                                   pin_memory=False
                                                   )
            print("only traning time cost:",time.time()-s_time)
            recall, ndcg = test_model(self.MFbase, now_test, topK=20)
            print("test result --------- reacll:{:.4f}  ndcg:{:.4f}".format(recall, ndcg))
            self.recall.append(recall)
            self.ndcg.append(ndcg.cpu().numpy())

            recall, ndcg = test_model(self.MFbase, now_test, topK=10)
            print("test result --------- @10 reacll:{:.4f}  ndcg:{:.4f}".format(recall, ndcg))
            self.recall_10.append(recall)
            self.ndcg_10.append(ndcg.cpu().numpy())

            recall, ndcg = test_model(self.MFbase, now_test, topK=5)
            print("test result --------- @5 reacll:{:.4f}  ndcg:{:.4f}".format(recall, ndcg))
            self.recall_5.append(recall)
            self.ndcg_5.append(ndcg.cpu().numpy())
            print("include test time cost:",time.time()-s_time)
            return True
        else:
            # args.MF_epochs = 10
            # args.TR_epochs = 5
            for phase in range(args.multi_num):
                self.MF_train_onestage(args, set_t, stage_id, val=val)
                self.MFbase.eval()
                self.save_MF_weight(save_as='hat')
                if self.need_writer:
                    if self.need_writer:
                        self.writer.add_scalars("Scale/user_weight", {"weight_hat": torch.norm(self.user_weight_hat),
                                                                      "weight_last": torch.norm(self.last_user_weight)},
                                                stage_id)
                        self.writer.add_scalars("Scale/item_weight",
                                                {"weight_hat": torch.norm(self.item_weight_hat).item(),
                                                 "weight_last": torch.norm(self.last_item_weight).item()}, stage_id)

                '''real test model '''
                self.updata()


                if phase==0:     # the first outer loop, we use our sml model to test D_(t+1) before using it to train trasfer!!
                    self.test_num.append(now_test.shape[0])
                    now_test = testDataset(now_test)
                    now_test = torch.utils.data.DataLoader(now_test,
                                                           batch_size=1024,
                                                           num_workers=args.numworkers,
                                                           pin_memory=False
                                                           )

                    recall, ndcg = test_model(self.MFbase, now_test, topK=20)
                    print("test result --------- reacll:{:.4f}  ndcg:{:.4f}".format(recall, ndcg))
                    self.recall.append(recall)
                    self.ndcg.append(ndcg.cpu().numpy())

                    recall, ndcg = test_model(self.MFbase, now_test, topK=10)
                    print("test result --------- @10 reacll:{:.4f}  ndcg:{:.4f}".format(recall, ndcg))
                    self.recall_10.append(recall)
                    self.ndcg_10.append(ndcg.cpu().numpy())

                    recall, ndcg = test_model(self.MFbase, now_test, topK=5)
                    print("test result --------- @5 reacll:{:.4f}  ndcg:{:.4f}".format(recall, ndcg))
                    self.recall_5.append(recall)
                    self.ndcg_5.append(ndcg.cpu().numpy())
                '''****************train transfer again*********************************
                 in this code ,we reload MFbase weight with weight_hat
                # it's really not need to load !!!!!!self.using weight_hat as input to transfer, 
                because we are using self.using weight_hat as input to transfer during traing 
                '''
                # self.load_MFbase_weight(self.user_weight_hat,self.item_weight_hat)
                self.transfer_train_onestage(args, set_tt, stage_id, val=val)
                if args.Load_W_hat:
                    self.load_MFbase_weight(self.user_weight_hat,self.item_weight_hat)

            ''' really update model'''
            self.updata()
            return True


    def updata(self):
        '''
        we will update MFbase model weights
        :return: None
        '''
        self.MFbase.eval()
        self.transfer.eval()
        if self.transfer_type == "transfer":
            new_user_weight, new_item_weight = self.transfer(self.last_user_weight,
                                                             self.user_weight_hat,
                                                             self.last_item_weight,
                                                             self.item_weight_hat)  # maybe out of menmory!!!
        elif self.transfer_type == 'transfer2':
            new_user_weight = self.transfer(self.last_user_weight,self.user_weight_hat,'user')
            new_item_weight = self.transfer(self.last_item_weight, self.item_weight_hat,'item')
        else:
            raise TypeError("No such type transfer!!!")

        self.load_MFbase_weight(new_user_weight,new_item_weight)       # update MFbase weight

        # self.MFbase.user_laten.weight.copy_(new_user_weight)
        # self.MFbase.item_laten.weight.copy_(new_item_weight)

    # def update_MF_weight_hat(self):
    #     self.MFbase.user_laten.weight.data = user_wight
    #     self.MFbase.item_laten.weight.data = item_weight

    def save_MF_weight(self, save_as="last"):
        '''
        save MFbase weights,usually it's used when the stage strating and when MFbase have trained over.
        if the save type is wrong.will raise type error
        :param save_as: "last":save as weight_last  "hat":save as weight_hat
        :return: None
        '''
        if self.with_MF_bias:
            if save_as == "hat":
                a = torch.cat([self.MFbase.user_laten.weight.data, self.MFbase.user_bais.weight.data], dim=-1)
                self.user_weight_hat.copy_(a)
                b = torch.cat([self.MFbase.item_laten.weight.data, self.MFbase.item_bais.weight.data], dim=-1)
                self.item_weight_hat.copy_(b)
        else:
            if save_as == "last":
                self.last_user_weight.copy_(self.MFbase.user_laten.weight.data)
                self.last_item_weight.copy_(self.MFbase.item_laten.weight.data)
            elif save_as == "hat":
                self.last_user_weight_hat.copy_(self.user_weight_hat.data)
                self.last_item_weight_hat.copy_(self.item_weight_hat.data)

                self.user_weight_hat.copy_(self.MFbase.user_laten.weight.data)
                self.item_weight_hat.copy_(self.MFbase.item_laten.weight.data)
                # user_last_norm = torch.sum(self.last_user_weight**2,dim=-1).mean().sqrt()
                # user_hat_norm = torch.sum(self.user_weight_hat**2 , dim=1).mean().sqrt()
                # self.user_weight_hat = self.user_weight_hat/user_hat_norm * user_last_norm
                #
                # item_last_norm = torch.sum(self.last_item_weight ** 2, dim=-1).mean().sqrt()
                # item_hat_norm = torch.sum(self.item_weight_hat ** 2, dim=1).mean().sqrt()
                # self.item_weight_hat = self.item_weight_hat / item_hat_norm * item_last_norm

            else:
                raise TypeError("save MFbase weight type is wrong")

    def load_MFbase_weight(self, user_weight, item_weight):
        '''
        set MFbase model weights as the input.
        :param user_weight: MF user weights
        :param item_weight: MF item weight
        :return: None
        '''
        if self.with_MF_bias:
            self.MFbase.user_laten.weight.data.copy_(user_weight[:, 0:-1])
            self.MFbase.item_laten.weight.data.copy_(item_weight[:, 0:-1])
            self.MFbase.user_bais.weight.data.copy_(user_weight[:, -1].unsqueeze(-1))
            self.MFbase.item_bais.weight.data.copy_(item_weight[:, -1].unsqueeze(-1))
        else:
            self.MFbase.user_laten.weight.data.copy_(user_weight)
            self.MFbase.item_laten.weight.data.copy_(item_weight)





    def run(self,args):
        '''
        run the full model , will print the model performance
        :param args: hyper-parameters
        :return: None
        '''

        pass_num = args.pass_num
        path_ = "MF-lr" + str(args.MF_lr) + "-l2-" + str(args.l2) + "e-" + str(args.MF_epochs) + "--TR-lr" + str(args.TR_lr) + "-l2-" + str(args.TR_l2) + "-e-" + str(args.TR_epochs) + str(args.TR_sample_type)
        MF_epochs = args.MF_epochs

        for pass_id in range(pass_num):
            stage_id = 0
            self.dataset.reinit()
            while(1):
                #np.save(str(stage_id)+"_weight.npy", self.transfer.user_transfer.weight.data.cpu().numpy())
                flag = self.train_one_stage3(args,stage_id)
                if flag:
                    stage_id += 1
                    # if pass_id < pass_num and stage_id == 2:
                    #     args.MF_epochs = 5
                    if pass_id < (pass_num-1) and stage_id >= 19 :   # news need be different ,not 19
                        #args.MF_epochs = 5
                        break
                else:
                    print(str(pass_id)+"--trained over!!!!!")
                    test_num = np.array(self.test_num)
                    N3 = round(test_num.shape[0]*1/3)
                    val_num=test_num[0:N3]
                    test_num=test_num[N3:-1]
                    recall = np.array(self.recall)
                    ndcg = np.array(self.ndcg)

                    print(test_num)
                    print(recall)
                    print(ndcg)
                    print("include stage 0 of test:")
                    val_num=val_num*1.0/val_num.sum()
                    test_num = test_num*1.0 / test_num.sum()
                    #test_num = test_num.reshape(-1,1)

                    print("val average recall@20:",(recall[0:N3]*val_num).sum())
                    print("val average ndcg@20:", (ndcg[0:N3]*val_num).sum())
                    print("test average recall@20:", (recall[N3:-1] * test_num).sum())
                    print("test average ndcg@20:", (ndcg[N3:-1] * test_num).sum())

                    recall = np.array(self.recall_10)
                    ndcg = np.array(self.ndcg_10)
                    print("\n")
                    print("val average recall@10:", (recall[0:N3] * val_num).sum())
                    print("val average ndcg@10:", (ndcg[0:N3] * val_num).sum())
                    print("test average recall@10:", (recall[N3:-1] * test_num).sum())
                    print("test average ndcg@10:", (ndcg[N3:-1] * test_num).sum())

                    recall = np.array(self.recall_5)
                    ndcg = np.array(self.ndcg_5)
                    print("\n")
                    print("val average recall@5:", (recall[0:N3] * val_num).sum())
                    print("val average ndcg@5:", (ndcg[0:N3] * val_num).sum())
                    print("test average recall@5:", (recall[N3:-1] * test_num).sum())
                    print("test average ndcg@5:", (ndcg[N3:-1] * test_num).sum())

                    #torch.save(self.MFbase.state_dict(),"./save_model/pass-"+str(pass_id)+path_+"-MF_base.pt")
                    #torch.save(self.transfer.state_dict(), "./save_model/pass-"+str(pass_id)+path_+"-transfer.pt")
                    break
            # self.MFbase.load_state_dict(torch.load("MF_base_init.pt"))
            # args.MF_epochs = MF_epochs