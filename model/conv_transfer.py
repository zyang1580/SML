'''
convolution neureal network transfer for our sml
we only use : ConvTransfer_com and one_transfer
'''
import torch
import torch.nn.functional as F
import torch.nn as nn

def Gelu(x):
    return x*torch.sigmoid(1.702*x)

def BCEloss(item_score,negitem_score):
    pos_loss = -torch.mean(torch.log(torch.sigmoid(item_score)+1e-15))
    neg_loss = -torch.mean(torch.log(1-torh.sigmoid(negitem_score)+1e-15))
    loss = pos_loss + neg_loss
    return loss

class one_transfer(nn.Module):
    '''
    one transfer that contain two cnn layers
    '''

    def __init__(self,input_dim,out_dim,kernel=2):
        super(one_transfer, self).__init__()
        self.hidden_dim = input_dim
        self.out_channel = 10
        self.conv1 = nn.Conv2d(1,self.out_channel,(kernel,1),stride=1)

        self.out_channel2 = 5
        self.conv2 = nn.Conv2d(self.out_channel,self.out_channel2,(1,1),stride=1)


        self.fc1 = nn.Linear(input_dim*self.out_channel2,512) # 128
        self.fc2 = nn.Linear(512,out_dim)

        print("kernel:",kernel)
    def forward(self,x):
        x = self.conv1(x)
        #x = x.view(-1,self.hidden_dim*self.out_channel)
        x = Gelu(x)

        x = self.conv2(x)
        x = x.view(-1,self.hidden_dim*self.out_channel2)
        x = Gelu(x)


        x = self.fc1(x)
        x = Gelu(x)
        x = self.fc2(x)
        return x

class ConvTransfer(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(ConvTransfer, self).__init__()
        self.user_transfer = one_transfer(in_dim,out_dim,kernel=2)
        self.item_transfer = one_transfer(in_dim,out_dim,kernel=2)
    def forward(self,x_t,x_hat,type):
        x = torch.cat((x_t,x_hat),dim=-1)
        x = x.view(-1,1,2,x_t.shape[-1])
        if type == "user":
            x = self.user_transfer(x)
            x_norm = (x ** 2).sum(dim=-1).sqrt()
            x = x / x_norm.detach().unsqueeze(-1)
        elif type == "item":
            x = self.item_transfer(x)
        else:
            raise TypeError("convtransfer has not this type")
        return x


    def run_MF(self, user_weight_last, user_weight_hat, item_weight_last, item_weight_hat, negitem_weight_last,
               negitem_weight_hat,norm=False):

        user_weight_new = self.forward(user_weight_last, user_weight_hat, 'user')
        item_weight_new = self.forward(item_weight_last, item_weight_hat, 'item')
        negitem_weight_new = self.forward(negitem_weight_last, negitem_weight_hat, 'item')

        item_score = torch.mul(user_weight_new, item_weight_new).sum(dim=-1)
        negitem_score = torch.mul(user_weight_new, negitem_weight_new).sum(dim=-1)
        score = item_score - negitem_score
        if norm:
            user_ = (user_weight_new**2).sum(dim=-1).sqrt()
            score = score/user_
        bpr_loss = -torch.sum(F.logsigmoid(score))
        return bpr_loss

class ConvTransfer_com(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(ConvTransfer_com, self).__init__()
        self.user_transfer = one_transfer(in_dim,out_dim,kernel=3)
        self.item_transfer = one_transfer(in_dim,out_dim,kernel=3)
    def forward(self,x_t,x_hat,type):
        x_com = torch.mul(x_t, x_hat.data.detach())
        x_t_norm = (x_t**2).sum(dim=-1).sqrt()
        #x_hat_norm = (x_hat.data.detach()**2).sum(dim=-1).sqrt()
        #x_norm = torch.mul(x_t_norm,x_hat_norm).sqrt() + 1e-20
        # print(x_com.shape)
        # print(x_t_norm.shape)
        x_com = x_com / x_t_norm.unsqueeze(-1)
        x_com.requires_grad = False
        x = torch.cat((x_t,x_hat,x_com),dim=-1)

        x = x.view(-1,1,3,x_t.shape[-1])
        if type == "user":
            x = self.user_transfer(x)
        elif type == "item":
            x = self.item_transfer(x)
        else:
            raise TypeError("convtransfer has not this type")
        return x


    def run_MF(self, user_weight_last, user_weight_hat, item_weight_last, item_weight_hat, negitem_weight_last,
               negitem_weight_hat,norm=False,adpative=False,BCE=True):

        user_weight_new = self.forward(user_weight_last, user_weight_hat, 'user')
        item_weight_new = self.forward(item_weight_last, item_weight_hat, 'item')
        negitem_weight_new = self.forward(negitem_weight_last, negitem_weight_hat, 'item')

        item_score = torch.mul(user_weight_new, item_weight_new).sum(dim=-1)
        negitem_score = torch.mul(user_weight_new, negitem_weight_new).sum(dim=-1)
        if BCE:
            #pass
            pos_loss = -torch.mean(torch.log(torch.sigmoid(item_score)+1e-15))
            neg_loss = -torch.mean(torch.log(1-torch.sigmoid(negitem_score)+1e-15))
            bpr_loss = pos_loss + neg_loss
        else:
            score = item_score - negitem_score
            if norm:
                user_ = (user_weight_new**2).sum(dim=-1).sqrt()
                score = score/user_
            if adpative:
                pass
            bpr_loss = -torch.sum(F.logsigmoid(score))
        return bpr_loss

class one_transfer_com(nn.Module):
    def __init__(self,input_dim,out_dim,kernel=2,need_com=False):
        self.add_com = need_com
        super(one_transfer_com, self).__init__()
        self.hidden_dim = input_dim
        self.out_channel = 10
        self.conv1 = nn.Conv2d(1,self.out_channel,(kernel,1),stride=1)
        if self.add_com:
            self.fc1 = nn.Linear(input_dim*self.out_channel+input_dim,1024) # 128
        else:
            self.fc1 = nn.Linear(input_dim * self.out_channel, 1024)  # 128
        self.fc2 = nn.Linear(1024,out_dim)
        print("kernel:",kernel)
    def forward(self,x,x_com=None):
        x = self.conv1(x)
        x = x.view(-1,self.hidden_dim * self.out_channel)
        if self.add_com: # add x_com
            if x_com is not None:
                x = torch.cat((x, x_com),dim=-1)
            else:
                raise RuntimeError("please input x_com")
        x = Gelu(x)
        x = self.fc1(x)
        x = Gelu(x)
        x = self.fc2(x)
        return x

class ConvTransfer_com2(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(ConvTransfer_com2, self).__init__()
        self.user_transfer = one_transfer_com(in_dim,out_dim,kernel=2,need_com=True)
        self.item_transfer = one_transfer_com(in_dim,out_dim,kernel=2,need_com=True)
    def forward(self,x_t,x_hat,type):
        x_com = torch.mul((x_t.detach()**2).sqrt().sqrt(), (x_hat.detach()**2).sqrt().sqrt()) # not need grad.
        #print("x_com max:",x_com.max())
        x = torch.cat((x_t, x_hat),dim=-1)
        x = x.view(-1,1,2,x_t.shape[-1])
        if type == "user":
            x = self.user_transfer(x,x_com=x_com)
        elif type == "item":
            x = self.item_transfer(x,x_com=x_com)
        else:
            raise TypeError("convtransfer has not this type")
        return x


    def run_MF(self, user_weight_last, user_weight_hat, item_weight_last, item_weight_hat, negitem_weight_last,
               negitem_weight_hat,norm=False):

        user_weight_new = self.forward(user_weight_last, user_weight_hat, 'user')
        item_weight_new = self.forward(item_weight_last, item_weight_hat, 'item')
        negitem_weight_new = self.forward(negitem_weight_last, negitem_weight_hat, 'item')

        item_score = torch.mul(user_weight_new, item_weight_new).sum(dim=-1)
        negitem_score = torch.mul(user_weight_new, negitem_weight_new).sum(dim=-1)
        score = item_score - negitem_score
        if norm:
            user_ = (user_weight_new**2).sum(dim=-1).sqrt()
            score = score/user_
        bpr_loss = -torch.sum(F.logsigmoid(score))
        return bpr_loss

class ConvTransfer_com3(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(ConvTransfer_com3, self).__init__()
        self.user_transfer = one_transfer_com(in_dim,out_dim,kernel=2,need_com=True)
        self.item_transfer = one_transfer_com(in_dim,out_dim,kernel=2,need_com=True)
    def forward(self,x_t,x_hat,type):
        x_com = torch.mul((x_t.detach()**2).sqrt().sqrt(), (x_hat.detach()**2).sqrt().sqrt()) # not need grad.
        #print("x_com max:",x_com.max())
        x = torch.cat((x_t, x_hat),dim=-1)
        x = x.view(-1,1,2,x_t.shape[-1])
        if type == "user":
            x = self.user_transfer(x,x_com=x_com)
        elif type == "item":
            x = self.item_transfer(x,x_com=x_com)
        else:
            raise TypeError("convtransfer has not this type")
        return x


    def run_MF(self, user_weight_last, user_weight_hat, item_weight_last, item_weight_hat, negitem_weight_last,
               negitem_weight_hat,norm=False):

        user_weight_new = self.forward(user_weight_last, user_weight_hat, 'user')
        item_weight_new = self.forward(item_weight_last, item_weight_hat, 'item')
        negitem_weight_new = self.forward(negitem_weight_last, negitem_weight_hat, 'item')

        item_score = torch.mul(user_weight_new, item_weight_new).sum(dim=-1)
        negitem_score = torch.mul(user_weight_new, negitem_weight_new).sum(dim=-1)
        score = item_score - negitem_score
        if norm:
            user_ = (user_weight_new**2).sum(dim=-1).sqrt()
            score = score/user_
        bpr_loss = -torch.sum(F.logsigmoid(score))
        return bpr_loss

if __name__=="__main__":
    x_t = torch.rand([100,64])
    x_hat = torch.rand_like(x_t)
    net = ConvTransfer(64,64)
    y = net(x_t,x_hat,"user")
