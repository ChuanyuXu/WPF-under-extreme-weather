import os
import time
import numpy as np
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import scipy.io as scio
import random
from ModelTool.Model import TemporalBlock_v2
from torch.nn.utils import weight_norm

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, mode='pre', kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock_v2(in_channels, out_channels, kernel_size,  stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout, mode=mode)]
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


## data processing
seed_torch(seed=1029)
dataFile = 'wf_4_train'
wf_1  = scio.loadmat(dataFile)
p=wf_1['p_1h']
p_conven_00=wf_1['p_conven']
p_conven_class_00=wf_1['p_conven_class']
p_extre_class1_00=wf_1['p_extre_class1']
p_extre_class2_00=wf_1['p_extre_class2']
p_extre_class3_00=wf_1['p_extre_class3']
p_extre_class4_00=wf_1['p_extre_class4']
nwp=wf_1['nwp_1h']
nwp_conven_00=wf_1['nwp_conven_']
nwp_conven_class_00=wf_1['nwp_conven_class_']
nwp_extre_class1_00=wf_1['nwp_extre_class1_']
nwp_extre_class2_00=wf_1['nwp_extre_class2_']
nwp_extre_class3_00=wf_1['nwp_extre_class3_']
nwp_extre_class4_00=wf_1['nwp_extre_class4_']
P_load1=p[:,1]
P_load=P_load1.reshape(np.size(P_load1,axis=0),-1)
P_nwp1=nwp
nwp_index=[25,20,2,11,6]
for i in range(np.size(nwp_conven_class_00)):
    if i==0:
        P_nwp=P_nwp1[:,nwp_index[i]].reshape(np.size(P_nwp1,axis=0),-1)
    else:
        P_nwp=np.concatenate((P_nwp,P_nwp1[:,nwp_index[i]].reshape(np.size(P_nwp1,axis=0),-1)),axis=1)


# Define training equipment
device=torch.device("cuda")
device0=torch.device("cpu")


# Define Parameters
dem_realp=1
len_realp=12
Cap=400.5
m=365
d=24
ooo=365
Series_day = P_load.reshape(-1,dem_realp)/Cap
nwp_day = (P_nwp/np.max(abs(P_nwp),axis=0)).reshape(-1,dem_realp*np.size(P_nwp,axis=1))
dem_realc=np.size(P_nwp,axis=1)
p_conven_=p_conven_00/Cap
nwp_conven_=np.empty([1,5],dtype=object)
for i in range(np.size(nwp_conven_00,axis=1)):
    nwp_conven_[0,i]=nwp_conven_00[0,i]/np.max(abs(P_nwp[:,i]),axis=0)
for i_nwp in range(np.size(nwp_conven_, axis=1)):
    if i_nwp == 0:
        nwp_conven_1 = nwp_conven_[0, i_nwp].transpose(1, 0)
        nwp_conven_1 = nwp_conven_1[:, :, np.newaxis]
    else:
        nwp_conven_0 = nwp_conven_[0, i_nwp].transpose(1, 0)
        nwp_conven_1 = np.concatenate((nwp_conven_1, nwp_conven_0[:, :, np.newaxis]), axis=2)
p_conven_1 = p_conven_.transpose(1, 0)
p_conven_1 = p_conven_1[:, :, np.newaxis]
train_input_c = nwp_conven_1
train_target_p = p_conven_1
test_target_p_=Series_day[m*d//dem_realp:(m*d+ooo*d)//dem_realp,:]
test_target_p=test_target_p_.reshape(-1,len_realp,dem_realp)
test_input_c_=nwp_day[m*d//dem_realp:(m*d+ooo*d)//dem_realp,:]
test_input_c=test_input_c_.reshape(-1,len_realp,dem_realc)
p_conven_class_=p_conven_class_00/Cap
nwp_conven_class_=np.empty([1,5],dtype=object)
for i in range(np.size(nwp_conven_class_00,axis=1)):
    nwp_conven_class_[0,i]=nwp_conven_class_00[0,i]/np.max(abs(P_nwp[:,i]),axis=0)
p_extre_class__=np.empty([1,4],dtype=object)
p_extre_class__[0,0]=p_extre_class1_00
p_extre_class__[0,1]=p_extre_class2_00
p_extre_class__[0,2]=p_extre_class3_00
p_extre_class__[0,3]=p_extre_class4_00
p_extre_class_=p_extre_class__/Cap
nwp_extre_class_00=np.empty([1,5],dtype=object)
for i_nwp in range(5):
    nwp_extre_class_00[0,i_nwp]=np.empty([1,4],dtype=object)
    for i_class in range(4):
        if i_class == 0:
            nwp_extre_class_00[0, i_nwp][0, i_class] = nwp_extre_class1_00[0, i_nwp]
        elif i_class == 1:
            nwp_extre_class_00[0, i_nwp][0, i_class] = nwp_extre_class2_00[0, i_nwp]
        elif i_class == 2:
            nwp_extre_class_00[0, i_nwp][0, i_class] = nwp_extre_class3_00[0, i_nwp]
        elif i_class == 3:
            nwp_extre_class_00[0, i_nwp][0, i_class] = nwp_extre_class4_00[0, i_nwp]
nwp_extre_class_=np.empty([1,5],dtype=object)
for i in range(np.size(nwp_extre_class_00,axis=1)):
    nwp_extre_class_[0,i]=nwp_extre_class_00[0,i]/np.max(abs(P_nwp[:,i]),axis=0)


## Define forecasting model
class model_fore(nn.Module):
    def __init__(self,input_channel_fore, output_channel_fore, mode, output_size_baselearner=1,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super().__init__()
        self.tcn = TemporalConvNet(input_channel_fore, output_channel_fore, mode, kernel_size, dropout)
        self.emb_dropout = emb_dropout
        self.drop = nn.Dropout(emb_dropout)
        self.fore_baselearner = nn.Linear(output_channel_fore[-1], output_size_baselearner)
        self.init_weights()
    def init_weights(self):
        self.fore_baselearner.bias.data.fill_(0)
        self.fore_baselearner.weight.data.normal_(0, 0.01)
    def forward(self, x):
        y = self.drop(x)
        y = self.tcn(y.transpose(1, 2)).transpose(1, 2)
        y = self.fore_baselearner(y)
        return y.contiguous()
model_fore_pre = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8],mode='pre')
model_fore_train_task_support = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8],mode='train_task_support')
model_fore_train_task_query = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8],mode='train_task_query')
model_fore_test_task_support = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8],mode='test_task_support')
model_fore_test_task_query = model_fore(input_channel_fore=dem_realc, output_channel_fore=[128, 96, 64, 48, 32, 16, 8],mode='test_task_support')


## Define loss
loss_fn_1=nn.MSELoss()
def penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss1 = loss_fn_1(logits[0::2] * scale, y[0::2])
    loss2 = loss_fn_1(logits[1::2] * scale, y[1::2])
    # grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    # return torch.sum(grad ** 2)
    grad_batch1 = torch.autograd.grad(loss1, [scale], create_graph=True)[0]
    grad_batch2 = torch.autograd.grad(loss2, [scale], create_graph=True)[0]
    return torch.sum(grad_batch1 * grad_batch2)


## Define optimizer
optimizer_fore_pre = torch.optim.Adam(filter(lambda p: p.requires_grad, model_fore_pre.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_fore_train_task_support = torch.optim.Adam(filter(lambda p: p.requires_grad, model_fore_train_task_support.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_fore_train_task_query = torch.optim.Adam(filter(lambda p: p.requires_grad, model_fore_train_task_query.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_fore_test_task_support = torch.optim.Adam(filter(lambda p: p.requires_grad, model_fore_test_task_support.parameters()), lr=0.0002, betas=(0.5, 0.999))
Train_target_p=torch.tensor(train_target_p,dtype=torch.float32)
Train_input_c=torch.tensor(train_input_c,dtype=torch.float32)
Test_target_p=torch.tensor(test_target_p,dtype=torch.float32)
Test_input_c=torch.tensor(test_input_c,dtype=torch.float32)
model_fore_pre = model_fore_pre.to(device)
model_fore_train_task_support = model_fore_train_task_support.to(device)
model_fore_train_task_query = model_fore_train_task_query.to(device)
model_fore_test_task_support = model_fore_test_task_support.to(device)
model_fore_test_task_query = model_fore_test_task_query.to(device)


## pre-train
print( "#########################################################################——————————pre-train——————————############################################################")
total_train_step=0
total_test_step=0
epoch1_pre = 80000
writer1=SummaryWriter("./logs_train/loss1")
writer2=SummaryWriter("./logs_train/loss2")
start_time=time.time()
for i in range(epoch1_pre):
    if i<10000:
        k = 0
    elif 10000<=i<20000:
        k = 1
    elif 20000<=i<30000:
        k = 5
    elif 30000 <= i:
        k = 10
    Train_target_p = Train_target_p.to(device)
    Train_input_c = Train_input_c.to(device)
    model_fore_pre.train()
    Train_outputs_pre=model_fore_pre(Train_input_c)
    loss1 = penalty(Train_outputs_pre,Train_target_p)
    loss2=loss_fn_1(Train_outputs_pre,Train_target_p)
    loss_en=k * loss1 + loss2
    optimizer_fore_pre.zero_grad()
    loss_en.backward()
    optimizer_fore_pre.step()
    #
    if (i + 1) % 100 == 0:
        end_time = time.time()
        print(end_time - start_time)
        print(
            "[Epoch %d/%d] [loss_mse: %f] "
            % (i, epoch1_pre, loss2.item())
        )
        writer1.add_scalar("loss_mse_pre", loss1.item(), i)
        writer2.add_scalar("loss_mse_pre", loss2.item(), i)
model_fore_pre.eval()
torch.save(model_fore_pre.state_dict(),"./model_fore_pre.pth")


## train_task_support
epoch_train_task=70000
for i_t in range(epoch_train_task):
    index_class = random.sample(range(0, 10), 5)
    nwp_conven_class=list()
    p_conven_class = list()
    for i_class in range(10):
        for i_nwp in range(np.size(nwp_conven_class_,axis=1)):
            if i_nwp==0:
                nwp_conven_class_1=nwp_conven_class_[0,i_nwp][0,i_class].transpose(1,0)
                nwp_conven_class_1=nwp_conven_class_1[:,:,np.newaxis]
            else:
                nwp_conven_class_0=nwp_conven_class_[0,i_nwp][0,i_class].transpose(1,0)
                nwp_conven_class_1=np.concatenate((nwp_conven_class_1, nwp_conven_class_0[:,:,np.newaxis]), axis=2)
        p_conven_class_1 = p_conven_class_[0, i_class].transpose(1, 0)
        p_conven_class_1 = p_conven_class_1[:, :, np.newaxis]
        nwp_conven_class.append(nwp_conven_class_1)
        p_conven_class.append(p_conven_class_1)
    train_input_dataset=list()
    train_target_dataset = list()
    for i_class in range(np.size(index_class)):
        train_input_dataset.append(nwp_conven_class[index_class[i_class]])
        train_target_dataset.append(p_conven_class[index_class[i_class]])
    train_input_support_=np.empty([10, 12, 5])
    train_input_query_ = np.empty([10, 12, 5])
    train_target_support_=np.empty([10, 12, 1])
    train_target_query_ = np.empty([10, 12, 1])
    for i_class in range(np.size(index_class)):
        index_shot = random.sample(range(0, np.size(train_input_dataset[i_class],axis=0)), 20)
        train_input_support_=train_input_dataset[i_class][index_shot[0:10],:,:]
        train_input_query_ = train_input_dataset[i_class][index_shot[10:20], :, :]
        train_target_support_=train_target_dataset[i_class][index_shot[0:10],:,:]
        train_target_query_ = train_target_dataset[i_class][index_shot[10:20], :, :]
        if i_class==0:
            train_input_support=train_input_support_
            train_input_query = train_input_query_
            train_target_support=train_target_support_
            train_target_query = train_target_query_
        else:
            train_input_support=np.concatenate((train_input_support,train_input_support_),axis=0)
            train_input_query = np.concatenate((train_input_query, train_input_query_), axis=0)
            train_target_support=np.concatenate((train_target_support,train_target_support_),axis=0)
            train_target_query = np.concatenate((train_target_query, train_target_query_), axis=0)
    Train_target_support = torch.tensor(train_target_support, dtype=torch.float32)
    Train_input_support = torch.tensor(train_input_support, dtype=torch.float32)
    Train_target_query = torch.tensor(train_target_query, dtype=torch.float32)
    Train_input_query = torch.tensor(train_input_query, dtype=torch.float32)
    print(
        "[##################################################################——————————train_task_support_Epoch %d/%d——————————############################################################]"
        % (i_t, epoch_train_task)
    )
    if i_t==0:
        model_fore_train_task_support.load_state_dict(torch.load("model_fore_pre.pth"))
    else:
        model_fore_train_task_support.load_state_dict(torch.load("model_fore_train_task_query.pth"))
    total_train_step=0
    total_test_step=0
    epoch1_train_task_support = 1
    start_time=time.time()
    for i in range(epoch1_train_task_support):
        k=10
        Train_target_support = Train_target_support.to(device)
        Train_input_support = Train_input_support.to(device)
        model_fore_train_task_support.train()
        Train_outputs_support=model_fore_train_task_support(Train_input_support)
        loss1=penalty(Train_outputs_support,Train_target_support)
        loss2=loss_fn_1(Train_outputs_support,Train_target_support)
        loss_en=k*loss1+loss2
        optimizer_fore_train_task_support.zero_grad()
        loss_en.backward()
        optimizer_fore_train_task_support.step()
        #
        if (i + 1) % 2 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print(
                "[Epoch %d/%d] [loss_mse: %f]"
                % (i, epoch1_train_task_support, loss2.item())
            )
            writer1.add_scalar("loss_mse_train_task_support", loss1.item(),
                              epoch1_pre + i_t * epoch1_train_task_support + i)
            writer2.add_scalar("loss_mse_train_task_support", loss2.item(), epoch1_pre+i_t*epoch1_train_task_support+i)
    model_fore_train_task_support.eval()
    torch.save(model_fore_train_task_support.state_dict(),"./model_fore_train_task_support.pth")


    ## train_task_query
    print(
        "[##################################################################——————————train_task_query_Epoch %d/%d——————————############################################################]"
        % (i_t, epoch_train_task)
    )
    if i_t==0:
        model_fore_train_task_query.load_state_dict(torch.load("model_fore_pre.pth"))
        model_fore_train_task_support.load_state_dict(torch.load("model_fore_train_task_support.pth"))
    else:
        model_fore_train_task_query.load_state_dict(torch.load("model_fore_train_task_query.pth"))
        model_fore_train_task_support.load_state_dict(torch.load("model_fore_train_task_support.pth"))
    total_train_step=0
    total_test_step=0
    epoch1_train_task_query = 1
    start_time=time.time()
    for i in range(epoch1_train_task_query):
        k=10
        Train_target_query = Train_target_query.to(device)
        Train_input_query = Train_input_query.to(device)
        model_fore_train_task_support.train()
        Train_outputs_query_=model_fore_train_task_support(Train_input_query)
        loss1 = penalty(Train_outputs_query_, Train_target_query)
        loss2=loss_fn_1(Train_outputs_query_,Train_target_query)
        loss_en=k*loss1+loss2
        optimizer_fore_train_task_query.zero_grad()
        loss_en.backward()
        optimizer_fore_train_task_query.step()
        Train_outputs_query=model_fore_train_task_query(Train_input_query)
        #
        if (i + 1) % 1 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print(
                "[Epoch %d/%d] [loss_mse: %f] "
                % (i, epoch1_train_task_query, loss2.item())
            )

            writer1.add_scalar("loss_mse_train_task_query", loss1.item(), epoch1_pre+i_t*(epoch1_train_task_support+epoch1_train_task_query)+epoch1_train_task_support+i)
            writer2.add_scalar("loss_mse_train_task_query", loss2.item(), epoch1_pre + i_t * (
                        epoch1_train_task_support + epoch1_train_task_query) + epoch1_train_task_support + i)
    model_fore_train_task_query.eval()
    torch.save(model_fore_train_task_query.state_dict(),"./model_fore_train_task_query.pth")


## test_task_support
print("##################################################################——————————test_task_support——————————############################################################")

Test_outputs_support_list=list()
for i_class in range(4):
    model_fore_test_task_support.load_state_dict(torch.load("model_fore_train_task_query.pth"))
    for i_nwp in range(np.size(nwp_extre_class_, axis=1)):
        if i_nwp == 0:
            nwp_extre_class_1 = nwp_extre_class_[0, i_nwp][0, i_class].transpose(1, 0)
            nwp_extre_class_1 = nwp_extre_class_1[:, :, np.newaxis]
        else:
            nwp_extre_class_0 = nwp_extre_class_[0, i_nwp][0, i_class].transpose(1, 0)
            nwp_extre_class_1 = np.concatenate((nwp_extre_class_1, nwp_extre_class_0[:, :, np.newaxis]), axis=2)
    p_extre_class_1 = p_extre_class_[0, i_class].transpose(1, 0)
    p_extre_class_1 = p_extre_class_1[:, :, np.newaxis]
    nwp_extre_class=nwp_extre_class_1
    p_extre_class=p_extre_class_1
    Test_target_support = torch.tensor(p_extre_class, dtype=torch.float32)
    Test_input_support = torch.tensor(nwp_extre_class, dtype=torch.float32)
    total_train_step=0
    total_test_step=0
    epoch1_test_task_support = 10
    start_time=time.time()
    for i in range(epoch1_test_task_support):
        k=0
        Test_target_support = Test_target_support.to(device)
        Test_input_support = Test_input_support.to(device)
        model_fore_test_task_support.train()
        Test_outputs_support=model_fore_test_task_support(Test_input_support)
        loss1 = penalty(Test_outputs_support, Test_target_support)
        loss2=loss_fn_1(Test_outputs_support,Test_target_support)
        loss_en=k*loss1+loss2
        optimizer_fore_test_task_support.zero_grad()
        loss_en.backward()
        optimizer_fore_test_task_support.step()
        if (i + 1) % 1 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print(
                "[Epoch %d/%d][loss_mse: %f]"
                % (i, epoch1_test_task_support,  loss2.item())
            )
            experiment_name=i_class
            writer2.add_scalar(f"loss_mse_test_task_support{experiment_name}", loss2.item(), epoch1_pre+epoch_train_task*(epoch1_train_task_support+epoch1_train_task_query)+i)
    Test_outputs_support_list.append(Test_outputs_support)
    model_fore_test_task_support.eval()
    writer1.close()
    writer2.close()
    torch.save(model_fore_test_task_support.state_dict(), "./model_fore_test_task_support_%d.pth"%(i_class))
test_outputs_query_00= np.empty([1, 6], dtype=object)
test_outputs_support_00= np.empty([1, 4], dtype=object)


## save
for i_model in range(6):
    with torch.no_grad():
        if i_model<4:
            model_fore_test_task_query.load_state_dict(torch.load("model_fore_test_task_support_%d.pth"%(i_model)))
        elif i_model==4:
            model_fore_test_task_query.load_state_dict(torch.load("model_fore_train_task_query.pth" ))
        elif i_model==5:
            model_fore_test_task_query.load_state_dict(torch.load("model_fore_pre.pth" ))
        Test_input_c = Test_input_c.to(device)
        Test_output_query=model_fore_test_task_query(Test_input_c)
        test_outputs_query=Test_output_query.to(device0)
        test_outputs_query_=np.array(test_outputs_query.reshape(-1,dem_realp))
        test_outputs_query_00[0,i_model]=test_outputs_query_
        if i_model < 4:
            test_outputs_support_list=Test_outputs_support_list[i_model].to(device0)
            test_outputs_support_ = np.array(test_outputs_support_list.reshape(-1, dem_realp))
            test_outputs_support_00[0,i_model]=test_outputs_support_
        train_outputs_pre=Train_outputs_pre.to(device0)
        train_outputs_support=Train_outputs_support.to(device0)
        train_outputs_query=Train_outputs_query.to(device0)
        train_outputs_pre_=np.array(train_outputs_pre.reshape(-1,dem_realp))
        train_outputs_support_=np.array(train_outputs_support.reshape(-1,dem_realp))
        train_outputs_query_=np.array(train_outputs_query.reshape(-1,dem_realp))
        train_target_p_=train_target_p.reshape(-1,dem_realp)
        test_target_p_=test_target_p.reshape(-1,dem_realp)

