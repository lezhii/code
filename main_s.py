import torch
from torch.utils.data import DataLoader
import torch.nn as nn


import scipy.io as io
import numpy as np
import os
import platform
import scipy.io as sio

from argparse import ArgumentParser
from utill_seperate import *
from data_500 import *
# parser = ArgumentParser(description='Learnable Optimization Algorithms (LOA)')

# parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
# parser.add_argument('--end_epoch', type=int, default=50, help='epoch number of end training')
# parser.add_argument('--start_phase', type=int, default=3, help='phase number of start training')
# parser.add_argument('--end_phase', type=int, default=15, help='phase number of end training')
# parser.add_argument('--layer_num', type=int, default=15, help='phase number of LDA-Net')
# parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
# parser.add_argument('--group_num', type=int, default=1, help='group number for training')
# parser.add_argument('--cs_ratio', type=int, default=25, help='from {1, 4, 10, 25, 40, 50}')
# parser.add_argument('--batch_size', type=int, default=64, help='batch size for loading data')
# parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
# parser.add_argument('--init', type=bool, default=True, help='initialization True by default')

# parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
# parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
# parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
# parser.add_argument('--log_dir', type=str, default='log', help='log directory')

# args = parser.parse_args()

# #%% experiement setup
# start_epoch = args.start_epoch
# end_epoch = args.end_epoch
# start_phase = args.start_phase
# end_phase = args.end_phase
# learning_rate = args.learning_rate
# layer_num = args.layer_num
# group_num = args.group_num
# cs_ratio = args.cs_ratio
# gpu_list = args.gpu_list
# batch_size = args.batch_size
# init = args.init



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Load Data

print('Load Data...')


mask = sio.loadmat('dataset_202/mask/mask_202.mat')
m = np.expand_dims(mask['mask_202'].astype(np.float32), axis=0)

#___________________________________________________________________________________Train

print("...................................")
print("Phase Number is %d" % (start_phase ))
print("...................................\n")
print('Load Data...')


#%% training dataloader
if (platform.system() == 'Windows'):
    rand_loader = DataLoader(dataset = RandomDataset(), 
                             batch_size=batch_size, num_workers=0,shuffle=False)
else:
    rand_loader = DataLoader(dataset=RandomDataset(), 
                             batch_size=batch_size, num_workers=1,shuffle=False)#?


#%% testing dataloader
if (platform.system() == 'Windows'):
    randtest_loader = DataLoader(dataset = TestingDataset(), 
                             batch_size=batch_size, num_workers=0,shuffle=False)
else:
    randtest_loader = DataLoader(dataset=TestingDataset(), 
                             batch_size=batch_size, num_workers=1,shuffle=False)#?                             

#%% initialize model

model = LDA(layer_num, start_phase)
model = nn.DataParallel(model)
model.to(device)

# print_flag = 1   # print parameter number

# if print_flag:
#     num_count = 0
#     for para in model.parameters():
#         num_count += 1
#         print('Layer %d' % num_count)
#         print(para.size())

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/LDA_layer_%d_group_%d_ratio_lr_%.4f" % \
    (args.model_dir, layer_num, group_num,  learning_rate)

log_file_name = "./%s/LDA_layer_%d_group_%d_ratio_lr_%.4f.txt" % \
    (args.log_dir, layer_num, group_num,  learning_rate)
    
log_file_name1 = "./%s/LDA_layer_%d_group_%d_ratio_lr_%.4f.txt" % \
    (args.log_dir, layer_num, group_num,  learning_rate)
# if not start from beginning load pretrained models

# start from checkpoint
# state_dict = torch.load('%s/net_params_epoch%d_phase%d.pkl' % \
#                                      (model_dir, 30, 13), 
#                                      map_location=device)
# state_dict['module.alphas1_bar'] = 0.1*torch.ones(15)
# state_dict['module.alphas2_bar'] = 0.1*torch.ones(15)
# model.load_state_dict(state_dict)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

#%% training
# for PhaseNo in range(start_phase, end_phase+1, 2):
#     # add new phases
#     model.module.set_PhaseNo(PhaseNo)
#     if PhaseNo == 3:
#         end_epoch = 100
#     else:
#         end_epoch = args.end_epoch
#     for epoch_i in range(start_epoch+1, end_epoch+1):
#         progress = 0
#         for kx1,kx2,t1,t2,k1,k2 in rand_loader:
            
#             progress += 1
            
#             kspace1 = k1
            
#             kspace1 = kspace1.to(device)
        
#             kspace2 = k2
#             kspace2 = kspace2.to(device)
            
#             kx1 = kx1.to(device)
        
#             kx2 = kx2.to(device)
            
#             k11,k22,out = model(PhaseNo,kspace1,kspace2,kx1,kx2)
           
#             t1 = t1.to(device)
#             t2 = t2.to(device)
            
#             cost_tr_rec1 = MSE(t1, k11)
#             cost_tr_rec2 = MSE(t2, k22)
            
#             s1=torch.reshape(torch.abs(k11),(160,180))
#             t1=torch.reshape(t1,(160,180))
#             t1= t1.cpu().detach().numpy()
#             s1= s1.cpu().detach().numpy()
            
#             s2=torch.reshape(torch.abs(k22),(160,180))
#             t2=torch.reshape(t2,(160,180))
#             t2= t2.cpu().detach().numpy()
#             s2= s2.cpu().detach().numpy()
            
        
#             ssim1= ssim(s1,t1, data_range=1)
#             ssim3= ssim(s2,t2, data_range=1)
            
#             # compute and print loss

#             loss_all = cost_tr_rec1 + cost_tr_rec2 + 0.1*(1-ssim1) + 0.1*(1-ssim3)
#             # loss_all = cost_tr_rec1 + cost_tr_rec2 
#             # Zero gradients, perform a backward pass, and update the weights.
#             optimizer.zero_grad()
#             loss_all.backward()
#             optimizer.step()
            
#             nrtrain = 2600
            
#             if progress % 20 == 0:
#                 output_data = "[Phase %02d] [Epoch %02d/%02d] Total Loss: %.4f" % \
#                     (PhaseNo, epoch_i, end_epoch, loss_all.item()) \
#                     + "\t progress: %02f" % (progress * batch_size/ nrtrain * 100) + "%\n"
                
#                 print(out)
#                 print(output_data)
     
            
#         output_file = open(log_file_name, 'a')
#         output_file.write(output_data)
#         output_file.close()
        
#         # output_file = open(log_file_name1, 'a')
#         # output_file.write(out)
#         # output_file.close()
        
#         if epoch_i % 1 == 0:
#             # save the parameters
#             torch.save(model.state_dict(), "./%s/net_params_epoch%d_phase%d.pkl" % \
#                       (model_dir, epoch_i, PhaseNo))
                
 #%% testing
print("Load Testing Data")
# model.load_state_dict(torch.load('%s/net_params_epoch%d_phase%d.pkl' % \
#                                      (model_dir, 100, start_phase), 
#                                      map_location=device))

model.load_state_dict(torch.load('%s/net_params_epoch%d_phase%d.pkl' % \
                                     (model_dir, 30, 15)))

# start from checkpoint
# state_dict = torch.load('%s/net_params_epoch%d_phase%d.pkl' % \
#                                      (model_dir, 30, 15), 
#                                      map_location=device)
# state_dict['module.alphas1_bar'] = 0.1*torch.ones(15)
# state_dict['module.alphas2_bar'] = 0.1*torch.ones(15)
# model.load_state_dict(state_dict)


model.to(device)

TIME_ALL  = []

PSNR1_All = []
SSIM1_All = []
NMSE1_All = []
RMSE1_All = []

PSNR2_All = []
SSIM2_All = []
NMSE2_All = []
RMSE2_All = []


init_x1 = []
init_x2 = []

progress = 0

with torch.no_grad():
    for kx1,kx2,t1,t2,k1,k2 in randtest_loader:
        
        progress += 1
        kspace1 = k1
        
        kspace1 = kspace1.to(device)
    
        kspace2 = k2
        kspace2 = kspace2.to(device)
        
        kx1 = kx1.to(device)
    
        kx2 = kx2.to(device)
        
        k11,k22,out = model(15,kspace1,kspace2,kx1,kx2)
       
        t1 = t1.to(device)
        t2 = t2.to(device)
        
        #reshape image size
        t1=torch.reshape(t1,(160,180))
        t2=torch.reshape(t2,(160,180))
        
        kx1=torch.reshape(torch.abs(kx1),(160,180))
        kx2=torch.reshape(torch.abs(kx2),(160,180))
        
        s1=torch.reshape(torch.abs(k11),(160,180))
        s2=torch.reshape(torch.abs(k22),(160,180))
        
        #NMSE & RMSE
        NMSE1=nmse(t1,s1)
        NMSE2=nmse(t2,s2)
        
        RMSE1=rmse(t1,s1)
        RMSE2=rmse(t2,s2)
        
        # NMSE1=nmse(t1,kx1)
        # NMSE2=nmse(t2,kx2)
        
        # RMSE1=rmse(t1,kx1)
        # RMSE2=rmse(t2,kx2)
        #transfer tensor to numpy
        t1= t1.cpu().detach().numpy()
        t2= t2.cpu().detach().numpy()
        
        s1= s1.cpu().detach().numpy()
        s2= s2.cpu().detach().numpy()
        
        kx1= kx1.cpu().detach().numpy()
        kx2= kx2.cpu().detach().numpy()
        
        #calculate metrics
        
        SSIM1= ssim(s1, t1, data_range=1)
        SSIM2= ssim(s2, t2, data_range=1)
        
        PSNR1=psnrr(t1, s1)
        PSNR2=psnrr(t2, s2)
        
        # SSIM1= ssim(kx1, t1, data_range=1)
        # SSIM2= ssim(kx2, t2, data_range=1)
        
        # PSNR1=psnrr(t1, kx1)
        # PSNR2=psnrr(t2, kx2)
        
        #metrics to numpy
        NMSE1= NMSE1.cpu().detach().numpy()
        NMSE2= NMSE2.cpu().detach().numpy()
        RMSE1= RMSE1.cpu().detach().numpy()
        RMSE2= RMSE2.cpu().detach().numpy()
        
        nrtrain = 1300
        
        result1 = "%s, PSNRt1:%.4f, SSIMt1:%.4f, NMSEt1:%.4f, RMSEt1:%.4f. \n" % (progress, PSNR1, SSIM1, NMSE1, RMSE1)
        result2 = "%s, PSNRt2:%.4f, SSIMt2:%.4f, NMSEt2:%.4f, RMSEt2:%.4f. \n" % (progress, PSNR2, SSIM2, NMSE2, RMSE2)
        
        
        print(result1)
        print(result2) 
        
        
        init_x1.append(s1)
        init_x2.append(s2)
        
        PSNR1_All.append(PSNR1)
        SSIM1_All.append(SSIM1)
        NMSE1_All.append(NMSE1)
        RMSE1_All.append(RMSE1)
    
        PSNR2_All.append(PSNR2)
        SSIM2_All.append(SSIM2)
        NMSE2_All.append(NMSE2)
        RMSE2_All.append(RMSE2)
        # output_data = "[Phase %02d] [Epoch %02d/%02d] Total Loss: %.4f" % \
        #     (PhaseNo, epoch_i, end_epoch, loss_all.item()) \
        #     + "\t progress: %02f" % (progress * batch_size/ nrtrain * 100) + "%\n"
        # print(output_data)
    
    

    output_data1 = " T1_Avg REC PSNR is %.4f dB std %.4f, SSIM is %.4f std %.4f, NMSE is %.4f std %.4f, RMSE is %.4f std %.4f\n" % (np.mean(PSNR1_All), np.std(PSNR1_All), np.mean(SSIM1_All), np.std(SSIM1_All), np.mean(NMSE1_All), np.std(NMSE1_All),np.mean(RMSE1_All), np.std(RMSE1_All))
    print(output_data1)
    output_data2 = " T2_Avg REC PSNR is %.4f dB std %.4f, SSIM is %.4f std %.4f, NMSE is %.4f std %.4f, RMSE is %.4f std %.4f\n" % (np.mean(PSNR2_All), np.std(PSNR2_All), np.mean(SSIM2_All), np.std(SSIM2_All), np.mean(NMSE2_All), np.std(NMSE2_All),np.mean(RMSE2_All), np.std(RMSE2_All))
    print(output_data2)
    
    
    output_file = open(log_file_name, 'a')
    output_file.write(output_data1)
    output_file.write(output_data2)
    output_file.close()
    
    X1_INIT = np.stack((init_x1), axis=0)
    X2_INIT = np.stack((init_x2),axis=0)
    
    # saveAsMat(X1_INIT, "T1_final_202.mat", 'T1_final_202',  mat_dict=None)
    # saveAsMat(X2_INIT, "T2_final_202.mat", 'T2_final_202',  mat_dict=None)