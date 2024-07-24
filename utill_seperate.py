import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from argparse import ArgumentParser

import numpy as np
import copy
import math
import scipy.io as io
import os
# from pytorch_msssim import SSIM,ssim,ms_ssim

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnrr
from skimage.metrics import normalized_root_mse as nmsee
import scipy.io as sio

parser = ArgumentParser(description='Learnable Optimization Algorithms (LOA)')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=30, help='epoch number of end training')
parser.add_argument('--start_phase', type=int, default=15, help='phase number of start training')
parser.add_argument('--end_phase', type=int, default=15, help='phase number of end training')
parser.add_argument('--layer_num', type=int, default=15, help='phase number of LDA-Net')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=25, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for loading data')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
# parser.add_argument('--init', type=bool, default=True, help='initialization True by default')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model_u_whole_seperate', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log_u_whole_seperate', help='log directory')

args = parser.parse_args()

#%% experiement setup
start_epoch = args.start_epoch
end_epoch = args.end_epoch
start_phase = args.start_phase
end_phase = args.end_phase
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
batch_size = args.batch_size
# init = args.init


#%% define LDA net
class LDA(torch.nn.Module):
    def __init__(self, LayerNo, PhaseNo):
        super(LDA, self).__init__()
        
        # soft threshold
        self.soft_thr1 = nn.Parameter(torch.Tensor([0.01]))
        self.soft_thr2 = nn.Parameter(torch.Tensor([0.01]))
        # sparcity bactracking
        self.gamma = 1.0
        # a parameter for backtracking
        self.sigma = 15000.0
        # parameter for activation function
        self.delta = 0.01
        # set phase number
        self.PhaseNo = PhaseNo
        self.init = True
        
        
        self.alphas1 = nn.Parameter(0.5 * torch.ones(LayerNo))
        self.alphas2 = nn.Parameter(0.5 * torch.ones(LayerNo))
        self.betas1 = nn.Parameter(0.1 * torch.ones(LayerNo))
        self.betas2 = nn.Parameter(0.1 * torch.ones(LayerNo))
        
        # self.alphas1_bar = nn.Parameter(0.1 * torch.ones(LayerNo))
        # self.alphas2_bar = nn.Parameter(0.1 * torch.ones(LayerNo))
        
        # size: out channels  x in channels x filter size x filter size
        # every block shares weights
        self.conv1r = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2r = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3r = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv4r = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        
        self.conv1i = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2i = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3i = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv4i = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        
        self.conv1r_ = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2r_ = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3r_ = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv4r_ = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        
        self.conv1i_ = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2i_ = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3i_ = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv4i_ = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        
        mask = sio.loadmat('dataset_202/mask/mask_202.mat')
        m = np.expand_dims(mask['mask_202'].astype(np.float32), axis=0)
        
        self.m = torch.tensor(m).cuda()
        
    def set_PhaseNo(self, PhaseNo):
        # used when adding more phases
        self.PhaseNo = PhaseNo
        
    # def set_init(self, init):
    #     self.init = init
        
    def activation(self, x):
        """ activation function from eq. (33) in paper """
        
        # index for x < -delta and x > delta
        index = torch.sign(F.relu(torch.abs(x)-self.delta))
        output = index * F.relu(x)
        # add parts when -delta <= x <= delta
        output += (1-index) * (1/(4*self.delta) * torch.square(x) + 1/2 * x + self.delta/4)
        return output
    
    def activation_der(self, x):
        """ derivative of activation function from eq. (33) in paper """
        
        # index for x < -delta and x > delta
        index = torch.sign(F.relu(torch.abs(x)-self.delta))
        output = index * torch.sign(F.relu(x))
        # add parts when -delta <= x <= delta
        output += (1-index) * (1/(2 * self.delta) * x + 1/2)
        return output
    

    def grad_r_x1(self, x1,gamma):
        # """ implementation of eq. (10) in paper  """
     
        
        # first obtain forward passs to get features g_i, i = 1, 2, ..., n_c
        # This is the feature extraction map, we can change it to other networks
        # x_input: n x 1 x 33 x 33
        
        conv1 = torch.complex(self.conv1r, self.conv1i)
        conv2 = torch.complex(self.conv2r, self.conv2i)
        conv3 = torch.complex(self.conv3r, self.conv3i)
        conv4 = torch.complex(self.conv4r, self.conv4i)
        
        x1_input = x1.view(-1, 1, 160, 180)
        
        # soft_thr = self.soft_thr * self.gamma
        soft_thr = self.soft_thr1 * gamma
        # shape from input to output: batch size x height x width x n channels
        w1= F.conv2d(x1_input, conv1, padding = 1)
        
        w1_real = torch.real(w1)
        w1_imag = torch.imag(w1)
        w1_real = self.activation(w1_real)
        w1_imag = self.activation(w1_imag)
        w1 = torch.complex(w1_real,w1_imag)
    
        w2= F.conv2d(w1, conv2, padding = 1)
        
        w2_real = torch.real(w2)
        w2_imag = torch.imag(w2)
        w2_real = self.activation(w2_real)
        w2_imag = self.activation(w2_imag)
        w2 = torch.complex(w2_real,w2_imag)
        
        w3= F.conv2d(w2, conv3, padding = 1)
        
        w3_real = torch.real(w3)
        w3_imag = torch.imag(w3)
        w3_real = self.activation(w3_real)
        w3_imag = self.activation(w3_imag)
        w3 = torch.complex(w3_real,w3_imag)
        
        w4= F.conv2d(w3, conv4, padding = 1)
        
        #n_channel = w4.shape[1]
        
        # compute norm over channel and compute g_factor
        # norm_g = torch.norm(w4, dim = 1)
        norm_g = torch.linalg.norm(w4, dim = 1,ord=2)
    
        I1 = torch.sign(F.relu(norm_g - soft_thr))
       
        ## I1 = torch.tile(I1, [1, n_channel, 1, 1])
        I0 = torch.ones_like(I1) - I1
       
        # g_factor= I1 * (norm_g - soft_thr/2) + I0 * 0.5 * torch.square(norm_g) / soft_thr
        # y = torch.sum(g_factor , axis =[-2,-1])
       
        g_factor = I1 * F.normalize(w4, dim=1) + I0 * w4 / soft_thr
        #shape of g_factor [1,32,160,180]
        
        # implementation for eq. (9): multiply grad_g to g_factor from the left
        # result derived from chain rule and that gradient of convolution is convolution transpose
        # g_r = F.conv_transpose2d(g_factor, self.conv4, padding = 1)
        # g_r *= self.activation_der(x3)
        # g_r = F.conv_transpose2d(g_r, self.conv3, padding = 1)
        # g_r *= self.activation_der(x2)
        # g_r = F.conv_transpose2d(g_r, self.conv2, padding = 1)
        # g_r *= self.activation_der(x1)
        # g_r = F.conv_transpose2d(g_r, self.conv1, padding = 1) 
       
        
        g1 = F.conv_transpose2d(g_factor, conv4, padding = 1) 
        
        w3_real= torch.real(w3)
        w3_imag= torch.imag(w3)
        w3_real= self.activation_der(w3_real)
        w3_imag= self.activation_der(w3_imag)
        w3=torch.complex(w3_real,w3_imag)
        g1 *= w3
        
        g2 = F.conv_transpose2d(g1, conv3, padding = 1) 
        
        w2_real= torch.real(w2)
        w2_imag= torch.imag(w2)
        w2_real= self.activation_der(w2_real)
        w2_imag= self.activation_der(w2_imag)
        w2=torch.complex(w2_real,w2_imag)
        g2 *= w2
        
        g3 = F.conv_transpose2d(g2, conv2, padding = 1) 
        
        w1_real= torch.real(w1)
        w1_imag= torch.imag(w1)
        w1_real= self.activation_der(w1_real)
        w1_imag= self.activation_der(w1_imag)
        w1=torch.complex(w1_real,w1_imag)
        g3 *= w1
       
        g4 = F.conv_transpose2d(g3, conv1, padding = 1) 
     
       
        # g_r *= self.activation_der(x3)
        # g_r = F.conv_transpose2d(g_r, self.conv3, padding = 1)
        # g_r *= self.activation_der(x2)
        # g_r = F.conv_transpose2d(g_r, self.conv2, padding = 1)
        # g_r *= self.activation_der(x1)
        # g_r = F.conv_transpose2d(g_r, self.conv1, padding = 1) 

        
        # first_dec= torch.complex(x1_real.grad, x1_imag.grad)
        # first_dec = first_dec.view(-1,1,160,180)
        # second_dec= torch.complex(x2_real.grad, x2_imag.grad)
        # second_dec = second_dec.view(-1,1,160,180)
        # return g_r.reshape(-1, 1089)
        
        # second_dec=second_dec.view(-1,1,160,180)
        return g4
    
    def grad_r_x2(self, x2,gamma):
        # """ implementation of eq. (9) in paper: the smoothed regularizer  """
        
        #         """ implementation of eq. (10) in paper  """
     
        
        # first obtain forward passs to get features g_i, i = 1, 2, ..., n_c
        # This is the feature extraction map, we can change it to other networks
        # x_input: n x 1 x 33 x 33
        
        conv1 = torch.complex(self.conv1r_, self.conv1i_)
        conv2 = torch.complex(self.conv2r_, self.conv2i_)
        conv3 = torch.complex(self.conv3r_, self.conv3i_)
        conv4 = torch.complex(self.conv4r_, self.conv4i_)
        
        x2_input = x2.view(-1, 1, 160, 180)
        
        # soft_thr = self.soft_thr * self.gamma
        soft_thr = self.soft_thr2 * gamma
        # shape from input to output: batch size x height x width x n channels
        w1= F.conv2d(x2_input, conv1, padding = 1)
        
        w1_real = torch.real(w1)
        w1_imag = torch.imag(w1)
        w1_real = self.activation(w1_real)
        w1_imag = self.activation(w1_imag)
        w1 = torch.complex(w1_real,w1_imag)
    
        w2= F.conv2d(w1, conv2, padding = 1)
        
        w2_real = torch.real(w2)
        w2_imag = torch.imag(w2)
        w2_real = self.activation(w2_real)
        w2_imag = self.activation(w2_imag)
        w2 = torch.complex(w2_real,w2_imag)
        
        w3= F.conv2d(w2, conv3, padding = 1)
        
        w3_real = torch.real(w3)
        w3_imag = torch.imag(w3)
        w3_real = self.activation(w3_real)
        w3_imag = self.activation(w3_imag)
        w3 = torch.complex(w3_real,w3_imag)
        
        w4= F.conv2d(w3, conv4, padding = 1)
        
        #n_channel = w4.shape[1]
        
        # compute norm over channel and compute g_factor
        # norm_g = torch.norm(w4, dim = 1)
        norm_g = torch.linalg.norm(w4, dim = 1,ord=2)
    
        I1 = torch.sign(F.relu(norm_g - soft_thr))
       
        ## I1 = torch.tile(I1, [1, n_channel, 1, 1])
        I0 = torch.ones_like(I1) - I1
       
        # g_factor= I1 * (norm_g - soft_thr/2) + I0 * 0.5 * torch.square(norm_g) / soft_thr
        # y = torch.sum(g_factor , axis =[-2,-1])
       
        g_factor = I1 * F.normalize(w4, dim=1) + I0 * w4 / soft_thr
        #shape of g_factor [1,32,160,180]
        
        # implementation for eq. (9): multiply grad_g to g_factor from the left
        # result derived from chain rule and that gradient of convolution is convolution transpose
        # g_r = F.conv_transpose2d(g_factor, self.conv4, padding = 1)
        # g_r *= self.activation_der(x3)
        # g_r = F.conv_transpose2d(g_r, self.conv3, padding = 1)
        # g_r *= self.activation_der(x2)
        # g_r = F.conv_transpose2d(g_r, self.conv2, padding = 1)
        # g_r *= self.activation_der(x1)
        # g_r = F.conv_transpose2d(g_r, self.conv1, padding = 1) 
       
        
        g1 = F.conv_transpose2d(g_factor, conv4, padding = 1) 
        
        w3_real= torch.real(w3)
        w3_imag= torch.imag(w3)
        w3_real= self.activation_der(w3_real)
        w3_imag= self.activation_der(w3_imag)
        w3=torch.complex(w3_real,w3_imag)
        g1 *= w3
        
        g2 = F.conv_transpose2d(g1, conv3, padding = 1) 
        
        w2_real= torch.real(w2)
        w2_imag= torch.imag(w2)
        w2_real= self.activation_der(w2_real)
        w2_imag= self.activation_der(w2_imag)
        w2=torch.complex(w2_real,w2_imag)
        g2 *= w2
        
        g3 = F.conv_transpose2d(g2, conv2, padding = 1) 
        
        w1_real= torch.real(w1)
        w1_imag= torch.imag(w1)
        w1_real= self.activation_der(w1_real)
        w1_imag= self.activation_der(w1_imag)
        w1=torch.complex(w1_real,w1_imag)
        g3 *= w1
       
        g4 = F.conv_transpose2d(g3, conv1, padding = 1) 
     
       
        # g_r *= self.activation_der(x3)
        # g_r = F.conv_transpose2d(g_r, self.conv3, padding = 1)
        # g_r *= self.activation_der(x2)
        # g_r = F.conv_transpose2d(g_r, self.conv2, padding = 1)
        # g_r *= self.activation_der(x1)
        # g_r = F.conv_transpose2d(g_r, self.conv1, padding = 1) 

        
        # first_dec= torch.complex(x1_real.grad, x1_imag.grad)
        # first_dec = first_dec.view(-1,1,160,180)
        # second_dec= torch.complex(x2_real.grad, x2_imag.grad)
        # second_dec = second_dec.view(-1,1,160,180)
        # return g_r.reshape(-1, 1089)
        
        # second_dec=second_dec.view(-1,1,160,180)
        return g4
        
    
    
    def R(self, x1,x2,gamma):
        """ implementation of eq. (9) in paper: the smoothed regularizer  """
        
        # first obtain forward passs to get features g_i, i = 1, 2, ..., n_c
        # x_input: n x 1 x 33 x 33
        # x_input = x.view(-1, 1, 160, 180)
        # soft_thr = self.soft_thr * self.gamma
        
        # shape from input to output: batch size x height x width x n channels
        # x1 = F.conv2d(x_input, self.conv1, padding = 1)                 # (batch,  1, 33, 33) -> (batch, 32, 33, 33)
        # x2 = F.conv2d(self.activation(x1), self.conv2, padding = 1)     # (batch, 32, 33, 33) -> (batch, 32, 33, 33)
        # x3 = F.conv2d(self.activation(x2), self.conv3, padding = 1)     # (batch, 32, 33, 33) -> (batch, 32, 33, 33)
        # g = F.conv2d(self.activation(x3), self.conv4, padding = 1)      # (batch, 32, 33, 33) -> (batch, 32, 33, 33)
        conv1 = torch.complex(self.conv1r, self.conv1i)
        conv2 = torch.complex(self.conv2r, self.conv2i)
        conv3 = torch.complex(self.conv3r, self.conv3i)
        conv4 = torch.complex(self.conv4r, self.conv4i)
        
        x1_input = x1.view(-1, 1, 160, 180)
        x2_input = x2.view(-1, 1, 160, 180)
        
        x = torch.cat((x1_input,x2_input),1)
        
        # soft_thr = self.soft_thr * self.gamma
        soft_thr = self.soft_thr * gamma
        # shape from input to output: batch size x height x width x n channels
        w1= F.conv2d(x, conv1, padding = 1)
        
        w1_real = torch.real(w1)
        w1_imag = torch.imag(w1)
        w1_real = self.activation(w1_real)
        w1_imag = self.activation(w1_imag)
        w1 = torch.complex(w1_real,w1_imag)
        
        w2= F.conv2d(w1, conv2, padding = 1)
       
        w2_real = torch.real(w2)
        w2_imag = torch.imag(w2)
        w2_real = self.activation(w2_real)
        w2_imag = self.activation(w2_imag)
        w2 = torch.complex(w2_real,w2_imag)
        
        w3= F.conv2d(w2, conv3, padding = 1)
        
        w3_real = torch.real(w3)
        w3_imag = torch.imag(w3)
        w3_real = self.activation(w3_real)
        w3_imag = self.activation(w3_imag)
        w3 = torch.complex(w3_real,w3_imag)
        
        w4= F.conv2d(w3, conv4, padding = 1)
        
        #n_channel = w4.shape[1]
        
        # compute norm over channel and compute g_factor
        # norm_g = torch.norm(w4, dim = 1)
        norm_g = torch.linalg.norm(w4, dim = 1,ord=2)
        
        I1 = torch.sign(F.relu(norm_g - soft_thr))
   
        # I1 = torch.tile(I1, [1, n_channel, 1, 1])
        I0 = 1 - I1
        
        
        r= I1 * (norm_g - soft_thr/2) + I0 * 0.5 * torch.square(norm_g) / soft_thr

        r = r.reshape(-1, 28800)
       
        r = torch.sum(r, -1)
        
        return r
    
    
    
    def data_fidelity(x):
        pfx = mriForwardOp(x1, m)    
        s = pfx - kspace1
        return 0.5*l2_norm_square(s)
        
    def l2_norm_square(self,x):#batch_size is 1
        x = x.reshape(-1, 28800)
        x = torch.norm( x, p=2, dim = 1)
        x = torch.square(x)
        # x = tf.reduce_sum(x , axis =[-2,-1]) #shape=(batch_size,), dtype=float32)
        return x   
        
    def l2_norm(self,x):#batch_size is 1
        x = x.reshape(-1, 28800)
        x = torch.norm( x, p=2, dim = 1)
        return x   
    
        
    def phi(self, x1, x2, k1, k2,gamma):
        """ The implementation for the loss function """
        # x is the reconstruction result
        # y is the ground truth
        
        r = self.R(x1,x2,gamma)
        f1 =  0.5*self.l2_norm_square(mriForwardOp(x1, self.m)- k1) 
        f2 = 0.5*self.l2_norm_square(mriForwardOp(x2, self.m)- k2)
    
        # f1 = 1/2 * torch.sum(torch.square(x @ torch.transpose(Phi, 0, 1) - y), 
        #                     dim = 1, keepdim=True)
        # f2 = 1/2 * torch.sum(torch.square(x @ torch.transpose(Phi, 0, 1) - y), 
        #                     dim = 1, keepdim=True)
        
        return f1 + f2 + r
    
    def phase(self, input1,input2, phase,kspace1,kspace2):
        
        a1= 1E+10
        c1= 1E+10
        """
        input1,input2 is the reconstruction output from last phase
        y is Phi True_x, the sampled ground truth
        
        """
        # alpha1 = torch.abs(self.alphas1[phase])
        # tau1 = torch.abs(self.betas1[phase])
        # alpha2 = torch.abs(self.alphas2[phase])
        # tau2 = torch.abs(self.betas2[phase])
        
        # alpha1_bar = torch.abs(self.alphasbar1[phase])
        # alpha2_bar = torch.abs(self.alphasbar2[phase])
      
        alpha1 = torch.abs(self.alphas1[phase])
        tau1 = torch.abs(self.betas1[phase])
        alpha2 = torch.abs(self.alphas2[phase])
        tau2 = torch.abs(self.betas2[phase])
        
        # alpha1_bar = torch.abs(self.alphas1_bar[phase])
        # alpha2_bar = torch.abs(self.alphas2_bar[phase])
       
        # self.alpha1_bar = torch.abs(self.alphas1_bar)
        # self.alpha2_bar = torch.abs(self.alphas2_bar)
        
        
        # Implementation of eq. 2/7 (ISTANet paper) Immediate reconstruction
        # here we obtain z (in LDA paper from eq. 12)
        
        #v
        # first_dec1,second_dec1=self.grad_r( input1, input2)
        # z1 = input1 - alpha1 * (ATAx1 - ATf1)-alpha1*first_dec1

        # ATAx2 = mriAdjointOp(mriForwardOp(input2, self.m), self.m)# FTPT(PFx1)
        # ATf2  = mriAdjointOp(kspace2, self.m)#FTPT(f1)
        
        # first_dec2,second_dec2= self.grad_r( z1, input2)
        # z2 = input2 - alpha2* (ATAx2 - ATf2)- alpha2*second_dec2
        
        # x1=z1
        # x2=z2

         #seperate   
        ATf1 = mriAdjointOp(kspace1, self.m)#FTPT(f1)
        ATAx1 = mriAdjointOp(mriForwardOp(input1, self.m), self.m)# FTPT(PFx1)
        
        z1 = input1 - alpha1 * (ATAx1 - ATf1)
    
        first_dec1=self.grad_r_x1(z1,self.gamma)
        u1 = z1 - tau1* first_dec1
        
        ATAx2 = mriAdjointOp(mriForwardOp(input2, self.m), self.m)# FTPT(PFx1)
        ATf2  = mriAdjointOp(kspace2, self.m)#FTPT(f1)
        
        z2 = input2 - alpha2* (ATAx2 - ATf2)
        second_dec2= self.grad_r_x2(z2,self.gamma)
        u2 = z2 - tau2*second_dec2
        
        x1=u1
        x2=u2
       

        # ATf1 = mriAdjointOp(kspace1, self.m)#FTPT(f1)
        # ATAx1 = mriAdjointOp(mriForwardOp(input1, self.m), self.m)# FTPT(PFx1)
        # z1 = input1 - alpha1 * (ATAx1 - ATf1)
    
        # first_dec1,second_dec1=self.grad_r( z1, input2,self.gamma)
        
        # u1 = z1 - tau1* first_dec1
        
        
        # ATAx2 = mriAdjointOp(mriForwardOp(input2, self.m), self.m)# FTPT(PFx1)
        # ATf2  = mriAdjointOp(kspace2, self.m)#FTPT(f1)
        
        # z2 = input2 - alpha2* (ATAx2 - ATf2)
        # first_dec2,second_dec2= self.grad_r( u1, z2,self.gamma)
        # u2 = z2 - tau2*second_dec2
        # #u2.size() == (1,1,160,180)
        
        # x1=u1
        # x2=u2
        # 10 conditions satisfy or not:
        # u1_x  = torch.mean(self.l2_norm_square(u1 - input1)+ self.l2_norm_square(u2 - input2))
        # phi_u1_x1 = torch.mean(self.phi(u1,u2,kspace1,kspace2,self.gamma) - self.phi(input1,input2,kspace1,kspace2,self.gamma))
     
        # # 11 conditions satisfy or not:
        # first_dec3,second_dec3=self.grad_r( input1, input2,self.gamma) 
        # gra_phi1=(ATAx1-ATf1)+first_dec3
        # gra_phi2=(ATAx2-ATf2)+second_dec3
        
        # norm_grad_deccc= torch.mean(torch.sqrt(self.l2_norm_square(gra_phi1)+self.l2_norm_square(gra_phi2)))
        
        # u1_xx = torch.mean(self.l2_norm(u1 - input1) + self.l2_norm(u2 - input2))
        
        # # print("kk")
        # # print(phi_u1_x1)
        # # print(u1_x)
    
        
        # if torch.le(norm_grad_deccc, a1 * u1_xx) and torch.le(phi_u1_x1, (-1/a1) * u1_x):
        #     x1=u1
        #     x2=u2
            
         
        # else:
        #     def calv1(alpha1_bar, alpha2_bar):
        #         v1 = input1- alpha1_bar*(ATAx1 - ATf1 + first_dec3)
        #         first_dec5,second_dec5=self.grad_r( v1 , input2, self.gamma)
        #         v2 = input2 -alpha2_bar*( ATAx2 -ATf2 + second_dec5)
        #         v1_x  = torch.mean(self.l2_norm_square(v1 - input1)+ self.l2_norm_square(v2 - input2))
        #         phi_v1_x1 = torch.mean(self.phi(v1,v2,kspace1,kspace2,self.gamma) - self.phi(input1,input2,kspace1,kspace2,self.gamma))
              
        #         return [v1,v2,v1_x,phi_v1_x1,alpha1_bar,alpha2_bar]
          
        #     [v1,v2,v1_x,phi_v1_x1,self.alpha1_bar,self.alpha2_bar]=calv1(alpha1_bar,alpha2_bar)
            
        #     while phi_v1_x1 > (-1/c1) * v1_x:
        #         alpha1_bar = 0.9 * alpha1_bar
        #         alpha2_bar = 0.9 * alpha2_bar
        #         [v1,v2,v1_x,phi_v1_x1,alpha1_bar,alpha2_bar]=calv1(alpha1_bar,alpha2_bar)
                
                
        #     x1=v1
        #     x2=v2
      
       
        # first_dec6,second_dec6=self.grad_r( x1, x2,self.gamma)
       
        # # norm1=torch.mean(self.l2_norm_square_t(first_dec6))
        # # norm2=torch.mean(self.l2_norm_square_t(second_dec6))
        # ATAu11 = mriAdjointOp(mriForwardOp(x1, self.m), self.m)# FTPT(PFx1)
        # ATAu22 = mriAdjointOp(mriForwardOp(x2, self.m), self.m)


        # gra_phi11=(ATAu11-ATf1)+first_dec6
        # gra_phi22=(ATAu22-ATf2)+second_dec6
    
        # norm_grad_decccc= torch.mean(torch.sqrt(self.l2_norm_square(gra_phi11)+self.l2_norm_square(gra_phi22)))
        
        # sig_gam_eps = self.sigma * self.gamma * self.soft_thr 
       
        # if torch.mean(norm_grad_decccc) < sig_gam_eps:
        #     self.gamma *= 0.9
        # else:
        #     self.gamma *= 1
  
        
        # self.gamma *= 0.9 if (torch.mean(norm_grad_decccc) < sig_gam_eps) else 1.0
       
        return x1,x2, [alpha1, alpha2, tau1, tau2, self.soft_thr1,self.soft_thr1, self.sigma]
        
        
    def forward(self, n,k1,k2,kx1,kx2):
        layers1 = []        
        layers2=[]
        
        gamma=[]
        out=[]
        
        kx1=torch.view_as_complex(kx1)
        kx2=torch.view_as_complex(kx2)
        k1=torch.view_as_complex(k1)
        k2=torch.view_as_complex(k2)
        
        layers1.append(kx1)
        layers2.append(kx2)
        
        gamma.append(0.01)
        
        for phase in range(self.PhaseNo):
            [k11,k22,o]=self.phase(layers1[-1],layers2[-1],phase,k1,k2)
            
            layers1.append(k11)
            layers2.append(k22)
            out.append(o)
            
        return layers1[-1], layers2[-1], out
#%% helper functions
def mriForwardOp(img, sampling_mask):
    # centered Fourier transform
    Fu = torch.fft.fftn(img)
    # apply sampling mask
    kspace = torch.complex(torch.real(Fu) * sampling_mask, torch.imag(Fu) * sampling_mask)
    return kspace

def mriAdjointOp(f, sampling_mask):
    # apply mask and perform inverse centered Fourier transform
    
    Finv = torch.fft.ifftn(torch.complex(torch.real(f) * sampling_mask, torch.imag(f) * sampling_mask))
    return Finv 

def MSE(y_true, y_pred):
    return torch.mean((torch.abs( y_true) - torch.abs(y_pred)) ** 2)
    
def psnr(img1, img2):
    mse = torch.mean((abs(img1) - abs(img2)) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

#order matters    
def nmse(img_t, img_s ):
    return torch.sum((abs(img_s) - abs(img_t)) ** 2) / torch.sum(abs(img_t)**2)

#RMSE
def rmse(img_t, img_s ):
    return torch.sqrt(torch.sum((abs(img_s) - abs(img_t)) ** 2) / (160*180) )
    
    
def saveAsMat(img, filename, matlab_id, mat_dict=None):
    """ Save mat files with ndim in [2,3,4]
        Args:
            img: image to be saved
            file_path: base directory
            matlab_id: identifer of variable
            mat_dict: additional variables to be saved
    """
    assert img.ndim in [2, 3, 4]

    img_arg = img.copy()
    if img.ndim == 3:
        img_arg = np.transpose(img_arg, (0,1,2))
    elif img.ndim == 4:
        img_arg = np.transpose(img_arg, (2, 3, 0, 1))

    if mat_dict == None:
        mat_dict = {matlab_id: img_arg}
    else:
        mat_dict[matlab_id] = img_arg

    dirname = os.path.dirname(filename) or '.'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    io.savemat(filename, mat_dict)