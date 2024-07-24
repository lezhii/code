from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import torch
##dataset
class RandomDataset(Dataset):
    def __init__(self):
        
        data_dir='dataset_202'
        
        x1_0='T1_0train.mat'
        x2_0='T2_0train.mat'
        
        mask = sio.loadmat('dataset_202/mask/mask_202.mat')
        m = np.expand_dims(mask['mask_202'].astype(np.float32), axis=0)
        X1_0 = sio.loadmat(x1_0)
        X2_0 = sio.loadmat(x2_0)
        
        #x_0
        df1 = X1_0['T1_0train']
        df2 = X2_0['T2_0train']
        
        T1 = './%s/train/T1.mat'%(data_dir)
        T1_ = './%s/train/T1_.mat'%(data_dir)
                
        T2 = './%s/train/T2.mat'%(data_dir)
        T2_ = './%s/train/T2_.mat'%(data_dir)
                
                
        gt_T1 = sio.loadmat(T1)['T1'].astype(np.float32)
        gt_T1_ = sio.loadmat(T1_)['T1_'].astype(np.float32)
        # gt_train_T1 = np.vstack((gt_T1, gt_T1_))
        gt_train_T1 = gt_T1
                
        gt_T2 = sio.loadmat(T2)['T2'].astype(np.float32)
        gt_T2_ = sio.loadmat(T2_)['T2_'].astype(np.float32)
        # gt_train_T2 = np.vstack((gt_T2, gt_T2_))
        gt_train_T2 = gt_T2
                
        ntrain = df1.shape[0]
        
        
        # T1 = './%s/validation/T1.mat'%(data_dir)
        # T1_ = './%s/validation/T1_.mat'%(data_dir)
                
        # T2 = './%s/validation/T2.mat'%(data_dir)
        # T2_ = './%s/validation/T2_.mat'%(data_dir)
                
        # gt_v_T1 = sio.loadmat(T1)['T1'].astype(np.float32)
        # gt_v_T1_ = sio.loadmat(T1_)['T1_'].astype(np.float32)
        # gt_val_T1 = np.vstack((gt_v_T1, gt_v_T1_))
                
        # gt_v_T2 = sio.loadmat(T2)['T2'].astype(np.float32)
        # gt_v_T2_ = sio.loadmat(T2_)['T2_'].astype(np.float32)
        # gt_val_T2 = np.vstack((gt_v_T2, gt_v_T2_))
                
        # gt_train_T11 = np.vstack((gt_T1, gt_T1_,gt_v_T1, gt_v_T1_))
        # gt_train_T22 = np.vstack((gt_T2, gt_T2_,gt_v_T2, gt_v_T2_))
        

        #kspace      
        K1 = './%s/train/K1.mat'%(data_dir)
        K1_ = './%s/train/K1_.mat'%(data_dir)
                
        K2 = './%s/train/K2.mat'%(data_dir)
        K2_ = './%s/train/K2_.mat'%(data_dir)
                
                # .astype(np.complex64)
       
        gt_K1 = sio.loadmat(K1)['K1']
        gt_K1_ = sio.loadmat(K1_)['K1_']
        # gt_train_K1 = np.vstack((gt_K1, gt_K1_))
        gt_train_K1 =gt_K1
              
        gt_K2 = sio.loadmat(K2)['K2']
        gt_K2_ = sio.loadmat(K2_)['K2_'].astype(np.complex64)
        # gt_train_K2 = np.vstack((gt_K2, gt_K2_))
        gt_train_K2 = gt_K2
        
        # K1 = './%s/validation/K1.mat'%(data_dir)
        # K1_ = './%s/validation/K1_.mat'%(data_dir)
                
        # K2 = './%s/validation/K2.mat'%(data_dir)
        # K2_ = './%s/validation/K2_.mat'%(data_dir)
                
        # gt_v_K1 = sio.loadmat(K1)['K1'].astype(np.complex64)
        # gt_v_K1_ = sio.loadmat(K1_)['K1_'].astype(np.complex64)
        # gt_val_K1 = np.vstack((gt_v_K1, gt_v_K1_))
                
        # gt_v_K2 = sio.loadmat(K2)['K2'].astype(np.complex64)
        # gt_v_K2_ = sio.loadmat(K2_)['K2_'].astype(np.complex64)
        # gt_val_K2 = np.vstack((gt_v_K2, gt_v_K2_))
                
        # gt_train_K11 = np.vstack((gt_K1, gt_K1_,gt_v_K1, gt_v_K1_))
        # gt_train_K22 = np.vstack((gt_K2, gt_K2_,gt_v_K2, gt_v_K2_))
        #data
        self.len = 500
        print('Total training data %d' % self.len)
        self.kx1=df1
        self.kx2=df2
        self.t1=gt_train_T1
        self.t2=gt_train_T2
        #self.k1=gt_train_K11
        self.k1=gt_train_K1
        self.k2=gt_train_K2
        
        
        
    def __getitem__(self, index):
        return [torch.tensor(self.kx1[index,:],dtype=torch.cfloat).unsqueeze_(0),
        torch.tensor(self.kx2[index,:],dtype=torch.cfloat).unsqueeze_(0),
        torch.tensor(self.t1[index,:],dtype=torch.float32).unsqueeze_(0),
        torch.tensor(self.t2[index,:],dtype=torch.float32).unsqueeze_(0),
        torch.tensor(self.k1[index,:],dtype=torch.cfloat).unsqueeze_(0),
        torch.tensor(self.k2[index,:],dtype=torch.cfloat).unsqueeze_(0)]
        # self.kx2[index,:],
        # self.t1[index,:],
        # self.t2[index,:],
        # self.k1[index,:],
        # self.k2[index,:]
    
    def __len__(self):
        return self.len


class TestingDataset(Dataset):
    def __init__(self):
        
        data_dir='dataset_202'
        
        x1_0='T1_0test.mat'
        x2_0='T2_0test.mat'
        
        # mask = sio.loadmat('dataset_202/mask/mask_202.mat')
        # m = np.expand_dims(mask['mask_202'].astype(np.float32), axis=0)
        X1_0 = sio.loadmat(x1_0)
        X2_0 = sio.loadmat(x2_0)
        
        #x_0
        df1 = X1_0['T1_0test']
        df2 = X2_0['T2_0test']
        #truth
        T1 = './%s/test/T1.mat'%(data_dir)

                
        T2 = './%s/test/T2.mat'%(data_dir)
                
        gt_T1 = sio.loadmat(T1)['T1'].astype(np.float32)
                
        gt_T2 = sio.loadmat(T2)['T2'].astype(np.float32)
 
        ntrain = df1.shape[0]
        print('Total testing data %d' % ntrain)
      
    
        #kspace      
        K1 = './%s/test/K1.mat'%(data_dir)
                
        K2 = './%s/test/K2.mat'%(data_dir)
                
                # .astype(np.complex64)
       
        gt_K1 = sio.loadmat(K1)['K1']
                
        gt_K2 = sio.loadmat(K2)['K2']
     
        #data
        self.len = ntrain
        self.kx1=df1
        self.kx2=df2
        self.t1=gt_T1
        self.t2=gt_T2
       
        self.k1=gt_K1
        self.k2=gt_K2
        
        
        
    def __getitem__(self, index):
        return [torch.tensor(self.kx1[index,:],dtype=torch.cfloat).unsqueeze_(0),
        torch.tensor(self.kx2[index,:],dtype=torch.cfloat).unsqueeze_(0),
        torch.tensor(self.t1[index,:],dtype=torch.float32).unsqueeze_(0),
        torch.tensor(self.t2[index,:],dtype=torch.float32).unsqueeze_(0),
        torch.tensor(self.k1[index,:],dtype=torch.cfloat).unsqueeze_(0),
        torch.tensor(self.k2[index,:],dtype=torch.cfloat).unsqueeze_(0)]
        # self.kx2[index,:],
        # self.t1[index,:],
        # self.t2[index,:],
        # self.k1[index,:],
        # self.k2[index,:]
    
    def __len__(self):
        return self.len

