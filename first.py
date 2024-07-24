import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import scipy.io as sio
import numpy as np
from scipy.io import savemat
import os
import tensorflow.contrib.slim as slim
from time import time
from PIL import Image
import math
from skimage.metrics import structural_similarity as ssim

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #0-2 1-3 2-0 3-0

batch_size = 1
global_step = tf.Variable(0, trainable=False, dtype=tf.int32)  
ckpt_model_number = 50
ckpt_batch = 300
ch = 32
EpochNum = 50
ddelta = 0.01
coeff_activation = 1.0 / (4.0 * ddelta)
data_dir = 'dataset_202'

mask = sio.loadmat('dataset_202/mask/mask_202.mat')
m = np.expand_dims(mask['mask_202'].astype(np.float32), axis=0)
kspace1 = tf.placeholder(tf.complex64, [None, 160, 180])#kspace1_rec
kspace2 = tf.placeholder(tf.complex64, [None, 160, 180])#kspace2_rec
rec_target1 = tf.placeholder(tf.float32, [None, 160, 180])
rec_target2 = tf.placeholder(tf.float32, [None, 160, 180])
syn_target = tf.placeholder(tf.float32, [None, 160, 180])

def mriForwardOp(img, sampling_mask):
    # centered Fourier transform
    Fu = tf.fft2d(img)
    # apply sampling mask
    kspace = tf.complex(tf.real(Fu) * sampling_mask, tf.imag(Fu) * sampling_mask)
    return kspace

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
    savemat(filename, mat_dict)
    
def mriAdjointOp(f, sampling_mask):
    # apply mask and perform inverse centered Fourier transform
    Finv = tf.ifft2d(tf.complex(tf.real(f) * sampling_mask, tf.imag(f) * sampling_mask))
    return Finv 

def nmse(img_t, img_s ):
    return np.sum((abs(img_s) - abs(img_t)) ** 2.) / np.sum(abs(img_t)**2)

def rmse(img_t, img_s):
    return np.linalg.norm( abs(img_t) - abs(img_s), 'fro' )/np.linalg.norm( abs(img_t), 'fro')

def MSE(y_true, y_pred):
    return tf.reduce_mean(tf.abs( y_true - y_pred) ** 2)

def MAE(y_true, y_pred):
    return tf.reduce_mean(tf.abs( y_true - y_pred))

def l1loss(y_true, y_pred):
    c=tf.reduce_sum(tf.abs( y_true- y_pred))
    return c
    
    
def psnr(img_t, img_s):
    mse = np.mean( ( abs(img_t) - abs(img_s) ) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = abs(img_t).max()
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def add_con2d_weight(w_shape, order_no):
    Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_%d' % order_no)
    return Weights

def sigma_activation(x_i): 
    return tf.where(tf.greater(tf.abs(x_i), ddelta), tf.nn.relu(x_i), coeff_activation * tf.square(x_i) + 0.5 * x_i + 0.25 * ddelta)

with tf.variable_scope('init', reuse=tf.AUTO_REUSE):

    w1_1 = add_con2d_weight([3, 3, 1, ch], 11)
    w1_1_ = add_con2d_weight([3, 3, 1, ch], 110)
    w1_2 = add_con2d_weight([3, 3, ch, ch], 21)
    w1_2_ = add_con2d_weight([3, 3, ch, ch], 210)
    w1_3 = add_con2d_weight([3, 3, ch, ch], 31)
    w1_3_ = add_con2d_weight([3, 3, ch, ch], 310)
    w1_4 = add_con2d_weight([3, 3, ch, 1], 41)
    w1_4_ = add_con2d_weight([3, 3, ch, 1], 410)
    w1_5 = add_con2d_weight([3, 3, 1, ch], 51)
    w1_5_ = add_con2d_weight([3, 3, 1, ch], 510)
    w1_6 = add_con2d_weight([3, 3, ch, ch], 61)
    w1_6_ = add_con2d_weight([3, 3, ch, ch], 610)
    w1_7 = add_con2d_weight([3, 3, ch, ch], 71)
    w1_7_ = add_con2d_weight([3, 3, ch, ch], 710)
    w1_8 = add_con2d_weight([3, 3, ch, 1], 81)
    w1_8_ = add_con2d_weight([3, 3, ch, 1], 810)
   
    
    w2_1 = add_con2d_weight([3, 3, 1, ch], 12)
    w2_1_ = add_con2d_weight([3, 3, 1, ch], 120)
    w2_2 = add_con2d_weight([3, 3, ch, ch], 22)
    w2_2_ = add_con2d_weight([3, 3, ch, ch], 220)
    w2_3 = add_con2d_weight([3, 3, ch, ch], 32)
    w2_3_ = add_con2d_weight([3, 3, ch, ch], 320)
    w2_4 = add_con2d_weight([3, 3, ch, 1], 42)
    w2_4_ = add_con2d_weight([3, 3, ch, 1], 420)  
    w2_5 = add_con2d_weight([3, 3, 1, ch], 52)
    w2_5_ = add_con2d_weight([3, 3, 1, ch], 520)
    w2_6 = add_con2d_weight([3, 3, ch, ch], 62)
    w2_6_ = add_con2d_weight([3, 3, ch, ch], 620)
    w2_7 = add_con2d_weight([3, 3, ch, ch], 72)
    w2_7_ = add_con2d_weight([3, 3, ch, ch], 720)
    w2_8 = add_con2d_weight([3, 3, ch, 1], 82)
    w2_8_ = add_con2d_weight([3, 3, ch, 1], 820)
  
    
    w3_1 = add_con2d_weight([3, 3, 1, 2*ch], 13)
    w3_1_ = add_con2d_weight([3, 3, 1, 2*ch], 130)
    w3_2 = add_con2d_weight([3, 3, 2*ch, 2*ch], 23)
    w3_2_ = add_con2d_weight([3, 3, 2*ch, 2*ch], 230)
    w3_3 = add_con2d_weight([3, 3, 2*ch, 2*ch], 33)
    w3_3_ = add_con2d_weight([3, 3, 2*ch, 2*ch], 330)
    w3_4 = add_con2d_weight([3, 3, 2*ch, 2*ch], 43)
    w3_4_ = add_con2d_weight([3, 3, 2*ch, 2*ch], 430)  
    w3_5 = add_con2d_weight([3, 3, 2*ch, 2*ch], 53)
    w3_5_ = add_con2d_weight([3, 3, 2*ch, 2*ch], 530)
    w3_6 = add_con2d_weight([3, 3, 2*ch, 2*ch], 63)
    w3_6_ = add_con2d_weight([3, 3, 2*ch, 2*ch], 630)
    w3_7 = add_con2d_weight([3, 3, 2*ch, 2*ch], 73)
    w3_7_ = add_con2d_weight([3, 3, 2*ch, 2*ch], 730)
    w3_8 = add_con2d_weight([3, 3, 2*ch, 1], 83)
    w3_8_ = add_con2d_weight([3, 3, 2*ch, 1], 830)
    
##############################################
# define our model
def x1_init(kspace1):
    
    k_real = tf.real(kspace1)
    k_imag = tf.imag(kspace1)

    k_real = tf.expand_dims(k_real, -1)
    k_imag = tf.expand_dims(k_imag, -1)#(?, 160, 180, 1)
        
    k1_real = tf.nn.relu(tf.nn.conv2d(k_real, w1_1, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(k_imag, w1_1_, strides=[1, 1, 1, 1], padding='SAME'))
    k1_imag = tf.nn.relu(tf.nn.conv2d(k_real, w1_1_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(k_imag, w1_1, strides=[1, 1, 1, 1], padding='SAME'))

    k2_real = tf.nn.relu(tf.nn.conv2d(k1_real, w1_2, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(k1_imag, w1_2_, strides=[1, 1, 1, 1], padding='SAME'))
    k2_imag = tf.nn.relu(tf.nn.conv2d(k1_real, w1_2_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(k1_imag, w1_2, strides=[1, 1, 1, 1], padding='SAME'))

    k3_real = tf.nn.relu(tf.nn.conv2d(k2_real, w1_3, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(k2_imag, w1_3_, strides=[1, 1, 1, 1], padding='SAME'))
    k3_imag = tf.nn.relu(tf.nn.conv2d(k2_real, w1_3_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(k2_imag, w1_3, strides=[1, 1, 1, 1], padding='SAME'))

    k4_real = tf.nn.conv2d(k3_real, w1_4, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(k3_imag, w1_4_, strides=[1, 1, 1, 1], padding='SAME')
    k4_imag = tf.nn.conv2d(k3_real, w1_4_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(k3_imag, w1_4, strides=[1, 1, 1, 1], padding='SAME')

    kk_real = tf.reshape(k4_real + k_real, shape=[ 160, 180])
    kk_imag = tf.reshape(k4_imag + k_imag, shape=[160, 180])
    k_space = tf.complex(kk_real, kk_imag)
    
    x1_0 = tf.ifft2d(k_space)
    x1_init = tf.reshape(x1_0, shape=[batch_size, 160,180])
    # ss = tf.reshape(tf.real(x1_0), shape=[ 1,160, 180])
    # tt= tf.reshape(tf.imag(x1_0), shape=[1,160, 180])
    # x_real = tf.expand_dims(ss, -1)
    # x_imag = tf.expand_dims(tt, -1)
    
    # k5_real = tf.nn.relu(tf.nn.conv2d(x_real, w1_5, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(x_imag, w1_5_, strides=[1, 1, 1, 1], padding='SAME'))
    # k5_imag = tf.nn.relu(tf.nn.conv2d(x_real, w1_5_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(x_imag, w1_5, strides=[1, 1, 1, 1], padding='SAME'))

    # k6_real = tf.nn.relu(tf.nn.conv2d(k5_real, w1_6, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(k5_imag, w1_6_, strides=[1, 1, 1, 1], padding='SAME'))
    # k6_imag = tf.nn.relu(tf.nn.conv2d(k5_real, w1_6_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(k5_imag, w1_6, strides=[1, 1, 1, 1], padding='SAME'))

    # k7_real = tf.nn.relu(tf.nn.conv2d(k6_real, w1_7, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(k6_imag, w1_7_, strides=[1, 1, 1, 1], padding='SAME'))
    # k7_imag = tf.nn.relu(tf.nn.conv2d(k6_real, w1_7_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(k6_imag, w1_7, strides=[1, 1, 1, 1], padding='SAME'))

    # k8_real = tf.nn.conv2d(k7_real, w1_8, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(k7_imag, w1_8_, strides=[1, 1, 1, 1], padding='SAME')
    # k8_imag = tf.nn.conv2d(k7_real, w1_8_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(k7_imag, w1_8, strides=[1, 1, 1, 1], padding='SAME')
      
    # k_real = tf.reshape(x_real + k8_real, shape=[batch_size, 160, 180])
    # k_imag = tf.reshape(x_imag + k8_imag, shape=[batch_size, 160, 180])
    # x1_init = tf.complex(k_real, k_imag)
    
    return x1_init 

def x2_init(kspace2):
    
    k_real = tf.real(kspace2)
    k_imag = tf.imag(kspace2)

    k_real = tf.expand_dims(k_real, -1)
    k_imag = tf.expand_dims(k_imag, -1)#(?, 160, 180, 1)
        
    k1_real = tf.nn.relu(tf.nn.conv2d(k_real, w2_1, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(k_imag, w2_1_, strides=[1, 1, 1, 1], padding='SAME'))
    k1_imag = tf.nn.relu(tf.nn.conv2d(k_real, w2_1_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(k_imag, w2_1, strides=[1, 1, 1, 1], padding='SAME'))

    k2_real = tf.nn.relu(tf.nn.conv2d(k1_real, w2_2, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(k1_imag, w2_2_, strides=[1, 1, 1, 1], padding='SAME'))
    k2_imag = tf.nn.relu(tf.nn.conv2d(k1_real, w2_2_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(k1_imag, w2_2, strides=[1, 1, 1, 1], padding='SAME'))

    k3_real = tf.nn.relu(tf.nn.conv2d(k2_real, w2_3, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(k2_imag, w2_3_, strides=[1, 1, 1, 1], padding='SAME'))
    k3_imag = tf.nn.relu(tf.nn.conv2d(k2_real, w2_3_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(k2_imag, w2_3, strides=[1, 1, 1, 1], padding='SAME'))

    k4_real = tf.nn.conv2d(k3_real, w2_4, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(k3_imag, w2_4_, strides=[1, 1, 1, 1], padding='SAME')
    k4_imag = tf.nn.conv2d(k3_real, w2_4_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(k3_imag, w2_4, strides=[1, 1, 1, 1], padding='SAME')
  
    k_real = tf.reshape(k4_real + k_real, shape=[160, 180])
    k_imag = tf.reshape(k4_imag + k_imag, shape=[160, 180])
    k_space = tf.complex(k_real, k_imag)
    
    x2_0 = tf.ifft2d(k_space)
    x2_init = tf.reshape(x2_0,shape=[batch_size,160,180])
    # ss = tf.reshape(tf.real(x2_0),shape=[1, 160, 180] )
    # tt = tf.reshape(tf.imag(x2_0), shape=[1, 160, 180])
    # x_real = tf.expand_dims(ss, -1)
    # x_imag = tf.expand_dims(tt, -1)
    
    # k5_real = tf.nn.relu(tf.nn.conv2d(x_real, w2_5, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(x_imag, w2_5_, strides=[1, 1, 1, 1], padding='SAME'))
    # k5_imag = tf.nn.relu(tf.nn.conv2d(x_real, w2_5_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(x_imag, w2_5, strides=[1, 1, 1, 1], padding='SAME'))

    # k6_real = tf.nn.relu(tf.nn.conv2d(k5_real, w2_6, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(k5_imag, w2_6_, strides=[1, 1, 1, 1], padding='SAME'))
    # k6_imag = tf.nn.relu(tf.nn.conv2d(k5_real, w2_6_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(k5_imag, w2_6, strides=[1, 1, 1, 1], padding='SAME'))

    # k7_real = tf.nn.relu(tf.nn.conv2d(k6_real, w2_7, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(k6_imag, w2_7_, strides=[1, 1, 1, 1], padding='SAME'))
    # k7_imag = tf.nn.relu(tf.nn.conv2d(k6_real, w2_7_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(k6_imag, w2_7, strides=[1, 1, 1, 1], padding='SAME'))

    # k8_real = tf.nn.conv2d(k7_real, w2_8, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(k7_imag, w2_8_, strides=[1, 1, 1, 1], padding='SAME')
    # k8_imag = tf.nn.conv2d(k7_real, w2_8_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(k7_imag, w2_8, strides=[1, 1, 1, 1], padding='SAME')
    
    # k_real = tf.reshape(x_real + k8_real, shape=[batch_size, 160, 180])
    # k_imag = tf.reshape(x_imag + k8_imag, shape=[batch_size, 160, 180])
    # x2_init = tf.complex(k_real, k_imag)
    
    return x2_init  

x1_0 = x1_init(kspace1)#t1
x2_0 = x2_init(kspace2)#flair

def compute_cost(x1_0, x2_0):

    cost_rec1 = MSE(rec_target1, tf.abs(x1_0))
    cost_rec2 = MSE(rec_target2, tf.abs(x2_0)) 
    return cost_rec1, cost_rec2

cost_rec1, cost_rec2 = compute_cost(x1_0, x2_0)

loss =  cost_rec1 + cost_rec2 

learning_rate = tf.train.exponential_decay(learning_rate = 0.001,
                                       global_step = global_step,
                                       decay_steps = 100,
                                       decay_rate=0.95, staircase=False) 

Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)#, name='Adam_%d' % PhaseNumber 
trainer = Optimizer.minimize(loss, global_step=global_step)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
sess = tf.Session(config=config) 
init = tf.global_variables_initializer()
sess.run(init)

model_dir = 'initial_kblock_weight' 
log_file_name = "Log1_%s_kblock.txt" % (model_dir)

#___________________________________________________________________________________Train
# print('Load Data...')

# K1_tr = './%s/train/K1.mat'%(data_dir)
# K1_tr_ = './%s/train/K1_.mat'%(data_dir)

# K2_tr = './%s/train/K2.mat'%(data_dir)
# K2_tr_ = './%s/train/K2_.mat'%(data_dir)

# T1_tr = './%s/train/T1.mat'%(data_dir)
# T1_tr_ = './%s/train/T1_.mat'%(data_dir)

# T2_tr = './%s/train/T2.mat'%(data_dir)
# T2_tr_ = './%s/train/T2_.mat'%(data_dir)

# k1 = sio.loadmat(K1_tr)['K1'].astype(np.complex64)
# k1_ = sio.loadmat(K1_tr_)['K1_'].astype(np.complex64)

# k2 = sio.loadmat(K2_tr)['K2'].astype(np.complex64)
# k2_ = sio.loadmat(K2_tr_)['K2_'].astype(np.complex64)

# gt_T1 = sio.loadmat(T1_tr)['T1'].astype(np.float32)
# gt_T1_ = sio.loadmat(T1_tr_)['T1_'].astype(np.float32)

# gt_T2 = sio.loadmat(T2_tr)['T2'].astype(np.float32)
# gt_T2_ = sio.loadmat(T2_tr_)['T2_'].astype(np.float32)

# # K1 = './%s/validation/K1.mat'%(data_dir)
# # K1_ = './%s/validation/K1_.mat'%(data_dir)

# # K2 = './%s/validation/K2.mat'%(data_dir)
# # K2_ = './%s/validation/K2_.mat'%(data_dir)

# # T1 = './%s/validation/T1.mat'%(data_dir)
# # T1_ = './%s/validation/T1_.mat'%(data_dir)

# # T2 = './%s/validation/T2.mat'%(data_dir)
# # T2_ = './%s/validation/T2_.mat'%(data_dir)

# # k1_v = sio.loadmat(K1)['K1'].astype(np.complex64)
# # k1_v_ = sio.loadmat(K1_)['K1_'].astype(np.complex64)
# # k1_train = np.vstack(( k1, k1_, k1_v, k1_v_))
# k1_train = k1
# # k2_v = sio.loadmat(K2)['K2'].astype(np.complex64)
# # k2_v_ = sio.loadmat(K2_)['K2_'].astype(np.complex64)
# # k2_train = np.vstack(( k2, k2_, k2_v, k2_v_))
# k2_train = k2

# # gt_v_T1 = sio.loadmat(T1)['T1'].astype(np.float32)
# # gt_v_T1_ = sio.loadmat(T1_)['T1_'].astype(np.float32)
# # gt_train_T1 = np.vstack((gt_T1, gt_T1_, gt_v_T1, gt_v_T1_))
# gt_train_T1 = gt_T1

# # gt_v_T2 = sio.loadmat(T2)['T2'].astype(np.float32)
# # gt_v_T2_ = sio.loadmat(T2_)['T2_'].astype(np.float32)
# # gt_train_T2 = np.vstack((gt_T2, gt_T2_, gt_v_T2, gt_v_T2_))
# gt_train_T2 = gt_T2

# ntrain = k1_train.shape[0]
# print('Total data %d' % ntrain)
# ntrain1= k2_train.shape[0]
# print('Total data %d' % ntrain1)
# ntrain2 = gt_train_T1.shape[0]
# print('Total data %d' % ntrain2)
# ntrain3 = gt_train_T2.shape[0]
# print('Total data %d' % ntrain3)


# for epoch_i in range(EpochNum+1):
#     randidx_all = range(ntrain3)
#     for batch_i in range(ntrain3// batch_size):
#         randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]
        
#         k1 = k1_train[randidx, :, :]
#         k2 = k2_train[randidx, :, :]
       
#         t1_true = gt_train_T1[randidx, :, :]
#         t2_true = gt_train_T2[randidx, :, :]
    
    
#         feed_dict = {kspace1: k1, kspace2: k2, rec_target1: t1_true, rec_target2: t2_true} 
        
#         sess.run(trainer, feed_dict=feed_dict)

    
#         output_data = "[%02d/%02d/%02d] t1_rec1: %.7f, t2_rec2: %.7f, lr: %.10f \n" % (epoch_i, EpochNum, batch_i, sess.run(cost_rec1, feed_dict=feed_dict),
#                       sess.run(cost_rec2, feed_dict=feed_dict),
#                       sess.run(learning_rate, feed_dict={global_step:epoch_i})) 
#         print(output_data)
            
#     output_file = open(log_file_name, 'a')
#     output_file.write(output_data)
#     output_file.close()

#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)
#     if epoch_i % 1 == 0:
#         saver.save(sess, './%s/CS_Saved_Model_%d.ckpt' % (model_dir, epoch_i), write_meta_graph=False)
        
# print("Training Finished" )

#____________________________________________________________________________________Test
print('Load Testing Data...')
# K1_tr = './%s/train/K1.mat'%(data_dir)
# K1_tr_ = './%s/train/K1_.mat'%(data_dir)

# K2_tr = './%s/train/K2.mat'%(data_dir)
# K2_tr_ = './%s/train/K2_.mat'%(data_dir)

# T1_tr = './%s/train/T1.mat'%(data_dir)
# T1_tr_ = './%s/train/T1_.mat'%(data_dir)

# T2_tr = './%s/train/T2.mat'%(data_dir)
# T2_tr_ = './%s/train/T2_.mat'%(data_dir)

# k1 = sio.loadmat(K1_tr)['K1'].astype(np.complex64)
# k1_ = sio.loadmat(K1_tr_)['K1_'].astype(np.complex64)

# k2 = sio.loadmat(K2_tr)['K2'].astype(np.complex64)
# k2_ = sio.loadmat(K2_tr_)['K2_'].astype(np.complex64)

# gt_T1 = sio.loadmat(T1_tr)['T1'].astype(np.float32)
# gt_T1_ = sio.loadmat(T1_tr_)['T1_'].astype(np.float32)

# gt_T2 = sio.loadmat(T2_tr)['T2'].astype(np.float32)
# gt_T2_ = sio.loadmat(T2_tr_)['T2_'].astype(np.float32)

# # K1 = './%s/validation/K1.mat'%(data_dir)
# # K1_ = './%s/validation/K1_.mat'%(data_dir)

# # K2 = './%s/validation/K2.mat'%(data_dir)
# # K2_ = './%s/validation/K2_.mat'%(data_dir)

# # T1 = './%s/validation/T1.mat'%(data_dir)
# # T1_ = './%s/validation/T1_.mat'%(data_dir)

# # T2 = './%s/validation/T2.mat'%(data_dir)
# # T2_ = './%s/validation/T2_.mat'%(data_dir)

# # k1_v = sio.loadmat(K1)['K1'].astype(np.complex64)
# # k1_v_ = sio.loadmat(K1_)['K1_'].astype(np.complex64)
# # k1_train = np.vstack(( k1, k1_, k1_v, k1_v_))
# k1_train = k1
# # k2_v = sio.loadmat(K2)['K2'].astype(np.complex64)
# # k2_v_ = sio.loadmat(K2_)['K2_'].astype(np.complex64)
# # k2_train = np.vstack(( k2, k2_, k2_v, k2_v_))
# k2_train = k2
# # gt_v_T1 = sio.loadmat(T1)['T1'].astype(np.float32)
# # gt_v_T1_ = sio.loadmat(T1_)['T1_'].astype(np.float32)
# # gt_train_T1 = np.vstack((gt_T1, gt_T1_, gt_v_T1, gt_v_T1_))
# gt_train_T1 = gt_T1
# # gt_v_T2 = sio.loadmat(T2)['T2'].astype(np.float32)
# # gt_v_T2_ = sio.loadmat(T2_)['T2_'].astype(np.float32)
# # gt_train_T2 = np.vstack((gt_T2, gt_T2_, gt_v_T2, gt_v_T2_))
# gt_train_T2 = gt_T2

# testdata
K1_tr = './%s/test/K1.mat'%(data_dir)

K2_tr = './%s/test/K2.mat'%(data_dir)

T1_tr = './%s/test/T1.mat'%(data_dir)

T2_tr = './%s/test/T2.mat'%(data_dir)


k1 = sio.loadmat(K1_tr)['K1'].astype(np.complex64)

k2 = sio.loadmat(K2_tr)['K2'].astype(np.complex64)

gt_T1 = sio.loadmat(T1_tr)['T1'].astype(np.float32)

gt_T2 = sio.loadmat(T2_tr)['T2'].astype(np.float32)


k1_train = k1

k2_train = k2


gt_train_T1 = gt_T1

gt_train_T2 = gt_T2
#testdata

ntrain = k1_train.shape[0]
print('Total data %d' % ntrain)
ntrain1= k2_train.shape[0]
print('Total data %d' % ntrain1)
ntrain2 = gt_train_T1.shape[0]
print('Total data %d' % ntrain2)
ntrain3 = gt_train_T2.shape[0]
print('Total data %d' % ntrain3)


TIME_ALL  = []
PSNR1_All = []
SSIM1_All = []
NMSE1_All = []
PSNR2_All = []
SSIM2_All = []
NMSE2_All = []
PSNR3_All = []
SSIM3_All = []
NMSE3_All = []

init_x1 = []
init_x2 = []


saver.restore(sess, './%s/CS_Saved_Model_%d.ckpt' % (model_dir, ckpt_model_number))

result_file_name = "Meta_Rec_Results.txt"
    
idx_all = np.arange(ntrain)
for imag_no in range(ntrain):
   
   
    randidx = idx_all[imag_no:imag_no+1]
    kspace_t1 = k1_train[randidx, :, :]
    kspace_flair = k2_train[randidx, :, :]
    target_t1 = gt_train_T1[randidx, :, :]
    target_t2 = gt_train_T2[randidx, :, :]
   

    feed_dict = {kspace1: kspace_t1, kspace2: kspace_flair, rec_target1: target_t1, rec_target2: target_t2}
    
    start = time()
    Reconstruction1_value = sess.run(x1_0, feed_dict=feed_dict) 
    Reconstruction2_value = sess.run(x2_0, feed_dict=feed_dict) 
    
    end = time()
    print(Reconstruction1_value)
    rec_1 = np.reshape(abs(Reconstruction1_value), (160,180)).astype(np.float32)
    rec_2 = np.reshape(abs(Reconstruction2_value), (160,180)).astype(np.float32)
   
    rec_111 = np.reshape(Reconstruction1_value, (160,180)).astype(np.complex64)
    rec_222 = np.reshape(Reconstruction2_value, (160,180)).astype(np.complex64)
    
    rec_reference_1 = np.reshape(target_t1, (160,180)).astype(np.float32)
    rec_reference_2 = np.reshape(target_t2, (160,180)).astype(np.float32)
    
   
    
    
    PSNRt1 =  psnr(rec_reference_1, rec_1)
    
    SSIMt1 =  ssim(rec_1, rec_reference_1)
    NMSEt1 =  nmse(rec_reference_1, rec_1)

    PSNRt2 =  psnr(rec_reference_2, rec_2)


    SSIMt2 =  ssim(rec_reference_2, rec_2)
    NMSEt2 =  nmse(rec_reference_2, rec_2)
    
    result1 = "Run time for %s:%.4f, PSNRt1:%.4f, SSIMt1:%.4f, NMSEt1:%.4f. \n" % (imag_no+1, (end - start), PSNRt1, SSIMt1, NMSEt1)
    result2 = "Run time for %s:%.4f, PSNRt2:%.4f, SSIMt2:%.4f, NMSEt2:%.4f. \n" % (imag_no+1, (end - start), PSNRt2, SSIMt2, NMSEt2)
   
    
    print(result1)
    print(result2) 

    im_rec_name = "%s_rec_%d.mat" % (imag_no+1, ckpt_model_number)  
    
    init_x1.append(rec_111)
    init_x2.append(rec_222)
    
    
    PSNR1_All.append(PSNRt1)
    SSIM1_All.append(SSIMt1)
    NMSE1_All.append(NMSEt1)

    PSNR2_All.append(PSNRt2)
    SSIM2_All.append(SSIMt2)
    NMSE2_All.append(NMSEt2)

   
    
    TIME_ALL.append((end - start))
 
output_data1 = "rec_t1 ckpt NO.%d, Avg REC PSNR is %.4f dB std %.4f, SSIM is %.4f std %.4f, NMSE is %.4f std %.4f, time: %.4f\n" % (ckpt_model_number, np.mean(PSNR1_All), np.std(PSNR1_All), np.mean(SSIM1_All), np.std(SSIM1_All), np.mean(NMSE1_All), np.std(NMSE1_All), np.mean(TIME_ALL))
print(output_data1)
output_data2 = "rec_t2 ckpt NO.%d, Avg REC PSNR is %.4f dB std %.4f, SSIM is %.4f std %.4f, NMSE is %.4f std %.4f, time: %.4f\n" % (ckpt_model_number, np.mean(PSNR2_All), np.std(PSNR2_All), np.mean(SSIM2_All), np.std(SSIM2_All), np.mean(NMSE2_All), np.std(NMSE2_All), np.mean(TIME_ALL))
print(output_data2)


output_file = open(result_file_name, 'a')
output_file.write(output_data1)
output_file.write(output_data2)

output_file.close()

X1_INIT = np.stack((init_x1), axis=0)
X2_INIT = np.stack((init_x2),axis=0)

print(X1_INIT.shape)


saveAsMat(X1_INIT, "T1_0train.mat", 'T1_0train',  mat_dict=None)
saveAsMat(X2_INIT, "T2_0train.mat", 'T2_0train',  mat_dict=None)

    
sess.close()

print("Reconstruction READY")