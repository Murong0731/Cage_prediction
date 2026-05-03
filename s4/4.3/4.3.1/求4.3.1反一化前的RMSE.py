#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from math import sqrt
from numpy import concatenate
from pandas import read_csv, DataFrame, concat
from datetime import datetime
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import keras.backend as K
from keras.layers import Multiply
from keras.models import Sequential ,Model
from keras.layers import Dense , Input , Reshape , Flatten ,Permute , Lambda , RepeatVector ,Conv1D , MaxPooling1D , Dropout, Bidirectional, Activation
from keras.layers import GRU, LSTM
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD,Adam
from keras.utils import np_utils   #np_utils
from keras.callbacks import TensorBoard  #TensorBoard可视化


# In[2]:


from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
def RMSE(valid_y,pre_Y):
    XP1 = valid_y.copy()
    XA1 = pre_Y.copy()
    print('RMSE(sklearn):',np.sqrt(mean_squared_error(XP1, XA1)))


# In[3]:


# 验证
# data1 = pd.read_csv('surgeN_自回归步长1.csv')
# surge_yuan = np.array(data1)[:, 0:1]
# surge1_1 = np.array(data1)[:, 1:2]
# evaluate(surge_yuan,surge1_1)


# In[4]:


# T:0
# H:1
# Surge:2
# Heave:4
# Force1:8
data = pd.read_csv('t_2_11.2_50.csv')
data.head()


# In[5]:


data.describe()


# In[6]:


data_distance = np.hstack((np.array(data)[:, 1:8], np.array(data)[:, 8:10]))
print(data_distance)
print(data_distance.shape)


# In[7]:


# H_scaler = MinMaxScaler(feature_range=(-1, 1))
# H = H_scaler.fit_transform(data_distance[:,0:1])
# Surge_scaler = MinMaxScaler(feature_range=(-1, 1))
# Surge = Surge_scaler.fit_transform(data_distance[:,1:2])
# # Sway_scaler = MinMaxScaler(feature_range=(-1, 1))
# # Sway = Sway_scaler.fit_transform(data_distance[:,2:3])
# Heave_scaler = MinMaxScaler(feature_range=(-1, 1))
# Heave = Heave_scaler.fit_transform(data_distance[:,3:4])
# # Roll_scaler = MinMaxScaler(feature_range=(-1, 1))
# # Roll = Roll_scaler.fit_transform(data_distance[:,4:5]*1e6)
# Pitch_scaler = MinMaxScaler(feature_range=(-1, 1))
# Pitch = Pitch_scaler.fit_transform(data_distance[:,5:6])
# Yaw_scaler = MinMaxScaler(feature_range=(-1, 1))
# Yaw = Yaw_scaler.fit_transform(data_distance[:,6:7]*1e6)
Force1_scaler = MinMaxScaler(feature_range=(-1, 1))
Force1 = Force1_scaler.fit_transform(data_distance[:,7:8])
Force2_scaler = MinMaxScaler(feature_range=(-1, 1))
Force2 = Force2_scaler.fit_transform(data_distance[:,8:9])
# zong_data = np.hstack(())


# In[8]:


data100_force1 = pd.read_csv('训练100_force1_步长1.csv')
data200_force1 = pd.read_csv('训练200_force1_步长1.csv')
data300_force1 = pd.read_csv('训练300_force1_步长1.csv')
data400_force1 = pd.read_csv('训练400_force1_步长1.csv')
data500_force1 = pd.read_csv('训练500_force1_步长1.csv')
data600_force1 = pd.read_csv('训练600_force1_步长1.csv')
data700_force1 = pd.read_csv('训练700_force1_步长1.csv')
data800_force1 = pd.read_csv('训练800_force1_步长1.csv')
data900_force1 = pd.read_csv('训练900_force1_步长1.csv')
data1000_force1 = pd.read_csv('训练1000_force1_步长1.csv')

data100_force2 = pd.read_csv('训练100_force2_步长1.csv')
data200_force2 = pd.read_csv('训练200_force2_步长1.csv')
data300_force2 = pd.read_csv('训练300_force2_步长1.csv')
data400_force2 = pd.read_csv('训练400_force2_步长1.csv')
data500_force2 = pd.read_csv('训练500_force2_步长1.csv')
data600_force2 = pd.read_csv('训练600_force2_步长1.csv')
data700_force2 = pd.read_csv('训练700_force2_步长1.csv')
data800_force2 = pd.read_csv('训练800_force2_步长1.csv')
data900_force2 = pd.read_csv('训练900_force2_步长1.csv')
data1000_force2 = pd.read_csv('训练1000_force2_步长1.csv')


# In[9]:


# 反归一化a=-1,b=1
# b-a=2；Hmax-Hmin=np.max(data_distance[:,0:1])-np.min(data_distance[:,0:1])
# guiyihua(输入步长)_变量(输出步长)

# guiyihua_H=2*(data_distance[:,0:1]-np.min(data_distance[:,0:1]))/(np.max(data_distance[:,0:1])-np.min(data_distance[:,0:1]))-1
# print(guiyihua_H)

# force1  data_distance[:,7:8]
yuan_force1 = 2*(np.array(data100_force1)[:, 0:1]-np.min(data_distance[:,7:8]))/(np.max(data_distance[:,7:8])-np.min(data_distance[:,7:8]))-1
guiyihua100_force1 = 2*(np.array(data100_force1)[:, 1:2]-np.min(data_distance[:,7:8]))/(np.max(data_distance[:,7:8])-np.min(data_distance[:,7:8]))-1
guiyihua200_force1 = 2*(np.array(data200_force1)[:, 1:2]-np.min(data_distance[:,7:8]))/(np.max(data_distance[:,7:8])-np.min(data_distance[:,7:8]))-1
guiyihua300_force1 = 2*(np.array(data300_force1)[:, 1:2]-np.min(data_distance[:,7:8]))/(np.max(data_distance[:,7:8])-np.min(data_distance[:,7:8]))-1
guiyihua400_force1 = 2*(np.array(data400_force1)[:, 1:2]-np.min(data_distance[:,7:8]))/(np.max(data_distance[:,7:8])-np.min(data_distance[:,7:8]))-1
guiyihua500_force1 = 2*(np.array(data500_force1)[:, 1:2]-np.min(data_distance[:,7:8]))/(np.max(data_distance[:,7:8])-np.min(data_distance[:,7:8]))-1
guiyihua600_force1 = 2*(np.array(data600_force1)[:, 1:2]-np.min(data_distance[:,7:8]))/(np.max(data_distance[:,7:8])-np.min(data_distance[:,7:8]))-1
guiyihua700_force1 = 2*(np.array(data700_force1)[:, 1:2]-np.min(data_distance[:,7:8]))/(np.max(data_distance[:,7:8])-np.min(data_distance[:,7:8]))-1
guiyihua800_force1 = 2*(np.array(data800_force1)[:, 1:2]-np.min(data_distance[:,7:8]))/(np.max(data_distance[:,7:8])-np.min(data_distance[:,7:8]))-1
guiyihua900_force1 = 2*(np.array(data900_force1)[:, 1:2]-np.min(data_distance[:,7:8]))/(np.max(data_distance[:,7:8])-np.min(data_distance[:,7:8]))-1
guiyihua1000_force1 = 2*(np.array(data1000_force1)[:, 1:2]-np.min(data_distance[:,7:8]))/(np.max(data_distance[:,7:8])-np.min(data_distance[:,7:8]))-1


# force2  data_distance[:,8:9]
yuan_force2 = 2*(np.array(data100_force2)[:, 0:1]-np.min(data_distance[:,8:9]))/(np.max(data_distance[:,8:9])-np.min(data_distance[:,8:9]))-1
guiyihua100_force2 = 2*(np.array(data100_force2)[:, 1:2]-np.min(data_distance[:,8:9]))/(np.max(data_distance[:,8:9])-np.min(data_distance[:,8:9]))-1
guiyihua200_force2 = 2*(np.array(data200_force2)[:, 1:2]-np.min(data_distance[:,8:9]))/(np.max(data_distance[:,8:9])-np.min(data_distance[:,8:9]))-1
guiyihua300_force2 = 2*(np.array(data300_force2)[:, 1:2]-np.min(data_distance[:,8:9]))/(np.max(data_distance[:,8:9])-np.min(data_distance[:,8:9]))-1
guiyihua400_force2 = 2*(np.array(data400_force2)[:, 1:2]-np.min(data_distance[:,8:9]))/(np.max(data_distance[:,8:9])-np.min(data_distance[:,8:9]))-1
guiyihua500_force2 = 2*(np.array(data500_force2)[:, 1:2]-np.min(data_distance[:,8:9]))/(np.max(data_distance[:,8:9])-np.min(data_distance[:,8:9]))-1
guiyihua600_force2 = 2*(np.array(data600_force2)[:, 1:2]-np.min(data_distance[:,8:9]))/(np.max(data_distance[:,8:9])-np.min(data_distance[:,8:9]))-1
guiyihua700_force2 = 2*(np.array(data700_force2)[:, 1:2]-np.min(data_distance[:,8:9]))/(np.max(data_distance[:,8:9])-np.min(data_distance[:,8:9]))-1
guiyihua800_force2 = 2*(np.array(data800_force2)[:, 1:2]-np.min(data_distance[:,8:9]))/(np.max(data_distance[:,8:9])-np.min(data_distance[:,8:9]))-1
guiyihua900_force2 = 2*(np.array(data900_force2)[:, 1:2]-np.min(data_distance[:,8:9]))/(np.max(data_distance[:,8:9])-np.min(data_distance[:,8:9]))-1
guiyihua1000_force2 = 2*(np.array(data1000_force2)[:, 1:2]-np.min(data_distance[:,8:9]))/(np.max(data_distance[:,8:9])-np.min(data_distance[:,8:9]))-1


# In[10]:


RMSE(yuan_force1,guiyihua100_force1)
RMSE(yuan_force1,guiyihua200_force1)
RMSE(yuan_force1,guiyihua300_force1)
RMSE(yuan_force1,guiyihua400_force1)
RMSE(yuan_force1,guiyihua500_force1)
RMSE(yuan_force1,guiyihua600_force1)
RMSE(yuan_force1,guiyihua700_force1)
RMSE(yuan_force1,guiyihua800_force1)
RMSE(yuan_force1,guiyihua900_force1)
RMSE(yuan_force1,guiyihua1000_force1)
print('-------------')
RMSE(yuan_force2,guiyihua100_force2)
RMSE(yuan_force2,guiyihua200_force2)
RMSE(yuan_force2,guiyihua300_force2)
RMSE(yuan_force2,guiyihua400_force2)
RMSE(yuan_force2,guiyihua500_force2)
RMSE(yuan_force2,guiyihua600_force2)
RMSE(yuan_force2,guiyihua700_force2)
RMSE(yuan_force2,guiyihua800_force2)
RMSE(yuan_force2,guiyihua900_force2)
RMSE(yuan_force2,guiyihua1000_force2)


# In[ ]:





# In[ ]:




