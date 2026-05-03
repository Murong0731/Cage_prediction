#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 网络结构和激活函数
# 网络结构：神经元个数、网络层数（网络深度）、每层神经元的个数又称为网络宽度；
# 不同宽度、深度的网络模型有不同的拟合能力
# 问题：运算量暴增、梯度消失、梯度爆炸


# In[2]:


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


# In[3]:


# T:0
# H:1
# Surge:2
# Heave:4
# Force1:8
data_1 = pd.read_csv('t_1.csv')
data_1.head()


# In[4]:


data_distance = np.hstack((np.array(data_1)[:, 1:8], np.array(data_1)[:, 8:10]))
print(data_distance)
print(data_distance.shape)

H_scaler = MinMaxScaler(feature_range=(-1, 1))
H = H_scaler.fit_transform(data_distance[:,0:1])
Surge_scaler = MinMaxScaler(feature_range=(-1, 1))
Surge = Surge_scaler.fit_transform(data_distance[:,1:2])
Sway_scaler = MinMaxScaler(feature_range=(-1, 1))
Sway = Sway_scaler.fit_transform(data_distance[:,2:3])
Heave_scaler = MinMaxScaler(feature_range=(-1, 1))
Heave = Heave_scaler.fit_transform(data_distance[:,3:4])
Roll_scaler = MinMaxScaler(feature_range=(-1, 1))
Roll = Roll_scaler.fit_transform(data_distance[:,4:5])
Pitch_scaler = MinMaxScaler(feature_range=(-1, 1))
Pitch = Pitch_scaler.fit_transform(data_distance[:,5:6])
Yaw_scaler = MinMaxScaler(feature_range=(-1, 1))
Yaw = Yaw_scaler.fit_transform(data_distance[:,6:7])
Force1_scaler = MinMaxScaler(feature_range=(-1, 1))
Force1 = Force1_scaler.fit_transform(data_distance[:,7:8])
Force2_scaler = MinMaxScaler(feature_range=(-1, 1))
Force2 = Force1_scaler.fit_transform(data_distance[:,8:9])
# zong_data = np.hstack(())


# In[5]:


np.savetxt('MinMaxScaler_data.csv',np.hstack((Heave,Surge,Pitch)),delimiter=',')


# #### 预处理 散点图

# In[88]:


def DataScatter_Plot(data_x, data_y, y_arange, scatter_size, color, Inter_xaxis, Inter_yaxis):
    fig = plt.figure(figsize=(5.2,3.8))
    # 将x周的刻度线方向设置向内
    plt.rcParams['xtick.direction'] = 'in'  
    # 将y轴的刻度方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  
    #设置字体以便支持中文
    plt.rcParams['font.sans-serif']=['SongNTR']
    #为正常显示负号
    plt.rcParams['axes.unicode_minus'] = False 
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    axes = plt.subplot()
    axes.minorticks_on()
    axes.tick_params(axis="both", which="major", direction="in", width=1, length=5)
    axes.tick_params(axis="both", which="minor", direction="in", width=1, length=3)
    axes.xaxis.set_minor_locator(MultipleLocator(Inter_xaxis))
    axes.yaxis.set_minor_locator(MultipleLocator(Inter_yaxis))
    plt.scatter(data_x, data_y, scatter_size, c=color)
    plt.xlabel("x", font='Times New Roman', fontsize=24)
    plt.ylabel("y", font='Times New Roman', fontsize=24)
    plt.xticks([0,2500,5000,7500,10000], fontsize=24)
    plt.yticks(y_arange, fontsize=24)
    plt.xlim(0, 11000)
    #plt.ylim(-2, 16)
    #plt.legend(loc='lower right', prop={'family':'Times New Roman', 'size':10})
    #plt.xlabel('Time(s)', fontdict={'family' : 'Times New Roman', 'size':10})
    #plt.grid(True, which="major", linestyle="--", color="gray", linewidth=0.75)
    #plt.grid(True, which="minor", linestyle=":", color="lightgray", linewidth=0.75)
    plt.show()


# In[91]:


DataScatter_Plot(np.arange(1,11001,1), data_distance[:,3:4], y_arange=[-1500,-1000,-500,0,500,1000], scatter_size=1, color="k", Inter_xaxis=2500, Inter_yaxis=250)


# In[92]:


DataScatter_Plot(np.arange(1,11001,1), Heave, y_arange=[-1,-0.5,0,0.5,1], scatter_size=1, color="k", Inter_xaxis=2500, Inter_yaxis=250)


# In[89]:


DataScatter_Plot(np.arange(1,11001,1), data_distance[:,1:2], y_arange=[-200,0,200,400], scatter_size=1, color="k", Inter_xaxis=2500, Inter_yaxis=250)
DataScatter_Plot(np.arange(1,11001,1), Surge, y_arange=[-1,-0.5,0,0.5,1], scatter_size=1, color="k", Inter_xaxis=2500, Inter_yaxis=250)


# In[90]:


DataScatter_Plot(np.arange(1,11001,1), data_distance[:,5:6],  y_arange=[-0.1,-0.05,0,0.05,0.1], scatter_size=1, color="k", Inter_xaxis=2500, Inter_yaxis=250)
DataScatter_Plot(np.arange(1,11001,1), Pitch, y_arange=[-1,-0.5,0,0.5,1], scatter_size=1, color="k", Inter_xaxis=2500, Inter_yaxis=250)


# In[41]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def tanh(x):
    return (np.exp(x)-np.exp(-x)) /(np.exp(x)+np.exp(-x)) 

def relu(x):
    return np.maximum(0, x)

def swish(x,beta=1.0):
    return x*sigmoid(beta*x)
def Function_Plot(function, a):
    x = np.arange(-10, 10, 0.01)
    #y = np.arange(-2, 10, 0.01)
    fig = plt.figure()
    # 将x周的刻度线方向设置向内
    plt.rcParams['xtick.direction'] = 'in'  
    # 将y轴的刻度方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  
    #设置字体以便支持中文
    plt.rcParams['font.sans-serif']=['SongNTR']
    #为正常显示负号
    plt.rcParams['axes.unicode_minus'] = False 
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    axes = plt.subplot()
    axes.minorticks_on()
    axes.tick_params(axis="both", which="major", direction="in", width=1, length=5)
    axes.tick_params(axis="both", which="minor", direction="in", width=1, length=3)
    axes.xaxis.set_minor_locator(MultipleLocator(1))
    plt.plot(x, function(x),'black')
    plt.xlabel("x", font='Times New Roman', fontsize=24)
    plt.ylabel("y", font='Times New Roman', fontsize=24)
    plt.xticks([-10,-6,-2,2,6,10], fontsize=24)
    plt.yticks(a, fontsize=24)
    plt.xlim(-10, 10)
    #plt.ylim(-2, 16)
    #plt.legend(loc='lower right', prop={'family':'Times New Roman', 'size':10})
    #plt.xlabel('Time(s)', fontdict={'family' : 'Times New Roman', 'size':10})
    plt.grid(True, which="major", linestyle="--", color="gray", linewidth=0.75)
    plt.grid(True, which="minor", linestyle=":", color="lightgray", linewidth=0.75)
    plt.show()
    

Function_Plot(sigmoid, [0,0.2,0.4,0.6,0.8,1])
Function_Plot(tanh, [-1,-0.5,0,0.5,1])
Function_Plot(relu, [0,2,4,6,8,10])
Function_Plot(swish, [0,2,4,6,8,10])


# In[2]:


#比较优化器：SGD, Momentum, AdaGrad, Adam(静态图)
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *


def f(x, y):
    return x**2 / 20.0 + y**2


def df(x, y):
    return x / 10.0, 2.0*y

init_pos = (-7.0, 2.0)
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
grads = {}
grads['x'], grads['y'] = 0, 0


optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["RMSprop"] = RMSprop(lr=1.5)  #AdaGrad
optimizers["Adam"] = Adam(lr=0.3)

idx = 1

for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'], params['y'] = init_pos[0], init_pos[1]
    
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        
        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads)
    

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    
    X, Y = np.meshgrid(x, y) 
    Z = f(X, Y)
    
    # for simple contour line  
    mask = Z > 7
    Z[mask] = 0
    
    # plot 
    #with plt.style.context(['science','no-latex']):
    fig = plt.figure()
    # 将x周的刻度线方向设置向内
    plt.rcParams['xtick.direction'] = 'in'  
    # 将y轴的刻度方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  
    #设置字体以便支持中文
    plt.rcParams['font.sans-serif']=['SimHei']
    #为正常显示负号
    plt.rcParams['axes.unicode_minus'] = False 
    #plt.subplot(2, 2, idx)
    idx += 1
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.contour(X, Y, Z, colors="black")
    plt.plot(x_history, y_history, 'o-', color="red")
    plt.ylim(-5, 5)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    #colorbar()
    #spring()
    #plt.title(key) 
    plt.xlabel("x", font='Times New Roman', fontsize=24)
    plt.ylabel("y", font='Times New Roman', fontsize=24)
    plt.xticks(np.arange(-10, 11, 5), fontproperties = 'Times New Roman', fontsize=24)
    plt.yticks(fontproperties = 'Times New Roman', fontsize=24)
    #plt.legend(loc='lower right', prop={'family':'Times New Roman', 'size':10})
    #plt.xlabel('Time(s)', fontdict={'family' : 'Times New Roman', 'size':10})
plt.show()


# In[26]:


#比较学习率：0.08, 0.3, 1, 5(静态图)
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *


def f(x, y):
    return x**2 / 20.0 + y**2


def df(x, y):
    return x / 10.0, 2.0*y

init_pos = (-7.0, 2.0)
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
grads = {}
grads['x'], grads['y'] = 0, 0


optimizers = OrderedDict()
optimizers["Adam1"] = Adam(lr=0.05)
optimizers["Adam2"] = Adam(lr=0.3)
optimizers["Adam3"] = Adam(lr=1)  #AdaGrad
optimizers["Adam4"] = Adam(lr=5)

idx = 1

for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'], params['y'] = init_pos[0], init_pos[1]
    
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        
        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads)
    

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    
    X, Y = np.meshgrid(x, y) 
    Z = f(X, Y)
    
    # for simple contour line  
    mask = Z > 7
    Z[mask] = 0
    
    # plot 
    #with plt.style.context(['science','no-latex']):
    fig = plt.figure(figsize=(8.5,6.55))
    # 将x周的刻度线方向设置向内
    plt.rcParams['xtick.direction'] = 'in'  
    # 将y轴的刻度方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  
    #设置字体以便支持中文
    plt.rcParams['font.sans-serif']=['SimHei']
    #为正常显示负号
    plt.rcParams['axes.unicode_minus'] = False 
    #plt.subplot(2, 2, idx)
    idx += 1
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.contour(X, Y, Z, colors="black")
    plt.plot(x_history, y_history, 'o-', color="red")
    plt.ylim(-5, 5)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    #colorbar()
    #spring()
    #plt.title(key) 
    plt.xlabel("x", font='Times New Roman', fontsize=32)
    plt.ylabel("y", font='Times New Roman', fontsize=32)
    plt.xticks(np.arange(-10, 11, 5), fontproperties = 'Times New Roman', fontsize=32)
    plt.yticks(fontproperties = 'Times New Roman', fontsize=32)
    #plt.legend(loc='lower right', prop={'family':'Times New Roman', 'size':10})
    #plt.xlabel('Time(s)', fontdict={'family' : 'Times New Roman', 'size':10})
plt.show()


# #### 时序模型A

# In[128]:


# GRU
def Model_GRU(train_X, train_Y, valid_X, valid_Y, lr=0.01, epochs=20, batch_size=256):
    model = Sequential()
    model.add(GRU(25, activation='tanh', return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))  #25×5的数据输入
    model.add(Dropout(0.3))
    model.add(GRU(100, activation='tanh', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(GRU(100, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(train_Y.shape[1])) 
    model.add(Activation('tanh'))
    adam = Adam(lr = lr)
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, validation_data=(valid_X, valid_Y), verbose=2, shuffle=False)
    model_structure = model.summary()
    pre_Y = model.predict(valid_X)
    return model, history, pre_Y

# LSTM
def Model_LSTM(train_X, train_Y, valid_X, valid_Y, lr=0.01, epochs=20, batch_size=256):
    model = Sequential()
    model.add(LSTM(25, activation='tanh', return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))  #25×5的数据输入
    model.add(Dropout(0.3))
    model.add(LSTM(100, activation='tanh', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(100, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(train_Y.shape[1])) 
    model.add(Activation('tanh'))
    adam = Adam(lr = lr)
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, validation_data=(valid_X, valid_Y), verbose=2, shuffle=False)
    model_structure = model.summary()
    pre_Y = model.predict(valid_X)
    return model, history, pre_Y

# BiLSTM
from keras.layers import LeakyReLU
def Model_BiLSTM(train_X, train_Y, valid_X, valid_Y, lr=0.01, epochs=20, batch_size=256):
    model = Sequential()
    model.add(Bidirectional(LSTM(25, batch_input_shape=(24, train_X.shape[1], train_X.shape[2]),stateful=False), merge_mode='concat'))  #25×5的数据输入
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(train_Y.shape[1])) 
    model.add(Activation('tanh'))
    adam = Adam(lr = lr)
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, validation_data=(valid_X, valid_Y), verbose=2, shuffle=False)
    model_structure = model.summary()
    pre_Y = model.predict(valid_X)
    return model, history, pre_Y

def series_to_supervised(data, n_in, n_out, dropnan=True):
#      '''
#      说明：将每一个特征（输入+输出）的单步-目标步长数全数列举出来
#      '''
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg
def deal_data1(data, features_number, time_steps):
    '''
    适用：单/多特征，单/多预测步长
    data：数据集（仅包括输入+输出，且输出仅有一列并在最后）
    feature_number：输入+输出变量数
    n_in：预测步长
    结果解释：输入+输出+所需预测步长的输出（当为单特征时，输入、输出为同一列）
    '''
    process_data = series_to_supervised(data, time_steps, 1, dropnan=True)
    a = []
    for i in range(features_number, features_number*(time_steps+1)-1):
        a.append(i)
        i = i+1
    process_data.drop(process_data.columns[a], axis=1, inplace=True)
    return process_data.values
def deal_data2(data, features_number, time_steps):
    '''
    适用：多特征，单/多预测步长，不适用于单特征
    data：数据集（仅包括输入+输出，且输出仅有一列并在最后）
    feature_number：输入+输出变量数
    n_in：预测步长
    结果解释：输入+所需预测步长的输出
    '''
    process_data = series_to_supervised(data, time_steps, 1, dropnan=True)
    a = []
    for i in range(features_number-1, features_number*(time_steps+1)-1):
        a.append(i)
        i = i+1
    process_data.drop(process_data.columns[a], axis=1, inplace=True)
    return process_data.values
def split_sequence(dataset, n_past):
    x, y = list(), list()
    for i in range(len(dataset)):
        end_ix = i + n_past
        if end_ix > len(dataset):
            break
        seq_x, seq_y = dataset[i:end_ix, :-1], dataset[i, -1]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)
def split_train_valid(data_X, data_Y, n_train_times, n_valid_times):
    '''
    说明：将数据集划分为训练集和验证集
    疑问：先数据切割重组再划分数据集，先划分数据集再数据切割重组，有何区别影响？
    '''
    train_x, valid_x = data_X[:n_train_times, :, :], data_X[n_train_times:n_valid_times, :, :]
    train_y, valid_y = data_Y[:n_train_times], data_Y[n_train_times:n_valid_times]
    train_y = train_y.reshape((n_train_times, 1))
    valid_y = valid_y.reshape((n_valid_times-n_train_times, 1))
    return train_x, train_y, valid_x, valid_y
def loss_plot(history, epo, length, width):
    # plot history
    plt.figure(figsize = (length, width))
    # 将x周的刻度线方向设置向内
    plt.rcParams['xtick.direction'] = 'in'  
    # 将y轴的刻度方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  
    #设置字体以便支持中文
    plt.rcParams['font.sans-serif']=['SimHei']
    #为正常显示负号
    plt.rcParams['axes.unicode_minus'] = False 
    plt.plot(history.history['loss'], label = 'train loss')
    plt.plot(history.history['val_loss'], label = 'valid loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    print(np.hstack((np.array(history.history['loss']).reshape(epo,1), np.array(history.history['val_loss']).reshape(epo,1))))

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
def evaluate(valid_y,pre_Y):
    XP1 = valid_y.copy()
    XA1 = pre_Y.copy()
    #c =abs((XP-XA)/XA)
    c1 =abs((XP1-XA1))
    #MAPE = (c.sum())/(c.shape[0])
    MAE1 = (c1.sum())/(c1.shape[0])
    print('MAE:',MAE1)
    print('MAE(sklearn):', mean_absolute_error(XP1, XA1))
    print('MAPE(sklearn):', mean_absolute_percentage_error(XP1, XA1))

    #MSE
    r1 = ((XP1-XA1)*(XP1-XA1)).sum()/(XP1.shape[0])
    #RMSE = np.sqrt(r)
    MSE1 = r1
    print('MSE:', MSE1)
    print('MSE(sklearn):', mean_squared_error(XP1, XA1))
    print('RMSE(sklearn):',np.sqrt(mean_squared_error(XP1, XA1)))
#     #R^2
#     XM1 = XA1.sum()/(XA1.shape[0])
#     R2_1 = 1-(((XP1-XA1)*(XP1-XA1)).sum()/((XM1-XA1)*(XM1-XA1)).sum())
#     print('R^2:', R2_1)
#     print('R^2(sklearn):', r2_score(XP1, XA1))

    real1 = np.trapz(abs((XP1 - (XP1.sum()/(XP1.shape[0])))).reshape(XP1.shape[0],), dx=0.1)
    pre1 = np.trapz(abs((XA1 - (XP1.sum()/(XP1.shape[0])))).reshape(XA1.shape[0],), dx=0.1)
    Acc1 = 1 - abs(1 - (pre1/real1))
    print('Acc:', Acc1)
    
def FanGuiHua_surge(valid_sur_y,sur_pre_Y):
    fan_surge_real=Surge_scaler.inverse_transform(valid_sur_y)
    fan_surge_pre=Surge_scaler.inverse_transform(sur_pre_Y)
    plt.figure(figsize=(30,4), dpi=100)
    plt.plot(fan_surge_real[:200],color='blue',label='real')   #真实曲线
    plt.plot(fan_surge_pre[:200],color='orange',label='prediction') #预测曲线
    plt.legend()
    plt.show()
    evaluate(fan_surge_real,fan_surge_pre)
    return fan_surge_real,fan_surge_pre
def FanGuiHua_pitch(valid_pit_y,pit_pre_Y):
    fan_pitch_real=Pitch_scaler.inverse_transform(valid_pit_y)
    fan_pitch_pre=Pitch_scaler.inverse_transform(pit_pre_Y)
    plt.figure(figsize=(30,4), dpi=100)
    plt.plot(fan_pitch_real[:200],color='blue',label='real')   #真实曲线
    plt.plot(fan_pitch_pre[:200],color='orange',label='prediction') #预测曲线
    plt.legend()
    plt.show()
    evaluate(fan_pitch_real,fan_pitch_pre)
    return fan_pitch_real,fan_pitch_pre
def FanGuiHua_sway(valid_sway_y,sway_pre_Y):
    fan_sway_real=Sway_scaler.inverse_transform(valid_sway_y)
    fan_sway_pre=Sway_scaler.inverse_transform(sway_pre_Y)
    plt.figure(figsize=(30,4), dpi=100)
    plt.plot(fan_sway_real[:200],color='blue',label='real')   #真实曲线
    plt.plot(fan_sway_pre[:200],color='orange',label='prediction') #预测曲线
    plt.legend()
    plt.show()
    evaluate(fan_sway_real,fan_sway_pre)
    return fan_sway_real,fan_sway_pre
def FanGuiHua_roll(valid_roll_y,roll_pre_Y):
    fan_roll_real=Roll_scaler.inverse_transform(valid_roll_y)
    fan_roll_pre=Roll_scaler.inverse_transform(roll_pre_Y)
    plt.figure(figsize=(30,4), dpi=100)
    plt.plot(fan_roll_real[:200],color='blue',label='real')   #真实曲线
    plt.plot(fan_roll_pre[:200],color='orange',label='prediction') #预测曲线
    plt.legend()
    plt.show()
    evaluate(fan_roll_real,fan_roll_pre)
    return fan_roll_real,fan_roll_pre
def FanGuiHua_yaw(valid_yaw_y,yaw_pre_Y):
    fan_yaw_real=Yaw_scaler.inverse_transform(valid_yaw_y)
    fan_yaw_pre=Yaw_scaler.inverse_transform(yaw_pre_Y)
    plt.figure(figsize=(30,4), dpi=100)
    plt.plot(fan_yaw_real[:200],color='blue',label='real')   #真实曲线
    plt.plot(fan_yaw_pre[:200],color='orange',label='prediction') #预测曲线
    plt.legend()
    plt.show()
    evaluate(fan_yaw_real,fan_yaw_pre)
    return fan_yaw_real,fan_yaw_pre
def FanGuiHua_heave(valid_hea_y,hea_pre_Y):
    fan_heave_real=Heave_scaler.inverse_transform(valid_hea_y)
    fan_heave_pre=Heave_scaler.inverse_transform(hea_pre_Y)
    plt.figure(figsize=(30,4), dpi=100)
    plt.plot(fan_heave_real[:200],color='blue',label='real')   #真实曲线
    plt.plot(fan_heave_pre[:200],color='orange',label='prediction') #预测曲线
    plt.legend()
    plt.show()
    evaluate(fan_heave_real,fan_heave_pre)
    return fan_heave_real,fan_heave_pre
def FanGuiHua_force(valid_for_y,for_pre_Y):
    fan_force_real=Force1_scaler.inverse_transform(valid_for_y)
    fan_force_pre=Force1_scaler.inverse_transform(for_pre_Y)
    plt.figure(figsize=(30,4), dpi=100)
    plt.plot(fan_force_real,color='blue',label='real')   #真实曲线
    plt.plot(fan_force_pre,color='orange',label='prediction') #预测曲线
    plt.legend()
    plt.show()
    evaluate(fan_force_real,fan_force_pre)
    return fan_force_real,fan_force_pre

# def train_1(H,Motion):
#     Motion_scaled1 = deal_data2(np.hstack((H,Motion))[345:,:], 2, 55)
#     Motion_x1,Motion_y1 = split_sequence(Motion_scaled1, 55)

#     train_Motion_x1, train_Motion_y1, valid_Motion_x1, valid_Motion_y1 = split_train_valid(Motion_x1,Motion_y1, 7500, 10500)
#     Motion1_model1, Motion1_history1, Motion1_pre_Y1 = Model_LSTM(train_Motion_x1, train_Motion_y1, valid_Motion_x1, valid_Motion_y1, lr=0.01, epochs=epo, batch_size=50)
#     return


# In[204]:


# 输出1步长pitch
pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)

train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1 = split_train_valid(pitch_x1,pitch_y1, 7500, 10500)
pit1_model1, pit1_history1, pit1_pre_Y1 = Model_LSTM(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, lr=0.01, epochs=60, batch_size=50)

fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pit_y1,pit1_pre_Y1)


# In[207]:


print(np.hstack((fan1_pitch_real1,fan1_pitch_pre1)))


# In[208]:


np.savetxt('pitch_步长1(3.2.2).csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pit1_history1, epo=60, length=10, width=6)


# In[205]:


# 输出1步长surge
surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)

train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1 = split_train_valid(surge_x1,surge_y1, 7500, 10500)
sur1_model1, sur1_history1, sur1_pre_Y1 = Model_LSTM(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, lr=0.01, epochs=60, batch_size=50)

fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_sur_y1,sur1_pre_Y1)


# In[209]:


print(np.hstack((fan1_surge_real1,fan1_surge_pre1)))


# In[210]:


np.savetxt('surge_步长1(3.2.2).csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(sur1_history1, epo=60, length=10, width=6)


# In[206]:


# 输出1步长heave
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)

train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 7500, 10500)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)

fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)


# In[211]:


print(np.hstack((fan1_heave_real1,fan1_heave_pre1)))


# In[212]:


np.savetxt('heave_步长1(3.2.2).csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)


# In[179]:


# 不同激活函数下的H-Heave预测
# LSTM
def Model_LSTM_Activation(train_X, train_Y, valid_X, valid_Y, activation='tanh', lr=0.01, epochs=20, batch_size=256):
    model = Sequential()
    model.add(LSTM(25, activation=activation, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))  #25×5的数据输入
    model.add(Dropout(0.3))
    model.add(LSTM(100, activation=activation, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(100, activation=activation))
    model.add(Dropout(0.3))
    model.add(Dense(train_Y.shape[1])) 
#     model.add(Activation('tanh'))
    adam = Adam(lr = lr)
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, validation_data=(valid_X, valid_Y), verbose=2, shuffle=False)
    model_structure = model.summary()
    pre_Y = model.predict(valid_X)
    return model, history, pre_Y

model_sigmoid, history_sigmoid, pre_Y_sigmoid = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='sigmoid', lr=0.01, epochs=60, batch_size=256)
model_tanh, history_tanh, pre_Y_tanh = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.01, epochs=60, batch_size=256)
model_relu, history_relu, pre_Y_relu = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='relu', lr=0.01, epochs=60, batch_size=256)
model_swish, history_swish, pre_Y_swish = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='swish', lr=0.01, epochs=60, batch_size=256)


# In[183]:


# 不同激活函数下的H-Surge预测
model_sigmoid2, history_sigmoid2, pre_Y_sigmoid2 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='sigmoid', lr=0.01, epochs=60, batch_size=256)
model_tanh2, history_tanh2, pre_Y_tanh2 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.01, epochs=60, batch_size=256)
model_relu2, history_relu2, pre_Y_relu2 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='relu', lr=0.01, epochs=60, batch_size=256)
model_swish2, history_swish2, pre_Y_swish2 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='swish', lr=0.01, epochs=60, batch_size=256)


# In[184]:


# 不同激活函数下的H-Pitch预测
model_sigmoid3, history_sigmoid3, pre_Y_sigmoid3 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='sigmoid', lr=0.01, epochs=60, batch_size=256)
model_tanh3, history_tanh3, pre_Y_tanh3 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.01, epochs=60, batch_size=256)
model_relu3, history_relu3, pre_Y_relu3 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='relu', lr=0.01, epochs=60, batch_size=256)
model_swish3, history_swish3, pre_Y_swish3 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='swish', lr=0.01, epochs=60, batch_size=256)


# In[182]:


print(np.hstack((np.array(history_sigmoid.history['loss']).reshape(60,1), np.array(history_sigmoid.history['val_loss']).reshape(60,1))))
print(np.hstack((np.array(history_tanh.history['loss']).reshape(60,1), np.array(history_tanh.history['val_loss']).reshape(60,1))))
print(np.hstack((np.array(history_relu.history['loss']).reshape(60,1), np.array(history_relu.history['val_loss']).reshape(60,1))))
print(np.hstack((np.array(history_swish.history['loss']).reshape(60,1), np.array(history_swish.history['val_loss']).reshape(60,1))))

# def evaluate_Ac(valid_y,pre_Y):
#     XP1 = valid_y.copy
#     XA1 = pre_Y.copy
#     pre1 = np.trapz(abs((XP1 - (XP1.sum()/(XP1.shape[0])))).reshape(XP1.shape[0],), dx=0.1)
#     real1 = np.trapz(abs((XA1 - (XA1.sum()/(XA1.shape[0])))).reshape(XA1.shape[0],), dx=0.1)
#     Acc1 = 1 - abs(1 - (pre1/real1))
#     return Acc1

# total_loss_Act1, Acc_Act1 = evaluate_Ac(valid_hea_y1, pre_Y_sigmoid)
# total_loss_Act2, Acc_Act2 = evaluate_Ac(valid_hea_y1, pre_Y_tanh)
# total_loss_Act3, Acc_Act3 = evaluate_Ac(valid_hea_y1, pre_Y_relu)
# total_loss_Act4, Acc_Act4 = evaluate_Ac(valid_hea_y1, pre_Y_swish)


# In[185]:


# 不同学习率下的H-Heave预测
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
def evaluate_lr(valid_y,pre_Y):
    XP1 = valid_y.copy()
    XA1 = pre_Y.copy()
    pre1 = np.trapz(abs((XP1 - (XP1.sum()/(XP1.shape[0])))).reshape(XP1.shape[0],), dx=0.1)
    real1 = np.trapz(abs((XA1 - (XA1.sum()/(XA1.shape[0])))).reshape(XA1.shape[0],), dx=0.1)
    Acc1 = 1 - abs(1 - (pre1/real1))
    return Acc1


# In[186]:


model_lr1, history_lr1, pre_Y_lr1 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0, epochs=10, batch_size=56)
model_lr2, history_lr2, pre_Y_lr2 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.05, epochs=10, batch_size=56)
model_lr3, history_lr3, pre_Y_lr3 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.08, epochs=10, batch_size=56)
model_lr4, history_lr4, pre_Y_lr4 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.1, epochs=10, batch_size=56)
model_lr5, history_lr5, pre_Y_lr5 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.15, epochs=10, batch_size=56)
model_lr6, history_lr6, pre_Y_lr6 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.2, epochs=10, batch_size=56)
model_lr7, history_lr7, pre_Y_lr7 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.25, epochs=10, batch_size=56)
model_lr8, history_lr8, pre_Y_lr8 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.3, epochs=10, batch_size=56)
model_lr9, history_lr9, pre_Y_lr9 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.35, epochs=10, batch_size=56)
model_lr10, history_lr10, pre_Y_lr10 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.4, epochs=10, batch_size=56)
model_lr11, history_lr11, pre_Y_lr11 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.45, epochs=10, batch_size=56)
model_lr12, history_lr12, pre_Y_lr12 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.5, epochs=10, batch_size=56)
model_lr13, history_lr13, pre_Y_lr13 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.55, epochs=10, batch_size=56)
model_lr14, history_lr14, pre_Y_lr14 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.6, epochs=10, batch_size=56)
model_lr15, history_lr15, pre_Y_lr15 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.65, epochs=10, batch_size=56)
model_lr16, history_lr16, pre_Y_lr16 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.7, epochs=10, batch_size=56)
model_lr17, history_lr17, pre_Y_lr17 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.75, epochs=10, batch_size=56)
model_lr18, history_lr18, pre_Y_lr18 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.8, epochs=10, batch_size=56)
model_lr19, history_lr19, pre_Y_lr19 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.85, epochs=10, batch_size=56)
model_lr20, history_lr20, pre_Y_lr20 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.9, epochs=10, batch_size=56)
model_lr21, history_lr21, pre_Y_lr21 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=0.95, epochs=10, batch_size=56)


# In[187]:


model_lr22, history_lr22, pre_Y_lr22 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1, epochs=10, batch_size=56)
model_lr23, history_lr23, pre_Y_lr23 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.05, epochs=10, batch_size=56)
model_lr24, history_lr24, pre_Y_lr24 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.1, epochs=10, batch_size=56)
model_lr25, history_lr25, pre_Y_lr25 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.15, epochs=10, batch_size=56)
model_lr26, history_lr26, pre_Y_lr26 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.2, epochs=10, batch_size=56)
model_lr27, history_lr27, pre_Y_lr27 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.25, epochs=10, batch_size=56)
model_lr28, history_lr28, pre_Y_lr28 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.3, epochs=10, batch_size=56)
model_lr29, history_lr29, pre_Y_lr29 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.35, epochs=10, batch_size=56)
model_lr30, history_lr30, pre_Y_lr30 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.4, epochs=10, batch_size=56)
model_lr31, history_lr31, pre_Y_lr31 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.45, epochs=10, batch_size=56)
model_lr32, history_lr32, pre_Y_lr32 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.5, epochs=10, batch_size=56)
model_lr33, history_lr33, pre_Y_lr33 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.55, epochs=10, batch_size=56)
model_lr34, history_lr34, pre_Y_lr34 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.6, epochs=10, batch_size=56)
model_lr35, history_lr35, pre_Y_lr35 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.65, epochs=10, batch_size=56)
model_lr36, history_lr36, pre_Y_lr36 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.7, epochs=10, batch_size=56)
model_lr37, history_lr37, pre_Y_lr37 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.75, epochs=10, batch_size=56)
model_lr38, history_lr38, pre_Y_lr38 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.8, epochs=10, batch_size=56)
model_lr39, history_lr39, pre_Y_lr39 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.85, epochs=10, batch_size=56)
model_lr40, history_lr40, pre_Y_lr40 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.9, epochs=10, batch_size=56)
model_lr41, history_lr41, pre_Y_lr41 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=1.95, epochs=10, batch_size=56)
model_lr42, history_lr42, pre_Y_lr42 = Model_LSTM_Activation(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, activation='tanh', lr=2, epochs=10, batch_size=56)


# In[189]:


def lr_infor(history_loss, valid_hea_y, pre_Y_lr):
    total_loss = np.hstack((np.array(history_loss.history['loss']).reshape(10,1), np.array(history_loss.history['val_loss']).reshape(10,1)))
    Acc = evaluate_lr(valid_hea_y,pre_Y_lr)
    return total_loss, Acc

total_loss1, Acc1 = lr_infor(history_lr1, valid_hea_y1, pre_Y_lr1)
total_loss2, Acc2 = lr_infor(history_lr2, valid_hea_y1, pre_Y_lr2)
total_loss3, Acc3 = lr_infor(history_lr3, valid_hea_y1, pre_Y_lr3)
total_loss4, Acc4 = lr_infor(history_lr4, valid_hea_y1, pre_Y_lr4)
total_loss5, Acc5 = lr_infor(history_lr5, valid_hea_y1, pre_Y_lr5)
total_loss6, Acc6 = lr_infor(history_lr6, valid_hea_y1, pre_Y_lr6)
total_loss7, Acc7 = lr_infor(history_lr7, valid_hea_y1, pre_Y_lr7)
total_loss8, Acc8 = lr_infor(history_lr8, valid_hea_y1, pre_Y_lr8)
total_loss9, Acc9 = lr_infor(history_lr9, valid_hea_y1, pre_Y_lr9)
total_loss10, Acc10 = lr_infor(history_lr10, valid_hea_y1, pre_Y_lr10)
total_loss11, Acc11 = lr_infor(history_lr11, valid_hea_y1, pre_Y_lr11)
total_loss12, Acc12 = lr_infor(history_lr12, valid_hea_y1, pre_Y_lr12)
total_loss13, Acc13 = lr_infor(history_lr13, valid_hea_y1, pre_Y_lr13)
total_loss14, Acc14 = lr_infor(history_lr14, valid_hea_y1, pre_Y_lr14)
total_loss15, Acc15 = lr_infor(history_lr15, valid_hea_y1, pre_Y_lr15)
total_loss16, Acc16 = lr_infor(history_lr16, valid_hea_y1, pre_Y_lr16)
total_loss17, Acc17 = lr_infor(history_lr17, valid_hea_y1, pre_Y_lr17)
total_loss18, Acc18 = lr_infor(history_lr18, valid_hea_y1, pre_Y_lr18)
total_loss19, Acc19 = lr_infor(history_lr19, valid_hea_y1, pre_Y_lr19)
total_loss20, Acc20 = lr_infor(history_lr20, valid_hea_y1, pre_Y_lr20)
total_loss21, Acc21 = lr_infor(history_lr31, valid_hea_y1, pre_Y_lr21)
total_loss22, Acc22 = lr_infor(history_lr32, valid_hea_y1, pre_Y_lr22)
total_loss23, Acc23 = lr_infor(history_lr35, valid_hea_y1, pre_Y_lr23)
total_loss24, Acc24 = lr_infor(history_lr34, valid_hea_y1, pre_Y_lr24)
total_loss25, Acc25 = lr_infor(history_lr35, valid_hea_y1, pre_Y_lr25)
total_loss26, Acc26 = lr_infor(history_lr36, valid_hea_y1, pre_Y_lr26)
total_loss27, Acc27 = lr_infor(history_lr37, valid_hea_y1, pre_Y_lr27)
total_loss28, Acc28 = lr_infor(history_lr38, valid_hea_y1, pre_Y_lr28)
total_loss29, Acc29 = lr_infor(history_lr39, valid_hea_y1, pre_Y_lr29)
total_loss30, Acc30 = lr_infor(history_lr40, valid_hea_y1, pre_Y_lr30)
total_loss31, Acc31 = lr_infor(history_lr31, valid_hea_y1, pre_Y_lr31)
total_loss32, Acc32 = lr_infor(history_lr32, valid_hea_y1, pre_Y_lr32)
total_loss33, Acc33 = lr_infor(history_lr35, valid_hea_y1, pre_Y_lr33)
total_loss34, Acc34 = lr_infor(history_lr34, valid_hea_y1, pre_Y_lr34)
total_loss35, Acc35 = lr_infor(history_lr35, valid_hea_y1, pre_Y_lr35)
total_loss36, Acc36 = lr_infor(history_lr36, valid_hea_y1, pre_Y_lr36)
total_loss37, Acc37 = lr_infor(history_lr37, valid_hea_y1, pre_Y_lr37)
total_loss38, Acc38 = lr_infor(history_lr38, valid_hea_y1, pre_Y_lr38)
total_loss39, Acc39 = lr_infor(history_lr39, valid_hea_y1, pre_Y_lr39)
total_loss40, Acc40 = lr_infor(history_lr40, valid_hea_y1, pre_Y_lr40)
total_loss41, Acc41 = lr_infor(history_lr41, valid_hea_y1, pre_Y_lr41)
total_loss42, Acc42 = lr_infor(history_lr42, valid_hea_y1, pre_Y_lr42)

print(total_loss1, Acc1)
print(total_loss2, Acc2)
print(total_loss3, Acc3)
print(total_loss4, Acc4)
print(total_loss5, Acc5)
print(total_loss6, Acc6)
print(total_loss7, Acc7)
print(total_loss8, Acc8)
print(total_loss9, Acc9)
print(total_loss10, Acc10)
print(total_loss11, Acc11)
print(total_loss12, Acc12)
print(total_loss13, Acc13)
print(total_loss14, Acc14)
print(total_loss15, Acc15)
print(total_loss16, Acc16)
print(total_loss17, Acc17)
print(total_loss18, Acc18)
print(total_loss19, Acc19)
print(total_loss20, Acc20)
print(total_loss21, Acc21)
print(total_loss22, Acc22)
print(total_loss23, Acc23)
print(total_loss24, Acc24)
print(total_loss25, Acc25)
print(total_loss26, Acc26)
print(total_loss27, Acc27)
print(total_loss28, Acc28)
print(total_loss29, Acc29)
print(total_loss30, Acc30)
print(total_loss31, Acc31)
print(total_loss32, Acc32)
print(total_loss33, Acc33)
print(total_loss34, Acc34)
print(total_loss35, Acc35)
print(total_loss36, Acc36)
print(total_loss37, Acc37)
print(total_loss38, Acc38)
print(total_loss39, Acc39)
print(total_loss40, Acc40)
print(total_loss41, Acc41)
print(total_loss42, Acc42)


# In[190]:


Act1 = [Acc1, Acc2, Acc3, Acc4, Acc5, Acc6, Acc7, 
       Acc8, Acc9, Acc10, Acc11, Acc12, Acc13, Acc14, 
       Acc15, Acc16, Acc17, Acc18, Acc19, Acc20, Acc21, 
       Acc22, Acc23, Acc24, Acc25, Acc26, Acc27, Acc28, 
       Acc29, Acc30, Acc31, Acc32, Acc33, Acc34, Acc35, 
       Acc36, Acc37, Acc38, Acc39, Acc40, Acc41, Acc42]


# In[193]:


print(np.array((Act1)).reshape(42,1))


# In[ ]:


model_lr1, history_lr1, pre_Y_lr1 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0, epochs=10, batch_size=56)
model_lr2, history_lr2, pre_Y_lr2 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.05, epochs=10, batch_size=56)
model_lr3, history_lr3, pre_Y_lr3 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.08, epochs=10, batch_size=56)
model_lr4, history_lr4, pre_Y_lr4 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.1, epochs=10, batch_size=56)
model_lr5, history_lr5, pre_Y_lr5 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.15, epochs=10, batch_size=56)
model_lr6, history_lr6, pre_Y_lr6 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.2, epochs=10, batch_size=56)


# In[196]:


model_lr7, history_lr7, pre_Y_lr7 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.25, epochs=10, batch_size=56)
model_lr8, history_lr8, pre_Y_lr8 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.3, epochs=10, batch_size=56)
model_lr9, history_lr9, pre_Y_lr9 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.35, epochs=10, batch_size=56)
model_lr10, history_lr10, pre_Y_lr10 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.4, epochs=10, batch_size=56)
model_lr11, history_lr11, pre_Y_lr11 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.45, epochs=10, batch_size=56)
model_lr12, history_lr12, pre_Y_lr12 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.5, epochs=10, batch_size=56)
model_lr13, history_lr13, pre_Y_lr13 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.55, epochs=10, batch_size=56)
model_lr14, history_lr14, pre_Y_lr14 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.6, epochs=10, batch_size=56)
model_lr15, history_lr15, pre_Y_lr15 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.65, epochs=10, batch_size=56)
model_lr16, history_lr16, pre_Y_lr16 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.7, epochs=10, batch_size=56)
model_lr17, history_lr17, pre_Y_lr17 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.75, epochs=10, batch_size=56)
model_lr18, history_lr18, pre_Y_lr18 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.8, epochs=10, batch_size=56)
model_lr19, history_lr19, pre_Y_lr19 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.85, epochs=10, batch_size=56)
model_lr20, history_lr20, pre_Y_lr20 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.9, epochs=10, batch_size=56)
model_lr21, history_lr21, pre_Y_lr21 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=0.95, epochs=10, batch_size=56)


# In[197]:


model_lr22, history_lr22, pre_Y_lr22 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1, epochs=10, batch_size=56)
model_lr23, history_lr23, pre_Y_lr23 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.05, epochs=10, batch_size=56)
model_lr24, history_lr24, pre_Y_lr24 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.1, epochs=10, batch_size=56)
model_lr25, history_lr25, pre_Y_lr25 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.15, epochs=10, batch_size=56)
model_lr26, history_lr26, pre_Y_lr26 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.2, epochs=10, batch_size=56)
model_lr27, history_lr27, pre_Y_lr27 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.25, epochs=10, batch_size=56)
model_lr28, history_lr28, pre_Y_lr28 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.3, epochs=10, batch_size=56)
model_lr29, history_lr29, pre_Y_lr29 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.35, epochs=10, batch_size=56)
model_lr30, history_lr30, pre_Y_lr30 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.4, epochs=10, batch_size=56)
model_lr31, history_lr31, pre_Y_lr31 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.45, epochs=10, batch_size=56)
model_lr32, history_lr32, pre_Y_lr32 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.5, epochs=10, batch_size=56)
model_lr33, history_lr33, pre_Y_lr33 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.55, epochs=10, batch_size=56)
model_lr34, history_lr34, pre_Y_lr34 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.6, epochs=10, batch_size=56)
model_lr35, history_lr35, pre_Y_lr35 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.65, epochs=10, batch_size=56)
model_lr36, history_lr36, pre_Y_lr36 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.7, epochs=10, batch_size=56)
model_lr37, history_lr37, pre_Y_lr37 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.75, epochs=10, batch_size=56)
model_lr38, history_lr38, pre_Y_lr38 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.8, epochs=10, batch_size=56)
model_lr39, history_lr39, pre_Y_lr39 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.85, epochs=10, batch_size=56)
model_lr40, history_lr40, pre_Y_lr40 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.9, epochs=10, batch_size=56)
model_lr41, history_lr41, pre_Y_lr41 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=1.95, epochs=10, batch_size=56)
model_lr42, history_lr42, pre_Y_lr42 = Model_LSTM_Activation(train_sur_x1, train_sur_y1, valid_sur_x1, valid_sur_y1, activation='tanh', lr=2, epochs=10, batch_size=56)


# In[198]:


def lr_infor(history_loss, valid_hea_y, pre_Y_lr):
    total_loss = np.hstack((np.array(history_loss.history['loss']).reshape(10,1), np.array(history_loss.history['val_loss']).reshape(10,1)))
    Acc = evaluate_lr(valid_hea_y,pre_Y_lr)
    return total_loss, Acc

total_loss1, Acc1 = lr_infor(history_lr1, valid_sur_y1, pre_Y_lr1)
total_loss2, Acc2 = lr_infor(history_lr2, valid_sur_y1, pre_Y_lr2)
total_loss3, Acc3 = lr_infor(history_lr3, valid_sur_y1, pre_Y_lr3)
total_loss4, Acc4 = lr_infor(history_lr4, valid_sur_y1, pre_Y_lr4)
total_loss5, Acc5 = lr_infor(history_lr5, valid_sur_y1, pre_Y_lr5)
total_loss6, Acc6 = lr_infor(history_lr6, valid_sur_y1, pre_Y_lr6)
total_loss7, Acc7 = lr_infor(history_lr7, valid_sur_y1, pre_Y_lr7)
total_loss8, Acc8 = lr_infor(history_lr8, valid_sur_y1, pre_Y_lr8)
total_loss9, Acc9 = lr_infor(history_lr9, valid_sur_y1, pre_Y_lr9)
total_loss10, Acc10 = lr_infor(history_lr10, valid_sur_y1, pre_Y_lr10)
total_loss11, Acc11 = lr_infor(history_lr11, valid_sur_y1, pre_Y_lr11)
total_loss12, Acc12 = lr_infor(history_lr12, valid_sur_y1, pre_Y_lr12)
total_loss13, Acc13 = lr_infor(history_lr13, valid_sur_y1, pre_Y_lr13)
total_loss14, Acc14 = lr_infor(history_lr14, valid_sur_y1, pre_Y_lr14)
total_loss15, Acc15 = lr_infor(history_lr15, valid_sur_y1, pre_Y_lr15)
total_loss16, Acc16 = lr_infor(history_lr16, valid_sur_y1, pre_Y_lr16)
total_loss17, Acc17 = lr_infor(history_lr17, valid_sur_y1, pre_Y_lr17)
total_loss18, Acc18 = lr_infor(history_lr18, valid_sur_y1, pre_Y_lr18)
total_loss19, Acc19 = lr_infor(history_lr19, valid_sur_y1, pre_Y_lr19)
total_loss20, Acc20 = lr_infor(history_lr20, valid_sur_y1, pre_Y_lr20)
total_loss21, Acc21 = lr_infor(history_lr31, valid_sur_y1, pre_Y_lr21)
total_loss22, Acc22 = lr_infor(history_lr32, valid_sur_y1, pre_Y_lr22)
total_loss23, Acc23 = lr_infor(history_lr35, valid_sur_y1, pre_Y_lr23)
total_loss24, Acc24 = lr_infor(history_lr34, valid_sur_y1, pre_Y_lr24)
total_loss25, Acc25 = lr_infor(history_lr35, valid_sur_y1, pre_Y_lr25)
total_loss26, Acc26 = lr_infor(history_lr36, valid_sur_y1, pre_Y_lr26)
total_loss27, Acc27 = lr_infor(history_lr37, valid_sur_y1, pre_Y_lr27)
total_loss28, Acc28 = lr_infor(history_lr38, valid_sur_y1, pre_Y_lr28)
total_loss29, Acc29 = lr_infor(history_lr39, valid_sur_y1, pre_Y_lr29)
total_loss30, Acc30 = lr_infor(history_lr40, valid_sur_y1, pre_Y_lr30)
total_loss31, Acc31 = lr_infor(history_lr31, valid_sur_y1, pre_Y_lr31)
total_loss32, Acc32 = lr_infor(history_lr32, valid_sur_y1, pre_Y_lr32)
total_loss33, Acc33 = lr_infor(history_lr35, valid_sur_y1, pre_Y_lr33)
total_loss34, Acc34 = lr_infor(history_lr34, valid_sur_y1, pre_Y_lr34)
total_loss35, Acc35 = lr_infor(history_lr35, valid_sur_y1, pre_Y_lr35)
total_loss36, Acc36 = lr_infor(history_lr36, valid_sur_y1, pre_Y_lr36)
total_loss37, Acc37 = lr_infor(history_lr37, valid_sur_y1, pre_Y_lr37)
total_loss38, Acc38 = lr_infor(history_lr38, valid_sur_y1, pre_Y_lr38)
total_loss39, Acc39 = lr_infor(history_lr39, valid_sur_y1, pre_Y_lr39)
total_loss40, Acc40 = lr_infor(history_lr40, valid_sur_y1, pre_Y_lr40)
total_loss41, Acc41 = lr_infor(history_lr41, valid_sur_y1, pre_Y_lr41)
total_loss42, Acc42 = lr_infor(history_lr42, valid_sur_y1, pre_Y_lr42)

print(total_loss1, Acc1)
print(total_loss2, Acc2)
print(total_loss3, Acc3)
print(total_loss4, Acc4)
print(total_loss5, Acc5)
print(total_loss6, Acc6)
print(total_loss7, Acc7)
print(total_loss8, Acc8)
print(total_loss9, Acc9)
print(total_loss10, Acc10)
print(total_loss11, Acc11)
print(total_loss12, Acc12)
print(total_loss13, Acc13)
print(total_loss14, Acc14)
print(total_loss15, Acc15)
print(total_loss16, Acc16)
print(total_loss17, Acc17)
print(total_loss18, Acc18)
print(total_loss19, Acc19)
print(total_loss20, Acc20)
print(total_loss21, Acc21)
print(total_loss22, Acc22)
print(total_loss23, Acc23)
print(total_loss24, Acc24)
print(total_loss25, Acc25)
print(total_loss26, Acc26)
print(total_loss27, Acc27)
print(total_loss28, Acc28)
print(total_loss29, Acc29)
print(total_loss30, Acc30)
print(total_loss31, Acc31)
print(total_loss32, Acc32)
print(total_loss33, Acc33)
print(total_loss34, Acc34)
print(total_loss35, Acc35)
print(total_loss36, Acc36)
print(total_loss37, Acc37)
print(total_loss38, Acc38)
print(total_loss39, Acc39)
print(total_loss40, Acc40)
print(total_loss41, Acc41)
print(total_loss42, Acc42)


# In[199]:


Act2 = [Acc1, Acc2, Acc3, Acc4, Acc5, Acc6, Acc7, 
       Acc8, Acc9, Acc10, Acc11, Acc12, Acc13, Acc14, 
       Acc15, Acc16, Acc17, Acc18, Acc19, Acc20, Acc21, 
       Acc22, Acc23, Acc24, Acc25, Acc26, Acc27, Acc28, 
       Acc29, Acc30, Acc31, Acc32, Acc33, Acc34, Acc35, 
       Acc36, Acc37, Acc38, Acc39, Acc40, Acc41, Acc42]
print(np.array((Act2)).reshape(42,1))


# In[200]:


model_lr1, history_lr1, pre_Y_lr1 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0, epochs=10, batch_size=56)
model_lr2, history_lr2, pre_Y_lr2 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.05, epochs=10, batch_size=56)
model_lr3, history_lr3, pre_Y_lr3 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.08, epochs=10, batch_size=56)
model_lr4, history_lr4, pre_Y_lr4 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.1, epochs=10, batch_size=56)
model_lr5, history_lr5, pre_Y_lr5 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.15, epochs=10, batch_size=56)
model_lr6, history_lr6, pre_Y_lr6 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.2, epochs=10, batch_size=56)
model_lr7, history_lr7, pre_Y_lr7 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.25, epochs=10, batch_size=56)
model_lr8, history_lr8, pre_Y_lr8 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.3, epochs=10, batch_size=56)
model_lr9, history_lr9, pre_Y_lr9 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.35, epochs=10, batch_size=56)
model_lr10, history_lr10, pre_Y_lr10 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.4, epochs=10, batch_size=56)
model_lr11, history_lr11, pre_Y_lr11 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.45, epochs=10, batch_size=56)
model_lr12, history_lr12, pre_Y_lr12 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.5, epochs=10, batch_size=56)
model_lr13, history_lr13, pre_Y_lr13 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.55, epochs=10, batch_size=56)
model_lr14, history_lr14, pre_Y_lr14 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.6, epochs=10, batch_size=56)
model_lr15, history_lr15, pre_Y_lr15 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.65, epochs=10, batch_size=56)
model_lr16, history_lr16, pre_Y_lr16 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.7, epochs=10, batch_size=56)
model_lr17, history_lr17, pre_Y_lr17 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.75, epochs=10, batch_size=56)
model_lr18, history_lr18, pre_Y_lr18 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.8, epochs=10, batch_size=56)
model_lr19, history_lr19, pre_Y_lr19 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.85, epochs=10, batch_size=56)
model_lr20, history_lr20, pre_Y_lr20 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.9, epochs=10, batch_size=56)
model_lr21, history_lr21, pre_Y_lr21 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=0.95, epochs=10, batch_size=56)


# In[201]:


model_lr22, history_lr22, pre_Y_lr22 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1, epochs=10, batch_size=56)
model_lr23, history_lr23, pre_Y_lr23 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.05, epochs=10, batch_size=56)
model_lr24, history_lr24, pre_Y_lr24 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.1, epochs=10, batch_size=56)
model_lr25, history_lr25, pre_Y_lr25 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.15, epochs=10, batch_size=56)
model_lr26, history_lr26, pre_Y_lr26 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.2, epochs=10, batch_size=56)
model_lr27, history_lr27, pre_Y_lr27 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.25, epochs=10, batch_size=56)
model_lr28, history_lr28, pre_Y_lr28 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.3, epochs=10, batch_size=56)
model_lr29, history_lr29, pre_Y_lr29 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.35, epochs=10, batch_size=56)
model_lr30, history_lr30, pre_Y_lr30 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.4, epochs=10, batch_size=56)
model_lr31, history_lr31, pre_Y_lr31 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.45, epochs=10, batch_size=56)
model_lr32, history_lr32, pre_Y_lr32 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.5, epochs=10, batch_size=56)
model_lr33, history_lr33, pre_Y_lr33 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.55, epochs=10, batch_size=56)
model_lr34, history_lr34, pre_Y_lr34 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.6, epochs=10, batch_size=56)
model_lr35, history_lr35, pre_Y_lr35 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.65, epochs=10, batch_size=56)
model_lr36, history_lr36, pre_Y_lr36 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.7, epochs=10, batch_size=56)
model_lr37, history_lr37, pre_Y_lr37 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.75, epochs=10, batch_size=56)
model_lr38, history_lr38, pre_Y_lr38 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.8, epochs=10, batch_size=56)
model_lr39, history_lr39, pre_Y_lr39 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.85, epochs=10, batch_size=56)
model_lr40, history_lr40, pre_Y_lr40 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.9, epochs=10, batch_size=56)
model_lr41, history_lr41, pre_Y_lr41 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=1.95, epochs=10, batch_size=56)
model_lr42, history_lr42, pre_Y_lr42 = Model_LSTM_Activation(train_pit_x1, train_pit_y1, valid_pit_x1, valid_pit_y1, activation='tanh', lr=2, epochs=10, batch_size=56)


# In[202]:


def lr_infor(history_loss, valid_hea_y, pre_Y_lr):
    total_loss = np.hstack((np.array(history_loss.history['loss']).reshape(10,1), np.array(history_loss.history['val_loss']).reshape(10,1)))
    Acc = evaluate_lr(valid_hea_y,pre_Y_lr)
    return total_loss, Acc

total_loss1, Acc1 = lr_infor(history_lr1, valid_pit_y1, pre_Y_lr1)
total_loss2, Acc2 = lr_infor(history_lr2, valid_pit_y1, pre_Y_lr2)
total_loss3, Acc3 = lr_infor(history_lr3, valid_pit_y1, pre_Y_lr3)
total_loss4, Acc4 = lr_infor(history_lr4, valid_pit_y1, pre_Y_lr4)
total_loss5, Acc5 = lr_infor(history_lr5, valid_pit_y1, pre_Y_lr5)
total_loss6, Acc6 = lr_infor(history_lr6, valid_pit_y1, pre_Y_lr6)
total_loss7, Acc7 = lr_infor(history_lr7, valid_pit_y1, pre_Y_lr7)
total_loss8, Acc8 = lr_infor(history_lr8, valid_pit_y1, pre_Y_lr8)
total_loss9, Acc9 = lr_infor(history_lr9, valid_pit_y1, pre_Y_lr9)
total_loss10, Acc10 = lr_infor(history_lr10, valid_pit_y1, pre_Y_lr10)
total_loss11, Acc11 = lr_infor(history_lr11, valid_pit_y1, pre_Y_lr11)
total_loss12, Acc12 = lr_infor(history_lr12, valid_pit_y1, pre_Y_lr12)
total_loss13, Acc13 = lr_infor(history_lr13, valid_pit_y1, pre_Y_lr13)
total_loss14, Acc14 = lr_infor(history_lr14, valid_pit_y1, pre_Y_lr14)
total_loss15, Acc15 = lr_infor(history_lr15, valid_pit_y1, pre_Y_lr15)
total_loss16, Acc16 = lr_infor(history_lr16, valid_pit_y1, pre_Y_lr16)
total_loss17, Acc17 = lr_infor(history_lr17, valid_pit_y1, pre_Y_lr17)
total_loss18, Acc18 = lr_infor(history_lr18, valid_pit_y1, pre_Y_lr18)
total_loss19, Acc19 = lr_infor(history_lr19, valid_pit_y1, pre_Y_lr19)
total_loss20, Acc20 = lr_infor(history_lr20, valid_pit_y1, pre_Y_lr20)
total_loss21, Acc21 = lr_infor(history_lr31, valid_pit_y1, pre_Y_lr21)
total_loss22, Acc22 = lr_infor(history_lr32, valid_pit_y1, pre_Y_lr22)
total_loss23, Acc23 = lr_infor(history_lr35, valid_pit_y1, pre_Y_lr23)
total_loss24, Acc24 = lr_infor(history_lr34, valid_pit_y1, pre_Y_lr24)
total_loss25, Acc25 = lr_infor(history_lr35, valid_pit_y1, pre_Y_lr25)
total_loss26, Acc26 = lr_infor(history_lr36, valid_pit_y1, pre_Y_lr26)
total_loss27, Acc27 = lr_infor(history_lr37, valid_pit_y1, pre_Y_lr27)
total_loss28, Acc28 = lr_infor(history_lr38, valid_pit_y1, pre_Y_lr28)
total_loss29, Acc29 = lr_infor(history_lr39, valid_pit_y1, pre_Y_lr29)
total_loss30, Acc30 = lr_infor(history_lr40, valid_pit_y1, pre_Y_lr30)
total_loss31, Acc31 = lr_infor(history_lr31, valid_pit_y1, pre_Y_lr31)
total_loss32, Acc32 = lr_infor(history_lr32, valid_pit_y1, pre_Y_lr32)
total_loss33, Acc33 = lr_infor(history_lr35, valid_pit_y1, pre_Y_lr33)
total_loss34, Acc34 = lr_infor(history_lr34, valid_pit_y1, pre_Y_lr34)
total_loss35, Acc35 = lr_infor(history_lr35, valid_pit_y1, pre_Y_lr35)
total_loss36, Acc36 = lr_infor(history_lr36, valid_pit_y1, pre_Y_lr36)
total_loss37, Acc37 = lr_infor(history_lr37, valid_pit_y1, pre_Y_lr37)
total_loss38, Acc38 = lr_infor(history_lr38, valid_pit_y1, pre_Y_lr38)
total_loss39, Acc39 = lr_infor(history_lr39, valid_pit_y1, pre_Y_lr39)
total_loss40, Acc40 = lr_infor(history_lr40, valid_pit_y1, pre_Y_lr40)
total_loss41, Acc41 = lr_infor(history_lr41, valid_pit_y1, pre_Y_lr41)
total_loss42, Acc42 = lr_infor(history_lr42, valid_pit_y1, pre_Y_lr42)

print(total_loss1, Acc1)
print(total_loss2, Acc2)
print(total_loss3, Acc3)
print(total_loss4, Acc4)
print(total_loss5, Acc5)
print(total_loss6, Acc6)
print(total_loss7, Acc7)
print(total_loss8, Acc8)
print(total_loss9, Acc9)
print(total_loss10, Acc10)
print(total_loss11, Acc11)
print(total_loss12, Acc12)
print(total_loss13, Acc13)
print(total_loss14, Acc14)
print(total_loss15, Acc15)
print(total_loss16, Acc16)
print(total_loss17, Acc17)
print(total_loss18, Acc18)
print(total_loss19, Acc19)
print(total_loss20, Acc20)
print(total_loss21, Acc21)
print(total_loss22, Acc22)
print(total_loss23, Acc23)
print(total_loss24, Acc24)
print(total_loss25, Acc25)
print(total_loss26, Acc26)
print(total_loss27, Acc27)
print(total_loss28, Acc28)
print(total_loss29, Acc29)
print(total_loss30, Acc30)
print(total_loss31, Acc31)
print(total_loss32, Acc32)
print(total_loss33, Acc33)
print(total_loss34, Acc34)
print(total_loss35, Acc35)
print(total_loss36, Acc36)
print(total_loss37, Acc37)
print(total_loss38, Acc38)
print(total_loss39, Acc39)
print(total_loss40, Acc40)
print(total_loss41, Acc41)
print(total_loss42, Acc42)


# In[203]:


Act3 = [Acc1, Acc2, Acc3, Acc4, Acc5, Acc6, Acc7, 
       Acc8, Acc9, Acc10, Acc11, Acc12, Acc13, Acc14, 
       Acc15, Acc16, Acc17, Acc18, Acc19, Acc20, Acc21, 
       Acc22, Acc23, Acc24, Acc25, Acc26, Acc27, Acc28, 
       Acc29, Acc30, Acc31, Acc32, Acc33, Acc34, Acc35, 
       Acc36, Acc37, Acc38, Acc39, Acc40, Acc41, Acc42]
print(np.array((Act3)).reshape(42,1))


# In[ ]:





# In[ ]:





# In[ ]:




