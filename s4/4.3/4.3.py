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


# T:0
# H:1
# Surge:2
# Heave:4
# Force1:8
data_1 = pd.read_csv('t_2_11.2_50.csv')
data_1.head()


# In[3]:


data_distance = np.hstack((np.array(data_1)[:, 1:8], np.array(data_1)[:, 8:10]))
print(data_distance)
print(data_distance.shape)


# In[4]:


H_scaler = MinMaxScaler(feature_range=(-1, 1))
H = H_scaler.fit_transform(data_distance[:,0:1])
Surge_scaler = MinMaxScaler(feature_range=(-1, 1))
Surge = Surge_scaler.fit_transform(data_distance[:,1:2])
Sway_scaler = MinMaxScaler(feature_range=(-1, 1))
Sway = Sway_scaler.fit_transform(data_distance[:,2:3])
Heave_scaler = MinMaxScaler(feature_range=(-1, 1))
Heave = Heave_scaler.fit_transform(data_distance[:,3:4])
Roll_scaler = MinMaxScaler(feature_range=(-1, 1))
Roll = Roll_scaler.fit_transform(data_distance[:,4:5]*1e6)
Pitch_scaler = MinMaxScaler(feature_range=(-1, 1))
Pitch = Pitch_scaler.fit_transform(data_distance[:,5:6])
Yaw_scaler = MinMaxScaler(feature_range=(-1, 1))
Yaw = Yaw_scaler.fit_transform(data_distance[:,6:7]*1e6)
Force1_scaler = MinMaxScaler(feature_range=(-1, 1))
Force1 = Force1_scaler.fit_transform(data_distance[:,7:8])
Force2_scaler = MinMaxScaler(feature_range=(-1, 1))
Force2 = Force2_scaler.fit_transform(data_distance[:,8:9])
# zong_data = np.hstack(())


# In[5]:


# LSTM
def Model_LSTM(train_X, train_Y, valid_X, valid_Y, lr=0.01, epochs=20, batch_size=256):
    model = Sequential()
    model.add(LSTM(25, activation='tanh', input_shape=(train_X.shape[1], train_X.shape[2])))  #25×5的数据输入
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
def FanGuiHua_force1(valid_for1_y,for1_pre_Y):
    fan_force1_real=Force1_scaler.inverse_transform(valid_for1_y)
    fan_force1_pre=Force1_scaler.inverse_transform(for1_pre_Y)
    plt.figure(figsize=(30,4), dpi=100)
    plt.plot(fan_force1_real,color='blue',label='real')   #真实曲线
    plt.plot(fan_force1_pre,color='orange',label='prediction') #预测曲线
    plt.legend()
    plt.show()
    evaluate(fan_force1_real,fan_force1_pre)
    return fan_force1_real,fan_force1_pre
def FanGuiHua_force2(valid_for2_y,for2_pre_Y):
    fan_force2_real=Force2_scaler.inverse_transform(valid_for2_y)
    fan_force2_pre=Force2_scaler.inverse_transform(for2_pre_Y)
    plt.figure(figsize=(30,4), dpi=100)
    plt.plot(fan_force2_real,color='blue',label='real')   #真实曲线
    plt.plot(fan_force2_pre,color='orange',label='prediction') #预测曲线
    plt.legend()
    plt.show()
    evaluate(fan_force2_real,fan_force2_pre)
    return fan_force2_real,fan_force2_pre


# #### Force1

# In[6]:


# 训练量=n_train_times_end-n_train_times_start
def split_train_valid(data_X, data_Y, n_train_times_start, n_train_times_end, n_valid_times):
    '''
    说明：将数据集划分为训练集和验证集
    疑问：先数据切割重组再划分数据集，先划分数据集再数据切割重组，有何区别影响？
    '''
    train_x, valid_x = data_X[n_train_times_start:n_train_times_end, :], data_X[n_train_times_end:n_valid_times, :]
    train_y, valid_y = data_Y[n_train_times_start:n_train_times_end], data_Y[n_train_times_end:n_valid_times]
    train_y = train_y.reshape((n_train_times_end-n_train_times_start, 1))
    valid_y = valid_y.reshape((n_valid_times-n_train_times_end, 1))
    return train_x, train_y, valid_x, valid_y


# In[14]:


# 100
force1_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[345:,:], 4, 55)
force1_x1,force1_y1 = split_sequence(force1_scaled1, 55)
print(force1_x1.shape)
train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1 = split_train_valid(force1_x1,force1_y1, 7400, 7500, 12000)
force11_model1, force11_history1, force11_pre_Y1 = Model_LSTM(train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force1_real1,fan1_force1_pre1 = FanGuiHua_force1(valid_force1_y1,force11_pre_Y1)
np.savetxt('gui训练100_force1_步长1.csv',np.hstack((valid_force1_y1,force11_pre_Y1)),delimiter=',')
np.savetxt('训练100_force1_步长1.csv',np.hstack((fan1_force1_real1,fan1_force1_pre1)),delimiter=',')
loss_plot(force11_history1, epo=30, length=10, width=6)


# In[15]:


# 200
force1_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[345:,:], 4, 55)
force1_x1,force1_y1 = split_sequence(force1_scaled1, 55)
print(force1_x1.shape)
train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1 = split_train_valid(force1_x1,force1_y1, 7300, 7500, 12000)
force11_model1, force11_history1, force11_pre_Y1 = Model_LSTM(train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force1_real1,fan1_force1_pre1 = FanGuiHua_force1(valid_force1_y1,force11_pre_Y1)
np.savetxt('gui训练200_force1_步长1.csv',np.hstack((valid_force1_y1,force11_pre_Y1)),delimiter=',')
np.savetxt('训练200_force1_步长1.csv',np.hstack((fan1_force1_real1,fan1_force1_pre1)),delimiter=',')
loss_plot(force11_history1, epo=30, length=10, width=6)


# In[16]:


# 300
force1_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[345:,:], 4, 55)
force1_x1,force1_y1 = split_sequence(force1_scaled1, 55)
print(force1_x1.shape)
train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1 = split_train_valid(force1_x1,force1_y1, 7200, 7500, 12000)
force11_model1, force11_history1, force11_pre_Y1 = Model_LSTM(train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force1_real1,fan1_force1_pre1 = FanGuiHua_force1(valid_force1_y1,force11_pre_Y1)
np.savetxt('gui训练300_force1_步长1.csv',np.hstack((valid_force1_y1,force11_pre_Y1)),delimiter=',')
np.savetxt('训练300_force1_步长1.csv',np.hstack((fan1_force1_real1,fan1_force1_pre1)),delimiter=',')
loss_plot(force11_history1, epo=30, length=10, width=6)


# In[24]:


# 400
force1_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[345:,:], 4, 55)
force1_x1,force1_y1 = split_sequence(force1_scaled1, 55)
print(force1_x1.shape)
train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1 = split_train_valid(force1_x1,force1_y1, 7100, 7500, 12000)
force11_model1, force11_history1, force11_pre_Y1 = Model_LSTM(train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force1_real1,fan1_force1_pre1 = FanGuiHua_force1(valid_force1_y1,force11_pre_Y1)
np.savetxt('gui训练400_force1_步长1.csv',np.hstack((valid_force1_y1,force11_pre_Y1)),delimiter=',')
np.savetxt('训练400_force1_步长1.csv',np.hstack((fan1_force1_real1,fan1_force1_pre1)),delimiter=',')
loss_plot(force11_history1, epo=30, length=10, width=6)


# In[18]:


# 500
force1_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[345:,:], 4, 55)
force1_x1,force1_y1 = split_sequence(force1_scaled1, 55)
print(force1_x1.shape)
train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1 = split_train_valid(force1_x1,force1_y1, 7000, 7500, 12000)
force11_model1, force11_history1, force11_pre_Y1 = Model_LSTM(train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force1_real1,fan1_force1_pre1 = FanGuiHua_force1(valid_force1_y1,force11_pre_Y1)
np.savetxt('gui训练500_force1_步长1.csv',np.hstack((valid_force1_y1,force11_pre_Y1)),delimiter=',')
np.savetxt('训练500_force1_步长1.csv',np.hstack((fan1_force1_real1,fan1_force1_pre1)),delimiter=',')
loss_plot(force11_history1, epo=30, length=10, width=6)


# In[19]:


# 600
force1_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[345:,:], 4, 55)
force1_x1,force1_y1 = split_sequence(force1_scaled1, 55)
print(force1_x1.shape)
train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1 = split_train_valid(force1_x1,force1_y1, 6900, 7500, 12000)
force11_model1, force11_history1, force11_pre_Y1 = Model_LSTM(train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force1_real1,fan1_force1_pre1 = FanGuiHua_force1(valid_force1_y1,force11_pre_Y1)
np.savetxt('gui训练600_force1_步长1.csv',np.hstack((valid_force1_y1,force11_pre_Y1)),delimiter=',')
np.savetxt('训练600_force1_步长1.csv',np.hstack((fan1_force1_real1,fan1_force1_pre1)),delimiter=',')
loss_plot(force11_history1, epo=30, length=10, width=6)


# In[20]:


# 700
force1_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[345:,:], 4, 55)
force1_x1,force1_y1 = split_sequence(force1_scaled1, 55)
print(force1_x1.shape)
train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1 = split_train_valid(force1_x1,force1_y1, 6800, 7500, 12000)
force11_model1, force11_history1, force11_pre_Y1 = Model_LSTM(train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force1_real1,fan1_force1_pre1 = FanGuiHua_force1(valid_force1_y1,force11_pre_Y1)
np.savetxt('gui训练700_force1_步长1.csv',np.hstack((valid_force1_y1,force11_pre_Y1)),delimiter=',')
np.savetxt('训练700_force1_步长1.csv',np.hstack((fan1_force1_real1,fan1_force1_pre1)),delimiter=',')
loss_plot(force11_history1, epo=30, length=10, width=6)


# In[21]:


# 800
force1_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[345:,:], 4, 55)
force1_x1,force1_y1 = split_sequence(force1_scaled1, 55)
print(force1_x1.shape)
train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1 = split_train_valid(force1_x1,force1_y1, 6700, 7500, 12000)
force11_model1, force11_history1, force11_pre_Y1 = Model_LSTM(train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force1_real1,fan1_force1_pre1 = FanGuiHua_force1(valid_force1_y1,force11_pre_Y1)
np.savetxt('gui训练800_force1_步长1.csv',np.hstack((valid_force1_y1,force11_pre_Y1)),delimiter=',')
np.savetxt('训练800_force1_步长1.csv',np.hstack((fan1_force1_real1,fan1_force1_pre1)),delimiter=',')
loss_plot(force11_history1, epo=30, length=10, width=6)


# In[22]:


# 900
force1_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[345:,:], 4, 55)
force1_x1,force1_y1 = split_sequence(force1_scaled1, 55)
print(force1_x1.shape)
train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1 = split_train_valid(force1_x1,force1_y1, 6600, 7500, 12000)
force11_model1, force11_history1, force11_pre_Y1 = Model_LSTM(train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force1_real1,fan1_force1_pre1 = FanGuiHua_force1(valid_force1_y1,force11_pre_Y1)
np.savetxt('gui训练900_force1_步长1.csv',np.hstack((valid_force1_y1,force11_pre_Y1)),delimiter=',')
np.savetxt('训练900_force1_步长1.csv',np.hstack((fan1_force1_real1,fan1_force1_pre1)),delimiter=',')
loss_plot(force11_history1, epo=30, length=10, width=6)


# In[23]:


# 1000
force1_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[345:,:], 4, 55)
force1_x1,force1_y1 = split_sequence(force1_scaled1, 55)
print(force1_x1.shape)
train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1 = split_train_valid(force1_x1,force1_y1, 6500, 7500, 12000)
force11_model1, force11_history1, force11_pre_Y1 = Model_LSTM(train_force1_x1, train_force1_y1, valid_force1_x1, valid_force1_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force1_real1,fan1_force1_pre1 = FanGuiHua_force1(valid_force1_y1,force11_pre_Y1)
np.savetxt('gui训练1000_force1_步长1.csv',np.hstack((valid_force1_y1,force11_pre_Y1)),delimiter=',')
np.savetxt('训练1000_force1_步长1.csv',np.hstack((fan1_force1_real1,fan1_force1_pre1)),delimiter=',')
loss_plot(force11_history1, epo=30, length=10, width=6)


# #### 输入特征

# In[25]:


# 输出1步长Force(Heave、Surge、Pitch)  训练量？？？
force1100_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[300:,:], 4, 100)
force1100_x1,force1100_y1 = split_sequence(force1100_scaled1, 100)
print(force1100_x1.shape)

train_force1100_x1, train_force1100_y1, valid_force1100_x1, valid_force1100_y1 = split_train_valid(force1100_x1,force1100_y1, 6900, 7500, 12000)
force1100_model1, force1100_history1, pre_force1100_Y1 = Model_LSTM(train_force1100_x1, train_force1100_y1, valid_force1100_x1, valid_force1100_y1, lr=0.01, epochs=30, batch_size=50)

fan_force1100_real1,fan_force1100_pre1 = FanGuiHua_force1(valid_force1100_y1,pre_force1100_Y1)

np.savetxt('gui单_force1_步长1.csv',np.hstack((valid_force1100_y1,pre_force1100_Y1)),delimiter=',')
np.savetxt('单_force1_步长1.csv',np.hstack((fan_force1100_real1,fan_force1100_pre1)),delimiter=',')
loss_plot(force1100_history1, epo=30, length=10, width=6)


# In[26]:


# 输出1步长Force(H、Heave、Surge、Pitch)  训练量？？？
past_force1100_scaled1 = deal_data2(np.hstack((H,Heave,Surge,Pitch,Force1))[300:,:], 5, 100)
past_force1100_x1,past_force1100_y1 = split_sequence(past_force1100_scaled1, 100)
print(past_force1100_x1.shape)

past_train_force1100_x1, past_train_force1100_y1, past_valid_force1100_x1, past_valid_force1100_y1 = split_train_valid(past_force1100_x1,past_force1100_y1, 6900, 7500, 12000)
past_force1100_model1, past_force1100_history1, past_pre_force1100_Y1 = Model_LSTM(past_train_force1100_x1, past_train_force1100_y1, past_valid_force1100_x1, past_valid_force1100_y1, lr=0.01, epochs=30, batch_size=50)

past_fan_force1100_real1,past_fan_force1100_pre1 = FanGuiHua_force1(past_valid_force1100_y1,past_pre_force1100_Y1)

np.savetxt('gui多_force1_步长1.csv',np.hstack((past_valid_force1100_y1,past_pre_force1100_Y1)),delimiter=',')
np.savetxt('多_force1_步长1.csv',np.hstack((past_fan_force1100_real1,past_fan_force1100_pre1)),delimiter=',')
loss_plot(past_force1100_history1, epo=30, length=10, width=6)


# In[ ]:


# # 输出1步长Force(H、Heave、Surge、Pitch、过去的Force)  训练量？？？
# past_force1100_scaled1 = deal_data1(np.hstack((H,Heave,Surge,Pitch,Force1))[300:,:], 5, 100)
# past_force1100_x1,past_force1100_y1 = split_sequence(past_force1100_scaled1, 100)
# print(past_force1100_x1.shape)

# past_train_force1100_x1, past_train_force1100_y1, past_valid_force1100_x1, past_valid_force1100_y1 = split_train_valid(past_force1100_x1,past_force1100_y1, 6900, 7500, 8000)
# past_force1100_model1, past_force1100_history1, past_pre_force1100_Y1 = Model_LSTM(past_train_force1100_x1, past_train_force1100_y1, past_valid_force1100_x1, past_valid_force1100_y1, lr=0.01, epochs=60, batch_size=50)

# past_fan_force1100_real1,past_fan_force1100_pre1 = FanGuiHua_force1(past_valid_force1100_y1,past_pre_force1100_Y1)

# np.savetxt('gui多2_force1_步长1.csv',np.hstack((past_valid_force1100_y1,past_pre_force1100_Y1)),delimiter=',')
# np.savetxt('多2_force1_步长1.csv',np.hstack((past_fan_force1100_real1,past_fan_force1100_pre1)),delimiter=',')
# loss_plot(past_force1100_history1, epo=60, length=10, width=6)


# #### 输入步长与输出步长

# In[28]:


# 归一化后的两个数据集
# 波浪-运动响应
# Force
# 输入1步长输出1步长
force11_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[399:,:], 4, 1)
force11_x1,force11_y1 = split_sequence(force11_scaled1, 1)
print(force11_x1.shape)
# 输入10步长输出1步长
force110_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[390:,:], 4, 10)
force110_x1,force110_y1 = split_sequence(force110_scaled1, 10)
print(force110_x1.shape)
# 输入30步长输出1步长
force130_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[370:,:], 4, 30)
force130_x1,force130_y1 = split_sequence(force130_scaled1, 30)
print(force130_x1.shape)
# 输入50步长输出1步长
force150_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[350:,:], 4, 50)
force150_x1,force150_y1 = split_sequence(force150_scaled1, 50)
print(force150_x1.shape)
# 输入100步长输出1步长
force1100_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[300:,:], 4, 100)
force1100_x1,force1100_y1 = split_sequence(force1100_scaled1, 100)
print(force1100_x1.shape)
# 输入200步长输出1步长
force1200_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[200:,:], 4, 200)
force1200_x1,force1200_y1 = split_sequence(force1200_scaled1, 200)
print(force1200_x1.shape)

# 输入1步长输出3步长
force11_scaled3 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[397:,:], 4, 3)
force11_x3,force11_y3 = split_sequence(force11_scaled3, 1)
print(force11_x3.shape)
# 输入10步长输出3步长
force110_scaled3 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[388:,:], 4, 12)
force110_x3,force110_y3 = split_sequence(force110_scaled3, 10)
print(force110_x3.shape)
# 输入30步长输出3步长
force130_scaled3 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[368:,:], 4, 32)
force130_x3,force130_y3 = split_sequence(force130_scaled3, 30)
print(force130_x3.shape)
# 输入50步长输出3步长
force150_scaled3 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[348:,:], 4, 52)
force150_x3,force150_y3 = split_sequence(force150_scaled3, 50)
print(force150_x3.shape)
# 输入100步长输出3步长
force1100_scaled3 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[298:,:], 4, 102)
force1100_x3,force1100_y3 = split_sequence(force1100_scaled3, 100)
print(force1100_x3.shape)
# 输入200步长输出3步长
force1200_scaled3 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[198:,:], 4, 202)
force1200_x3,force1200_y3 = split_sequence(force1200_scaled3, 200)
print(force1200_x3.shape)

# 输入1步长输出5步长
force11_scaled5 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[395:,:], 4, 5)
force11_x5,force11_y5 = split_sequence(force11_scaled5, 1)
print(force11_x5.shape)
# 输入10步长输出5步长
force110_scaled5 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[386:,:], 4, 14)
force110_x5,force110_y5 = split_sequence(force110_scaled5, 10)
print(force110_x5.shape)
# 输入30步长输出5步长
force130_scaled5 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[366:,:], 4, 34)
force130_x5,force130_y5 = split_sequence(force130_scaled5, 30)
print(force130_x5.shape)
# 输入50步长输出5步长
force150_scaled5 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[346:,:], 4, 54)
force150_x5,force150_y5 = split_sequence(force150_scaled5, 50)
print(force150_x5.shape)
# 输入100步长输出5步长
force1100_scaled5 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[296:,:], 4, 104)
force1100_x5,force1100_y5 = split_sequence(force1100_scaled5, 100)
print(force1100_x5.shape)
# 输入200步长输出5步长
force1200_scaled5 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[196:,:], 4, 204)
force1200_x5,force1200_y5 = split_sequence(force1200_scaled5, 200)
print(force1200_x5.shape)

# 输入1步长输出7步长
force11_scaled7 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[393:,:], 4, 7)
force11_x7,force11_y7 = split_sequence(force11_scaled7, 1)
print(force11_x7.shape)
# 输入10步长输出7步长
force110_scaled7 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[384:,:], 4, 16)
force110_x7,force110_y7 = split_sequence(force110_scaled7, 10)
print(force110_x7.shape)
# 输入30步长输出7步长
force130_scaled7 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[364:,:], 4, 36)
force130_x7,force130_y7 = split_sequence(force130_scaled7, 30)
print(force130_x7.shape)
# 输入50步长输出7步长
force150_scaled7 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[344:,:], 4, 56)
force150_x7,force150_y7 = split_sequence(force150_scaled7, 50)
print(force150_x7.shape)
# 输入100步长输出7步长
force1100_scaled7 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[294:,:], 4, 106)
force1100_x7,force1100_y7 = split_sequence(force1100_scaled7, 100)
print(force1100_x7.shape)
# 输入200步长输出7步长
force1200_scaled7 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[194:,:], 4, 206)
force1200_x7,force1200_y7 = split_sequence(force1200_scaled7, 200)
print(force1200_x7.shape)

# 输入1步长输出9步长
force11_scaled9 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[391:,:], 4, 9)
force11_x9,force11_y9 = split_sequence(force11_scaled9, 1)
print(force11_x9.shape)
# 输入10步长输出9步长
force110_scaled9 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[382:,:], 4, 18)
force110_x9,force110_y9 = split_sequence(force110_scaled9, 10)
print(force110_x9.shape)
# 输入30步长输出9步长
force130_scaled9 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[362:,:], 4, 38)
force130_x9,force130_y9 = split_sequence(force130_scaled9, 30)
print(force130_x9.shape)
# 输入50步长输出9步长
force150_scaled9 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[342:,:], 4, 58)
force150_x9,force150_y9 = split_sequence(force150_scaled9, 50)
print(force150_x9.shape)
# 输入100步长输出9步长
force1100_scaled9 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[292:,:], 4, 108)
force1100_x9,force1100_y9 = split_sequence(force1100_scaled9, 100)
print(force1100_x9.shape)
# 输入200步长输出9步长
force1200_scaled9 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[192:,:], 4, 208)
force1200_x9,force1200_y9 = split_sequence(force1200_scaled9, 200)
print(force1200_x9.shape)

# 输入1步长输出2步长
force11_scaled2 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[398:,:], 4, 2)
force11_x2,force11_y2 = split_sequence(force11_scaled2, 1)
print(force11_x2.shape)
# 输入10步长输出2步长
force110_scaled2 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[389:,:], 4, 11)
force110_x2,force110_y2 = split_sequence(force110_scaled2, 10)
print(force110_x2.shape)
# 输入30步长输出2步长
force130_scaled2 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[369:,:], 4, 31)
force130_x2,force130_y2 = split_sequence(force130_scaled2, 30)
print(force130_x2.shape)
# 输入50步长输出2步长
force150_scaled2 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[349:,:], 4, 51)
force150_x2,force150_y2 = split_sequence(force150_scaled2, 50)
print(force150_x2.shape)
# 输入100步长输出2步长
force1100_scaled2 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[299:,:], 4, 101)
force1100_x2,force1100_y2 = split_sequence(force1100_scaled2, 100)
print(force1100_x2.shape)
# 输入200步长输出2步长
force1200_scaled2 = deal_data1(np.hstack((Heave,Surge,Pitch,Force1))[199:,:], 4, 201)
force1200_x2,force1200_y2 = split_sequence(force1200_scaled2, 200)
print(force1200_x2.shape)

# 输入1步长输出4步长
force11_scaled4 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[396:,:], 4, 4)
force11_x4,force11_y4 = split_sequence(force11_scaled4, 1)
print(force11_x4.shape)
# 输入10步长输出4步长
force110_scaled4 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[387:,:], 4, 13)
force110_x4,force110_y4 = split_sequence(force110_scaled4, 10)
print(force110_x4.shape)
# 输入30步长输出4步长
force130_scaled4 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[367:,:], 4, 33)
force130_x4,force130_y4 = split_sequence(force130_scaled4, 30)
print(force130_x4.shape)
# 输入50步长输出4步长
force150_scaled4 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[347:,:], 4, 53)
force150_x4,force150_y4 = split_sequence(force150_scaled4, 50)
print(force150_x4.shape)
# 输入100步长输出4步长
force1100_scaled4 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[297:,:], 4, 103)
force1100_x4,force1100_y4 = split_sequence(force1100_scaled4, 100)
print(force1100_x4.shape)
# 输入200步长输出4步长
force1200_scaled4 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[197:,:], 4, 203)
force1200_x4,force1200_y4 = split_sequence(force1200_scaled4, 200)
print(force1200_x4.shape)

# 输入1步长输出6步长
force11_scaled6 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[394:,:], 4, 6)
force11_x6,force11_y6 = split_sequence(force11_scaled6, 1)
print(force11_x6.shape)
# 输入10步长输出6步长
force110_scaled6 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[385:,:], 4, 15)
force110_x6,force110_y6 = split_sequence(force110_scaled6, 10)
print(force110_x6.shape)
# 输入30步长输出6步长
force130_scaled6 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[365:,:], 4, 35)
force130_x6,force130_y6 = split_sequence(force130_scaled6, 30)
print(force130_x6.shape)
# 输入50步长输出6步长
force150_scaled6 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[345:,:], 4, 55)
force150_x6,force150_y6 = split_sequence(force150_scaled6, 50)
print(force150_x6.shape)
# 输入100步长输出6步长
force1100_scaled6 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[295:,:], 4, 105)
force1100_x6,force1100_y6 = split_sequence(force1100_scaled6, 100)
print(force1100_x6.shape)
# 输入200步长输出6步长
force1200_scaled6 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[195:,:], 4, 205)
force1200_x6,force1200_y6 = split_sequence(force1200_scaled6, 200)
print(force1200_x6.shape)

# 输入1步长输出8步长
force11_scaled8 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[392:,:], 4, 8)
force11_x8,force11_y8 = split_sequence(force11_scaled8, 1)
print(force11_x8.shape)
# 输入10步长输出8步长
force110_scaled8 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[383:,:], 4, 17)
force110_x8,force110_y8 = split_sequence(force110_scaled8, 10)
print(force110_x8.shape)
# 输入30步长输出8步长
force130_scaled8 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[363:,:], 4, 37)
force130_x8,force130_y8 = split_sequence(force130_scaled8, 30)
print(force130_x8.shape)
# 输入50步长输出8步长
force150_scaled8 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[343:,:], 4, 57)
force150_x8,force150_y8 = split_sequence(force150_scaled8, 50)
print(force150_x8.shape)
# 输入100步长输出8步长
force1100_scaled8 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[293:,:], 4, 107)
force1100_x8,force1100_y8 = split_sequence(force1100_scaled8, 100)
print(force1100_x8.shape)
# 输入200步长输出8步长
force1200_scaled8 = deal_data2(np.hstack((Heave,Surge,Pitch,Force1))[193:,:], 4, 207)
force1200_x8,force1200_y8 = split_sequence(force1200_scaled8, 200)
print(force1200_x8.shape)


# In[29]:


# force1
# 1
train_force11_x1, train_force11_y1, valid_force11_x1, valid_force11_y1 = split_train_valid(force11_x1,force11_y1, 6900, 7500, 12000)
force11_model1, force11_history1, pre_force11_Y1 = Model_LSTM(train_force11_x1, train_force11_y1, valid_force11_x1, valid_force11_y1, lr=1.3, epochs=30, batch_size=50)
train_force110_x1, train_force110_y1, valid_force110_x1, valid_force110_y1 = split_train_valid(force110_x1,force110_y1, 6900, 7500, 12000)
force110_model1, force110_history1, pre_force110_Y1 = Model_LSTM(train_force110_x1, train_force110_y1, valid_force110_x1, valid_force110_y1, lr=1.3, epochs=30, batch_size=50)
train_force130_x1, train_force130_y1, valid_force130_x1, valid_force130_y1 = split_train_valid(force130_x1,force130_y1, 6900, 7500, 12000)
force130_model1, force130_history1, pre_force130_Y1 = Model_LSTM(train_force130_x1, train_force130_y1, valid_force130_x1, valid_force130_y1, lr=1.3, epochs=30, batch_size=50)
train_force150_x1, train_force150_y1, valid_force150_x1, valid_force150_y1 = split_train_valid(force150_x1,force150_y1, 6900, 7500, 12000)
force150_model1, force150_history1, pre_force150_Y1 = Model_LSTM(train_force150_x1, train_force150_y1, valid_force150_x1, valid_force150_y1, lr=1.3, epochs=30, batch_size=50)
train_force1100_x1, train_force1100_y1, valid_force1100_x1, valid_force1100_y1 = split_train_valid(force1100_x1,force1100_y1, 6900, 7500, 12000)
force1100_model1, force1100_history1, pre_force1100_Y1 = Model_LSTM(train_force1100_x1, train_force1100_y1, valid_force1100_x1, valid_force1100_y1, lr=1.3, epochs=30, batch_size=50)
train_force1200_x1, train_force1200_y1, valid_force1200_x1, valid_force1200_y1 = split_train_valid(force1200_x1,force1200_y1, 6900, 7500, 12000)
force1200_model1, force1200_history1, pre_force1200_Y1 = Model_LSTM(train_force1200_x1, train_force1200_y1, valid_force1200_x1, valid_force1200_y1, lr=1.3, epochs=30, batch_size=50)

fan_force11_real1,fan_force11_pre1 = FanGuiHua_force1(valid_force11_y1,pre_force11_Y1)
fan_force110_real1,fan_force110_pre1 = FanGuiHua_force1(valid_force110_y1,pre_force110_Y1)
fan_force130_real1,fan_force130_pre1 = FanGuiHua_force1(valid_force130_y1,pre_force130_Y1)
fan_force150_real1,fan_force150_pre1 = FanGuiHua_force1(valid_force150_y1,pre_force150_Y1)
fan_force1100_real1,fan_force1100_pre1 = FanGuiHua_force1(valid_force1100_y1,pre_force1100_Y1)
fan_force1200_real1,fan_force1200_pre1 = FanGuiHua_force1(valid_force1200_y1,pre_force1200_Y1)
np.savetxt('force1N_其他步长1.csv',np.hstack((fan_force11_real1,fan_force11_pre1,fan_force110_pre1,
                                        fan_force130_pre1,fan_force150_pre1,fan_force1100_pre1,fan_force1200_pre1)),delimiter=',')


# In[30]:


# force1
# 2
train_force11_x2, train_force11_y2, valid_force11_x2, valid_force11_y2 = split_train_valid(force11_x2,force11_y2, 6900, 7500, 12000)
force11_model2, force11_history2, pre_force11_Y2 = Model_LSTM(train_force11_x2, train_force11_y2, valid_force11_x2, valid_force11_y2, lr=0.1, epochs=30, batch_size=50)
train_force110_x2, train_force110_y2, valid_force110_x2, valid_force110_y2 = split_train_valid(force110_x2,force110_y2, 6900, 7500, 12000)
force110_model2, force110_history2, pre_force110_Y2 = Model_LSTM(train_force110_x2, train_force110_y2, valid_force110_x2, valid_force110_y2, lr=0.1, epochs=30, batch_size=50)
train_force130_x2, train_force130_y2, valid_force130_x2, valid_force130_y2 = split_train_valid(force130_x2,force130_y2, 6900, 7500, 12000)
force130_model2, force130_history2, pre_force130_Y2 = Model_LSTM(train_force130_x2, train_force130_y2, valid_force130_x2, valid_force130_y2, lr=0.1, epochs=30, batch_size=50)
train_force150_x2, train_force150_y2, valid_force150_x2, valid_force150_y2 = split_train_valid(force150_x2,force150_y2, 6900, 7500, 12000)
force150_model2, force150_history2, pre_force150_Y2 = Model_LSTM(train_force150_x2, train_force150_y2, valid_force150_x2, valid_force150_y2, lr=0.1, epochs=30, batch_size=50)
train_force1100_x2, train_force1100_y2, valid_force1100_x2, valid_force1100_y2 = split_train_valid(force1100_x2,force1100_y2, 6900, 7500, 12000)
force1100_model2, force1100_history2, pre_force1100_Y2 = Model_LSTM(train_force1100_x2, train_force1100_y2, valid_force1100_x2, valid_force1100_y2, lr=0.1, epochs=30, batch_size=50)
train_force1200_x2, train_force1200_y2, valid_force1200_x2, valid_force1200_y2 = split_train_valid(force1200_x2,force1200_y2, 6900, 7500, 12000)
force1200_model2, force1200_history2, pre_force1200_Y2 = Model_LSTM(train_force1200_x2, train_force1200_y2, valid_force1200_x2, valid_force1200_y2, lr=0.1, epochs=30, batch_size=50)

fan_force11_real2,fan_force11_pre2 = FanGuiHua_force1(valid_force11_y2,pre_force11_Y2)
fan_force110_real2,fan_force110_pre2 = FanGuiHua_force1(valid_force110_y2,pre_force110_Y2)
fan_force130_real2,fan_force130_pre2 = FanGuiHua_force1(valid_force130_y2,pre_force130_Y2)
fan_force150_real2,fan_force150_pre2 = FanGuiHua_force1(valid_force150_y2,pre_force150_Y2)
fan_force1100_real2,fan_force1100_pre2 = FanGuiHua_force1(valid_force1100_y2,pre_force1100_Y2)
fan_force1200_real2,fan_force1200_pre2 = FanGuiHua_force1(valid_force1200_y2,pre_force1200_Y2)
np.savetxt('force1N_其他步长2.csv',np.hstack((fan_force11_real2,fan_force11_pre2,fan_force110_pre2,
                                        fan_force130_pre2,fan_force150_pre2,fan_force1100_pre2,fan_force1200_pre2)),delimiter=',')


# In[31]:


# # 3
# train_force11_x3, train_force11_y3, valid_force11_x3, valid_force11_y3 = split_train_valid(force11_x3,force11_y3, 6900, 7500, 8000)
# force11_model3, force11_history3, pre_force11_Y3 = Model_LSTM(train_force11_x3, train_force11_y3, valid_force11_x3, valid_force11_y3, lr=1.3, epochs=60, batch_size=50)
# train_force110_x3, train_force110_y3, valid_force110_x3, valid_force110_y3 = split_train_valid(force110_x3,force110_y3, 6900, 7500, 8000)
# force110_model3, force110_history3, pre_force110_Y3 = Model_LSTM(train_force110_x3, train_force110_y3, valid_force110_x3, valid_force110_y3, lr=1.3, epochs=60, batch_size=50)
# train_force130_x3, train_force130_y3, valid_force130_x3, valid_force130_y3 = split_train_valid(force130_x3,force130_y3, 6900, 7500, 8000)
# force130_model3, force130_history3, pre_force130_Y3 = Model_LSTM(train_force130_x3, train_force130_y3, valid_force130_x3, valid_force130_y3, lr=1.3, epochs=60, batch_size=50)
# train_force150_x3, train_force150_y3, valid_force150_x3, valid_force150_y3 = split_train_valid(force150_x3,force150_y3, 6900, 7500, 8000)
# force150_model3, force150_history3, pre_force150_Y3 = Model_LSTM(train_force150_x3, train_force150_y3, valid_force150_x3, valid_force150_y3, lr=1.3, epochs=60, batch_size=50)
# train_force1100_x3, train_force1100_y3, valid_force1100_x3, valid_force1100_y3 = split_train_valid(force1100_x3,force1100_y3, 6900, 7500, 8000)
# force1100_model3, force1100_history3, pre_force1100_Y3 = Model_LSTM(train_force1100_x3, train_force1100_y3, valid_force1100_x3, valid_force1100_y3, lr=1.3, epochs=60, batch_size=50)
# train_force1200_x3, train_force1200_y3, valid_force1200_x3, valid_force1200_y3 = split_train_valid(force1200_x3,force1200_y3, 6900, 7500, 8000)
# force1200_model3, force1200_history3, pre_force1200_Y3 = Model_LSTM(train_force1200_x3, train_force1200_y3, valid_force1200_x3, valid_force1200_y3, lr=1.3, epochs=60, batch_size=50)

# fan_force11_real3,fan_force11_pre3 = FanGuiHua_force1(valid_force11_y3,pre_force11_Y3)
# fan_force110_real3,fan_force110_pre3 = FanGuiHua_force1(valid_force110_y3,pre_force110_Y3)
# fan_force130_real3,fan_force130_pre3 = FanGuiHua_force1(valid_force130_y3,pre_force130_Y3)
# fan_force150_real3,fan_force150_pre3 = FanGuiHua_force1(valid_force150_y3,pre_force150_Y3)
# fan_force1100_real3,fan_force1100_pre3 = FanGuiHua_force1(valid_force1100_y3,pre_force1100_Y3)
# fan_force1200_real3,fan_force1200_pre3 = FanGuiHua_force1(valid_force1200_y3,pre_force1200_Y3)
# np.savetxt('force1N_其他步长3.csv',np.hstack((fan_force11_real3,fan_force11_pre3,fan_force110_pre3,
#                                         fan_force130_pre3,fan_force150_pre3,fan_force1100_pre3,fan_force1200_pre3)),delimiter=',')


# In[32]:


# force1
# 4
train_force11_x4, train_force11_y4, valid_force11_x4, valid_force11_y4 = split_train_valid(force11_x4,force11_y4, 6900, 7500, 12000)
force11_model4, force11_history4, pre_force11_Y4 = Model_LSTM(train_force11_x4, train_force11_y4, valid_force11_x4, valid_force11_y4, lr=0.1, epochs=30, batch_size=50)
train_force110_x4, train_force110_y4, valid_force110_x4, valid_force110_y4 = split_train_valid(force110_x4,force110_y4, 6900, 7500, 12000)
force110_model4, force110_history4, pre_force110_Y4 = Model_LSTM(train_force110_x4, train_force110_y4, valid_force110_x4, valid_force110_y4, lr=0.1, epochs=30, batch_size=50)
train_force130_x4, train_force130_y4, valid_force130_x4, valid_force130_y4 = split_train_valid(force130_x4,force130_y4, 6900, 7500, 12000)
force130_model4, force130_history4, pre_force130_Y4 = Model_LSTM(train_force130_x4, train_force130_y4, valid_force130_x4, valid_force130_y4, lr=0.1, epochs=30, batch_size=50)
train_force150_x4, train_force150_y4, valid_force150_x4, valid_force150_y4 = split_train_valid(force150_x4,force150_y4, 6900, 7500, 12000)
force150_model4, force150_history4, pre_force150_Y4 = Model_LSTM(train_force150_x4, train_force150_y4, valid_force150_x4, valid_force150_y4, lr=0.1, epochs=30, batch_size=50)
train_force1100_x4, train_force1100_y4, valid_force1100_x4, valid_force1100_y4 = split_train_valid(force1100_x4,force1100_y4, 6900, 7500, 12000)
force1100_model4, force1100_history4, pre_force1100_Y4 = Model_LSTM(train_force1100_x4, train_force1100_y4, valid_force1100_x4, valid_force1100_y4, lr=0.1, epochs=30, batch_size=50)
train_force1200_x4, train_force1200_y4, valid_force1200_x4, valid_force1200_y4 = split_train_valid(force1200_x4,force1200_y4, 6900, 7500, 12000)
force1200_model4, force1200_history4, pre_force1200_Y4 = Model_LSTM(train_force1200_x4, train_force1200_y4, valid_force1200_x4, valid_force1200_y4, lr=0.1, epochs=30, batch_size=50)

fan_force11_real4,fan_force11_pre4 = FanGuiHua_force1(valid_force11_y4,pre_force11_Y4)
fan_force110_real4,fan_force110_pre4 = FanGuiHua_force1(valid_force110_y4,pre_force110_Y4)
fan_force130_real4,fan_force130_pre4 = FanGuiHua_force1(valid_force130_y4,pre_force130_Y4)
fan_force150_real4,fan_force150_pre4 = FanGuiHua_force1(valid_force150_y4,pre_force150_Y4)
fan_force1100_real4,fan_force1100_pre4 = FanGuiHua_force1(valid_force1100_y4,pre_force1100_Y4)
fan_force1200_real4,fan_force1200_pre4 = FanGuiHua_force1(valid_force1200_y4,pre_force1200_Y4)
np.savetxt('force1N_其他步长4.csv',np.hstack((fan_force11_real4,fan_force11_pre4,fan_force110_pre4,
                                        fan_force130_pre4,fan_force150_pre4,fan_force1100_pre4,fan_force1200_pre4)),delimiter=',')


# In[33]:


# # 5
# train_force11_x5, train_force11_y5, valid_force11_x5, valid_force11_y5 = split_train_valid(force11_x5,force11_y5, 6900, 7500, 8000)
# force11_model5, force11_history5, pre_force11_Y5 = Model_LSTM(train_force11_x5, train_force11_y5, valid_force11_x5, valid_force11_y5, lr=1.3, epochs=60, batch_size=50)
# train_force110_x5, train_force110_y5, valid_force110_x5, valid_force110_y5 = split_train_valid(force110_x5,force110_y5, 6900, 7500, 8000)
# force110_model5, force110_history5, pre_force110_Y5 = Model_LSTM(train_force110_x5, train_force110_y5, valid_force110_x5, valid_force110_y5, lr=1.3, epochs=60, batch_size=50)
# train_force130_x5, train_force130_y5, valid_force130_x5, valid_force130_y5 = split_train_valid(force130_x5,force130_y5, 6900, 7500, 8000)
# force130_model5, force130_history5, pre_force130_Y5 = Model_LSTM(train_force130_x5, train_force130_y5, valid_force130_x5, valid_force130_y5, lr=1.3, epochs=60, batch_size=50)
# train_force150_x5, train_force150_y5, valid_force150_x5, valid_force150_y5 = split_train_valid(force150_x5,force150_y5, 6900, 7500, 8000)
# force150_model5, force150_history5, pre_force150_Y5 = Model_LSTM(train_force150_x5, train_force150_y5, valid_force150_x5, valid_force150_y5, lr=1.3, epochs=60, batch_size=50)
# train_force1100_x5, train_force1100_y5, valid_force1100_x5, valid_force1100_y5 = split_train_valid(force1100_x5,force1100_y5, 6900, 7500, 8000)
# force1100_model5, force1100_history5, pre_force1100_Y5 = Model_LSTM(train_force1100_x5, train_force1100_y5, valid_force1100_x5, valid_force1100_y5, lr=1.3, epochs=60, batch_size=50)
# train_force1200_x5, train_force1200_y5, valid_force1200_x5, valid_force1200_y5 = split_train_valid(force1200_x5,force1200_y5, 6900, 7500, 8000)
# force1200_model5, force1200_history5, pre_force1200_Y5 = Model_LSTM(train_force1200_x5, train_force1200_y5, valid_force1200_x5, valid_force1200_y5, lr=1.3, epochs=60, batch_size=50)

# fan_force11_real5,fan_force11_pre5 = FanGuiHua_force1(valid_force11_y5,pre_force11_Y5)
# fan_force110_real5,fan_force110_pre5 = FanGuiHua_force1(valid_force110_y5,pre_force110_Y5)
# fan_force130_real5,fan_force130_pre5 = FanGuiHua_force1(valid_force130_y5,pre_force130_Y5)
# fan_force150_real5,fan_force150_pre5 = FanGuiHua_force1(valid_force150_y5,pre_force150_Y5)
# fan_force1100_real5,fan_force1100_pre5 = FanGuiHua_force1(valid_force1100_y5,pre_force1100_Y5)
# fan_force1200_real5,fan_force1200_pre5 = FanGuiHua_force1(valid_force1200_y5,pre_force1200_Y5)
# np.savetxt('force1N_其他步长5.csv',np.hstack((fan_force11_real5,fan_force11_pre5,fan_force110_pre5,
#                                         fan_force130_pre5,fan_force150_pre5,fan_force1100_pre5,fan_force1200_pre5)),delimiter=',')


# In[34]:


# force1
# 6
train_force11_x6, train_force11_y6, valid_force11_x6, valid_force11_y6 = split_train_valid(force11_x6,force11_y6, 6900, 7500, 12000)
force11_model6, force11_history6, pre_force11_Y6 = Model_LSTM(train_force11_x6, train_force11_y6, valid_force11_x6, valid_force11_y6, lr=0.1, epochs=30, batch_size=50)
train_force110_x6, train_force110_y6, valid_force110_x6, valid_force110_y6 = split_train_valid(force110_x6,force110_y6, 6900, 7500, 12000)
force110_model6, force110_history6, pre_force110_Y6 = Model_LSTM(train_force110_x6, train_force110_y6, valid_force110_x6, valid_force110_y6, lr=0.1, epochs=30, batch_size=50)
train_force130_x6, train_force130_y6, valid_force130_x6, valid_force130_y6 = split_train_valid(force130_x6,force130_y6, 6900, 7500, 12000)
force130_model6, force130_history6, pre_force130_Y6 = Model_LSTM(train_force130_x6, train_force130_y6, valid_force130_x6, valid_force130_y6, lr=0.1, epochs=30, batch_size=50)
train_force150_x6, train_force150_y6, valid_force150_x6, valid_force150_y6 = split_train_valid(force150_x6,force150_y6, 6900, 7500, 12000)
force150_model6, force150_history6, pre_force150_Y6 = Model_LSTM(train_force150_x6, train_force150_y6, valid_force150_x6, valid_force150_y6, lr=0.1, epochs=30, batch_size=50)
train_force1100_x6, train_force1100_y6, valid_force1100_x6, valid_force1100_y6 = split_train_valid(force1100_x6,force1100_y6, 6900, 7500, 12000)
force1100_model6, force1100_history6, pre_force1100_Y6 = Model_LSTM(train_force1100_x6, train_force1100_y6, valid_force1100_x6, valid_force1100_y6, lr=0.1, epochs=30, batch_size=50)
train_force1200_x6, train_force1200_y6, valid_force1200_x6, valid_force1200_y6 = split_train_valid(force1200_x6,force1200_y6, 6900, 7500, 12000)
force1200_model6, force1200_history6, pre_force1200_Y6 = Model_LSTM(train_force1200_x6, train_force1200_y6, valid_force1200_x6, valid_force1200_y6, lr=0.1, epochs=30, batch_size=50)

fan_force11_real6,fan_force11_pre6 = FanGuiHua_force1(valid_force11_y6,pre_force11_Y6)
fan_force110_real6,fan_force110_pre6 = FanGuiHua_force1(valid_force110_y6,pre_force110_Y6)
fan_force130_real6,fan_force130_pre6 = FanGuiHua_force1(valid_force130_y6,pre_force130_Y6)
fan_force150_real6,fan_force150_pre6 = FanGuiHua_force1(valid_force150_y6,pre_force150_Y6)
fan_force1100_real6,fan_force1100_pre6 = FanGuiHua_force1(valid_force1100_y6,pre_force1100_Y6)
fan_force1200_real6,fan_force1200_pre6 = FanGuiHua_force1(valid_force1200_y6,pre_force1200_Y6)
np.savetxt('force1N_其他步长6.csv',np.hstack((fan_force11_real6,fan_force11_pre6,fan_force110_pre6,
                                        fan_force130_pre6,fan_force150_pre6,fan_force1100_pre6,fan_force1200_pre6)),delimiter=',')


# In[35]:


# # 7
# train_force11_x7, train_force11_y7, valid_force11_x7, valid_force11_y7 = split_train_valid(force11_x7,force11_y7, 6900, 7500, 8000)
# force11_model7, force11_history7, pre_force11_Y7 = Model_LSTM(train_force11_x7, train_force11_y7, valid_force11_x7, valid_force11_y7, lr=1.3, epochs=60, batch_size=50)
# train_force110_x7, train_force110_y7, valid_force110_x7, valid_force110_y7 = split_train_valid(force110_x7,force110_y7, 6900, 7500, 8000)
# force110_model7, force110_history7, pre_force110_Y7 = Model_LSTM(train_force110_x7, train_force110_y7, valid_force110_x7, valid_force110_y7, lr=1.3, epochs=60, batch_size=50)
# train_force130_x7, train_force130_y7, valid_force130_x7, valid_force130_y7 = split_train_valid(force130_x7,force130_y7, 6900, 7500, 8000)
# force130_model7, force130_history7, pre_force130_Y7 = Model_LSTM(train_force130_x7, train_force130_y7, valid_force130_x7, valid_force130_y7, lr=1.3, epochs=60, batch_size=50)
# train_force150_x7, train_force150_y7, valid_force150_x7, valid_force150_y7 = split_train_valid(force150_x7,force150_y7, 6900, 7500, 8000)
# force150_model7, force150_histor7, pre_force150_Y7 = Model_LSTM(train_force150_x7, train_force150_y7, valid_force150_x7, valid_force150_y7, lr=1.3, epochs=60, batch_size=50)
# train_force1100_x7, train_force1100_y7, valid_force1100_x7, valid_force1100_y7 = split_train_valid(force1100_x7,force1100_y7, 6900, 7500, 8000)
# force1100_model7, force1100_history7, pre_force1100_Y7 = Model_LSTM(train_force1100_x7, train_force1100_y7, valid_force1100_x7, valid_force1100_y7, lr=1.3, epochs=60, batch_size=50)
# train_force1200_x7, train_force1200_y7, valid_force1200_x7, valid_force1200_y7 = split_train_valid(force1200_x7,force1200_y7, 6900, 7500, 8000)
# force1200_model7, force1200_history7, pre_force1200_Y7 = Model_LSTM(train_force1200_x7, train_force1200_y7, valid_force1200_x7, valid_force1200_y7, lr=1.3, epochs=60, batch_size=50)

# fan_force11_real7,fan_force11_pre7 = FanGuiHua_force1(valid_force11_y7,pre_force11_Y7)
# fan_force110_real7,fan_force110_pre7 = FanGuiHua_force1(valid_force110_y7,pre_force110_Y7)
# fan_force130_real7,fan_force130_pre7 = FanGuiHua_force1(valid_force130_y7,pre_force130_Y7)
# fan_force150_real7,fan_force150_pre7 = FanGuiHua_force1(valid_force150_y7,pre_force150_Y7)
# fan_force1100_real7,fan_force1100_pre7 = FanGuiHua_force1(valid_force1100_y7,pre_force1100_Y7)
# fan_force1200_real7,fan_force1200_pre7 = FanGuiHua_force1(valid_force1200_y7,pre_force1200_Y7)
# np.savetxt('force1N_其他步长7.csv',np.hstack((fan_force11_real7,fan_force11_pre7,fan_force110_pre7,
#                                         fan_force130_pre7,fan_force150_pre7,fan_force1100_pre7,fan_force1200_pre7)),delimiter=',')


# In[36]:


# force1
# 8
train_force11_x8, train_force11_y8, valid_force11_x8, valid_force11_y8 = split_train_valid(force11_x8,force11_y8, 6900, 7500, 12000)
force11_model8, force11_history8, pre_force11_Y8 = Model_LSTM(train_force11_x8, train_force11_y8, valid_force11_x8, valid_force11_y8, lr=0.1, epochs=30, batch_size=50)
train_force110_x8, train_force110_y8, valid_force110_x8, valid_force110_y8 = split_train_valid(force110_x8,force110_y8, 6900, 7500, 12000)
force110_model8, force110_history8, pre_force110_Y8 = Model_LSTM(train_force110_x8, train_force110_y8, valid_force110_x8, valid_force110_y8, lr=0.1, epochs=30, batch_size=50)
train_force130_x8, train_force130_y8, valid_force130_x8, valid_force130_y8 = split_train_valid(force130_x8,force130_y8, 6900, 7500, 12000)
force130_model8, force130_history8, pre_force130_Y8 = Model_LSTM(train_force130_x8, train_force130_y8, valid_force130_x8, valid_force130_y8, lr=0.1, epochs=30, batch_size=50)
train_force150_x8, train_force150_y8, valid_force150_x8, valid_force150_y8 = split_train_valid(force150_x8,force150_y8, 6900, 7500, 12000)
force150_model8, force150_history8, pre_force150_Y8 = Model_LSTM(train_force150_x8, train_force150_y8, valid_force150_x8, valid_force150_y8, lr=0.1, epochs=30, batch_size=50)
train_force1100_x8, train_force1100_y8, valid_force1100_x8, valid_force1100_y8 = split_train_valid(force1100_x8,force1100_y8, 6900, 7500, 12000)
force1100_model8, force1100_history8, pre_force1100_Y8 = Model_LSTM(train_force1100_x8, train_force1100_y8, valid_force1100_x8, valid_force1100_y8, lr=0.1, epochs=30, batch_size=50)
train_force1200_x8, train_force1200_y8, valid_force1200_x8, valid_force1200_y8 = split_train_valid(force1200_x8,force1200_y8, 6900, 7500, 12000)
force1200_model8, force1200_history8, pre_force1200_Y8 = Model_LSTM(train_force1200_x8, train_force1200_y8, valid_force1200_x8, valid_force1200_y8, lr=0.1, epochs=30, batch_size=50)

fan_force11_real8,fan_force11_pre8 = FanGuiHua_force1(valid_force11_y8,pre_force11_Y8)
fan_force110_real8,fan_force110_pre8 = FanGuiHua_force1(valid_force110_y8,pre_force110_Y8)
fan_force130_real8,fan_force130_pre8 = FanGuiHua_force1(valid_force130_y8,pre_force130_Y8)
fan_force150_real8,fan_force150_pre8 = FanGuiHua_force1(valid_force150_y8,pre_force150_Y8)
fan_force1100_real8,fan_force1100_pre8 = FanGuiHua_force1(valid_force1100_y8,pre_force1100_Y8)
fan_force1200_real8,fan_force1200_pre8 = FanGuiHua_force1(valid_force1200_y8,pre_force1200_Y8)
np.savetxt('force1N_其他步长8.csv',np.hstack((fan_force11_real8,fan_force11_pre8,fan_force110_pre8,
                                        fan_force130_pre8,fan_force150_pre8,fan_force1100_pre8,fan_force1200_pre8)),delimiter=',')


# In[37]:


# # 9
# train_force11_x9, train_force11_y9, valid_force11_x9, valid_force11_y9 = split_train_valid(force11_x9,force11_y9, 6900, 7500, 8000)
# force11_model9, force11_history9, pre_force11_Y9 = Model_LSTM(train_force11_x9, train_force11_y9, valid_force11_x9, valid_force11_y9, lr=1.3, epochs=60, batch_size=50)
# train_force110_x9, train_force110_y9, valid_force110_x9, valid_force110_y9 = split_train_valid(force110_x9,force110_y9, 6900, 7500, 8000)
# force110_model9, force110_history9, pre_force110_Y9 = Model_LSTM(train_force110_x9, train_force110_y9, valid_force110_x9, valid_force110_y9, lr=1.3, epochs=60, batch_size=50)
# train_force130_x9, train_force130_y9, valid_force130_x9, valid_force130_y9 = split_train_valid(force130_x9,force130_y9, 6900, 7500, 8000)
# force130_model9, force130_history9, pre_force130_Y9 = Model_LSTM(train_force130_x9, train_force130_y9, valid_force130_x9, valid_force130_y9, lr=1.3, epochs=60, batch_size=50)
# train_force150_x9, train_force150_y9, valid_force150_x9, valid_force150_y9 = split_train_valid(force150_x9,force150_y9, 6900, 7500, 8000)
# force150_model9, force150_history9, pre_force150_Y9 = Model_LSTM(train_force150_x9, train_force150_y9, valid_force150_x9, valid_force150_y9, lr=1.3, epochs=60, batch_size=50)
# train_force1100_x9, train_force1100_y9, valid_force1100_x9, valid_force1100_y9 = split_train_valid(force1100_x9,force1100_y9, 6900, 7500, 8000)
# force1100_model9, force1100_history9, pre_force1100_Y9 = Model_LSTM(train_force1100_x9, train_force1100_y9, valid_force1100_x9, valid_force1100_y9, lr=1.3, epochs=60, batch_size=50)
# train_force1200_x9, train_force1200_y9, valid_force1200_x9, valid_force1200_y9 = split_train_valid(force1200_x9,force1200_y9, 6900, 7500, 8000)
# force1200_model9, force1200_history9, pre_force1200_Y9 = Model_LSTM(train_force1200_x9, train_force1200_y9, valid_force1200_x9, valid_force1200_y9, lr=1.3, epochs=60, batch_size=50)

# fan_force11_real9,fan_force11_pre9 = FanGuiHua_force1(valid_force11_y9,pre_force11_Y9)
# fan_force110_real9,fan_force110_pre9 = FanGuiHua_force1(valid_force110_y9,pre_force110_Y9)
# fan_force130_real9,fan_force130_pre9 = FanGuiHua_force1(valid_force130_y9,pre_force130_Y9)
# fan_force150_real9,fan_force150_pre9 = FanGuiHua_force1(valid_force150_y9,pre_force150_Y9)
# fan_force1100_real9,fan_force1100_pre9 = FanGuiHua_force1(valid_force1100_y9,pre_force1100_Y9)
# fan_force1200_real9,fan_force1200_pre9 = FanGuiHua_force1(valid_force1200_y9,pre_force1200_Y9)
# np.savetxt('force1N_其他步长9.csv',np.hstack((fan_force11_real9,fan_force11_pre9,fan_force110_pre9,
#                                         fan_force130_pre9,fan_force150_pre9,fan_force1100_pre9,fan_force1200_pre9)),delimiter=',')


# In[ ]:


# # 输出1步长Force(H、Heave、Surge、Pitch)  训练量？？？
# # 输入200步长输出9步长[342:,:], 4, 58  /50
# force1200_scaled9_1 = deal_data2(np.hstack((H,Heave,Surge,Pitch,Force1))[342:,:], 5, 58)
# force1200_x9_1,force1200_y9_1 = split_sequence(force1200_scaled9_1, 50)
# print(force1200_x9_1.shape)

# past_train_force1200_x1_1, past_train_force1200_y1_1, past_valid_force1200_x1_1, past_valid_force1200_y1_1 = split_train_valid(force1200_x9_1,force1200_y9_1, 6900, 7500, 9000)
# past_force1200_model1_1, past_force1200_history1_1, past_pre_force1200_Y1_1 = Model_LSTM(past_train_force1200_x1_1, past_train_force1200_y1_1, past_valid_force1200_x1_1, past_valid_force1200_y1_1, lr=0.01, epochs=110, batch_size=50)

# past_fan_force1200_real1_1,past_fan_force1200_pre1_1 = FanGuiHua_force1(past_valid_force1200_y1_1,past_pre_force1200_Y1_1)

# np.savetxt('gui多1_force1_步长9.csv',np.hstack((past_valid_force1200_y1_1,past_pre_force1200_Y1_1)),delimiter=',')
# np.savetxt('多1_force1_步长9.csv',np.hstack((past_fan_force1200_real1_1,past_fan_force1200_pre1_1)),delimiter=',')


# In[ ]:


# loss_plot(past_force1200_history1_1, epo=110, length=10, width=6)


# In[ ]:


# # 输出1步长Force(H、Heave、Surge、Pitch、过去的Force)  训练量？？？
# # 输入200步长输出9步长
# force1200_scaled9_2 = deal_data1(np.hstack((H,Heave,Surge,Pitch,Force1))[342:,:], 5, 58)
# past_force1200_x9_2,past_force1200_y9_2 = split_sequence(force1200_scaled9_2, 50)
# print(past_force1200_x9_2.shape)

# past_train_force1200_x1_2, past_train_force1200_y1_2, past_valid_force1200_x1_2, past_valid_force1200_y1_2 = split_train_valid(past_force1200_x9_2,past_force1200_y9_2, 6900, 7500, 9000)
# past_force1200_model1_2, past_force1200_history1_2, past_pre_force1200_Y1_2 = Model_LSTM(past_train_force1200_x1_2, past_train_force1200_y1_2, past_valid_force1200_x1_2, past_valid_force1200_y1_2, lr=0.01, epochs=110, batch_size=50)

# past_fan_force1200_real1_2,past_fan_force1200_pre1_2 = FanGuiHua_force1(past_valid_force1200_y1_2,past_pre_force1200_Y1_2)

# np.savetxt('gui多2_force1_步长9.csv',np.hstack((past_valid_force1200_y1_2,past_pre_force1200_Y1_2)),delimiter=',')
# np.savetxt('多2_force1_步长9.csv',np.hstack((past_fan_force1200_real1_2,past_fan_force1200_pre1_2)),delimiter=',')
# loss_plot(past_force1200_history1_2, epo=110, length=10, width=6)


# #### Force2

# In[38]:


# # 10
# force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
# force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
# print(force2_x1.shape)
# train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 7490, 7500, 8000)
# force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=60, batch_size=50)
# fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
# np.savetxt('gui训练10_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
# np.savetxt('训练10_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
# loss_plot(force21_history1, epo=60, length=10, width=6)

# # 20
# force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
# force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
# print(force2_x1.shape)
# train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 7480, 7500, 8000)
# force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=60, batch_size=50)
# fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
# np.savetxt('gui训练20_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
# np.savetxt('训练20_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
# loss_plot(force21_history1, epo=60, length=10, width=6)

# # 30
# force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
# force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
# print(force2_x1.shape)
# train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 7470, 7500, 8000)
# force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=60, batch_size=50)
# fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
# np.savetxt('gui训练30_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
# np.savetxt('训练30_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
# loss_plot(force21_history1, epo=60, length=10, width=6)

# # 40
# force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
# force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
# print(force2_x1.shape)
# train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 7460, 7500, 8000)
# force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=60, batch_size=50)
# fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
# np.savetxt('gui训练40_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
# np.savetxt('训练40_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
# loss_plot(force21_history1, epo=60, length=10, width=6)

# # 50
# force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
# force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
# print(force2_x1.shape)
# train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 7450, 7500, 8000)
# force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=60, batch_size=50)
# fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
# np.savetxt('gui训练50_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
# np.savetxt('训练50_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
# loss_plot(force21_history1, epo=60, length=10, width=6)

# # 60
# force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
# force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
# print(force2_x1.shape)
# train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 7440, 7500, 8000)
# force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=60, batch_size=50)
# fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
# np.savetxt('gui训练60_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
# np.savetxt('训练60_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
# loss_plot(force21_history1, epo=60, length=10, width=6)

# # 70
# force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
# force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
# print(force2_x1.shape)
# train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 7430, 7500, 8000)
# force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=60, batch_size=50)
# fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
# np.savetxt('gui训练70_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
# np.savetxt('训练70_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
# loss_plot(force21_history1, epo=60, length=10, width=6)

# # 80
# force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
# force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
# print(force2_x1.shape)
# train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 7420, 7500, 8000)
# force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=60, batch_size=50)
# fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
# np.savetxt('gui训练80_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
# np.savetxt('训练80_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
# loss_plot(force21_history1, epo=60, length=10, width=6)

# # 90
# force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
# force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
# print(force2_x1.shape)
# train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 7410, 7500, 8000)
# force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=60, batch_size=50)
# fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
# np.savetxt('gui训练90_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
# np.savetxt('训练90_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
# loss_plot(force21_history1, epo=60, length=10, width=6)

# 100
force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
print(force2_x1.shape)
train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 7400, 7500, 12000)
force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
np.savetxt('gui训练100_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
np.savetxt('训练100_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
loss_plot(force21_history1, epo=30, length=10, width=6)


# In[39]:


# 200
force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
print(force2_x1.shape)
train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 7300, 7500, 12000)
force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
np.savetxt('gui训练200_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
np.savetxt('训练200_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
loss_plot(force21_history1, epo=30, length=10, width=6)


# In[40]:


# 300
force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
print(force2_x1.shape)
train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 7200, 7500, 12000)
force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
np.savetxt('gui训练300_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
np.savetxt('训练300_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
loss_plot(force21_history1, epo=30, length=10, width=6)


# In[41]:


# 400
force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
print(force2_x1.shape)
train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 7100, 7500, 12000)
force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
np.savetxt('gui训练400_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
np.savetxt('训练400_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
loss_plot(force21_history1, epo=30, length=10, width=6)


# In[42]:


# 500
force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
print(force2_x1.shape)
train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 7000, 7500, 12000)
force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
np.savetxt('gui训练500_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
np.savetxt('训练500_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
loss_plot(force21_history1, epo=30, length=10, width=6)


# In[43]:


# 600
force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
print(force2_x1.shape)
train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 6900, 7500, 12000)
force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
np.savetxt('gui训练600_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
np.savetxt('训练600_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
loss_plot(force21_history1, epo=30, length=10, width=6)


# In[44]:


# 700
force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
print(force2_x1.shape)
train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 6800, 7500, 12000)
force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
np.savetxt('gui训练700_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
np.savetxt('训练700_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
loss_plot(force21_history1, epo=30, length=10, width=6)


# In[45]:


# 800
force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
print(force2_x1.shape)
train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 6700, 7500, 12000)
force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
np.savetxt('gui训练800_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
np.savetxt('训练800_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
loss_plot(force21_history1, epo=30, length=10, width=6)


# In[46]:


# 900
force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
print(force2_x1.shape)
train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 6600, 7500, 12000)
force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
np.savetxt('gui训练900_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
np.savetxt('训练900_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
loss_plot(force21_history1, epo=30, length=10, width=6)


# In[47]:


# 1000
force2_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
force2_x1,force2_y1 = split_sequence(force2_scaled1, 55)
print(force2_x1.shape)
train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1 = split_train_valid(force2_x1,force2_y1, 6500, 7500, 12000)
force21_model1, force21_history1, force21_pre_Y1 = Model_LSTM(train_force2_x1, train_force2_y1, valid_force2_x1, valid_force2_y1, lr=0.01, epochs=30, batch_size=50)
fan1_force2_real1,fan1_force2_pre1 = FanGuiHua_force2(valid_force2_y1,force21_pre_Y1)
np.savetxt('gui训练1000_force2_步长1.csv',np.hstack((valid_force2_y1,force21_pre_Y1)),delimiter=',')
np.savetxt('训练1000_force2_步长1.csv',np.hstack((fan1_force2_real1,fan1_force2_pre1)),delimiter=',')
loss_plot(force21_history1, epo=30, length=10, width=6)


# In[49]:


# 输出1步长Force(Heave、Surge、Pitch)  训练量？？？
force2100_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[300:,:], 4, 100)
force2100_x1,force2100_y1 = split_sequence(force2100_scaled1, 100)
print(force2100_x1.shape)

train_force2100_x1, train_force2100_y1, valid_force2100_x1, valid_force2100_y1 = split_train_valid(force2100_x1,force2100_y1, 7000, 7500, 12000)
force2100_model1, force2100_history1, pre_force2100_Y1 = Model_LSTM(train_force2100_x1, train_force2100_y1, valid_force2100_x1, valid_force2100_y1, lr=0.01, epochs=30, batch_size=50)

fan_force2100_real1,fan_force2100_pre1 = FanGuiHua_force2(valid_force2100_y1,pre_force2100_Y1)

np.savetxt('gui单_force2_步长1.csv',np.hstack((valid_force2100_y1,pre_force2100_Y1)),delimiter=',')
np.savetxt('单_force2_步长1.csv',np.hstack((fan_force2100_real1,fan_force2100_pre1)),delimiter=',')
loss_plot(force2100_history1, epo=30, length=10, width=6)




# In[51]:


# 输出1步长Force(H、Heave、Surge、Pitch)  训练量？？？
past_force2100_scaled1 = deal_data2(np.hstack((H,Heave,Surge,Pitch,Force2))[300:,:], 5, 100)
past_force2100_x1,past_force2100_y1 = split_sequence(past_force2100_scaled1, 100)
print(past_force2100_x1.shape)

past_train_force2100_x1, past_train_force2100_y1, past_valid_force2100_x1, past_valid_force2100_y1 = split_train_valid(past_force2100_x1,past_force2100_y1, 7000, 7500, 12000)
past_force2100_model1, past_force2100_history1, past_pre_force2100_Y1 = Model_LSTM(past_train_force2100_x1, past_train_force2100_y1, past_valid_force2100_x1, past_valid_force2100_y1, lr=0.01, epochs=30, batch_size=50)

past_fan_force2100_real1,past_fan_force2100_pre1 = FanGuiHua_force2(past_valid_force2100_y1,past_pre_force2100_Y1)

np.savetxt('gui多_force2_步长1.csv',np.hstack((past_valid_force2100_y1,past_pre_force2100_Y1)),delimiter=',')
np.savetxt('多_force2_步长1.csv',np.hstack((past_fan_force2100_real1,past_fan_force2100_pre1)),delimiter=',')
loss_plot(past_force2100_history1, epo=30, length=10, width=6)


# In[52]:


# # 输出1步长Force(H、Heave、Surge、Pitch、过去的Force)  训练量？？？
# past_force2100_scaled1 = deal_data1(np.hstack((H,Heave,Surge,Pitch,Force2))[300:,:], 5, 100)
# past_force2100_x1,past_force2100_y1 = split_sequence(past_force2100_scaled1, 100)
# print(past_force2100_x1.shape)

# past_train_force2100_x1, past_train_force2100_y1, past_valid_force2100_x1, past_valid_force2100_y1 = split_train_valid(past_force2100_x1,past_force2100_y1, 6800, 7500, 8000)
# past_force2100_model1, past_force2100_history1, past_pre_force2100_Y1 = Model_LSTM(past_train_force2100_x1, past_train_force2100_y1, past_valid_force2100_x1, past_valid_force2100_y1, lr=0.01, epochs=60, batch_size=50)

# past_fan_force2100_real1,past_fan_force2100_pre1 = FanGuiHua_force2(past_valid_force2100_y1,past_pre_force2100_Y1)

# np.savetxt('gui多_force2_步长1.csv',np.hstack((past_valid_force2100_y1,past_pre_force2100_Y1)),delimiter=',')
# np.savetxt('多_force2_步长1.csv',np.hstack((past_fan_force2100_real1,past_fan_force2100_pre1)),delimiter=',')
# loss_plot(past_force2100_history1, epo=60, length=10, width=6)


# In[7]:


# 归一化后的两个数据集
# 波浪-运动响应
# Force
# 输入1步长输出1步长
force21_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[399:,:], 4, 1)
force21_x1,force21_y1 = split_sequence(force21_scaled1, 1)
print(force21_x1.shape)
# 输入10步长输出1步长
force210_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[390:,:], 4, 10)
force210_x1,force210_y1 = split_sequence(force210_scaled1, 10)
print(force210_x1.shape)
# 输入30步长输出1步长
force230_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[370:,:], 4, 30)
force230_x1,force230_y1 = split_sequence(force230_scaled1, 30)
print(force230_x1.shape)
# 输入50步长输出1步长
force250_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[350:,:], 4, 50)
force250_x1,force250_y1 = split_sequence(force250_scaled1, 50)
print(force250_x1.shape)
# 输入100步长输出1步长
force2100_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[300:,:], 4, 100)
force2100_x1,force2100_y1 = split_sequence(force2100_scaled1, 100)
print(force2100_x1.shape)
# 输入200步长输出1步长
force2200_scaled1 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[200:,:], 4, 200)
force2200_x1,force2200_y1 = split_sequence(force2200_scaled1, 200)
print(force2200_x1.shape)

# 输入1步长输出3步长
force21_scaled3 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[397:,:], 4, 3)
force21_x3,force21_y3 = split_sequence(force21_scaled3, 1)
print(force21_x3.shape)
# 输入10步长输出3步长
force210_scaled3 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[388:,:], 4, 12)
force210_x3,force210_y3 = split_sequence(force210_scaled3, 10)
print(force210_x3.shape)
# 输入30步长输出3步长
force230_scaled3 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[368:,:], 4, 32)
force230_x3,force230_y3 = split_sequence(force230_scaled3, 30)
print(force230_x3.shape)
# 输入50步长输出3步长
force250_scaled3 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[348:,:], 4, 52)
force250_x3,force250_y3 = split_sequence(force250_scaled3, 50)
print(force250_x3.shape)
# 输入100步长输出3步长
force2100_scaled3 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[298:,:], 4, 102)
force2100_x3,force2100_y3 = split_sequence(force2100_scaled3, 100)
print(force2100_x3.shape)
# 输入200步长输出3步长
force2200_scaled3 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[198:,:], 4, 202)
force2200_x3,force2200_y3 = split_sequence(force2200_scaled3, 200)
print(force2200_x3.shape)

# 输入1步长输出5步长
force21_scaled5 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[395:,:], 4, 5)
force21_x5,force21_y5 = split_sequence(force21_scaled5, 1)
print(force21_x5.shape)
# 输入10步长输出5步长
force210_scaled5 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[386:,:], 4, 14)
force210_x5,force210_y5 = split_sequence(force210_scaled5, 10)
print(force210_x5.shape)
# 输入30步长输出5步长
force230_scaled5 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[366:,:], 4, 34)
force230_x5,force230_y5 = split_sequence(force230_scaled5, 30)
print(force230_x5.shape)
# 输入50步长输出5步长
force250_scaled5 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[346:,:], 4, 54)
force250_x5,force250_y5 = split_sequence(force250_scaled5, 50)
print(force250_x5.shape)
# 输入100步长输出5步长
force2100_scaled5 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[296:,:], 4, 104)
force2100_x5,force2100_y5 = split_sequence(force2100_scaled5, 100)
print(force2100_x5.shape)
# 输入200步长输出5步长
force2200_scaled5 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[196:,:], 4, 204)
force2200_x5,force2200_y5 = split_sequence(force2200_scaled5, 200)
print(force2200_x5.shape)

# 输入1步长输出7步长
force21_scaled7 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[393:,:], 4, 7)
force21_x7,force21_y7 = split_sequence(force21_scaled7, 1)
print(force21_x7.shape)
# 输入10步长输出7步长
force210_scaled7 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[384:,:], 4, 16)
force210_x7,force210_y7 = split_sequence(force210_scaled7, 10)
print(force210_x7.shape)
# 输入30步长输出7步长
force230_scaled7 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[364:,:], 4, 36)
force230_x7,force230_y7 = split_sequence(force230_scaled7, 30)
print(force230_x7.shape)
# 输入50步长输出7步长
force250_scaled7 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[344:,:], 4, 56)
force250_x7,force250_y7 = split_sequence(force250_scaled7, 50)
print(force250_x7.shape)
# 输入100步长输出7步长
force2100_scaled7 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[294:,:], 4, 106)
force2100_x7,force2100_y7 = split_sequence(force2100_scaled7, 100)
print(force2100_x7.shape)
# 输入200步长输出7步长
force2200_scaled7 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[194:,:], 4, 206)
force2200_x7,force2200_y7 = split_sequence(force2200_scaled7, 200)
print(force2200_x7.shape)

# 输入1步长输出9步长
force21_scaled9 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[391:,:], 4, 9)
force21_x9,force21_y9 = split_sequence(force21_scaled9, 1)
print(force21_x9.shape)
# 输入10步长输出9步长
force210_scaled9 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[382:,:], 4, 18)
force210_x9,force210_y9 = split_sequence(force210_scaled9, 10)
print(force210_x9.shape)
# 输入30步长输出9步长
force230_scaled9 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[362:,:], 4, 38)
force230_x9,force230_y9 = split_sequence(force230_scaled9, 30)
print(force230_x9.shape)
# 输入50步长输出9步长
force250_scaled9 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[342:,:], 4, 58)
force250_x9,force250_y9 = split_sequence(force250_scaled9, 50)
print(force250_x9.shape)
# 输入100步长输出9步长
force2100_scaled9 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[292:,:], 4, 108)
force2100_x9,force2100_y9 = split_sequence(force2100_scaled9, 100)
print(force2100_x9.shape)
# 输入200步长输出9步长
force2200_scaled9 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[192:,:], 4, 208)
force2200_x9,force2200_y9 = split_sequence(force2200_scaled9, 200)
print(force2200_x9.shape)

# 输入1步长输出2步长
force21_scaled2 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[398:,:], 4, 2)
force21_x2,force21_y2 = split_sequence(force21_scaled2, 1)
print(force21_x2.shape)
# 输入10步长输出2步长
force210_scaled2 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[389:,:], 4, 11)
force210_x2,force210_y2 = split_sequence(force210_scaled2, 10)
print(force210_x2.shape)
# 输入30步长输出2步长
force230_scaled2 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[369:,:], 4, 31)
force230_x2,force230_y2 = split_sequence(force230_scaled2, 30)
print(force230_x2.shape)
# 输入50步长输出2步长
force250_scaled2 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[349:,:], 4, 51)
force250_x2,force250_y2 = split_sequence(force250_scaled2, 50)
print(force250_x2.shape)
# 输入100步长输出2步长
force2100_scaled2 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[299:,:], 4, 101)
force2100_x2,force2100_y2 = split_sequence(force2100_scaled2, 100)
print(force2100_x2.shape)
# 输入200步长输出2步长
force2200_scaled2 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[199:,:], 4, 201)
force2200_x2,force2200_y2 = split_sequence(force2200_scaled2, 200)
print(force2200_x2.shape)

# 输入1步长输出4步长
force21_scaled4 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[396:,:], 4, 4)
force21_x4,force21_y4 = split_sequence(force21_scaled4, 1)
print(force21_x4.shape)
# 输入10步长输出4步长
force210_scaled4 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[387:,:], 4, 13)
force210_x4,force210_y4 = split_sequence(force210_scaled4, 10)
print(force210_x4.shape)
# 输入30步长输出4步长
force230_scaled4 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[367:,:], 4, 33)
force230_x4,force230_y4 = split_sequence(force230_scaled4, 30)
print(force230_x4.shape)
# 输入50步长输出4步长
force250_scaled4 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[347:,:], 4, 53)
force250_x4,force250_y4 = split_sequence(force250_scaled4, 50)
print(force250_x4.shape)
# 输入100步长输出4步长
force2100_scaled4 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[297:,:], 4, 103)
force2100_x4,force2100_y4 = split_sequence(force2100_scaled4, 100)
print(force2100_x4.shape)
# 输入200步长输出4步长
force2200_scaled4 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[197:,:], 4, 203)
force2200_x4,force2200_y4 = split_sequence(force2200_scaled4, 200)
print(force2200_x4.shape)

# 输入1步长输出6步长
force21_scaled6 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[394:,:], 4, 6)
force21_x6,force21_y6 = split_sequence(force21_scaled6, 1)
print(force21_x6.shape)
# 输入10步长输出6步长
force210_scaled6 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[385:,:], 4, 15)
force210_x6,force210_y6 = split_sequence(force210_scaled6, 10)
print(force210_x6.shape)
# 输入30步长输出6步长
force230_scaled6 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[365:,:], 4, 35)
force230_x6,force230_y6 = split_sequence(force230_scaled6, 30)
print(force230_x6.shape)
# 输入50步长输出6步长
force250_scaled6 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[345:,:], 4, 55)
force250_x6,force250_y6 = split_sequence(force250_scaled6, 50)
print(force250_x6.shape)
# 输入100步长输出6步长
force2100_scaled6 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[295:,:], 4, 105)
force2100_x6,force2100_y6 = split_sequence(force2100_scaled6, 100)
print(force2100_x6.shape)
# 输入200步长输出6步长
force2200_scaled6 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[195:,:], 4, 205)
force2200_x6,force2200_y6 = split_sequence(force2200_scaled6, 200)
print(force2200_x6.shape)

# 输入1步长输出8步长
force21_scaled8 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[392:,:], 4, 8)
force21_x8,force21_y8 = split_sequence(force21_scaled8, 1)
print(force21_x8.shape)
# 输入10步长输出8步长
force210_scaled8 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[383:,:], 4, 17)
force210_x8,force210_y8 = split_sequence(force210_scaled8, 10)
print(force210_x8.shape)
# 输入30步长输出8步长
force230_scaled8 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[363:,:], 4, 37)
force230_x8,force230_y8 = split_sequence(force230_scaled8, 30)
print(force230_x8.shape)
# 输入50步长输出8步长
force250_scaled8 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[343:,:], 4, 57)
force250_x8,force250_y8 = split_sequence(force250_scaled8, 50)
print(force250_x8.shape)
# 输入100步长输出8步长
force2100_scaled8 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[293:,:], 4, 107)
force2100_x8,force2100_y8 = split_sequence(force2100_scaled8, 100)
print(force2100_x8.shape)
# 输入200步长输出8步长
force2200_scaled8 = deal_data2(np.hstack((Heave,Surge,Pitch,Force2))[193:,:], 4, 207)
force2200_x8,force2200_y8 = split_sequence(force2200_scaled8, 200)
print(force2200_x8.shape)


# In[54]:


# force2
# 1
train_force21_x1, train_force21_y1, valid_force21_x1, valid_force21_y1 = split_train_valid(force21_x1,force21_y1, 7000, 7500, 12000)
force21_model1, force21_history1, pre_force21_Y1 = Model_LSTM(train_force21_x1, train_force21_y1, valid_force21_x1, valid_force21_y1, lr=1.3, epochs=30, batch_size=50)
train_force210_x1, train_force210_y1, valid_force210_x1, valid_force210_y1 = split_train_valid(force210_x1,force210_y1, 7000, 7500, 12000)
force210_model1, force210_history1, pre_force210_Y1 = Model_LSTM(train_force210_x1, train_force210_y1, valid_force210_x1, valid_force210_y1, lr=1.3, epochs=30, batch_size=50)
train_force230_x1, train_force230_y1, valid_force230_x1, valid_force230_y1 = split_train_valid(force230_x1,force230_y1, 7000, 7500, 12000)
force230_model1, force230_history1, pre_force230_Y1 = Model_LSTM(train_force230_x1, train_force230_y1, valid_force230_x1, valid_force230_y1, lr=1.3, epochs=30, batch_size=50)
train_force250_x1, train_force250_y1, valid_force250_x1, valid_force250_y1 = split_train_valid(force250_x1,force250_y1, 7000, 7500, 12000)
force250_model1, force250_history1, pre_force250_Y1 = Model_LSTM(train_force250_x1, train_force250_y1, valid_force250_x1, valid_force250_y1, lr=1.3, epochs=30, batch_size=50)
train_force2100_x1, train_force2100_y1, valid_force2100_x1, valid_force2100_y1 = split_train_valid(force2100_x1,force2100_y1, 7000, 7500, 12000)
force2100_model1, force2100_history1, pre_force2100_Y1 = Model_LSTM(train_force2100_x1, train_force2100_y1, valid_force2100_x1, valid_force2100_y1, lr=1.3, epochs=30, batch_size=50)
train_force2200_x1, train_force2200_y1, valid_force2200_x1, valid_force2200_y1 = split_train_valid(force2200_x1,force2200_y1, 7000, 7500, 12000)
force2200_model1, force2200_history1, pre_force2200_Y1 = Model_LSTM(train_force2200_x1, train_force2200_y1, valid_force2200_x1, valid_force2200_y1, lr=1.3, epochs=30, batch_size=50)

fan_force21_real1,fan_force21_pre1 = FanGuiHua_force2(valid_force21_y1,pre_force21_Y1)
fan_force210_real1,fan_force210_pre1 = FanGuiHua_force2(valid_force210_y1,pre_force210_Y1)
fan_force230_real1,fan_force230_pre1 = FanGuiHua_force2(valid_force230_y1,pre_force230_Y1)
fan_force250_real1,fan_force250_pre1 = FanGuiHua_force2(valid_force250_y1,pre_force250_Y1)
fan_force2100_real1,fan_force2100_pre1 = FanGuiHua_force2(valid_force2100_y1,pre_force2100_Y1)
fan_force2200_real1,fan_force2200_pre1 = FanGuiHua_force2(valid_force2200_y1,pre_force2200_Y1)
np.savetxt('force2N_其他步长1.csv',np.hstack((fan_force21_real1,fan_force21_pre1,fan_force210_pre1,
                                        fan_force230_pre1,fan_force250_pre1,fan_force2100_pre1,fan_force2200_pre1)),delimiter=',')


# In[55]:


# force2
# 2
train_force21_x2, train_force21_y2, valid_force21_x2, valid_force21_y2 = split_train_valid(force21_x2,force21_y2, 7000, 7500, 12000)
force21_model2, force21_history2, pre_force21_Y2 = Model_LSTM(train_force21_x2, train_force21_y2, valid_force21_x2, valid_force21_y2, lr=0.1, epochs=30, batch_size=50)
train_force210_x2, train_force210_y2, valid_force210_x2, valid_force210_y2 = split_train_valid(force210_x2,force210_y2, 7000, 7500, 12000)
force210_model2, force210_history2, pre_force210_Y2 = Model_LSTM(train_force210_x2, train_force210_y2, valid_force210_x2, valid_force210_y2, lr=0.1, epochs=30, batch_size=50)
train_force230_x2, train_force230_y2, valid_force230_x2, valid_force230_y2 = split_train_valid(force230_x2,force230_y2, 7000, 7500, 12000)
force230_model2, force230_history2, pre_force230_Y2 = Model_LSTM(train_force230_x2, train_force230_y2, valid_force230_x2, valid_force230_y2, lr=0.1, epochs=30, batch_size=50)
train_force250_x2, train_force250_y2, valid_force250_x2, valid_force250_y2 = split_train_valid(force250_x2,force250_y2, 7000, 7500, 12000)
force250_model2, force250_history2, pre_force250_Y2 = Model_LSTM(train_force250_x2, train_force250_y2, valid_force250_x2, valid_force250_y2, lr=0.1, epochs=30, batch_size=50)
train_force2100_x2, train_force2100_y2, valid_force2100_x2, valid_force2100_y2 = split_train_valid(force2100_x2,force2100_y2, 7000, 7500, 12000)
force2100_model2, force2100_history2, pre_force2100_Y2 = Model_LSTM(train_force2100_x2, train_force2100_y2, valid_force2100_x2, valid_force2100_y2, lr=0.1, epochs=30, batch_size=50)
train_force2200_x2, train_force2200_y2, valid_force2200_x2, valid_force2200_y2 = split_train_valid(force2200_x2,force2200_y2, 7000, 7500, 12000)
force2200_model2, force2200_history2, pre_force2200_Y2 = Model_LSTM(train_force2200_x2, train_force2200_y2, valid_force2200_x2, valid_force2200_y2, lr=0.1, epochs=30, batch_size=50)

fan_force21_real2,fan_force21_pre2 = FanGuiHua_force2(valid_force21_y2,pre_force21_Y2)
fan_force210_real2,fan_force210_pre2 = FanGuiHua_force2(valid_force210_y2,pre_force210_Y2)
fan_force230_real2,fan_force230_pre2 = FanGuiHua_force2(valid_force230_y2,pre_force230_Y2)
fan_force250_real2,fan_force250_pre2 = FanGuiHua_force2(valid_force250_y2,pre_force250_Y2)
fan_force2100_real2,fan_force2100_pre2 = FanGuiHua_force2(valid_force2100_y2,pre_force2100_Y2)
fan_force2200_real2,fan_force2200_pre2 = FanGuiHua_force2(valid_force2200_y2,pre_force2200_Y2)
np.savetxt('force2N_其他步长2.csv',np.hstack((fan_force21_real2,fan_force21_pre2,fan_force210_pre2,
                                        fan_force230_pre2,fan_force250_pre2,fan_force2100_pre2,fan_force2200_pre2)),delimiter=',')


# In[56]:


# # 3
# train_force21_x3, train_force21_y3, valid_force21_x3, valid_force21_y3 = split_train_valid(force21_x3,force21_y3, 6800, 7500, 8000)
# force21_model3, force21_history3, pre_force21_Y3 = Model_LSTM(train_force21_x3, train_force21_y3, valid_force21_x3, valid_force21_y3, lr=1.3, epochs=60, batch_size=50)
# train_force210_x3, train_force210_y3, valid_force210_x3, valid_force210_y3 = split_train_valid(force210_x3,force210_y3, 6800, 7500, 8000)
# force210_model3, force210_history3, pre_force210_Y3 = Model_LSTM(train_force210_x3, train_force210_y3, valid_force210_x3, valid_force210_y3, lr=1.3, epochs=60, batch_size=50)
# train_force230_x3, train_force230_y3, valid_force230_x3, valid_force230_y3 = split_train_valid(force230_x3,force230_y3, 6800, 7500, 8000)
# force230_model3, force230_history3, pre_force230_Y3 = Model_LSTM(train_force230_x3, train_force230_y3, valid_force230_x3, valid_force230_y3, lr=1.3, epochs=60, batch_size=50)
# train_force250_x3, train_force250_y3, valid_force250_x3, valid_force250_y3 = split_train_valid(force250_x3,force250_y3, 6800, 7500, 8000)
# force250_model3, force250_history3, pre_force250_Y3 = Model_LSTM(train_force250_x3, train_force250_y3, valid_force250_x3, valid_force250_y3, lr=1.3, epochs=60, batch_size=50)
# train_force2100_x3, train_force2100_y3, valid_force2100_x3, valid_force2100_y3 = split_train_valid(force2100_x3,force2100_y3, 6800, 7500, 8000)
# force2100_model3, force2100_history3, pre_force2100_Y3 = Model_LSTM(train_force2100_x3, train_force2100_y3, valid_force2100_x3, valid_force2100_y3, lr=1.3, epochs=60, batch_size=50)
# train_force2200_x3, train_force2200_y3, valid_force2200_x3, valid_force2200_y3 = split_train_valid(force2200_x3,force2200_y3, 6800, 7500, 8000)
# force2200_model3, force2200_history3, pre_force2200_Y3 = Model_LSTM(train_force2200_x3, train_force2200_y3, valid_force2200_x3, valid_force2200_y3, lr=1.3, epochs=60, batch_size=50)

# fan_force21_real3,fan_force21_pre3 = FanGuiHua_force2(valid_force21_y3,pre_force21_Y3)
# fan_force210_real3,fan_force210_pre3 = FanGuiHua_force2(valid_force210_y3,pre_force210_Y3)
# fan_force230_real3,fan_force230_pre3 = FanGuiHua_force2(valid_force230_y3,pre_force230_Y3)
# fan_force250_real3,fan_force250_pre3 = FanGuiHua_force2(valid_force250_y3,pre_force250_Y3)
# fan_force2100_real3,fan_force2100_pre3 = FanGuiHua_force2(valid_force2100_y3,pre_force2100_Y3)
# fan_force2200_real3,fan_force2200_pre3 = FanGuiHua_force2(valid_force2200_y3,pre_force2200_Y3)
# np.savetxt('force2N_其他步长3.csv',np.hstack((fan_force21_real3,fan_force21_pre3,fan_force210_pre3,
#                                         fan_force230_pre3,fan_force250_pre3,fan_force2100_pre3,fan_force2200_pre3)),delimiter=',')


# In[57]:


# force2
# 4
train_force21_x4, train_force21_y4, valid_force21_x4, valid_force21_y4 = split_train_valid(force21_x4,force21_y4, 7000, 7500, 12000)
force21_model4, force21_history4, pre_force21_Y4 = Model_LSTM(train_force21_x4, train_force21_y4, valid_force21_x4, valid_force21_y4, lr=0.1, epochs=30, batch_size=50)
train_force210_x4, train_force210_y4, valid_force210_x4, valid_force210_y4 = split_train_valid(force210_x4,force210_y4, 7000, 7500, 12000)
force210_model4, force210_history4, pre_force210_Y4 = Model_LSTM(train_force210_x4, train_force210_y4, valid_force210_x4, valid_force210_y4, lr=0.1, epochs=30, batch_size=50)
train_force230_x4, train_force230_y4, valid_force230_x4, valid_force230_y4 = split_train_valid(force230_x4,force230_y4, 7000, 7500, 12000)
force230_model4, force230_history4, pre_force230_Y4 = Model_LSTM(train_force230_x4, train_force230_y4, valid_force230_x4, valid_force230_y4, lr=0.1, epochs=30, batch_size=50)
train_force250_x4, train_force250_y4, valid_force250_x4, valid_force250_y4 = split_train_valid(force250_x4,force250_y4, 7000, 7500, 12000)
force250_model4, force250_history4, pre_force250_Y4 = Model_LSTM(train_force250_x4, train_force250_y4, valid_force250_x4, valid_force250_y4, lr=0.1, epochs=30, batch_size=50)
train_force2100_x4, train_force2100_y4, valid_force2100_x4, valid_force2100_y4 = split_train_valid(force2100_x4,force2100_y4, 7000, 7500, 12000)
force2100_model4, force2100_history4, pre_force2100_Y4 = Model_LSTM(train_force2100_x4, train_force2100_y4, valid_force2100_x4, valid_force2100_y4, lr=0.1, epochs=30, batch_size=50)
train_force2200_x4, train_force2200_y4, valid_force2200_x4, valid_force2200_y4 = split_train_valid(force2200_x4,force2200_y4, 7000, 7500, 12000)
force2200_model4, force2200_history4, pre_force2200_Y4 = Model_LSTM(train_force2200_x4, train_force2200_y4, valid_force2200_x4, valid_force2200_y4, lr=0.1, epochs=30, batch_size=50)

fan_force21_real4,fan_force21_pre4 = FanGuiHua_force2(valid_force21_y4,pre_force21_Y4)
fan_force210_real4,fan_force210_pre4 = FanGuiHua_force2(valid_force210_y4,pre_force210_Y4)
fan_force230_real4,fan_force230_pre4 = FanGuiHua_force2(valid_force230_y4,pre_force230_Y4)
fan_force250_real4,fan_force250_pre4 = FanGuiHua_force2(valid_force250_y4,pre_force250_Y4)
fan_force2100_real4,fan_force2100_pre4 = FanGuiHua_force2(valid_force2100_y4,pre_force2100_Y4)
fan_force2200_real4,fan_force2200_pre4 = FanGuiHua_force2(valid_force2200_y4,pre_force2200_Y4)
np.savetxt('force2N_其他步长4.csv',np.hstack((fan_force21_real4,fan_force21_pre4,fan_force210_pre4,
                                        fan_force230_pre4,fan_force250_pre4,fan_force2100_pre4,fan_force2200_pre4)),delimiter=',')


# In[ ]:


# # 5
# train_force21_x5, train_force21_y5, valid_force21_x5, valid_force21_y5 = split_train_valid(force21_x5,force21_y5, 6800, 7500, 8000)
# force21_model5, force21_history5, pre_force21_Y5 = Model_LSTM(train_force21_x5, train_force21_y5, valid_force21_x5, valid_force21_y5, lr=1.3, epochs=60, batch_size=50)
# train_force210_x5, train_force210_y5, valid_force210_x5, valid_force210_y5 = split_train_valid(force210_x5,force210_y5, 6800, 7500, 8000)
# force210_model5, force210_history5, pre_force210_Y5 = Model_LSTM(train_force210_x5, train_force210_y5, valid_force210_x5, valid_force210_y5, lr=1.3, epochs=60, batch_size=50)
# train_force230_x5, train_force230_y5, valid_force230_x5, valid_force230_y5 = split_train_valid(force230_x5,force230_y5, 6800, 7500, 8000)
# force230_model5, force230_history5, pre_force230_Y5 = Model_LSTM(train_force230_x5, train_force230_y5, valid_force230_x5, valid_force230_y5, lr=1.3, epochs=60, batch_size=50)
# train_force250_x5, train_force250_y5, valid_force250_x5, valid_force250_y5 = split_train_valid(force250_x5,force250_y5, 6800, 7500, 8000)
# force250_model5, force250_history5, pre_force250_Y5 = Model_LSTM(train_force250_x5, train_force250_y5, valid_force250_x5, valid_force250_y5, lr=1.3, epochs=60, batch_size=50)
# train_force2100_x5, train_force2100_y5, valid_force2100_x5, valid_force2100_y5 = split_train_valid(force2100_x5,force2100_y5, 6800, 7500, 8000)
# force2100_model5, force2100_history5, pre_force2100_Y5 = Model_LSTM(train_force2100_x5, train_force2100_y5, valid_force2100_x5, valid_force2100_y5, lr=1.3, epochs=60, batch_size=50)
# train_force2200_x5, train_force2200_y5, valid_force2200_x5, valid_force2200_y5 = split_train_valid(force2200_x5,force2200_y5, 6800, 7500, 8000)
# force2200_model5, force2200_history5, pre_force2200_Y5 = Model_LSTM(train_force2200_x5, train_force2200_y5, valid_force2200_x5, valid_force2200_y5, lr=1.3, epochs=60, batch_size=50)

# fan_force21_real5,fan_force21_pre5 = FanGuiHua_force2(valid_force21_y5,pre_force21_Y5)
# fan_force210_real5,fan_force210_pre5 = FanGuiHua_force2(valid_force210_y5,pre_force210_Y5)
# fan_force230_real5,fan_force230_pre5 = FanGuiHua_force2(valid_force230_y5,pre_force230_Y5)
# fan_force250_real5,fan_force250_pre5 = FanGuiHua_force2(valid_force250_y5,pre_force250_Y5)
# fan_force2100_real5,fan_force2100_pre5 = FanGuiHua_force2(valid_force2100_y5,pre_force2100_Y5)
# fan_force2200_real5,fan_force2200_pre5 = FanGuiHua_force2(valid_force2200_y5,pre_force2200_Y5)
# np.savetxt('force2N_其他步长5.csv',np.hstack((fan_force21_real5,fan_force21_pre5,fan_force210_pre5,
#                                         fan_force230_pre5,fan_force250_pre5,fan_force2100_pre5,fan_force2200_pre5)),delimiter=',')


# In[8]:


# force2
# 6
train_force21_x6, train_force21_y6, valid_force21_x6, valid_force21_y6 = split_train_valid(force21_x6,force21_y6, 7000, 7500, 12000)
force21_model6, force21_history6, pre_force21_Y6 = Model_LSTM(train_force21_x6, train_force21_y6, valid_force21_x6, valid_force21_y6, lr=0.1, epochs=30, batch_size=50)
train_force210_x6, train_force210_y6, valid_force210_x6, valid_force210_y6 = split_train_valid(force210_x6,force210_y6, 7000, 7500, 12000)
force210_model6, force210_history6, pre_force210_Y6 = Model_LSTM(train_force210_x6, train_force210_y6, valid_force210_x6, valid_force210_y6, lr=0.1, epochs=30, batch_size=50)
train_force230_x6, train_force230_y6, valid_force230_x6, valid_force230_y6 = split_train_valid(force230_x6,force230_y6, 7000, 7500, 12000)
force230_model6, force230_history6, pre_force230_Y6 = Model_LSTM(train_force230_x6, train_force230_y6, valid_force230_x6, valid_force230_y6, lr=0.1, epochs=30, batch_size=50)
train_force250_x6, train_force250_y6, valid_force250_x6, valid_force250_y6 = split_train_valid(force250_x6,force250_y6, 7000, 7500, 12000)
force250_model6, force250_history6, pre_force250_Y6 = Model_LSTM(train_force250_x6, train_force250_y6, valid_force250_x6, valid_force250_y6, lr=0.1, epochs=30, batch_size=50)
train_force2100_x6, train_force2100_y6, valid_force2100_x6, valid_force2100_y6 = split_train_valid(force2100_x6,force2100_y6, 7000, 7500, 12000)
force2100_model6, force2100_history6, pre_force2100_Y6 = Model_LSTM(train_force2100_x6, train_force2100_y6, valid_force2100_x6, valid_force2100_y6, lr=0.1, epochs=30, batch_size=50)
train_force2200_x6, train_force2200_y6, valid_force2200_x6, valid_force2200_y6 = split_train_valid(force2200_x6,force2200_y6, 7000, 7500, 12000)
force2200_model6, force2200_history6, pre_force2200_Y6 = Model_LSTM(train_force2200_x6, train_force2200_y6, valid_force2200_x6, valid_force2200_y6, lr=0.1, epochs=30, batch_size=50)

fan_force21_real6,fan_force21_pre6 = FanGuiHua_force2(valid_force21_y6,pre_force21_Y6)
fan_force210_real6,fan_force210_pre6 = FanGuiHua_force2(valid_force210_y6,pre_force210_Y6)
fan_force230_real6,fan_force230_pre6 = FanGuiHua_force2(valid_force230_y6,pre_force230_Y6)
fan_force250_real6,fan_force250_pre6 = FanGuiHua_force2(valid_force250_y6,pre_force250_Y6)
fan_force2100_real6,fan_force2100_pre6 = FanGuiHua_force2(valid_force2100_y6,pre_force2100_Y6)
fan_force2200_real6,fan_force2200_pre6 = FanGuiHua_force2(valid_force2200_y6,pre_force2200_Y6)
np.savetxt('force2N_其他步长6.csv',np.hstack((fan_force21_real6,fan_force21_pre6,fan_force210_pre6,
                                        fan_force230_pre6,fan_force250_pre6,fan_force2100_pre6,fan_force2200_pre6)),delimiter=',')


# In[9]:


# # 7
# train_force21_x7, train_force21_y7, valid_force21_x7, valid_force21_y7 = split_train_valid(force21_x7,force21_y7, 6800, 7500, 8000)
# force21_model7, force21_history7, pre_force21_Y7 = Model_LSTM(train_force21_x7, train_force21_y7, valid_force21_x7, valid_force21_y7, lr=1.3, epochs=60, batch_size=50)
# train_force210_x7, train_force210_y7, valid_force210_x7, valid_force210_y7 = split_train_valid(force210_x7,force210_y7, 6800, 7500, 8000)
# force210_model7, force210_history7, pre_force210_Y7 = Model_LSTM(train_force210_x7, train_force210_y7, valid_force210_x7, valid_force210_y7, lr=1.3, epochs=60, batch_size=50)
# train_force230_x7, train_force230_y7, valid_force230_x7, valid_force230_y7 = split_train_valid(force230_x7,force230_y7, 6800, 7500, 8000)
# force230_model7, force230_history7, pre_force230_Y7 = Model_LSTM(train_force230_x7, train_force230_y7, valid_force230_x7, valid_force230_y7, lr=1.3, epochs=60, batch_size=50)
# train_force250_x7, train_force250_y7, valid_force250_x7, valid_force250_y7 = split_train_valid(force250_x7,force250_y7, 6800, 7500, 8000)
# force250_model7, force250_histor7, pre_force250_Y7 = Model_LSTM(train_force250_x7, train_force250_y7, valid_force250_x7, valid_force250_y7, lr=1.3, epochs=60, batch_size=50)
# train_force2100_x7, train_force2100_y7, valid_force2100_x7, valid_force2100_y7 = split_train_valid(force2100_x7,force2100_y7, 6800, 7500, 8000)
# force2100_model7, force2100_history7, pre_force2100_Y7 = Model_LSTM(train_force2100_x7, train_force2100_y7, valid_force2100_x7, valid_force2100_y7, lr=1.3, epochs=60, batch_size=50)
# train_force2200_x7, train_force2200_y7, valid_force2200_x7, valid_force2200_y7 = split_train_valid(force2200_x7,force2200_y7, 6800, 7500, 8000)
# force2200_model7, force2200_history7, pre_force2200_Y7 = Model_LSTM(train_force2200_x7, train_force2200_y7, valid_force2200_x7, valid_force2200_y7, lr=1.3, epochs=60, batch_size=50)

# fan_force21_real7,fan_force21_pre7 = FanGuiHua_force2(valid_force21_y7,pre_force21_Y7)
# fan_force210_real7,fan_force210_pre7 = FanGuiHua_force2(valid_force210_y7,pre_force210_Y7)
# fan_force230_real7,fan_force230_pre7 = FanGuiHua_force2(valid_force230_y7,pre_force230_Y7)
# fan_force250_real7,fan_force250_pre7 = FanGuiHua_force2(valid_force250_y7,pre_force250_Y7)
# fan_force2100_real7,fan_force2100_pre7 = FanGuiHua_force2(valid_force2100_y7,pre_force2100_Y7)
# fan_force2200_real7,fan_force2200_pre7 = FanGuiHua_force2(valid_force2200_y7,pre_force2200_Y7)
# np.savetxt('force2N_其他步长7.csv',np.hstack((fan_force21_real7,fan_force21_pre7,fan_force210_pre7,
#                                         fan_force230_pre7,fan_force250_pre7,fan_force2100_pre7,fan_force2200_pre7)),delimiter=',')


# In[10]:


# force2
# 8
train_force21_x8, train_force21_y8, valid_force21_x8, valid_force21_y8 = split_train_valid(force21_x8,force21_y8, 7000, 7500, 12000)
force21_model8, force21_history8, pre_force21_Y8 = Model_LSTM(train_force21_x8, train_force21_y8, valid_force21_x8, valid_force21_y8, lr=0.1, epochs=30, batch_size=50)
train_force210_x8, train_force210_y8, valid_force210_x8, valid_force210_y8 = split_train_valid(force210_x8,force210_y8, 7000, 7500, 12000)
force210_model8, force210_history8, pre_force210_Y8 = Model_LSTM(train_force210_x8, train_force210_y8, valid_force210_x8, valid_force210_y8, lr=0.1, epochs=30, batch_size=50)
train_force230_x8, train_force230_y8, valid_force230_x8, valid_force230_y8 = split_train_valid(force230_x8,force230_y8, 7000, 7500, 12000)
force230_model8, force230_history8, pre_force230_Y8 = Model_LSTM(train_force230_x8, train_force230_y8, valid_force230_x8, valid_force230_y8, lr=0.1, epochs=30, batch_size=50)
train_force250_x8, train_force250_y8, valid_force250_x8, valid_force250_y8 = split_train_valid(force250_x8,force250_y8, 7000, 7500, 12000)
force250_model8, force250_history8, pre_force250_Y8 = Model_LSTM(train_force250_x8, train_force250_y8, valid_force250_x8, valid_force250_y8, lr=0.1, epochs=30, batch_size=50)
train_force2100_x8, train_force2100_y8, valid_force2100_x8, valid_force2100_y8 = split_train_valid(force2100_x8,force2100_y8, 7000, 7500, 12000)
force2100_model8, force2100_history8, pre_force2100_Y8 = Model_LSTM(train_force2100_x8, train_force2100_y8, valid_force2100_x8, valid_force2100_y8, lr=0.1, epochs=30, batch_size=50)
train_force2200_x8, train_force2200_y8, valid_force2200_x8, valid_force2200_y8 = split_train_valid(force2200_x8,force2200_y8, 7000, 7500, 12000)
force2200_model8, force2200_history8, pre_force2200_Y8 = Model_LSTM(train_force2200_x8, train_force2200_y8, valid_force2200_x8, valid_force2200_y8, lr=0.1, epochs=30, batch_size=50)

fan_force21_real8,fan_force21_pre8 = FanGuiHua_force2(valid_force21_y8,pre_force21_Y8)
fan_force210_real8,fan_force210_pre8 = FanGuiHua_force2(valid_force210_y8,pre_force210_Y8)
fan_force230_real8,fan_force230_pre8 = FanGuiHua_force2(valid_force230_y8,pre_force230_Y8)
fan_force250_real8,fan_force250_pre8 = FanGuiHua_force2(valid_force250_y8,pre_force250_Y8)
fan_force2100_real8,fan_force2100_pre8 = FanGuiHua_force2(valid_force2100_y8,pre_force2100_Y8)
fan_force2200_real8,fan_force2200_pre8 = FanGuiHua_force2(valid_force2200_y8,pre_force2200_Y8)
np.savetxt('force2N_其他步长8.csv',np.hstack((fan_force21_real8,fan_force21_pre8,fan_force210_pre8,
                                        fan_force230_pre8,fan_force250_pre8,fan_force2100_pre8,fan_force2200_pre8)),delimiter=',')


# In[ ]:


# # 9
# train_force21_x9, train_force21_y9, valid_force21_x9, valid_force21_y9 = split_train_valid(force21_x9,force21_y9, 6800, 7500, 8000)
# force21_model9, force21_history9, pre_force21_Y9 = Model_LSTM(train_force21_x9, train_force21_y9, valid_force21_x9, valid_force21_y9, lr=1.3, epochs=60, batch_size=50)
# train_force210_x9, train_force210_y9, valid_force210_x9, valid_force210_y9 = split_train_valid(force210_x9,force210_y9, 6800, 7500, 8000)
# force210_model9, force210_history9, pre_force210_Y9 = Model_LSTM(train_force210_x9, train_force210_y9, valid_force210_x9, valid_force210_y9, lr=1.3, epochs=60, batch_size=50)
# train_force230_x9, train_force230_y9, valid_force230_x9, valid_force230_y9 = split_train_valid(force230_x9,force230_y9, 6800, 7500, 8000)
# force230_model9, force230_history9, pre_force230_Y9 = Model_LSTM(train_force230_x9, train_force230_y9, valid_force230_x9, valid_force230_y9, lr=1.3, epochs=60, batch_size=50)
# train_force250_x9, train_force250_y9, valid_force250_x9, valid_force250_y9 = split_train_valid(force250_x9,force250_y9, 6800, 7500, 8000)
# force250_model9, force250_history9, pre_force250_Y9 = Model_LSTM(train_force250_x9, train_force250_y9, valid_force250_x9, valid_force250_y9, lr=1.3, epochs=60, batch_size=50)
# train_force2100_x9, train_force2100_y9, valid_force2100_x9, valid_force2100_y9 = split_train_valid(force2100_x9,force2100_y9, 6800, 7500, 8000)
# force2100_model9, force2100_history9, pre_force2100_Y9 = Model_LSTM(train_force2100_x9, train_force2100_y9, valid_force2100_x9, valid_force2100_y9, lr=1.3, epochs=60, batch_size=50)
# train_force2200_x9, train_force2200_y9, valid_force2200_x9, valid_force2200_y9 = split_train_valid(force2200_x9,force2200_y9, 6800, 7500, 8000)
# force2200_model9, force2200_history9, pre_force2200_Y9 = Model_LSTM(train_force2200_x9, train_force2200_y9, valid_force2200_x9, valid_force2200_y9, lr=1.3, epochs=60, batch_size=50)

# fan_force21_real9,fan_force21_pre9 = FanGuiHua_force2(valid_force21_y9,pre_force21_Y9)
# fan_force210_real9,fan_force210_pre9 = FanGuiHua_force2(valid_force210_y9,pre_force210_Y9)
# fan_force230_real9,fan_force230_pre9 = FanGuiHua_force2(valid_force230_y9,pre_force230_Y9)
# fan_force250_real9,fan_force250_pre9 = FanGuiHua_force2(valid_force250_y9,pre_force250_Y9)
# fan_force2100_real9,fan_force2100_pre9 = FanGuiHua_force2(valid_force2100_y9,pre_force2100_Y9)
# fan_force2200_real9,fan_force2200_pre9 = FanGuiHua_force2(valid_force2200_y9,pre_force2200_Y9)
# np.savetxt('force2N_其他步长9.csv',np.hstack((fan_force21_real9,fan_force21_pre9,fan_force210_pre9,
#                                         fan_force230_pre9,fan_force250_pre9,fan_force2100_pre9,fan_force2200_pre9)),delimiter=',')


# In[ ]:


# # 输出1步长Force(H、Heave、Surge、Pitch)  训练量？？？[346:,:], 4, 54
# # 输入200步长输出9步长[342:,:], 4, 58  /50
# force2200_scaled9_1 = deal_data2(np.hstack((H,Heave,Surge,Pitch,Force2))[347:,:], 5, 53)
# force2200_x9_1,force2200_y9_1 = split_sequence(force2200_scaled9_1, 50)
# print(force2200_x9_1.shape)

# past_train_force2200_x1_1, past_train_force2200_y1_1, past_valid_force2200_x1_1, past_valid_force2200_y1_1 = split_train_valid(force2200_x9_1,force2200_y9_1, 6900, 7500, 8000)
# past_force2200_model1_1, past_force2200_history1_1, past_pre_force2200_Y1_1 = Model_LSTM(past_train_force2200_x1_1, past_train_force2200_y1_1, past_valid_force2200_x1_1, past_valid_force2200_y1_1, lr=0.01, epochs=110, batch_size=50)

# past_fan_force2200_real1_1,past_fan_force2200_pre1_1 = FanGuiHua_force2(past_valid_force2200_y1_1,past_pre_force2200_Y1_1)

# np.savetxt('gui多1_force2_步长9.csv',np.hstack((past_valid_force2200_y1_1,past_pre_force2200_Y1_1)),delimiter=',')
# np.savetxt('多1_force2_步长9.csv',np.hstack((past_fan_force2200_real1_1,past_fan_force2200_pre1_1)),delimiter=',')
# loss_plot(past_force2200_history1_1, epo=110, length=10, width=6)


# In[ ]:


# # 输出1步长Force(H、Heave、Surge、Pitch、过去的Force)  训练量？？？
# # 输入200步长输出9步长
# force2200_scaled9_2 = deal_data1(np.hstack((H,Heave,Surge,Pitch,Force2))[347:,:], 5, 53)
# past_force2200_x9_2,past_force2200_y9_2 = split_sequence(force2200_scaled9_2, 50)
# print(past_force2200_x9_2.shape)

# past_train_force2200_x1_2, past_train_force2200_y1_2, past_valid_force2200_x1_2, past_valid_force2200_y1_2 = split_train_valid(past_force2200_x9_2,past_force2200_y9_2, 6900, 7500, 8000)
# past_force2200_model1_2, past_force2200_history1_2, past_pre_force2200_Y1_2 = Model_LSTM(past_train_force2200_x1_2, past_train_force2200_y1_2, past_valid_force2200_x1_2, past_valid_force2200_y1_2, lr=0.01, epochs=110, batch_size=50)

# past_fan_force2200_real1_2,past_fan_force2200_pre1_2 = FanGuiHua_force2(past_valid_force2200_y1_2,past_pre_force2200_Y1_2)

# np.savetxt('gui多2_force2_步长9.csv',np.hstack((past_valid_force2200_y1_2,past_pre_force2200_Y1_2)),delimiter=',')
# np.savetxt('多2_force2_步长9.csv',np.hstack((past_fan_force2200_real1_2,past_fan_force2200_pre1_2)),delimiter=',')
# loss_plot(past_force2200_history1_2, epo=110, length=10, width=6)


# In[ ]:





# In[ ]:





# In[ ]:




