#!/usr/bin/env python
# coding: utf-8

# In[47]:


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
data_1 = pd.read_csv('t_1.csv')
data_1.head()


# In[5]:


print(np.min(data_distance[:,0:1]))
print(np.max(data_distance[:,0:1]))


# In[4]:


data_distance = np.hstack((np.array(data_1)[:, 1:8], np.array(data_1)[:, 8:10]))
print(data_distance)
print(data_distance.shape)


# In[6]:


print(data_distance[:,6:7])
print(data_distance[:,6:7]*1e6)
print(np.max(abs(data_distance[:,6:7])))
print(np.min(abs(data_distance[:,6:7])))

print(np.max(abs(data_distance[:,6:7])*1e6))
print(np.min(abs(data_distance[:,6:7])*1e6))


# In[7]:


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


# In[8]:


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

# Bi-StatefulLSTM
def Model_Bi_StatefulLSTM(train_X, train_Y, valid_X, valid_Y, lr=0.01, epochs=20, batch_size=256):
    model = Sequential()
    model.add(Bidirectional(LSTM(25, batch_input_shape=(batch_size, train_X.shape[1], train_X.shape[2]),stateful=True,return_sequences=True), merge_mode='concat'))  
    model.add(Bidirectional(LSTM(25,stateful=True), merge_mode='concat'))  
    model.add(Dense(train_Y.shape[1])) 
    model.add(Activation('tanh'))
    adam = Adam(lr = lr)
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, validation_data=(valid_X, valid_Y), verbose=2, shuffle=False)
    model.reset_states()
    model_structure = model.summary()
    pre_Y = model.predict(valid_X,batch_size=batch_size)
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


# In[9]:


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


# In[ ]:


train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 0, 7500, 10500)


# #### 不同训练量情况的未来运动预测结果与分析

# In[42]:


# 10
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 7490, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练10_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练10_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 7490, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练10_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练10_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 7490, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练10_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练10_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# In[10]:


# 20
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 7480, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练20_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练20_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 7480, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练20_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练20_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 7480, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练20_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练20_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# In[11]:


# 30
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 7470, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练30_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练30_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 7470, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练30_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练30_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 7470, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练30_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练30_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# In[12]:


# 40
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 7460, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练40_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练40_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 7460, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练40_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练40_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 7460, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练40_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练40_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# In[13]:


# 50
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 7450, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练50_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练50_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 7450, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练50_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练50_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 7450, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练50_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练50_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# In[15]:


# 60
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 7440, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练60_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练60_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 7440, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练60_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练60_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 7440, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练60_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练60_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# In[16]:


# 70
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 7430, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练70_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练70_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 7430, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练70_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练70_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 7430, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练70_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练70_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# In[19]:


# 80
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 7420, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练80_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练80_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 7420, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练80_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练80_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 7420, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练80_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练80_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# In[20]:


# 90
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 7410, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练90_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练90_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 7410, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练90_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练90_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 7410, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练90_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练90_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# In[21]:


# 100
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 7400, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练100_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练100_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 7400, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练100_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练100_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 7400, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练100_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练100_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# In[22]:


# 200
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 7300, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练200_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练200_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 7300, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练200_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练200_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 7300, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练200_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练200_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# In[23]:


# 300
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 7200, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练300_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练300_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 7200, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练300_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练300_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 7200, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练300_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练300_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# In[24]:


# 400
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 7100, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练400_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练400_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 7100, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练400_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练400_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 7100, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练400_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练400_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# In[25]:


# 500
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 7000, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练500_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练500_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 7000, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练500_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练500_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 7000, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练500_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练500_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# In[26]:


# 600
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 6900, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练600_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练600_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 6900, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练600_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练600_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 6900, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练600_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练600_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# In[27]:


# 700
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 6800, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练700_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练700_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 6800, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练700_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练700_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 6800, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练700_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练700_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# In[17]:


# 800
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 6700, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练800_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练800_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 6700, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练800_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练800_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 6700, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练800_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练800_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# In[18]:


# 900
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 6600, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练900_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练900_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 6600, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练900_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练900_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 6600, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练900_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练900_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# In[14]:


# 1000
heave_scaled1 = deal_data2(np.hstack((H,Heave))[345:,:], 2, 55)
heave_x1,heave_y1 = split_sequence(heave_scaled1, 55)
print(heave_x1.shape)
train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1 = split_train_valid(heave_x1,heave_y1, 6500, 7500, 8000)
hea1_model1, hea1_history1, hea1_pre_Y1 = Model_LSTM(train_hea_x1, train_hea_y1, valid_hea_x1, valid_hea_y1, lr=0.01, epochs=60, batch_size=50)
fan1_heave_real1,fan1_heave_pre1 = FanGuiHua_heave(valid_hea_y1,hea1_pre_Y1)
np.savetxt('gui训练1000_heave_步长1.csv',np.hstack((valid_hea_y1,hea1_pre_Y1)),delimiter=',')
np.savetxt('训练1000_heave_步长1.csv',np.hstack((fan1_heave_real1,fan1_heave_pre1)),delimiter=',')
loss_plot(hea1_history1, epo=60, length=10, width=6)

pitch_scaled1 = deal_data2(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch_x1,pitch_y1 = split_sequence(pitch_scaled1, 55)
print(pitch_x1.shape)
train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1 = split_train_valid(pitch_x1,pitch_y1, 6500, 7500, 8000)
pitch1_model1, pitch1_history1, pitch1_pre_Y1 = Model_LSTM(train_pitch_x1, train_pitch_y1, valid_pitch_x1, valid_pitch_y1, lr=0.01, epochs=60, batch_size=50)
fan1_pitch_real1,fan1_pitch_pre1 = FanGuiHua_pitch(valid_pitch_y1,pitch1_pre_Y1)
np.savetxt('gui训练1000_pitch_步长1.csv',np.hstack((valid_pitch_y1,pitch1_pre_Y1)),delimiter=',')
np.savetxt('训练1000_pitch_步长1.csv',np.hstack((fan1_pitch_real1,fan1_pitch_pre1)),delimiter=',')
loss_plot(pitch1_history1, epo=60, length=10, width=6)

surge_scaled1 = deal_data2(np.hstack((H,Surge))[345:,:], 2, 55)
surge_x1,surge_y1 = split_sequence(surge_scaled1, 55)
print(surge_x1.shape)
train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1 = split_train_valid(surge_x1,surge_y1, 6500, 7500, 8000)
surge1_model1, surge1_history1, surge1_pre_Y1 = Model_LSTM(train_surge_x1, train_surge_y1, valid_surge_x1, valid_surge_y1, lr=0.01, epochs=60, batch_size=50)
fan1_surge_real1,fan1_surge_pre1 = FanGuiHua_surge(valid_surge_y1,surge1_pre_Y1)
np.savetxt('gui训练1000_surge_步长1.csv',np.hstack((valid_surge_y1,surge1_pre_Y1)),delimiter=',')
np.savetxt('训练1000_surge_步长1.csv',np.hstack((fan1_surge_real1,fan1_surge_pre1)),delimiter=',')
loss_plot(surge1_history1, epo=60, length=10, width=6)


# #### 不同输入特征下的未来运动预测结果与分析
# (1.波高-运动响应、2.波高和运动响应-运动响应)

# #### pitch

# In[45]:


# 输出1步长pitch  训练量600
pitch100_scaled1 = deal_data2(np.hstack((H,Pitch))[300:,:], 2, 100)
pitch100_x1,pitch100_y1 = split_sequence(pitch100_scaled1, 100)
print(pitch100_x1.shape)

train_pitch100_x1, train_pitch100_y1, valid_pitch100_x1, valid_pitch100_y1 = split_train_valid(pitch100_x1,pitch100_y1, 6900, 7500, 10400)
pitch100_model1, pitch100_history1, pre_pitch100_Y1 = Model_LSTM(train_pitch100_x1, train_pitch100_y1, valid_pitch100_x1, valid_pitch100_y1, lr=0.01, epochs=60, batch_size=50)

fan_pitch100_real1,fan_pitch100_pre1 = FanGuiHua_pitch(valid_pitch100_y1,pre_pitch100_Y1)


# In[46]:


np.savetxt('gui单_pitch_步长1.csv',np.hstack((valid_pitch100_y1,pre_pitch100_Y1)),delimiter=',')
np.savetxt('单_pitch_步长1.csv',np.hstack((fan_pitch100_real1,fan_pitch100_pre1)),delimiter=',')
loss_plot(pitch100_history1, epo=60, length=10, width=6)


# In[47]:


# 输出1步长pitch(加入过去的pitch)  训练量600
past_pitch100_scaled1 = deal_data1(np.hstack((H,Pitch))[300:,:], 2, 100)
past_pitch100_x1,past_pitch100_y1 = split_sequence(past_pitch100_scaled1, 100)
print(past_pitch100_x1.shape)

past_train_pitch100_x1, past_train_pitch100_y1, past_valid_pitch100_x1, past_valid_pitch100_y1 = split_train_valid(past_pitch100_x1,past_pitch100_y1, 6900, 7500, 10400)
past_pitch100_model1, past_pitch100_history1, past_pre_pitch100_Y1 = Model_LSTM(past_train_pitch100_x1, past_train_pitch100_y1, past_valid_pitch100_x1, past_valid_pitch100_y1, lr=0.01, epochs=60, batch_size=50)

past_fan_pitch100_real1,past_fan_pitch100_pre1 = FanGuiHua_pitch(past_valid_pitch100_y1,past_pre_pitch100_Y1)


# In[48]:


np.savetxt('gui多_pitch_步长1.csv',np.hstack((past_valid_pitch100_y1,past_pre_pitch100_Y1)),delimiter=',')
np.savetxt('多_pitch_步长1.csv',np.hstack((past_fan_pitch100_real1,past_fan_pitch100_pre1)),delimiter=',')
loss_plot(past_pitch100_history1, epo=60, length=10, width=6)


# In[ ]:





# In[ ]:





# #### surge

# In[49]:


# 输出1步长surge  训练量900
surge100_scaled1 = deal_data2(np.hstack((H,Surge))[300:,:], 2, 100)
surge100_x1,surge100_y1 = split_sequence(surge100_scaled1, 100)
print(surge100_x1.shape)

train_surge100_x1, train_surge100_y1, valid_surge100_x1, valid_surge100_y1 = split_train_valid(surge100_x1,surge100_y1, 6600, 7500, 10400)
surge100_model1, surge100_history1, pre_surge100_Y1 = Model_LSTM(train_surge100_x1, train_surge100_y1, valid_surge100_x1, valid_surge100_y1, lr=0.01, epochs=60, batch_size=50)

fan_surge100_real1,fan_surge100_pre1 = FanGuiHua_surge(valid_surge100_y1,pre_surge100_Y1)


# In[50]:


np.savetxt('gui单_surge_步长1.csv',np.hstack((valid_surge100_y1,pre_surge100_Y1)),delimiter=',')
np.savetxt('单_surge_步长1.csv',np.hstack((fan_surge100_real1,fan_surge100_pre1)),delimiter=',')
loss_plot(surge100_history1, epo=60, length=10, width=6)


# In[24]:


# 输出1步长surge(加入过去的surge)  训练量900
past_surge100_scaled1 = deal_data1(np.hstack((H,Surge))[300:,:], 2, 100)
past_surge100_x1,past_surge100_y1 = split_sequence(past_surge100_scaled1, 100)
print(past_surge100_x1.shape)

past_train_surge100_x1, past_train_surge100_y1, past_valid_surge100_x1, past_valid_surge100_y1 = split_train_valid(past_surge100_x1,past_surge100_y1, 6600, 7500, 10400)
past_surge100_model1, past_surge100_history1, past_pre_surge100_Y1 = Model_LSTM(past_train_surge100_x1, past_train_surge100_y1, past_valid_surge100_x1, past_valid_surge100_y1, lr=0.01, epochs=60, batch_size=50)

past_fan_surge100_real1,past_fan_surge100_pre1 = FanGuiHua_surge(past_valid_surge100_y1,past_pre_surge100_Y1)


# In[25]:


np.savetxt('gui多_surge_步长1.csv',np.hstack((past_valid_surge100_y1,past_pre_surge100_Y1)),delimiter=',')
np.savetxt('多_surge_步长1.csv',np.hstack((past_fan_surge100_real1,past_fan_surge100_pre1)),delimiter=',')
loss_plot(past_surge100_history1, epo=60, length=10, width=6)


# #### heave

# In[53]:


# 输出1步长heave  训练量600
heave100_scaled1 = deal_data2(np.hstack((H,Heave))[300:,:], 2, 100)
heave100_x1,heave100_y1 = split_sequence(heave100_scaled1, 100)
print(heave100_x1.shape)

train_heave100_x1, train_heave100_y1, valid_heave100_x1, valid_heave100_y1 = split_train_valid(heave100_x1,heave100_y1, 6900, 7500, 10400)
heave100_model1, heave100_history1, pre_heave100_Y1 = Model_LSTM(train_heave100_x1, train_heave100_y1, valid_heave100_x1, valid_heave100_y1, lr=0.01, epochs=60, batch_size=50)

fan_heave100_real1,fan_heave100_pre1 = FanGuiHua_heave(valid_heave100_y1,pre_heave100_Y1)


# In[54]:


np.savetxt('gui单_heave_步长1.csv',np.hstack((valid_heave100_y1,pre_heave100_Y1)),delimiter=',')
np.savetxt('单_heave_步长1.csv',np.hstack((fan_heave100_real1,fan_heave100_pre1)),delimiter=',')
loss_plot(heave100_history1, epo=60, length=10, width=6)


# In[55]:


# 输出1步长heave(加入过去的heave)  训练量600
past_heave100_scaled1 = deal_data1(np.hstack((H,Heave))[300:,:], 2, 100)
past_heave100_x1,past_heave100_y1 = split_sequence(past_heave100_scaled1, 100)
print(past_heave100_x1.shape)

past_train_heave100_x1, past_train_heave100_y1, past_valid_heave100_x1, past_valid_heave100_y1 = split_train_valid(past_heave100_x1,past_heave100_y1, 6900, 7500, 10400)
past_heave100_model1, past_heave100_history1, past_pre_heave100_Y1 = Model_LSTM(past_train_heave100_x1, past_train_heave100_y1, past_valid_heave100_x1, past_valid_heave100_y1, lr=0.01, epochs=60, batch_size=50)

past_fan_heave100_real1,past_fan_heave100_pre1 = FanGuiHua_heave(past_valid_heave100_y1,past_pre_heave100_Y1)


# In[56]:


np.savetxt('gui多_heave_步长1.csv',np.hstack((past_valid_heave100_y1,past_pre_heave100_Y1)),delimiter=',')
np.savetxt('多_heave_步长1.csv',np.hstack((past_fan_heave100_real1,past_fan_heave100_pre1)),delimiter=',')
loss_plot(past_heave100_history1, epo=60, length=10, width=6)


# In[ ]:





# #### sway

# In[23]:


# 输出1步长sway
sway100_scaled1 = deal_data2(np.hstack((H,Sway))[300:,:], 2, 100)
sway100_x1,sway100_y1 = split_sequence(sway100_scaled1, 100)
print(sway100_x1.shape)

train_sway100_x1, train_sway100_y1, valid_sway100_x1, valid_sway100_y1 = split_train_valid(sway100_x1,sway100_y1, 0, 6000, 10400)
sway100_model1, sway100_history1, pre_sway100_Y1 = Model_LSTM(train_sway100_x1, train_sway100_y1, valid_sway100_x1, valid_sway100_y1, lr=0.01, epochs=60, batch_size=50)

fan_sway100_real1,fan_sway100_pre1 = FanGuiHua_sway(valid_sway100_y1,pre_sway100_Y1)


# In[50]:


# 输出1步长sway(加入过去的sway)
past_sway100_scaled1 = deal_data1(np.hstack((H,Sway))[300:,:], 2, 100)
past_sway100_x1,past_sway100_y1 = split_sequence(past_sway100_scaled1, 100)
print(past_sway100_x1.shape)

past_train_sway100_x1, past_train_sway100_y1, past_valid_sway100_x1, past_valid_sway100_y1 = split_train_valid(past_sway100_x1,past_sway100_y1, 5500, 6000, 6200)
past_sway100_model1, past_sway100_history1, past_pre_sway100_Y1 = Model_LSTM(past_train_sway100_x1, past_train_sway100_y1, past_valid_sway100_x1, past_valid_sway100_y1, lr=0.01, epochs=600, batch_size=50)

past_fan_sway100_real1,past_fan_sway100_pre1 = FanGuiHua_sway(past_valid_sway100_y1,past_pre_sway100_Y1)


# In[29]:


# 输出1步长sway(加入过去的sway)
past_sway100_scaled1 = deal_data1(np.hstack((H,Sway))[300:,:], 2, 100)
past_sway100_x1,past_sway100_y1 = split_sequence(past_sway100_scaled1, 100)
print(past_sway100_x1.shape)

past_train_sway100_x1, past_train_sway100_y1, past_valid_sway100_x1, past_valid_sway100_y1 = split_train_valid(past_sway100_x1,past_sway100_y1, 0, 6000, 10400)
past_sway100_model1, past_sway100_history1, past_pre_sway100_Y1 = Model_LSTM(past_train_sway100_x1, past_train_sway100_y1, past_valid_sway100_x1, past_valid_sway100_y1, lr=0.01, epochs=200, batch_size=50)

past_fan_sway100_real1,past_fan_sway100_pre1 = FanGuiHua_sway(past_valid_sway100_y1,past_pre_sway100_Y1)


# In[32]:


# 输出1步长sway(加入过去的sway)
past_sway100_scaled1 = deal_data1(Sway[300:,:], 1, 100)
past_sway100_x1,past_sway100_y1 = split_sequence(past_sway100_scaled1, 100)
print(past_sway100_x1.shape)

past_train_sway100_x1, past_train_sway100_y1, past_valid_sway100_x1, past_valid_sway100_y1 = split_train_valid(past_sway100_x1,past_sway100_y1, 5000, 6000, 6500)
past_sway100_model1, past_sway100_history1, past_pre_sway100_Y1 = Model_LSTM(past_train_sway100_x1, past_train_sway100_y1, past_valid_sway100_x1, past_valid_sway100_y1, lr=0.01, epochs=600, batch_size=50)

past_fan_sway100_real1,past_fan_sway100_pre1 = FanGuiHua_sway(past_valid_sway100_y1,past_pre_sway100_Y1)


# #### roll

# In[25]:


# 输出1步长roll
roll100_scaled1 = deal_data2(np.hstack((H,Roll))[300:,:], 2, 100)
roll100_x1,roll100_y1 = split_sequence(roll100_scaled1, 100)
print(roll100_x1.shape)

train_roll100_x1, train_roll100_y1, valid_roll100_x1, valid_roll100_y1 = split_train_valid(roll100_x1,roll100_y1, 0, 6000, 10400)
roll100_model1, roll100_history1, pre_roll100_Y1 = Model_LSTM(train_roll100_x1, train_roll100_y1, valid_roll100_x1, valid_roll100_y1, lr=0.01, epochs=60, batch_size=50)

# fan_roll100_real1,fan_roll100_pre1 = FanGuiHua_roll(valid_roll100_y1,pre_roll100_Y1)


# In[33]:


plt.figure(figsize=(30,4), dpi=100)
plt.plot(valid_roll100_y1[:200,:],color='blue',label='real')   #真实曲线
plt.plot(pre_roll100_Y1[:200,:],color='orange',label='prediction') #预测曲线
plt.legend()
plt.show()


# In[26]:


# 输出1步长roll(加入过去的roll)
past_roll100_scaled1 = deal_data1(np.hstack((H,Roll))[300:,:], 2, 100)
past_roll100_x1,past_roll100_y1 = split_sequence(past_roll100_scaled1, 100)
print(past_roll100_x1.shape)

past_train_roll100_x1, past_train_roll100_y1, past_valid_roll100_x1, past_valid_roll100_y1 = split_train_valid(past_sway100_x1,past_sway100_y1, 0, 6000, 10400)
past_roll100_model1, past_roll100_history1, past_pre_roll100_Y1 = Model_LSTM(past_train_roll100_x1, past_train_roll100_y1, past_valid_roll100_x1, past_valid_roll100_y1, lr=0.01, epochs=60, batch_size=50)

# past_fan_roll100_real1,past_fan_roll100_pre1 = FanGuiHua_roll(past_valid_roll100_y1,past_pre_roll100_Y1)


# In[ ]:





# In[ ]:





# #### yaw

# In[27]:


# 输出1步长yaw
yaw100_scaled1 = deal_data2(np.hstack((H,Yaw))[300:,:], 2, 100)
yaw100_x1,yaw100_y1 = split_sequence(yaw100_scaled1, 100)
print(yaw100_x1.shape)

train_yaw100_x1, train_yaw100_y1, valid_yaw100_x1, valid_yaw100_y1 = split_train_valid(yaw100_x1,yaw100_y1, 0, 6000, 10400)
yaw100_model1, yaw100_history1, pre_yaw100_Y1 = Model_LSTM(train_yaw100_x1, train_yaw100_y1, valid_yaw100_x1, valid_yaw100_y1, lr=0.01, epochs=60, batch_size=50)

fan_yaw100_real1,fan_yaw100_pre1 = FanGuiHua_yaw(valid_yaw100_y1,pre_yaw100_Y1)


# In[28]:


# 输出1步长yaw(加入过去的yaw)
past_yaw100_scaled1 = deal_data1(np.hstack((H,Yaw))[300:,:], 2, 100)
past_yaw100_x1,past_yaw100_y1 = split_sequence(past_yaw100_scaled1, 100)
print(past_yaw100_x1.shape)

past_train_yaw100_x1, past_train_yaw100_y1, past_valid_yaw100_x1, past_valid_yaw100_y1 = split_train_valid(past_yaw100_x1,past_yaw100_y1, 0, 6000, 10400)
past_yaw100_model1, past_yaw100_history1, past_pre_yaw100_Y1 = Model_LSTM(past_train_yaw100_x1, past_train_yaw100_y1, past_valid_yaw100_x1, past_valid_yaw100_y1, lr=0.01, epochs=60, batch_size=50)

past_fan_yaw100_real1,past_fan_yaw100_pre1 = FanGuiHua_yaw(past_valid_yaw100_y1,past_pre_yaw100_Y1)


# In[ ]:





# In[ ]:





# #### 不同时间窗情况的未来运动预测结果与分析

# In[18]:


# 归一化后的两个数据集
# 波浪-运动响应
# surge
# 输入1步长输出1步长
surge1_scaled1 = deal_data1(np.hstack((H,Surge))[399:,:], 2, 1)
surge1_x1,surge1_y1 = split_sequence(surge1_scaled1, 1)
print(surge1_x1.shape)
# 输入10步长输出1步长
surge10_scaled1 = deal_data1(np.hstack((H,Surge))[390:,:], 2, 10)
surge10_x1,surge10_y1 = split_sequence(surge10_scaled1, 10)
print(surge10_x1.shape)
# 输入30步长输出1步长
surge30_scaled1 = deal_data1(np.hstack((H,Surge))[370:,:], 2, 30)
surge30_x1,surge30_y1 = split_sequence(surge30_scaled1, 30)
print(surge30_x1.shape)
# 输入50步长输出1步长
surge50_scaled1 = deal_data1(np.hstack((H,Surge))[350:,:], 2, 50)
surge50_x1,surge50_y1 = split_sequence(surge50_scaled1, 50)
print(surge50_x1.shape)
# 输入100步长输出1步长
surge100_scaled1 = deal_data1(np.hstack((H,Surge))[300:,:], 2, 100)
surge100_x1,surge100_y1 = split_sequence(surge100_scaled1, 100)
print(surge100_x1.shape)
# 输入200步长输出1步长
surge200_scaled1 = deal_data1(np.hstack((H,Surge))[200:,:], 2, 200)
surge200_x1,surge200_y1 = split_sequence(surge200_scaled1, 200)
print(surge200_x1.shape)

# 输入1步长输出3步长
surge1_scaled3 = deal_data1(np.hstack((H,Surge))[397:,:], 2, 3)
surge1_x3,surge1_y3 = split_sequence(surge1_scaled3, 1)
print(surge1_x3.shape)
# 输入10步长输出3步长
surge10_scaled3 = deal_data1(np.hstack((H,Surge))[388:,:], 2, 12)
surge10_x3,surge10_y3 = split_sequence(surge10_scaled3, 10)
print(surge10_x3.shape)
# 输入30步长输出3步长
surge30_scaled3 = deal_data1(np.hstack((H,Surge))[368:,:], 2, 32)
surge30_x3,surge30_y3 = split_sequence(surge30_scaled3, 30)
print(surge30_x3.shape)
# 输入50步长输出3步长
surge50_scaled3 = deal_data1(np.hstack((H,Surge))[348:,:], 2, 52)
surge50_x3,surge50_y3 = split_sequence(surge50_scaled3, 50)
print(surge50_x3.shape)
# 输入100步长输出3步长
surge100_scaled3 = deal_data1(np.hstack((H,Surge))[298:,:], 2, 102)
surge100_x3,surge100_y3 = split_sequence(surge100_scaled3, 100)
print(surge100_x3.shape)
# 输入200步长输出3步长
surge200_scaled3 = deal_data1(np.hstack((H,Surge))[198:,:], 2, 202)
surge200_x3,surge200_y3 = split_sequence(surge200_scaled3, 200)
print(surge200_x3.shape)

# 输入1步长输出5步长
surge1_scaled5 = deal_data1(np.hstack((H,Surge))[395:,:], 2, 5)
surge1_x5,surge1_y5 = split_sequence(surge1_scaled5, 1)
print(surge1_x5.shape)
# 输入10步长输出5步长
surge10_scaled5 = deal_data1(np.hstack((H,Surge))[386:,:], 2, 14)
surge10_x5,surge10_y5 = split_sequence(surge10_scaled5, 10)
print(surge10_x5.shape)
# 输入30步长输出5步长
surge30_scaled5 = deal_data1(np.hstack((H,Surge))[366:,:], 2, 34)
surge30_x5,surge30_y5 = split_sequence(surge30_scaled5, 30)
print(surge30_x5.shape)
# 输入50步长输出5步长
surge50_scaled5 = deal_data1(np.hstack((H,Surge))[346:,:], 2, 54)
surge50_x5,surge50_y5 = split_sequence(surge50_scaled5, 50)
print(surge50_x5.shape)
# 输入100步长输出5步长
surge100_scaled5 = deal_data1(np.hstack((H,Surge))[296:,:], 2, 104)
surge100_x5,surge100_y5 = split_sequence(surge100_scaled5, 100)
print(surge100_x5.shape)
# 输入200步长输出5步长
surge200_scaled5 = deal_data1(np.hstack((H,Surge))[196:,:], 2, 204)
surge200_x5,surge200_y5 = split_sequence(surge200_scaled5, 200)
print(surge200_x5.shape)

# 输入1步长输出7步长
surge1_scaled7 = deal_data1(np.hstack((H,Surge))[393:,:], 2, 7)
surge1_x7,surge1_y7 = split_sequence(surge1_scaled7, 1)
print(surge1_x7.shape)
# 输入10步长输出7步长
surge10_scaled7 = deal_data1(np.hstack((H,Surge))[384:,:], 2, 16)
surge10_x7,surge10_y7 = split_sequence(surge10_scaled7, 10)
print(surge10_x7.shape)
# 输入30步长输出7步长
surge30_scaled7 = deal_data1(np.hstack((H,Surge))[364:,:], 2, 36)
surge30_x7,surge30_y7 = split_sequence(surge30_scaled7, 30)
print(surge30_x7.shape)
# 输入50步长输出7步长
surge50_scaled7 = deal_data1(np.hstack((H,Surge))[344:,:], 2, 56)
surge50_x7,surge50_y7 = split_sequence(surge50_scaled7, 50)
print(surge50_x7.shape)
# 输入100步长输出7步长
surge100_scaled7 = deal_data1(np.hstack((H,Surge))[294:,:], 2, 106)
surge100_x7,surge100_y7 = split_sequence(surge100_scaled7, 100)
print(surge100_x7.shape)
# 输入200步长输出7步长
surge200_scaled7 = deal_data1(np.hstack((H,Surge))[194:,:], 2, 206)
surge200_x7,surge200_y7 = split_sequence(surge200_scaled7, 200)
print(surge200_x7.shape)

# 输入1步长输出9步长
surge1_scaled9 = deal_data1(np.hstack((H,Surge))[391:,:], 2, 9)
surge1_x9,surge1_y9 = split_sequence(surge1_scaled9, 1)
print(surge1_x9.shape)
# 输入10步长输出9步长
surge10_scaled9 = deal_data1(np.hstack((H,Surge))[382:,:], 2, 18)
surge10_x9,surge10_y9 = split_sequence(surge10_scaled9, 10)
print(surge10_x9.shape)
# 输入30步长输出9步长
surge30_scaled9 = deal_data1(np.hstack((H,Surge))[362:,:], 2, 38)
surge30_x9,surge30_y9 = split_sequence(surge30_scaled9, 30)
print(surge30_x9.shape)
# 输入50步长输出9步长
surge50_scaled9 = deal_data1(np.hstack((H,Surge))[342:,:], 2, 58)
surge50_x9,surge50_y9 = split_sequence(surge50_scaled9, 50)
print(surge50_x9.shape)
# 输入100步长输出9步长
surge100_scaled9 = deal_data1(np.hstack((H,Surge))[292:,:], 2, 108)
surge100_x9,surge100_y9 = split_sequence(surge100_scaled9, 100)
print(surge100_x9.shape)
# 输入200步长输出9步长
surge200_scaled9 = deal_data1(np.hstack((H,Surge))[192:,:], 2, 208)
surge200_x9,surge200_y9 = split_sequence(surge200_scaled9, 200)
print(surge200_x9.shape)


# In[66]:


# surge
# 1
train_surge1_x1, train_surge1_y1, valid_surge1_x1, valid_surge1_y1 = split_train_valid(surge1_x1,surge1_y1, 6600, 7500, 8000)
surge1_model1, surge1_history1, pre_surge1_Y1 = Model_LSTM(train_surge1_x1, train_surge1_y1, valid_surge1_x1, valid_surge1_y1, lr=1.3, epochs=60, batch_size=50)
train_surge10_x1, train_surge10_y1, valid_surge10_x1, valid_surge10_y1 = split_train_valid(surge10_x1,surge10_y1, 6600, 7500, 8000)
surge10_model1, surge10_history1, pre_surge10_Y1 = Model_LSTM(train_surge10_x1, train_surge10_y1, valid_surge10_x1, valid_surge10_y1, lr=1.3, epochs=60, batch_size=50)
train_surge30_x1, train_surge30_y1, valid_surge30_x1, valid_surge30_y1 = split_train_valid(surge30_x1,surge30_y1, 6600, 7500, 8000)
surge30_model1, surge30_history1, pre_surge30_Y1 = Model_LSTM(train_surge30_x1, train_surge30_y1, valid_surge30_x1, valid_surge30_y1, lr=1.3, epochs=60, batch_size=50)
train_surge50_x1, train_surge50_y1, valid_surge50_x1, valid_surge50_y1 = split_train_valid(surge50_x1,surge50_y1, 6600, 7500, 8000)
surge50_model1, surge50_history1, pre_surge50_Y1 = Model_LSTM(train_surge50_x1, train_surge50_y1, valid_surge50_x1, valid_surge50_y1, lr=1.3, epochs=60, batch_size=50)
train_surge100_x1, train_surge100_y1, valid_surge100_x1, valid_surge100_y1 = split_train_valid(surge100_x1,surge100_y1, 6600, 7500, 8000)
surge100_model1, surge100_history1, pre_surge100_Y1 = Model_LSTM(train_surge100_x1, train_surge100_y1, valid_surge100_x1, valid_surge100_y1, lr=1.3, epochs=60, batch_size=50)
train_surge200_x1, train_surge200_y1, valid_surge200_x1, valid_surge200_y1 = split_train_valid(surge200_x1,surge200_y1, 6600, 7500, 8000)
surge200_model1, surge200_history1, pre_surge200_Y1 = Model_LSTM(train_surge200_x1, train_surge200_y1, valid_surge200_x1, valid_surge200_y1, lr=1.3, epochs=60, batch_size=50)


# In[67]:


fan_surge1_real1,fan_surge1_pre1 = FanGuiHua_surge(valid_surge1_y1,pre_surge1_Y1)
fan_surge10_real1,fan_surge10_pre1 = FanGuiHua_surge(valid_surge10_y1,pre_surge10_Y1)
fan_surge30_real1,fan_surge30_pre1 = FanGuiHua_surge(valid_surge30_y1,pre_surge30_Y1)
fan_surge50_real1,fan_surge50_pre1 = FanGuiHua_surge(valid_surge50_y1,pre_surge50_Y1)
fan_surge100_real1,fan_surge100_pre1 = FanGuiHua_surge(valid_surge100_y1,pre_surge100_Y1)
fan_surge200_real1,fan_surge200_pre1 = FanGuiHua_surge(valid_surge200_y1,pre_surge200_Y1)
np.savetxt('surgeN_其他步长1.csv',np.hstack((fan_surge1_real1,fan_surge1_pre1,fan_surge10_pre1,
                                        fan_surge30_pre1,fan_surge50_pre1,fan_surge100_pre1,fan_surge200_pre1)),delimiter=',')


# In[68]:


# 3
train_surge1_x3, train_surge1_y3, valid_surge1_x3, valid_surge1_y3 = split_train_valid(surge1_x3,surge1_y3, 6600, 7500, 8000)
surge1_model3, surge1_history3, pre_surge1_Y3 = Model_LSTM(train_surge1_x3, train_surge1_y3, valid_surge1_x3, valid_surge1_y3, lr=1.3, epochs=60, batch_size=50)
train_surge10_x3, train_surge10_y3, valid_surge10_x3, valid_surge10_y3 = split_train_valid(surge10_x3,surge10_y3, 6600, 7500, 8000)
surge10_model3, surge10_history3, pre_surge10_Y3 = Model_LSTM(train_surge10_x3, train_surge10_y3, valid_surge10_x3, valid_surge10_y3, lr=1.3, epochs=60, batch_size=50)
train_surge30_x3, train_surge30_y3, valid_surge30_x3, valid_surge30_y3 = split_train_valid(surge30_x3,surge30_y3, 6600, 7500, 8000)
surge30_model3, surge30_history3, pre_surge30_Y3 = Model_LSTM(train_surge30_x3, train_surge30_y3, valid_surge30_x3, valid_surge30_y3, lr=1.3, epochs=60, batch_size=50)
train_surge50_x3, train_surge50_y3, valid_surge50_x3, valid_surge50_y3 = split_train_valid(surge50_x3,surge50_y3, 6600, 7500, 8000)
surge50_model3, surge50_history3, pre_surge50_Y3 = Model_LSTM(train_surge50_x3, train_surge50_y3, valid_surge50_x3, valid_surge50_y3, lr=1.3, epochs=60, batch_size=50)
train_surge100_x3, train_surge100_y3, valid_surge100_x3, valid_surge100_y3 = split_train_valid(surge100_x3,surge100_y3, 6600, 7500, 8000)
surge100_model3, surge100_history3, pre_surge100_Y3 = Model_LSTM(train_surge100_x3, train_surge100_y3, valid_surge100_x3, valid_surge100_y3, lr=1.3, epochs=60, batch_size=50)
train_surge200_x3, train_surge200_y3, valid_surge200_x3, valid_surge200_y3 = split_train_valid(surge200_x3,surge200_y3, 6600, 7500, 8000)
surge200_model3, surge200_history3, pre_surge200_Y3 = Model_LSTM(train_surge200_x3, train_surge200_y3, valid_surge200_x3, valid_surge200_y3, lr=1.3, epochs=60, batch_size=50)


# In[69]:


fan_surge1_real3,fan_surge1_pre3 = FanGuiHua_surge(valid_surge1_y3,pre_surge1_Y3)
fan_surge10_real3,fan_surge10_pre3 = FanGuiHua_surge(valid_surge10_y3,pre_surge10_Y3)
fan_surge30_real3,fan_surge30_pre3 = FanGuiHua_surge(valid_surge30_y3,pre_surge30_Y3)
fan_surge50_real3,fan_surge50_pre3 = FanGuiHua_surge(valid_surge50_y3,pre_surge50_Y3)
fan_surge100_real3,fan_surge100_pre3 = FanGuiHua_surge(valid_surge100_y3,pre_surge100_Y3)
fan_surge200_real3,fan_surge200_pre3 = FanGuiHua_surge(valid_surge200_y3,pre_surge200_Y3)
np.savetxt('surgeN_其他步长3.csv',np.hstack((fan_surge1_real3,fan_surge1_pre3,fan_surge10_pre3,
                                        fan_surge30_pre3,fan_surge50_pre3,fan_surge100_pre3,fan_surge200_pre3)),delimiter=',')


# In[70]:


# 5
train_surge1_x5, train_surge1_y5, valid_surge1_x5, valid_surge1_y5 = split_train_valid(surge1_x5,surge1_y5, 6600, 7500, 8000)
surge1_model5, surge1_history5, pre_surge1_Y5 = Model_LSTM(train_surge1_x5, train_surge1_y5, valid_surge1_x5, valid_surge1_y5, lr=1.3, epochs=60, batch_size=50)
train_surge10_x5, train_surge10_y5, valid_surge10_x5, valid_surge10_y5 = split_train_valid(surge10_x5,surge10_y5, 6600, 7500, 8000)
surge10_model5, surge10_history5, pre_surge10_Y5 = Model_LSTM(train_surge10_x5, train_surge10_y5, valid_surge10_x5, valid_surge10_y5, lr=1.3, epochs=60, batch_size=50)
train_surge30_x5, train_surge30_y5, valid_surge30_x5, valid_surge30_y5 = split_train_valid(surge30_x5,surge30_y5, 6600, 7500, 8000)
surge30_model5, surge30_history5, pre_surge30_Y5 = Model_LSTM(train_surge30_x5, train_surge30_y5, valid_surge30_x5, valid_surge30_y5, lr=1.3, epochs=60, batch_size=50)
train_surge50_x5, train_surge50_y5, valid_surge50_x5, valid_surge50_y5 = split_train_valid(surge50_x5,surge50_y5, 6600, 7500, 8000)
surge50_model5, surge50_history5, pre_surge50_Y5 = Model_LSTM(train_surge50_x5, train_surge50_y5, valid_surge50_x5, valid_surge50_y5, lr=1.3, epochs=60, batch_size=50)
train_surge100_x5, train_surge100_y5, valid_surge100_x5, valid_surge100_y5 = split_train_valid(surge100_x5,surge100_y5, 6600, 7500, 8000)
surge100_model5, surge100_history5, pre_surge100_Y5 = Model_LSTM(train_surge100_x5, train_surge100_y5, valid_surge100_x5, valid_surge100_y5, lr=1.3, epochs=60, batch_size=50)
train_surge200_x5, train_surge200_y5, valid_surge200_x5, valid_surge200_y5 = split_train_valid(surge200_x5,surge200_y5, 6600, 7500, 8000)
surge200_model5, surge200_history5, pre_surge200_Y5 = Model_LSTM(train_surge200_x5, train_surge200_y5, valid_surge200_x5, valid_surge200_y5, lr=1.3, epochs=60, batch_size=50)


# In[71]:


fan_surge1_real5,fan_surge1_pre5 = FanGuiHua_surge(valid_surge1_y5,pre_surge1_Y5)
fan_surge10_real5,fan_surge10_pre5 = FanGuiHua_surge(valid_surge10_y5,pre_surge10_Y5)
fan_surge30_real5,fan_surge30_pre5 = FanGuiHua_surge(valid_surge30_y5,pre_surge30_Y5)
fan_surge50_real5,fan_surge50_pre5 = FanGuiHua_surge(valid_surge50_y5,pre_surge50_Y5)
fan_surge100_real5,fan_surge100_pre5 = FanGuiHua_surge(valid_surge100_y5,pre_surge100_Y5)
fan_surge200_real5,fan_surge200_pre5 = FanGuiHua_surge(valid_surge200_y5,pre_surge200_Y5)
np.savetxt('surgeN_其他步长5.csv',np.hstack((fan_surge1_real5,fan_surge1_pre5,fan_surge10_pre5,
                                        fan_surge30_pre5,fan_surge50_pre5,fan_surge100_pre5,fan_surge200_pre5)),delimiter=',')


# In[72]:


# 7
train_surge1_x7, train_surge1_y7, valid_surge1_x7, valid_surge1_y7 = split_train_valid(surge1_x7,surge1_y7, 6600, 7500, 8000)
surge1_model7, surge1_history7, pre_surge1_Y7 = Model_LSTM(train_surge1_x7, train_surge1_y7, valid_surge1_x7, valid_surge1_y7, lr=1.3, epochs=60, batch_size=50)
train_surge10_x7, train_surge10_y7, valid_surge10_x7, valid_surge10_y7 = split_train_valid(surge10_x7,surge10_y7, 6600, 7500, 8000)
surge10_model7, surge10_history7, pre_surge10_Y7 = Model_LSTM(train_surge10_x7, train_surge10_y7, valid_surge10_x7, valid_surge10_y7, lr=1.3, epochs=60, batch_size=50)
train_surge30_x7, train_surge30_y7, valid_surge30_x7, valid_surge30_y7 = split_train_valid(surge30_x7,surge30_y7, 6600, 7500, 8000)
surge30_model7, surge30_history7, pre_surge30_Y7 = Model_LSTM(train_surge30_x7, train_surge30_y7, valid_surge30_x7, valid_surge30_y7, lr=1.3, epochs=60, batch_size=50)
train_surge50_x7, train_surge50_y7, valid_surge50_x7, valid_surge50_y7 = split_train_valid(surge50_x7,surge50_y7, 6600, 7500, 8000)
surge50_model7, surge50_histor7, pre_surge50_Y7 = Model_LSTM(train_surge50_x7, train_surge50_y7, valid_surge50_x7, valid_surge50_y7, lr=1.3, epochs=60, batch_size=50)
train_surge100_x7, train_surge100_y7, valid_surge100_x7, valid_surge100_y7 = split_train_valid(surge100_x7,surge100_y7, 6600, 7500, 8000)
surge100_model7, surge100_history7, pre_surge100_Y7 = Model_LSTM(train_surge100_x7, train_surge100_y7, valid_surge100_x7, valid_surge100_y7, lr=1.3, epochs=60, batch_size=50)
train_surge200_x7, train_surge200_y7, valid_surge200_x7, valid_surge200_y7 = split_train_valid(surge200_x7,surge200_y7, 6600, 7500, 8000)
surge200_model7, surge200_history7, pre_surge200_Y7 = Model_LSTM(train_surge200_x7, train_surge200_y7, valid_surge200_x7, valid_surge200_y7, lr=1.3, epochs=60, batch_size=50)


# In[73]:


fan_surge1_real7,fan_surge1_pre7 = FanGuiHua_surge(valid_surge1_y7,pre_surge1_Y7)
fan_surge10_real7,fan_surge10_pre7 = FanGuiHua_surge(valid_surge10_y7,pre_surge10_Y7)
fan_surge30_real7,fan_surge30_pre7 = FanGuiHua_surge(valid_surge30_y7,pre_surge30_Y7)
fan_surge50_real7,fan_surge50_pre7 = FanGuiHua_surge(valid_surge50_y7,pre_surge50_Y7)
fan_surge100_real7,fan_surge100_pre7 = FanGuiHua_surge(valid_surge100_y7,pre_surge100_Y7)
fan_surge200_real7,fan_surge200_pre7 = FanGuiHua_surge(valid_surge200_y7,pre_surge200_Y7)
np.savetxt('surgeN_其他步长7.csv',np.hstack((fan_surge1_real7,fan_surge1_pre7,fan_surge10_pre7,
                                        fan_surge30_pre7,fan_surge50_pre7,fan_surge100_pre7,fan_surge200_pre7)),delimiter=',')


# In[74]:


# 9
train_surge1_x9, train_surge1_y9, valid_surge1_x9, valid_surge1_y9 = split_train_valid(surge1_x9,surge1_y9, 6600, 7500, 8000)
surge1_model9, surge1_history9, pre_surge1_Y9 = Model_LSTM(train_surge1_x9, train_surge1_y9, valid_surge1_x9, valid_surge1_y9, lr=1.3, epochs=60, batch_size=50)
train_surge10_x9, train_surge10_y9, valid_surge10_x9, valid_surge10_y9 = split_train_valid(surge10_x9,surge10_y9, 6600, 7500, 8000)
surge10_model9, surge10_history9, pre_surge10_Y9 = Model_LSTM(train_surge10_x9, train_surge10_y9, valid_surge10_x9, valid_surge10_y9, lr=1.3, epochs=60, batch_size=50)
train_surge30_x9, train_surge30_y9, valid_surge30_x9, valid_surge30_y9 = split_train_valid(surge30_x9,surge30_y9, 6600, 7500, 8000)
surge30_model9, surge30_history9, pre_surge30_Y9 = Model_LSTM(train_surge30_x9, train_surge30_y9, valid_surge30_x9, valid_surge30_y9, lr=1.3, epochs=60, batch_size=50)
train_surge50_x9, train_surge50_y9, valid_surge50_x9, valid_surge50_y9 = split_train_valid(surge50_x9,surge50_y9, 6600, 7500, 8000)
surge50_model9, surge50_history9, pre_surge50_Y9 = Model_LSTM(train_surge50_x9, train_surge50_y9, valid_surge50_x9, valid_surge50_y9, lr=1.3, epochs=60, batch_size=50)
train_surge100_x9, train_surge100_y9, valid_surge100_x9, valid_surge100_y9 = split_train_valid(surge100_x9,surge100_y9, 6600, 7500, 8000)
surge100_model9, surge100_history9, pre_surge100_Y9 = Model_LSTM(train_surge100_x9, train_surge100_y9, valid_surge100_x9, valid_surge100_y9, lr=1.3, epochs=60, batch_size=50)
train_surge200_x9, train_surge200_y9, valid_surge200_x9, valid_surge200_y9 = split_train_valid(surge200_x9,surge200_y9, 6600, 7500, 8000)
surge200_model9, surge200_history9, pre_surge200_Y9 = Model_LSTM(train_surge200_x9, train_surge200_y9, valid_surge200_x9, valid_surge200_y9, lr=1.3, epochs=60, batch_size=50)


# In[75]:


fan_surge1_real9,fan_surge1_pre9 = FanGuiHua_surge(valid_surge1_y9,pre_surge1_Y9)
fan_surge10_real9,fan_surge10_pre9 = FanGuiHua_surge(valid_surge10_y9,pre_surge10_Y9)
fan_surge30_real9,fan_surge30_pre9 = FanGuiHua_surge(valid_surge30_y9,pre_surge30_Y9)
fan_surge50_real9,fan_surge50_pre9 = FanGuiHua_surge(valid_surge50_y9,pre_surge50_Y9)
fan_surge100_real9,fan_surge100_pre9 = FanGuiHua_surge(valid_surge100_y9,pre_surge100_Y9)
fan_surge200_real9,fan_surge200_pre9 = FanGuiHua_surge(valid_surge200_y9,pre_surge200_Y9)
np.savetxt('surgeN_其他步长9.csv',np.hstack((fan_surge1_real9,fan_surge1_pre9,fan_surge10_pre9,
                                        fan_surge30_pre9,fan_surge50_pre9,fan_surge100_pre9,fan_surge200_pre9)),delimiter=',')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


# pitch
# 输入1步长输出1步长
pitch1_scaled1 = deal_data1(np.hstack((H,Pitch))[399:,:], 2, 1)
pitch1_x1,pitch1_y1 = split_sequence(pitch1_scaled1, 1)
print(pitch1_x1.shape)
# 输入10步长输出1步长
pitch10_scaled1 = deal_data1(np.hstack((H,Pitch))[390:,:], 2, 10)
pitch10_x1,pitch10_y1 = split_sequence(pitch10_scaled1, 10)
print(pitch10_x1.shape)
# 输入30步长输出1步长
pitch30_scaled1 = deal_data1(np.hstack((H,Pitch))[370:,:], 2, 30)
pitch30_x1,pitch30_y1 = split_sequence(pitch30_scaled1, 30)
print(pitch30_x1.shape)
# 输入50步长输出1步长
pitch50_scaled1 = deal_data1(np.hstack((H,Pitch))[350:,:], 2, 50)
pitch50_x1,pitch50_y1 = split_sequence(pitch50_scaled1, 50)
print(pitch50_x1.shape)
# 输入100步长输出1步长
pitch100_scaled1 = deal_data1(np.hstack((H,Pitch))[300:,:], 2, 100)
pitch100_x1,pitch100_y1 = split_sequence(pitch100_scaled1, 100)
print(pitch100_x1.shape)
# 输入200步长输出1步长
pitch200_scaled1 = deal_data1(np.hstack((H,Pitch))[200:,:], 2, 200)
pitch200_x1,pitch200_y1 = split_sequence(pitch200_scaled1, 200)
print(pitch200_x1.shape)

# 输入1步长输出3步长
pitch1_scaled3 = deal_data1(np.hstack((H,Pitch))[397:,:], 2, 3)
pitch1_x3,pitch1_y3 = split_sequence(pitch1_scaled3, 1)
print(pitch1_x3.shape)
# 输入10步长输出3步长
pitch10_scaled3 = deal_data1(np.hstack((H,Pitch))[388:,:], 2, 12)
pitch10_x3,pitch10_y3 = split_sequence(pitch10_scaled3, 10)
print(pitch10_x3.shape)
# 输入30步长输出3步长
pitch30_scaled3 = deal_data1(np.hstack((H,Pitch))[368:,:], 2, 32)
pitch30_x3,pitch30_y3 = split_sequence(pitch30_scaled3, 30)
print(pitch30_x3.shape)
# 输入50步长输出3步长
pitch50_scaled3 = deal_data1(np.hstack((H,Pitch))[348:,:], 2, 52)
pitch50_x3,pitch50_y3 = split_sequence(pitch50_scaled3, 50)
print(pitch50_x3.shape)
# 输入100步长输出3步长
pitch100_scaled3 = deal_data1(np.hstack((H,Pitch))[298:,:], 2, 102)
pitch100_x3,pitch100_y3 = split_sequence(pitch100_scaled3, 100)
print(pitch100_x3.shape)
# 输入200步长输出3步长
pitch200_scaled3 = deal_data1(np.hstack((H,Pitch))[198:,:], 2, 202)
pitch200_x3,pitch200_y3 = split_sequence(pitch200_scaled3, 200)
print(pitch200_x3.shape)

# 输入1步长输出5步长
pitch1_scaled5 = deal_data1(np.hstack((H,Pitch))[395:,:], 2, 5)
pitch1_x5,pitch1_y5 = split_sequence(pitch1_scaled5, 1)
print(pitch1_x5.shape)
# 输入10步长输出5步长
pitch10_scaled5 = deal_data1(np.hstack((H,Pitch))[386:,:], 2, 14)
pitch10_x5,pitch10_y5 = split_sequence(pitch10_scaled5, 10)
print(pitch10_x5.shape)
# 输入30步长输出5步长
pitch30_scaled5 = deal_data1(np.hstack((H,Pitch))[366:,:], 2, 34)
pitch30_x5,pitch30_y5 = split_sequence(pitch30_scaled5, 30)
print(pitch30_x5.shape)
# 输入50步长输出5步长
pitch50_scaled5 = deal_data1(np.hstack((H,Pitch))[346:,:], 2, 54)
pitch50_x5,pitch50_y5 = split_sequence(pitch50_scaled5, 50)
print(pitch50_x5.shape)
# 输入100步长输出5步长
pitch100_scaled5 = deal_data1(np.hstack((H,Pitch))[296:,:], 2, 104)
pitch100_x5,pitch100_y5 = split_sequence(pitch100_scaled5, 100)
print(pitch100_x5.shape)
# 输入200步长输出5步长
pitch200_scaled5 = deal_data1(np.hstack((H,Pitch))[196:,:], 2, 204)
pitch200_x5,pitch200_y5 = split_sequence(pitch200_scaled5, 200)
print(pitch200_x5.shape)

# 输入1步长输出7步长
pitch1_scaled7 = deal_data1(np.hstack((H,Pitch))[393:,:], 2, 7)
pitch1_x7,pitch1_y7 = split_sequence(pitch1_scaled7, 1)
print(pitch1_x7.shape)
# 输入10步长输出7步长
pitch10_scaled7 = deal_data1(np.hstack((H,Pitch))[384:,:], 2, 16)
pitch10_x7,pitch10_y7 = split_sequence(pitch10_scaled7, 10)
print(pitch10_x7.shape)
# 输入30步长输出7步长
pitch30_scaled7 = deal_data1(np.hstack((H,Pitch))[364:,:], 2, 36)
pitch30_x7,pitch30_y7 = split_sequence(pitch30_scaled7, 30)
print(pitch30_x7.shape)
# 输入50步长输出7步长
pitch50_scaled7 = deal_data1(np.hstack((H,Pitch))[344:,:], 2, 56)
pitch50_x7,pitch50_y7 = split_sequence(pitch50_scaled7, 50)
print(pitch50_x7.shape)
# 输入100步长输出7步长
pitch100_scaled7 = deal_data1(np.hstack((H,Pitch))[294:,:], 2, 106)
pitch100_x7,pitch100_y7 = split_sequence(pitch100_scaled7, 100)
print(pitch100_x7.shape)
# 输入200步长输出7步长
pitch200_scaled7 = deal_data1(np.hstack((H,Pitch))[194:,:], 2, 206)
pitch200_x7,pitch200_y7 = split_sequence(pitch200_scaled7, 200)
print(pitch200_x7.shape)

# 输入1步长输出9步长
pitch1_scaled9 = deal_data1(np.hstack((H,Pitch))[391:,:], 2, 9)
pitch1_x9,pitch1_y9 = split_sequence(pitch1_scaled9, 1)
print(pitch1_x9.shape)
# 输入10步长输出9步长
pitch10_scaled9 = deal_data1(np.hstack((H,Pitch))[382:,:], 2, 18)
pitch10_x9,pitch10_y9 = split_sequence(pitch10_scaled9, 10)
print(pitch10_x9.shape)
# 输入30步长输出9步长
pitch30_scaled9 = deal_data1(np.hstack((H,Pitch))[362:,:], 2, 38)
pitch30_x9,pitch30_y9 = split_sequence(pitch30_scaled9, 30)
print(pitch30_x9.shape)
# 输入50步长输出9步长
pitch50_scaled9 = deal_data1(np.hstack((H,Pitch))[342:,:], 2, 58)
pitch50_x9,pitch50_y9 = split_sequence(pitch50_scaled9, 50)
print(pitch50_x9.shape)
# 输入100步长输出9步长
pitch100_scaled9 = deal_data1(np.hstack((H,Pitch))[292:,:], 2, 108)
pitch100_x9,pitch100_y9 = split_sequence(pitch100_scaled9, 100)
print(pitch100_x9.shape)
# 输入200步长输出9步长
pitch200_scaled9 = deal_data1(np.hstack((H,Pitch))[192:,:], 2, 208)
pitch200_x9,pitch200_y9 = split_sequence(pitch200_scaled9, 200)
print(pitch200_x9.shape)


# In[77]:


# pitch
# 1
train_pitch1_x1, train_pitch1_y1, valid_pitch1_x1, valid_pitch1_y1 = split_train_valid(pitch1_x1,pitch1_y1, 6900, 7500, 8000)
pitch1_model1, pitch1_history1, pre_pitch1_Y1 = Model_LSTM(train_pitch1_x1, train_pitch1_y1, valid_pitch1_x1, valid_pitch1_y1, lr=0.35, epochs=60, batch_size=50)
train_pitch10_x1, train_pitch10_y1, valid_pitch10_x1, valid_pitch10_y1 = split_train_valid(pitch10_x1,pitch10_y1, 6900, 7500, 8000)
pitch10_model1, pitch10_history1, pre_pitch10_Y1 = Model_LSTM(train_pitch10_x1, train_pitch10_y1, valid_pitch10_x1, valid_pitch10_y1, lr=0.35, epochs=60, batch_size=50)
train_pitch30_x1, train_pitch30_y1, valid_pitch30_x1, valid_pitch30_y1 = split_train_valid(pitch30_x1,pitch30_y1, 6900, 7500, 8000)
pitch30_model1, pitch30_history1, pre_pitch30_Y1 = Model_LSTM(train_pitch30_x1, train_pitch30_y1, valid_pitch30_x1, valid_pitch30_y1, lr=0.35, epochs=60, batch_size=50)
train_pitch50_x1, train_pitch50_y1, valid_pitch50_x1, valid_pitch50_y1 = split_train_valid(pitch50_x1,pitch50_y1, 6900, 7500, 8000)
pitch50_model1, pitch50_history1, pre_pitch50_Y1 = Model_LSTM(train_pitch50_x1, train_pitch50_y1, valid_pitch50_x1, valid_pitch50_y1, lr=0.35, epochs=60, batch_size=50)
train_pitch100_x1, train_pitch100_y1, valid_pitch100_x1, valid_pitch100_y1 = split_train_valid(pitch100_x1,pitch100_y1, 6900, 7500, 8000)
pitch100_model1, pitch100_history1, pre_pitch100_Y1 = Model_LSTM(train_pitch100_x1, train_pitch100_y1, valid_pitch100_x1, valid_pitch100_y1, lr=0.35, epochs=60, batch_size=50)
train_pitch200_x1, train_pitch200_y1, valid_pitch200_x1, valid_pitch200_y1 = split_train_valid(pitch200_x1,pitch200_y1, 6900, 7500, 8000)
pitch200_model1, pitch200_history1, pre_pitch200_Y1 = Model_LSTM(train_pitch200_x1, train_pitch200_y1, valid_pitch200_x1, valid_pitch200_y1, lr=0.35, epochs=60, batch_size=50)


# In[78]:


fan_pitch1_real1,fan_pitch1_pre1 = FanGuiHua_pitch(valid_pitch1_y1,pre_pitch1_Y1)
fan_pitch10_real1,fan_pitch10_pre1 = FanGuiHua_pitch(valid_pitch10_y1,pre_pitch10_Y1)
fan_pitch30_real1,fan_pitch30_pre1 = FanGuiHua_pitch(valid_pitch30_y1,pre_pitch30_Y1)
fan_pitch50_real1,fan_pitch50_pre1 = FanGuiHua_pitch(valid_pitch50_y1,pre_pitch50_Y1)
fan_pitch100_real1,fan_pitch100_pre1 = FanGuiHua_pitch(valid_pitch100_y1,pre_pitch100_Y1)
fan_pitch200_real1,fan_pitch200_pre1 = FanGuiHua_pitch(valid_pitch200_y1,pre_pitch200_Y1)
np.savetxt('pitchN_其他步长1.csv',np.hstack((fan_pitch1_real1,fan_pitch1_pre1,fan_pitch10_pre1,
                                        fan_pitch30_pre1,fan_pitch50_pre1,fan_pitch100_pre1,fan_pitch200_pre1)),delimiter=',')


# In[79]:


# 3
train_pitch1_x3, train_pitch1_y3, valid_pitch1_x3, valid_pitch1_y3 = split_train_valid(pitch1_x3,pitch1_y3, 6900, 7500, 8000)
pitch1_model3, pitch1_history3, pre_pitch1_Y3 = Model_LSTM(train_pitch1_x3, train_pitch1_y3, valid_pitch1_x3, valid_pitch1_y3, lr=0.35, epochs=60, batch_size=50)
train_pitch10_x3, train_pitch10_y3, valid_pitch10_x3, valid_pitch10_y3 = split_train_valid(pitch10_x3,pitch10_y3, 6900, 7500, 8000)
pitch10_model3, pitch10_history3, pre_pitch10_Y3 = Model_LSTM(train_pitch10_x3, train_pitch10_y3, valid_pitch10_x3, valid_pitch10_y3, lr=0.35, epochs=60, batch_size=50)
train_pitch30_x3, train_pitch30_y3, valid_pitch30_x3, valid_pitch30_y3 = split_train_valid(pitch30_x3,pitch30_y3, 6900, 7500, 8000)
pitch30_model3, pitch30_history3, pre_pitch30_Y3 = Model_LSTM(train_pitch30_x3, train_pitch30_y3, valid_pitch30_x3, valid_pitch30_y3, lr=0.35, epochs=60, batch_size=50)
train_pitch50_x3, train_pitch50_y3, valid_pitch50_x3, valid_pitch50_y3 = split_train_valid(pitch50_x3,pitch50_y3, 6900, 7500, 8000)
pitch50_model3, pitch50_history3, pre_pitch50_Y3 = Model_LSTM(train_pitch50_x3, train_pitch50_y3, valid_pitch50_x3, valid_pitch50_y3, lr=0.35, epochs=60, batch_size=50)
train_pitch100_x3, train_pitch100_y3, valid_pitch100_x3, valid_pitch100_y3 = split_train_valid(pitch100_x3,pitch100_y3, 6900, 7500, 8000)
pitch100_model3, pitch100_history3, pre_pitch100_Y3 = Model_LSTM(train_pitch100_x3, train_pitch100_y3, valid_pitch100_x3, valid_pitch100_y3, lr=0.35, epochs=60, batch_size=50)
train_pitch200_x3, train_pitch200_y3, valid_pitch200_x3, valid_pitch200_y3 = split_train_valid(pitch200_x3,pitch200_y3, 6900, 7500, 8000)
pitch200_model3, pitch200_history3, pre_pitch200_Y3 = Model_LSTM(train_pitch200_x3, train_pitch200_y3, valid_pitch200_x3, valid_pitch200_y3, lr=0.35, epochs=60, batch_size=50)


# In[80]:


fan_pitch1_real3,fan_pitch1_pre3 = FanGuiHua_pitch(valid_pitch1_y3,pre_pitch1_Y3)
fan_pitch10_real3,fan_pitch10_pre3 = FanGuiHua_pitch(valid_pitch10_y3,pre_pitch10_Y3)
fan_pitch30_real3,fan_pitch30_pre3 = FanGuiHua_pitch(valid_pitch30_y3,pre_pitch30_Y3)
fan_pitch50_real3,fan_pitch50_pre3 = FanGuiHua_pitch(valid_pitch50_y3,pre_pitch50_Y3)
fan_pitch100_real3,fan_pitch100_pre3 = FanGuiHua_pitch(valid_pitch100_y3,pre_pitch100_Y3)
fan_pitch200_real3,fan_pitch200_pre3 = FanGuiHua_pitch(valid_pitch200_y3,pre_pitch200_Y3)
np.savetxt('pitchN_其他步长3.csv',np.hstack((fan_pitch1_real3,fan_pitch1_pre3,fan_pitch10_pre3,
                                        fan_pitch30_pre3,fan_pitch50_pre3,fan_pitch100_pre3,fan_pitch200_pre3)),delimiter=',')


# In[81]:


# 5
train_pitch1_x5, train_pitch1_y5, valid_pitch1_x5, valid_pitch1_y5 = split_train_valid(pitch1_x5,pitch1_y5, 6900, 7500, 8000)
pitch1_model5, pitch1_history5, pre_pitch1_Y5 = Model_LSTM(train_pitch1_x5, train_pitch1_y5, valid_pitch1_x5, valid_pitch1_y5, lr=0.35, epochs=60, batch_size=50)
train_pitch10_x5, train_pitch10_y5, valid_pitch10_x5, valid_pitch10_y5 = split_train_valid(pitch10_x5,pitch10_y5, 6900, 7500, 8000)
pitch10_model5, pitch10_history5, pre_pitch10_Y5 = Model_LSTM(train_pitch10_x5, train_pitch10_y5, valid_pitch10_x5, valid_pitch10_y5, lr=0.35, epochs=60, batch_size=50)
train_pitch30_x5, train_pitch30_y5, valid_pitch30_x5, valid_pitch30_y5 = split_train_valid(pitch30_x5,pitch30_y5, 6900, 7500, 8000)
pitch30_model5, pitch30_history5, pre_pitch30_Y5 = Model_LSTM(train_pitch30_x5, train_pitch30_y5, valid_pitch30_x5, valid_pitch30_y5, lr=0.35, epochs=60, batch_size=50)
train_pitch50_x5, train_pitch50_y5, valid_pitch50_x5, valid_pitch50_y5 = split_train_valid(pitch50_x5,pitch50_y5, 6900, 7500, 8000)
pitch50_model5, pitch50_history5, pre_pitch50_Y5 = Model_LSTM(train_pitch50_x5, train_pitch50_y5, valid_pitch50_x5, valid_pitch50_y5, lr=0.35, epochs=60, batch_size=50)
train_pitch100_x5, train_pitch100_y5, valid_pitch100_x5, valid_pitch100_y5 = split_train_valid(pitch100_x5,pitch100_y5, 6900, 7500, 8000)
pitch100_model5, pitch100_history5, pre_pitch100_Y5 = Model_LSTM(train_pitch100_x5, train_pitch100_y5, valid_pitch100_x5, valid_pitch100_y5, lr=0.35, epochs=60, batch_size=50)
train_pitch200_x5, train_pitch200_y5, valid_pitch200_x5, valid_pitch200_y5 = split_train_valid(pitch200_x5,pitch200_y5, 6900, 7500, 8000)
pitch200_model5, pitch200_history5, pre_pitch200_Y5 = Model_LSTM(train_pitch200_x5, train_pitch200_y5, valid_pitch200_x5, valid_pitch200_y5, lr=0.35, epochs=60, batch_size=50)


# In[82]:


fan_pitch1_real5,fan_pitch1_pre5 = FanGuiHua_pitch(valid_pitch1_y5,pre_pitch1_Y5)
fan_pitch10_real5,fan_pitch10_pre5 = FanGuiHua_pitch(valid_pitch10_y5,pre_pitch10_Y5)
fan_pitch30_real5,fan_pitch30_pre5 = FanGuiHua_pitch(valid_pitch30_y5,pre_pitch30_Y5)
fan_pitch50_real5,fan_pitch50_pre5 = FanGuiHua_pitch(valid_pitch50_y5,pre_pitch50_Y5)
fan_pitch100_real5,fan_pitch100_pre5 = FanGuiHua_pitch(valid_pitch100_y5,pre_pitch100_Y5)
fan_pitch200_real5,fan_pitch200_pre5 = FanGuiHua_pitch(valid_pitch200_y5,pre_pitch200_Y5)
np.savetxt('pitchN_其他步长5.csv',np.hstack((fan_pitch1_real5,fan_pitch1_pre5,fan_pitch10_pre5,
                                        fan_pitch30_pre5,fan_pitch50_pre5,fan_pitch100_pre5,fan_pitch200_pre5)),delimiter=',')


# In[83]:


# 7
train_pitch1_x7, train_pitch1_y7, valid_pitch1_x7, valid_pitch1_y7 = split_train_valid(pitch1_x7,pitch1_y7, 6900, 7500, 8000)
pitch1_model7, pitch1_history7, pre_pitch1_Y7 = Model_LSTM(train_pitch1_x7, train_pitch1_y7, valid_pitch1_x7, valid_pitch1_y7, lr=0.35, epochs=60, batch_size=50)
train_pitch10_x7, train_pitch10_y7, valid_pitch10_x7, valid_pitch10_y7 = split_train_valid(pitch10_x7,pitch10_y7, 6900, 7500, 8000)
pitch10_model7, pitch10_history7, pre_pitch10_Y7 = Model_LSTM(train_pitch10_x7, train_pitch10_y7, valid_pitch10_x7, valid_pitch10_y7, lr=0.35, epochs=60, batch_size=50)
train_pitch30_x7, train_pitch30_y7, valid_pitch30_x7, valid_pitch30_y7 = split_train_valid(pitch30_x7,pitch30_y7, 6900, 7500, 8000)
pitch30_model7,pitch30_history7, pre_pitch30_Y7 = Model_LSTM(train_pitch30_x7, train_pitch30_y7, valid_pitch30_x7, valid_pitch30_y7, lr=0.35, epochs=60, batch_size=50)
train_pitch50_x7, train_pitch50_y7, valid_pitch50_x7, valid_pitch50_y7 = split_train_valid(pitch50_x7,pitch50_y7, 6900, 7500, 8000)
pitch50_model7, pitch50_histor7, pre_pitch50_Y7 = Model_LSTM(train_pitch50_x7, train_pitch50_y7, valid_pitch50_x7, valid_pitch50_y7, lr=0.35, epochs=60, batch_size=50)
train_pitch100_x7, train_pitch100_y7, valid_pitch100_x7, valid_pitch100_y7 = split_train_valid(pitch100_x7,pitch100_y7, 6900, 7500, 8000)
pitch100_model7, pitch100_history7, pre_pitch100_Y7 = Model_LSTM(train_pitch100_x7, train_pitch100_y7, valid_pitch100_x7, valid_pitch100_y7, lr=0.35, epochs=60, batch_size=50)
train_pitch200_x7, train_pitch200_y7, valid_pitch200_x7, valid_pitch200_y7 = split_train_valid(pitch200_x7,pitch200_y7, 6900, 7500, 8000)
pitch200_model7, pitch200_history7, pre_pitch200_Y7 = Model_LSTM(train_pitch200_x7, train_pitch200_y7, valid_pitch200_x7, valid_pitch200_y7, lr=0.35, epochs=60, batch_size=50)


# In[84]:


fan_pitch1_real7,fan_pitch1_pre7 = FanGuiHua_pitch(valid_pitch1_y7,pre_pitch1_Y7)
fan_pitch10_real7,fan_pitch10_pre7 = FanGuiHua_pitch(valid_pitch10_y7,pre_pitch10_Y7)
fan_pitch30_real7,fan_pitch30_pre7 = FanGuiHua_pitch(valid_pitch30_y7,pre_pitch30_Y7)
fan_pitch50_real7,fan_pitch50_pre7 = FanGuiHua_pitch(valid_pitch50_y7,pre_pitch50_Y7)
fan_pitch100_real7,fan_pitch100_pre7 = FanGuiHua_pitch(valid_pitch100_y7,pre_pitch100_Y7)
fan_pitch200_real7,fan_pitch200_pre7 = FanGuiHua_pitch(valid_pitch200_y7,pre_pitch200_Y7)
np.savetxt('pitchN_其他步长7.csv',np.hstack((fan_pitch1_real7,fan_pitch1_pre7,fan_pitch10_pre7,
                                        fan_pitch30_pre7,fan_pitch50_pre7,fan_pitch100_pre7,fan_pitch200_pre7)),delimiter=',')


# In[85]:


# 9
train_pitch1_x9, train_pitch1_y9, valid_pitch1_x9, valid_pitch1_y9 = split_train_valid(pitch1_x9,pitch1_y9, 6900, 7500, 8000)
pitch1_model9, pitch1_history9, pre_pitch1_Y9 = Model_LSTM(train_pitch1_x9, train_pitch1_y9, valid_pitch1_x9, valid_pitch1_y9, lr=0.35, epochs=60, batch_size=50)
train_pitch10_x9, train_pitch10_y9, valid_pitch10_x9, valid_pitch10_y9 = split_train_valid(pitch10_x9,pitch10_y9, 6900, 7500, 8000)
pitch10_model9, pitch10_history9, pre_pitch10_Y9 = Model_LSTM(train_pitch10_x9, train_pitch10_y9, valid_pitch10_x9, valid_pitch10_y9, lr=0.35, epochs=60, batch_size=50)
train_pitch30_x9, train_pitch30_y9, valid_pitch30_x9, valid_pitch30_y9 = split_train_valid(pitch30_x9,pitch30_y9, 6900, 7500, 8000)
pitch30_model9, pitch30_history9, pre_pitch30_Y9 = Model_LSTM(train_pitch30_x9, train_pitch30_y9, valid_pitch30_x9, valid_pitch30_y9, lr=0.35, epochs=60, batch_size=50)
train_pitch50_x9, train_pitch50_y9, valid_pitch50_x9, valid_pitch50_y9 = split_train_valid(pitch50_x9,pitch50_y9, 6900, 7500, 8000)
pitch50_model9, pitch50_history9, pre_pitch50_Y9 = Model_LSTM(train_pitch50_x9, train_pitch50_y9, valid_pitch50_x9, valid_pitch50_y9, lr=0.35, epochs=60, batch_size=50)
train_pitch100_x9, train_pitch100_y9, valid_pitch100_x9, valid_pitch100_y9 = split_train_valid(pitch100_x9,pitch100_y9, 6900, 7500, 8000)
pitch100_model9, pitch100_history9, pre_pitch100_Y9 = Model_LSTM(train_pitch100_x9, train_pitch100_y9, valid_pitch100_x9, valid_pitch100_y9, lr=0.35, epochs=60, batch_size=50)
train_pitch200_x9, train_pitch200_y9, valid_pitch200_x9, valid_pitch200_y9 = split_train_valid(pitch200_x9,pitch200_y9, 6900, 7500, 8000)
pitch200_model9, pitch200_history9, pre_pitch200_Y9 = Model_LSTM(train_pitch200_x9, train_pitch200_y9, valid_pitch200_x9, valid_pitch200_y9, lr=0.35, epochs=60, batch_size=50)


# In[86]:


fan_pitch1_real9,fan_pitch1_pre9 = FanGuiHua_pitch(valid_pitch1_y9,pre_pitch1_Y9)
fan_pitch10_real9,fan_pitch10_pre9 = FanGuiHua_pitch(valid_pitch10_y9,pre_pitch10_Y9)
fan_pitch30_real9,fan_pitch30_pre9 = FanGuiHua_pitch(valid_pitch30_y9,pre_pitch30_Y9)
fan_pitch50_real9,fan_pitch50_pre9 = FanGuiHua_pitch(valid_pitch50_y9,pre_pitch50_Y9)
fan_pitch100_real9,fan_pitch100_pre9 = FanGuiHua_pitch(valid_pitch100_y9,pre_pitch100_Y9)
fan_pitch200_real9,fan_pitch200_pre9 = FanGuiHua_pitch(valid_pitch200_y9,pre_pitch200_Y9)
np.savetxt('pitchN_其他步长9.csv',np.hstack((fan_pitch1_real9,fan_pitch1_pre9,fan_pitch10_pre9,
                                        fan_pitch30_pre9,fan_pitch50_pre9,fan_pitch100_pre9,fan_pitch200_pre9)),delimiter=',')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


# heave
# 输入1步长输出1步长
heave1_scaled1 = deal_data1(np.hstack((H,Heave))[399:,:], 2, 1)
heave1_x1,heave1_y1 = split_sequence(heave1_scaled1, 1)
print(heave1_x1.shape)
# 输入10步长输出1步长
heave10_scaled1 = deal_data1(np.hstack((H,Heave))[390:,:], 2, 10)
heave10_x1,heave10_y1 = split_sequence(heave10_scaled1, 10)
print(heave10_x1.shape)
# 输入30步长输出1步长
heave30_scaled1 = deal_data1(np.hstack((H,Heave))[370:,:], 2, 30)
heave30_x1,heave30_y1 = split_sequence(heave30_scaled1, 30)
print(heave30_x1.shape)
# 输入50步长输出1步长
heave50_scaled1 = deal_data1(np.hstack((H,Heave))[350:,:], 2, 50)
heave50_x1,heave50_y1 = split_sequence(heave50_scaled1, 50)
print(heave50_x1.shape)
# 输入100步长输出1步长
heave100_scaled1 = deal_data1(np.hstack((H,Heave))[300:,:], 2, 100)
heave100_x1,heave100_y1 = split_sequence(heave100_scaled1, 100)
print(heave100_x1.shape)
# 输入200步长输出1步长
heave200_scaled1 = deal_data1(np.hstack((H,Heave))[200:,:], 2, 200)
heave200_x1,heave200_y1 = split_sequence(heave200_scaled1, 200)
print(heave200_x1.shape)

# 输入1步长输出3步长
heave1_scaled3 = deal_data1(np.hstack((H,Heave))[397:,:], 2, 3)
heave1_x3,heave1_y3 = split_sequence(heave1_scaled3, 1)
print(heave1_x3.shape)
# 输入10步长输出3步长
heave10_scaled3 = deal_data1(np.hstack((H,Heave))[388:,:], 2, 12)
heave10_x3,heave10_y3 = split_sequence(heave10_scaled3, 10)
print(heave10_x3.shape)
# 输入30步长输出3步长
heave30_scaled3 = deal_data1(np.hstack((H,Heave))[368:,:], 2, 32)
heave30_x3,heave30_y3 = split_sequence(heave30_scaled3, 30)
print(heave30_x3.shape)
# 输入50步长输出3步长
heave50_scaled3 = deal_data1(np.hstack((H,Heave))[348:,:], 2, 52)
heave50_x3,heave50_y3 = split_sequence(heave50_scaled3, 50)
print(heave50_x3.shape)
# 输入100步长输出3步长
heave100_scaled3 = deal_data1(np.hstack((H,Heave))[298:,:], 2, 102)
heave100_x3,heave100_y3 = split_sequence(heave100_scaled3, 100)
print(heave100_x3.shape)
# 输入200步长输出3步长
heave200_scaled3 = deal_data1(np.hstack((H,Heave))[198:,:], 2, 202)
heave200_x3,heave200_y3 = split_sequence(heave200_scaled3, 200)
print(heave200_x3.shape)

# 输入1步长输出5步长
heave1_scaled5 = deal_data1(np.hstack((H,Heave))[395:,:], 2, 5)
heave1_x5,heave1_y5 = split_sequence(heave1_scaled5, 1)
print(heave1_x5.shape)
# 输入10步长输出5步长
heave10_scaled5 = deal_data1(np.hstack((H,Heave))[386:,:], 2, 14)
heave10_x5,heave10_y5 = split_sequence(heave10_scaled5, 10)
print(heave10_x5.shape)
# 输入30步长输出5步长
heave30_scaled5 = deal_data1(np.hstack((H,Heave))[366:,:], 2, 34)
heave30_x5,heave30_y5 = split_sequence(heave30_scaled5, 30)
print(heave30_x5.shape)
# 输入50步长输出5步长
heave50_scaled5 = deal_data1(np.hstack((H,Heave))[346:,:], 2, 54)
heave50_x5,heave50_y5 = split_sequence(heave50_scaled5, 50)
print(heave50_x5.shape)
# 输入100步长输出5步长
heave100_scaled5 = deal_data1(np.hstack((H,Heave))[296:,:], 2, 104)
heave100_x5,heave100_y5 = split_sequence(heave100_scaled5, 100)
print(heave100_x5.shape)
# 输入200步长输出5步长
heave200_scaled5 = deal_data1(np.hstack((H,Heave))[196:,:], 2, 204)
heave200_x5,heave200_y5 = split_sequence(heave200_scaled5, 200)
print(heave200_x5.shape)

# 输入1步长输出7步长
heave1_scaled7 = deal_data1(np.hstack((H,Heave))[393:,:], 2, 7)
heave1_x7,heave1_y7 = split_sequence(heave1_scaled7, 1)
print(heave1_x7.shape)
# 输入10步长输出7步长
heave10_scaled7 = deal_data1(np.hstack((H,Heave))[384:,:], 2, 16)
heave10_x7,heave10_y7 = split_sequence(heave10_scaled7, 10)
print(heave10_x7.shape)
# 输入30步长输出7步长
heave30_scaled7 = deal_data1(np.hstack((H,Heave))[364:,:], 2, 36)
heave30_x7,heave30_y7 = split_sequence(heave30_scaled7, 30)
print(heave30_x7.shape)
# 输入50步长输出7步长
heave50_scaled7 = deal_data1(np.hstack((H,Heave))[344:,:], 2, 56)
heave50_x7,heave50_y7 = split_sequence(heave50_scaled7, 50)
print(heave50_x7.shape)
# 输入100步长输出7步长
heave100_scaled7 = deal_data1(np.hstack((H,Heave))[294:,:], 2, 106)
heave100_x7,heave100_y7 = split_sequence(heave100_scaled7, 100)
print(heave100_x7.shape)
# 输入200步长输出7步长
heave200_scaled7 = deal_data1(np.hstack((H,Heave))[194:,:], 2, 206)
heave200_x7,heave200_y7 = split_sequence(heave200_scaled7, 200)
print(heave200_x7.shape)

# 输入1步长输出9步长
heave1_scaled9 = deal_data1(np.hstack((H,Heave))[391:,:], 2, 9)
heave1_x9,heave1_y9 = split_sequence(heave1_scaled9, 1)
print(heave1_x9.shape)
# 输入10步长输出9步长
heave10_scaled9 = deal_data1(np.hstack((H,Heave))[382:,:], 2, 18)
heave10_x9,heave10_y9 = split_sequence(heave10_scaled9, 10)
print(heave10_x9.shape)
# 输入30步长输出9步长
heave30_scaled9 = deal_data1(np.hstack((H,Heave))[362:,:], 2, 38)
heave30_x9,heave30_y9 = split_sequence(heave30_scaled9, 30)
print(heave30_x9.shape)
# 输入50步长输出9步长
heave50_scaled9 = deal_data1(np.hstack((H,Heave))[342:,:], 2, 58)
heave50_x9,heave50_y9 = split_sequence(heave50_scaled9, 50)
print(heave50_x9.shape)
# 输入100步长输出9步长
heave100_scaled9 = deal_data1(np.hstack((H,Heave))[292:,:], 2, 108)
heave100_x9,heave100_y9 = split_sequence(heave100_scaled9, 100)
print(heave100_x9.shape)
# 输入200步长输出9步长
heave200_scaled9 = deal_data1(np.hstack((H,Heave))[192:,:], 2, 208)
heave200_x9,heave200_y9 = split_sequence(heave200_scaled9, 200)
print(heave200_x9.shape)


# In[12]:


# 输入1步长输出2步长
heave1_scaled2 = deal_data1(np.hstack((H,Heave))[398:,:], 2, 2)
heave1_x2,heave1_y2 = split_sequence(heave1_scaled2, 1)
print(heave1_x2.shape)
# 输入10步长输出2步长
heave10_scaled2 = deal_data1(np.hstack((H,Heave))[389:,:], 2, 11)
heave10_x2,heave10_y2 = split_sequence(heave10_scaled2, 10)
print(heave10_x2.shape)
# 输入30步长输出2步长
heave30_scaled2 = deal_data1(np.hstack((H,Heave))[369:,:], 2, 31)
heave30_x2,heave30_y2 = split_sequence(heave30_scaled2, 30)
print(heave30_x2.shape)
# 输入50步长输出2步长
heave50_scaled2 = deal_data1(np.hstack((H,Heave))[349:,:], 2, 51)
heave50_x2,heave50_y2 = split_sequence(heave50_scaled2, 50)
print(heave50_x2.shape)
# 输入100步长输出2步长
heave100_scaled2 = deal_data1(np.hstack((H,Heave))[299:,:], 2, 101)
heave100_x2,heave100_y2 = split_sequence(heave100_scaled2, 100)
print(heave100_x2.shape)
# 输入200步长输出2步长
heave200_scaled2 = deal_data1(np.hstack((H,Heave))[199:,:], 2, 201)
heave200_x2,heave200_y2 = split_sequence(heave200_scaled2, 200)
print(heave200_x2.shape)

# 输入1步长输出4步长
heave1_scaled4 = deal_data1(np.hstack((H,Heave))[396:,:], 2, 4)
heave1_x4,heave1_y4 = split_sequence(heave1_scaled4, 1)
print(heave1_x4.shape)
# 输入10步长输出4步长
heave10_scaled4 = deal_data1(np.hstack((H,Heave))[387:,:], 2, 13)
heave10_x4,heave10_y4 = split_sequence(heave10_scaled4, 10)
print(heave10_x4.shape)
# 输入30步长输出4步长
heave30_scaled4 = deal_data1(np.hstack((H,Heave))[367:,:], 2, 33)
heave30_x4,heave30_y4 = split_sequence(heave30_scaled4, 30)
print(heave30_x4.shape)
# 输入50步长输出4步长
heave50_scaled4 = deal_data1(np.hstack((H,Heave))[347:,:], 2, 53)
heave50_x4,heave50_y4 = split_sequence(heave50_scaled4, 50)
print(heave50_x4.shape)
# 输入100步长输出4步长
heave100_scaled4 = deal_data1(np.hstack((H,Heave))[297:,:], 2, 103)
heave100_x4,heave100_y4 = split_sequence(heave100_scaled4, 100)
print(heave100_x4.shape)
# 输入200步长输出4步长
heave200_scaled4 = deal_data1(np.hstack((H,Heave))[197:,:], 2, 203)
heave200_x4,heave200_y4 = split_sequence(heave200_scaled4, 200)
print(heave200_x4.shape)

# 输入1步长输出6步长
heave1_scaled6 = deal_data1(np.hstack((H,Heave))[394:,:], 2, 6)
heave1_x6,heave1_y6 = split_sequence(heave1_scaled6, 1)
print(heave1_x6.shape)
# 输入10步长输出6步长
heave10_scaled6 = deal_data1(np.hstack((H,Heave))[385:,:], 2, 15)
heave10_x6,heave10_y6 = split_sequence(heave10_scaled6, 10)
print(heave10_x6.shape)
# 输入30步长输出6步长
heave30_scaled6 = deal_data1(np.hstack((H,Heave))[365:,:], 2, 35)
heave30_x6,heave30_y6 = split_sequence(heave30_scaled6, 30)
print(heave30_x6.shape)
# 输入50步长输出6步长
heave50_scaled6 = deal_data1(np.hstack((H,Heave))[345:,:], 2, 55)
heave50_x6,heave50_y6 = split_sequence(heave50_scaled6, 50)
print(heave50_x6.shape)
# 输入100步长输出6步长
heave100_scaled6 = deal_data1(np.hstack((H,Heave))[295:,:], 2, 105)
heave100_x6,heave100_y6 = split_sequence(heave100_scaled6, 100)
print(heave100_x6.shape)
# 输入200步长输出6步长
heave200_scaled6 = deal_data1(np.hstack((H,Heave))[195:,:], 2, 205)
heave200_x6,heave200_y6 = split_sequence(heave200_scaled6, 200)
print(heave200_x6.shape)

# 输入1步长输出8步长
heave1_scaled8 = deal_data1(np.hstack((H,Heave))[392:,:], 2, 8)
heave1_x8,heave1_y8 = split_sequence(heave1_scaled8, 1)
print(heave1_x8.shape)
# 输入10步长输出8步长
heave10_scaled8 = deal_data1(np.hstack((H,Heave))[383:,:], 2, 17)
heave10_x8,heave10_y8 = split_sequence(heave10_scaled8, 10)
print(heave10_x8.shape)
# 输入30步长输出8步长
heave30_scaled8 = deal_data1(np.hstack((H,Heave))[363:,:], 2, 37)
heave30_x8,heave30_y8 = split_sequence(heave30_scaled8, 30)
print(heave30_x8.shape)
# 输入50步长输出8步长
heave50_scaled8 = deal_data1(np.hstack((H,Heave))[343:,:], 2, 57)
heave50_x8,heave50_y8 = split_sequence(heave50_scaled8, 50)
print(heave50_x8.shape)
# 输入100步长输出8步长
heave100_scaled8 = deal_data1(np.hstack((H,Heave))[293:,:], 2, 107)
heave100_x8,heave100_y8 = split_sequence(heave100_scaled8, 100)
print(heave100_x8.shape)
# 输入200步长输出8步长
heave200_scaled8 = deal_data1(np.hstack((H,Heave))[193:,:], 2, 207)
heave200_x8,heave200_y8 = split_sequence(heave200_scaled8, 200)
print(heave200_x8.shape)


# In[88]:


# heave
# 1
train_heave1_x1, train_heave1_y1, valid_heave1_x1, valid_heave1_y1 = split_train_valid(heave1_x1,heave1_y1, 6900, 7500, 8000)
heave1_model1, heave1_history1, pre_heave1_Y1 = Model_LSTM(train_heave1_x1, train_heave1_y1, valid_heave1_x1, valid_heave1_y1, lr=0.1, epochs=60, batch_size=50)
train_heave10_x1, train_heave10_y1, valid_heave10_x1, valid_heave10_y1 = split_train_valid(heave10_x1,heave10_y1, 6900, 7500, 8000)
heave10_model1, heave10_history1, pre_heave10_Y1 = Model_LSTM(train_heave10_x1, train_heave10_y1, valid_heave10_x1, valid_heave10_y1, lr=0.1, epochs=60, batch_size=50)
train_heave30_x1, train_heave30_y1, valid_heave30_x1, valid_heave30_y1 = split_train_valid(heave30_x1,heave30_y1, 6900, 7500, 8000)
heave30_model1, heave30_history1, pre_heave30_Y1 = Model_LSTM(train_heave30_x1, train_heave30_y1, valid_heave30_x1, valid_heave30_y1, lr=0.1, epochs=60, batch_size=50)
train_heave50_x1, train_heave50_y1, valid_heave50_x1, valid_heave50_y1 = split_train_valid(heave50_x1,heave50_y1, 6900, 7500, 8000)
heave50_model1, heave50_history1, pre_heave50_Y1 = Model_LSTM(train_heave50_x1, train_heave50_y1, valid_heave50_x1, valid_heave50_y1, lr=0.1, epochs=60, batch_size=50)
train_heave100_x1, train_heave100_y1, valid_heave100_x1, valid_heave100_y1 = split_train_valid(heave100_x1,heave100_y1, 6900, 7500, 8000)
heave100_model1, heave100_history1, pre_heave100_Y1 = Model_LSTM(train_heave100_x1, train_heave100_y1, valid_heave100_x1, valid_heave100_y1, lr=0.1, epochs=60, batch_size=50)
train_heave200_x1, train_heave200_y1, valid_heave200_x1, valid_heave200_y1 = split_train_valid(heave200_x1,heave200_y1, 6900, 7500, 8000)
heave200_model1, heave200_history1, pre_heave200_Y1 = Model_LSTM(train_heave200_x1, train_heave200_y1, valid_heave200_x1, valid_heave200_y1, lr=0.1, epochs=60, batch_size=50)

fan_heave1_real1,fan_heave1_pre1 = FanGuiHua_heave(valid_heave1_y1,pre_heave1_Y1)
fan_heave10_real1,fan_heave10_pre1 = FanGuiHua_heave(valid_heave10_y1,pre_heave10_Y1)
fan_heave30_real1,fan_heave30_pre1 = FanGuiHua_heave(valid_heave30_y1,pre_heave30_Y1)
fan_heave50_real1,fan_heave50_pre1 = FanGuiHua_heave(valid_heave50_y1,pre_heave50_Y1)
fan_heave100_real1,fan_heave100_pre1 = FanGuiHua_heave(valid_heave100_y1,pre_heave100_Y1)
fan_heave200_real1,fan_heave200_pre1 = FanGuiHua_heave(valid_heave200_y1,pre_heave200_Y1)
np.savetxt('heaveN_其他步长1.csv',np.hstack((fan_heave1_real1,fan_heave1_pre1,fan_heave10_pre1,
                                        fan_heave30_pre1,fan_heave50_pre1,fan_heave100_pre1,fan_heave200_pre1)),delimiter=',')


# In[89]:


# 3
train_heave1_x3, train_heave1_y3, valid_heave1_x3, valid_heave1_y3 = split_train_valid(heave1_x3,heave1_y3, 6900, 7500, 8000)
heave1_model3, heave1_history3, pre_heave1_Y3 = Model_LSTM(train_heave1_x3, train_heave1_y3, valid_heave1_x3, valid_heave1_y3, lr=0.1, epochs=60, batch_size=50)
train_heave10_x3, train_heave10_y3, valid_heave10_x3, valid_heave10_y3 = split_train_valid(heave10_x3,heave10_y3, 6900, 7500, 8000)
heave10_model3, heave10_history3, pre_heave10_Y3 = Model_LSTM(train_heave10_x3, train_heave10_y3, valid_heave10_x3, valid_heave10_y3, lr=0.1, epochs=60, batch_size=50)
train_heave30_x3, train_heave30_y3, valid_heave30_x3, valid_heave30_y3 = split_train_valid(heave30_x3,heave30_y3, 6900, 7500, 8000)
heave30_model3, heave30_history3, pre_heave30_Y3 = Model_LSTM(train_heave30_x3, train_heave30_y3, valid_heave30_x3, valid_heave30_y3, lr=0.1, epochs=60, batch_size=50)
train_heave50_x3, train_heave50_y3, valid_heave50_x3, valid_heave50_y3 = split_train_valid(heave50_x3,heave50_y3, 6900, 7500, 8000)
heave50_model3, heave50_history3, pre_heave50_Y3 = Model_LSTM(train_heave50_x3, train_heave50_y3, valid_heave50_x3, valid_heave50_y3, lr=0.1, epochs=60, batch_size=50)
train_heave100_x3, train_heave100_y3, valid_heave100_x3, valid_heave100_y3 = split_train_valid(heave100_x3,heave100_y3, 6900, 7500, 8000)
heave100_model3, heave100_history3, pre_heave100_Y3 = Model_LSTM(train_heave100_x3, train_heave100_y3, valid_heave100_x3, valid_heave100_y3, lr=0.1, epochs=60, batch_size=50)
train_heave200_x3, train_heave200_y3, valid_heave200_x3, valid_heave200_y3 = split_train_valid(heave200_x3,heave200_y3, 6900, 7500, 8000)
heave200_model3, heave200_history3, pre_heave200_Y3 = Model_LSTM(train_heave200_x3, train_heave200_y3, valid_heave200_x3, valid_heave200_y3, lr=0.1, epochs=60, batch_size=50)

fan_heave1_real3,fan_heave1_pre3 = FanGuiHua_heave(valid_heave1_y3,pre_heave1_Y3)
fan_heave10_real3,fan_heave10_pre3 = FanGuiHua_heave(valid_heave10_y3,pre_heave10_Y3)
fan_heave30_real3,fan_heave30_pre3 = FanGuiHua_heave(valid_heave30_y3,pre_heave30_Y3)
fan_heave50_real3,fan_heave50_pre3 = FanGuiHua_heave(valid_heave50_y3,pre_heave50_Y3)
fan_heave100_real3,fan_heave100_pre3 = FanGuiHua_heave(valid_heave100_y3,pre_heave100_Y3)
fan_heave200_real3,fan_heave200_pre3 = FanGuiHua_heave(valid_heave200_y3,pre_heave200_Y3)
np.savetxt('heaveN_其他步长3.csv',np.hstack((fan_heave1_real3,fan_heave1_pre3,fan_heave10_pre3,
                                        fan_heave30_pre3,fan_heave50_pre3,fan_heave100_pre3,fan_heave200_pre3)),delimiter=',')


# In[90]:


# 5
train_heave1_x5, train_heave1_y5, valid_heave1_x5, valid_heave1_y5 = split_train_valid(heave1_x5,heave1_y5, 6900, 7500, 8000)
heave1_model5, heave1_history5, pre_heave1_Y5 = Model_LSTM(train_heave1_x5, train_heave1_y5, valid_heave1_x5, valid_heave1_y5, lr=0.1, epochs=60, batch_size=50)
train_heave10_x5, train_heave10_y5, valid_heave10_x5, valid_heave10_y5 = split_train_valid(heave10_x5,heave10_y5, 6900, 7500, 8000)
heave10_model5, heave10_history5, pre_heave10_Y5 = Model_LSTM(train_heave10_x5, train_heave10_y5, valid_heave10_x5, valid_heave10_y5, lr=0.1, epochs=60, batch_size=50)
train_heave30_x5, train_heave30_y5, valid_heave30_x5, valid_heave30_y5 = split_train_valid(heave30_x5,heave30_y5, 6900, 7500, 8000)
heave30_model5, heave30_history5, pre_heave30_Y5 = Model_LSTM(train_heave30_x5, train_heave30_y5, valid_heave30_x5, valid_heave30_y5, lr=0.1, epochs=60, batch_size=50)
train_heave50_x5, train_heave50_y5, valid_heave50_x5, valid_heave50_y5 = split_train_valid(heave50_x5,heave50_y5, 6900, 7500, 8000)
heave50_model5, heave50_history5, pre_heave50_Y5 = Model_LSTM(train_heave50_x5, train_heave50_y5, valid_heave50_x5, valid_heave50_y5, lr=0.1, epochs=60, batch_size=50)
train_heave100_x5, train_heave100_y5, valid_heave100_x5, valid_heave100_y5 = split_train_valid(heave100_x5,heave100_y5, 6900, 7500, 8000)
heave100_model5, heave100_history5, pre_heave100_Y5 = Model_LSTM(train_heave100_x5, train_heave100_y5, valid_heave100_x5, valid_heave100_y5, lr=0.1, epochs=60, batch_size=50)
train_heave200_x5, train_heave200_y5, valid_heave200_x5, valid_heave200_y5 = split_train_valid(heave200_x5,heave200_y5, 6900, 7500, 8000)
heave200_model5, heave200_history5, pre_heave200_Y5 = Model_LSTM(train_heave200_x5, train_heave200_y5, valid_heave200_x5, valid_heave200_y5, lr=0.1, epochs=60, batch_size=50)

fan_heave1_real5,fan_heave1_pre5 = FanGuiHua_heave(valid_heave1_y5,pre_heave1_Y5)
fan_heave10_real5,fan_heave10_pre5 = FanGuiHua_heave(valid_heave10_y5,pre_heave10_Y5)
fan_heave30_real5,fan_heave30_pre5 = FanGuiHua_heave(valid_heave30_y5,pre_heave30_Y5)
fan_heave50_real5,fan_heave50_pre5 = FanGuiHua_heave(valid_heave50_y5,pre_heave50_Y5)
fan_heave100_real5,fan_heave100_pre5 = FanGuiHua_heave(valid_heave100_y5,pre_heave100_Y5)
fan_heave200_real5,fan_heave200_pre5 = FanGuiHua_heave(valid_heave200_y5,pre_heave200_Y5)
np.savetxt('heaveN_其他步长5.csv',np.hstack((fan_heave1_real5,fan_heave1_pre5,fan_heave10_pre5,
                                        fan_heave30_pre5,fan_heave50_pre5,fan_heave100_pre5,fan_heave200_pre5)),delimiter=',')


# In[91]:


# 7
train_heave1_x7, train_heave1_y7, valid_heave1_x7, valid_heave1_y7 = split_train_valid(heave1_x7,heave1_y7, 6900, 7500, 8000)
heave1_model7, heave1_history7, pre_heave1_Y7 = Model_LSTM(train_heave1_x7, train_heave1_y7, valid_heave1_x7, valid_heave1_y7, lr=0.1, epochs=60, batch_size=50)
train_heave10_x7, train_heave10_y7, valid_heave10_x7, valid_heave10_y7 = split_train_valid(heave10_x7,heave10_y7, 6900, 7500, 8000)
heave10_model7, heave10_history7, pre_heave10_Y7 = Model_LSTM(train_heave10_x7, train_heave10_y7, valid_heave10_x7, valid_heave10_y7, lr=0.1, epochs=60, batch_size=50)
train_heave30_x7, train_heave30_y7, valid_heave30_x7, valid_heave30_y7 = split_train_valid(heave30_x7,heave30_y7, 6900, 7500, 8000)
heave30_model7, heave30_history7, pre_heave30_Y7 = Model_LSTM(train_heave30_x7, train_heave30_y7, valid_heave30_x7, valid_heave30_y7, lr=0.1, epochs=60, batch_size=50)
train_heave50_x7, train_heave50_y7, valid_heave50_x7, valid_heave50_y7 = split_train_valid(heave50_x7,heave50_y7, 6900, 7500, 8000)
heave50_model7, heave50_histor7, pre_heave50_Y7 = Model_LSTM(train_heave50_x7, train_heave50_y7, valid_heave50_x7, valid_heave50_y7, lr=0.1, epochs=60, batch_size=50)
train_heave100_x7, train_heave100_y7, valid_heave100_x7, valid_heave100_y7 = split_train_valid(heave100_x7,heave100_y7, 6900, 7500, 8000)
heave100_model7, heave100_history7, pre_heave100_Y7 = Model_LSTM(train_heave100_x7, train_heave100_y7, valid_heave100_x7, valid_heave100_y7, lr=0.1, epochs=60, batch_size=50)
train_heave200_x7, train_heave200_y7, valid_heave200_x7, valid_heave200_y7 = split_train_valid(heave200_x7,heave200_y7, 6900, 7500, 8000)
heave200_model7, heave200_history7, pre_heave200_Y7 = Model_LSTM(train_heave200_x7, train_heave200_y7, valid_heave200_x7, valid_heave200_y7, lr=0.1, epochs=60, batch_size=50)

fan_heave1_real7,fan_heave1_pre7 = FanGuiHua_heave(valid_heave1_y7,pre_heave1_Y7)
fan_heave10_real7,fan_heave10_pre7 = FanGuiHua_heave(valid_heave10_y7,pre_heave10_Y7)
fan_heave30_heave7,fan_heave30_pre7 = FanGuiHua_heave(valid_heave30_y7,pre_heave30_Y7)
fan_heave50_real7,fan_heave50_pre7 = FanGuiHua_heave(valid_heave50_y7,pre_heave50_Y7)
fan_heave100_real7,fan_heave100_pre7 = FanGuiHua_heave(valid_heave100_y7,pre_heave100_Y7)
fan_heave200_real7,fan_heave200_pre7 = FanGuiHua_heave(valid_heave200_y7,pre_heave200_Y7)
np.savetxt('heaveN_其他步长7.csv',np.hstack((fan_heave1_real7,fan_heave1_pre7,fan_heave10_pre7,
                                        fan_heave30_pre7,fan_heave50_pre7,fan_heave100_pre7,fan_heave200_pre7)),delimiter=',')


# In[93]:


# 9
train_heave1_x9, train_heave1_y9, valid_heave1_x9, valid_heave1_y9 = split_train_valid(heave1_x9,heave1_y9, 6900, 7500, 8000)
heave1_model9, heave1_history9, pre_heave1_Y9 = Model_LSTM(train_heave1_x9, train_heave1_y9, valid_heave1_x9, valid_heave1_y9, lr=0.1, epochs=60, batch_size=50)
train_heave10_x9, train_heave10_y9, valid_heave10_x9, valid_heave10_y9 = split_train_valid(heave10_x9,heave10_y9, 6900, 7500, 8000)
heave10_model9, heave10_history9, pre_heave10_Y9 = Model_LSTM(train_heave10_x9, train_heave10_y9, valid_heave10_x9, valid_heave10_y9, lr=0.1, epochs=60, batch_size=50)
train_heave30_x9, train_heave30_y9, valid_heave30_x9, valid_heave30_y9 = split_train_valid(heave30_x9,heave30_y9, 6900, 7500, 8000)
heave30_model9, heave30_history9, pre_heave30_Y9 = Model_LSTM(train_heave30_x9, train_heave30_y9, valid_heave30_x9, valid_heave30_y9, lr=0.1, epochs=60, batch_size=50)
train_heave50_x9, train_heave50_y9, valid_heave50_x9, valid_heave50_y9 = split_train_valid(heave50_x9,heave50_y9, 6900, 7500, 8000)
heave50_model9, heave50_history9, pre_heave50_Y9 = Model_LSTM(train_heave50_x9, train_heave50_y9, valid_heave50_x9, valid_heave50_y9, lr=0.1, epochs=60, batch_size=50)
train_heave100_x9, train_heave100_y9, valid_heave100_x9, valid_heave100_y9 = split_train_valid(heave100_x9,heave100_y9, 6900, 7500, 8000)
heave100_model9, heave100_history9, pre_heave100_Y9 = Model_LSTM(train_heave100_x9, train_heave100_y9, valid_heave100_x9, valid_heave100_y9, lr=0.1, epochs=60, batch_size=50)
train_heave200_x9, train_heave200_y9, valid_heave200_x9, valid_heave200_y9 = split_train_valid(heave200_x9,heave200_y9,6900, 7500, 8000)
heave200_model9, heave200_history9, pre_heave200_Y9 = Model_LSTM(train_heave200_x9, train_heave200_y9, valid_heave200_x9, valid_heave200_y9, lr=0.1, epochs=60, batch_size=50)

fan_heave1_real9,fan_heave1_pre9 = FanGuiHua_heave(valid_heave1_y9,pre_heave1_Y9)
fan_heave10_real9,fan_heave10_pre9 = FanGuiHua_heave(valid_heave10_y9,pre_heave10_Y9)
fan_heave30_real9,fan_heave30_pre9 = FanGuiHua_heave(valid_heave30_y9,pre_heave30_Y9)
fan_heave50_real9,fan_heave50_pre9 = FanGuiHua_heave(valid_heave50_y9,pre_heave50_Y9)
fan_heave100_real9,fan_heave100_pre9 = FanGuiHua_heave(valid_heave100_y9,pre_heave100_Y9)
fan_heave200_real9,fan_heave200_pre9 = FanGuiHua_heave(valid_heave200_y9,pre_heave200_Y9)
np.savetxt('heaveN_其他步长9.csv',np.hstack((fan_heave1_real9,fan_heave1_pre9,fan_heave10_pre9,
                                        fan_heave30_pre9,fan_heave50_pre9,fan_heave100_pre9,fan_heave200_pre9)),delimiter=',')


# In[13]:


# heave
# 2
train_heave1_x2, train_heave1_y2, valid_heave1_x2, valid_heave1_y2 = split_train_valid(heave1_x2,heave1_y2, 6900, 7500, 8000)
heave1_model2, heave1_history2, pre_heave1_Y2 = Model_LSTM(train_heave1_x2, train_heave1_y2, valid_heave1_x2, valid_heave1_y2, lr=0.1, epochs=60, batch_size=50)
train_heave10_x2, train_heave10_y2, valid_heave10_x2, valid_heave10_y2 = split_train_valid(heave10_x2,heave10_y2, 6900, 7500, 8000)
heave10_model2, heave10_history2, pre_heave10_Y2 = Model_LSTM(train_heave10_x2, train_heave10_y2, valid_heave10_x2, valid_heave10_y2, lr=0.1, epochs=60, batch_size=50)
train_heave30_x2, train_heave30_y2, valid_heave30_x2, valid_heave30_y2 = split_train_valid(heave30_x2,heave30_y2, 6900, 7500, 8000)
heave30_model2, heave30_history2, pre_heave30_Y2 = Model_LSTM(train_heave30_x2, train_heave30_y2, valid_heave30_x2, valid_heave30_y2, lr=0.1, epochs=60, batch_size=50)
train_heave50_x2, train_heave50_y2, valid_heave50_x2, valid_heave50_y2 = split_train_valid(heave50_x2,heave50_y2, 6900, 7500, 8000)
heave50_model2, heave50_history2, pre_heave50_Y2 = Model_LSTM(train_heave50_x2, train_heave50_y2, valid_heave50_x2, valid_heave50_y2, lr=0.1, epochs=60, batch_size=50)
train_heave100_x2, train_heave100_y2, valid_heave100_x2, valid_heave100_y2 = split_train_valid(heave100_x2,heave100_y2, 6900, 7500, 8000)
heave100_model2, heave100_history2, pre_heave100_Y2 = Model_LSTM(train_heave100_x2, train_heave100_y2, valid_heave100_x2, valid_heave100_y2, lr=0.1, epochs=60, batch_size=50)


# In[14]:


train_heave200_x2, train_heave200_y2, valid_heave200_x2, valid_heave200_y2 = split_train_valid(heave200_x2,heave200_y2, 6900, 7500, 8000)
heave200_model2, heave200_history2, pre_heave200_Y2 = Model_LSTM(train_heave200_x2, train_heave200_y2, valid_heave200_x2, valid_heave200_y2, lr=0.1, epochs=60, batch_size=50)

fan_heave1_real2,fan_heave1_pre2 = FanGuiHua_heave(valid_heave1_y2,pre_heave1_Y2)
fan_heave10_real2,fan_heave10_pre2 = FanGuiHua_heave(valid_heave10_y2,pre_heave10_Y2)
fan_heave30_real2,fan_heave30_pre2 = FanGuiHua_heave(valid_heave30_y2,pre_heave30_Y2)
fan_heave50_real2,fan_heave50_pre2 = FanGuiHua_heave(valid_heave50_y2,pre_heave50_Y2)
fan_heave100_real2,fan_heave100_pre2 = FanGuiHua_heave(valid_heave100_y2,pre_heave100_Y2)
fan_heave200_real2,fan_heave200_pre2 = FanGuiHua_heave(valid_heave200_y2,pre_heave200_Y2)
np.savetxt('heaveN_其他步长2.csv',np.hstack((fan_heave1_real2,fan_heave1_pre2,fan_heave10_pre2,
                                        fan_heave30_pre2,fan_heave50_pre2,fan_heave100_pre2,fan_heave200_pre2)),delimiter=',')


# In[15]:


# heave
# 4
train_heave1_x4, train_heave1_y4, valid_heave1_x4, valid_heave1_y4 = split_train_valid(heave1_x4,heave1_y4, 6900, 7500, 8000)
heave1_model4, heave1_history4, pre_heave1_Y4 = Model_LSTM(train_heave1_x4, train_heave1_y4, valid_heave1_x4, valid_heave1_y4, lr=0.1, epochs=60, batch_size=50)
train_heave10_x4, train_heave10_y4, valid_heave10_x4, valid_heave10_y4 = split_train_valid(heave10_x4,heave10_y4, 6900, 7500, 8000)
heave10_model4, heave10_history4, pre_heave10_Y4 = Model_LSTM(train_heave10_x4, train_heave10_y4, valid_heave10_x4, valid_heave10_y4, lr=0.1, epochs=60, batch_size=50)
train_heave30_x4, train_heave30_y4, valid_heave30_x4, valid_heave30_y4 = split_train_valid(heave30_x4,heave30_y4, 6900, 7500, 8000)
heave30_model4, heave30_history4, pre_heave30_Y4 = Model_LSTM(train_heave30_x4, train_heave30_y4, valid_heave30_x4, valid_heave30_y4, lr=0.1, epochs=60, batch_size=50)
train_heave50_x4, train_heave50_y4, valid_heave50_x4, valid_heave50_y4 = split_train_valid(heave50_x4,heave50_y4, 6900, 7500, 8000)
heave50_model4, heave50_history4, pre_heave50_Y4 = Model_LSTM(train_heave50_x4, train_heave50_y4, valid_heave50_x4, valid_heave50_y4, lr=0.1, epochs=60, batch_size=50)
train_heave100_x4, train_heave100_y4, valid_heave100_x4, valid_heave100_y4 = split_train_valid(heave100_x4,heave100_y4, 6900, 7500, 8000)
heave100_model4, heave100_history4, pre_heave100_Y4 = Model_LSTM(train_heave100_x4, train_heave100_y4, valid_heave100_x4, valid_heave100_y4, lr=0.1, epochs=60, batch_size=50)
train_heave200_x4, train_heave200_y4, valid_heave200_x4, valid_heave200_y4 = split_train_valid(heave200_x4,heave200_y4, 6900, 7500, 8000)
heave200_model4, heave200_history4, pre_heave200_Y4 = Model_LSTM(train_heave200_x4, train_heave200_y4, valid_heave200_x4, valid_heave200_y4, lr=0.1, epochs=60, batch_size=50)

fan_heave1_real4,fan_heave1_pre4 = FanGuiHua_heave(valid_heave1_y4,pre_heave1_Y4)
fan_heave10_real4,fan_heave10_pre4 = FanGuiHua_heave(valid_heave10_y4,pre_heave10_Y4)
fan_heave30_real4,fan_heave30_pre4 = FanGuiHua_heave(valid_heave30_y4,pre_heave30_Y4)
fan_heave50_real4,fan_heave50_pre4 = FanGuiHua_heave(valid_heave50_y4,pre_heave50_Y4)
fan_heave100_real4,fan_heave100_pre4 = FanGuiHua_heave(valid_heave100_y4,pre_heave100_Y4)
fan_heave200_real4,fan_heave200_pre4 = FanGuiHua_heave(valid_heave200_y4,pre_heave200_Y4)
np.savetxt('heaveN_其他步长4.csv',np.hstack((fan_heave1_real4,fan_heave1_pre4,fan_heave10_pre4,
                                        fan_heave30_pre4,fan_heave50_pre4,fan_heave100_pre4,fan_heave200_pre4)),delimiter=',')


# In[16]:


# heave
# 6
train_heave1_x6, train_heave1_y6, valid_heave1_x6, valid_heave1_y6 = split_train_valid(heave1_x6,heave1_y6, 6900, 7500, 8000)
heave1_model6, heave1_history6, pre_heave1_Y6 = Model_LSTM(train_heave1_x6, train_heave1_y6, valid_heave1_x6, valid_heave1_y6, lr=0.1, epochs=60, batch_size=50)
train_heave10_x6, train_heave10_y6, valid_heave10_x6, valid_heave10_y6 = split_train_valid(heave10_x6,heave10_y6, 6900, 7500, 8000)
heave10_model6, heave10_history6, pre_heave10_Y6 = Model_LSTM(train_heave10_x6, train_heave10_y6, valid_heave10_x6, valid_heave10_y6, lr=0.1, epochs=60, batch_size=50)
train_heave30_x6, train_heave30_y6, valid_heave30_x6, valid_heave30_y6 = split_train_valid(heave30_x6,heave30_y6, 6900, 7500, 8000)
heave30_model6, heave30_history6, pre_heave30_Y6 = Model_LSTM(train_heave30_x6, train_heave30_y6, valid_heave30_x6, valid_heave30_y6, lr=0.1, epochs=60, batch_size=50)
train_heave50_x6, train_heave50_y6, valid_heave50_x6, valid_heave50_y6 = split_train_valid(heave50_x6,heave50_y6, 6900, 7500, 8000)
heave50_model6, heave50_history6, pre_heave50_Y6 = Model_LSTM(train_heave50_x6, train_heave50_y6, valid_heave50_x6, valid_heave50_y6, lr=0.1, epochs=60, batch_size=50)
train_heave100_x6, train_heave100_y6, valid_heave100_x6, valid_heave100_y6 = split_train_valid(heave100_x6,heave100_y6, 6900, 7500, 8000)
heave100_model6, heave100_history6, pre_heave100_Y6 = Model_LSTM(train_heave100_x6, train_heave100_y6, valid_heave100_x6, valid_heave100_y6, lr=0.1, epochs=60, batch_size=50)
train_heave200_x6, train_heave200_y6, valid_heave200_x6, valid_heave200_y6 = split_train_valid(heave200_x6,heave200_y6, 6900, 7500, 8000)
heave200_model6, heave200_history6, pre_heave200_Y6 = Model_LSTM(train_heave200_x6, train_heave200_y6, valid_heave200_x6, valid_heave200_y6, lr=0.1, epochs=60, batch_size=50)

fan_heave1_real6,fan_heave1_pre6 = FanGuiHua_heave(valid_heave1_y6,pre_heave1_Y6)
fan_heave10_real6,fan_heave10_pre6 = FanGuiHua_heave(valid_heave10_y6,pre_heave10_Y6)
fan_heave30_real6,fan_heave30_pre6 = FanGuiHua_heave(valid_heave30_y6,pre_heave30_Y6)
fan_heave50_real6,fan_heave50_pre6 = FanGuiHua_heave(valid_heave50_y6,pre_heave50_Y6)
fan_heave100_real6,fan_heave100_pre6 = FanGuiHua_heave(valid_heave100_y6,pre_heave100_Y6)
fan_heave200_real6,fan_heave200_pre6 = FanGuiHua_heave(valid_heave200_y6,pre_heave200_Y6)
np.savetxt('heaveN_其他步长6.csv',np.hstack((fan_heave1_real6,fan_heave1_pre6,fan_heave10_pre6,
                                        fan_heave30_pre6,fan_heave50_pre6,fan_heave100_pre6,fan_heave200_pre6)),delimiter=',')


# In[17]:


# heave
# 8
train_heave1_x8, train_heave1_y8, valid_heave1_x8, valid_heave1_y8 = split_train_valid(heave1_x8,heave1_y8, 6900, 7500, 8000)
heave1_model8, heave1_history8, pre_heave1_Y8 = Model_LSTM(train_heave1_x8, train_heave1_y8, valid_heave1_x8, valid_heave1_y8, lr=0.1, epochs=60, batch_size=50)
train_heave10_x8, train_heave10_y8, valid_heave10_x8, valid_heave10_y8 = split_train_valid(heave10_x8,heave10_y8, 6900, 7500, 8000)
heave10_model8, heave10_history8, pre_heave10_Y8 = Model_LSTM(train_heave10_x8, train_heave10_y8, valid_heave10_x8, valid_heave10_y8, lr=0.1, epochs=60, batch_size=50)
train_heave30_x8, train_heave30_y8, valid_heave30_x8, valid_heave30_y8 = split_train_valid(heave30_x8,heave30_y8, 6900, 7500, 8000)
heave30_model8, heave30_history8, pre_heave30_Y8 = Model_LSTM(train_heave30_x8, train_heave30_y8, valid_heave30_x8, valid_heave30_y8, lr=0.1, epochs=60, batch_size=50)
train_heave50_x8, train_heave50_y8, valid_heave50_x8, valid_heave50_y8 = split_train_valid(heave50_x8,heave50_y8, 6900, 7500, 8000)
heave50_model8, heave50_history8, pre_heave50_Y8 = Model_LSTM(train_heave50_x8, train_heave50_y8, valid_heave50_x8, valid_heave50_y8, lr=0.1, epochs=60, batch_size=50)
train_heave100_x8, train_heave100_y8, valid_heave100_x8, valid_heave100_y8 = split_train_valid(heave100_x8,heave100_y8, 6900, 7500, 8000)
heave100_model8, heave100_history8, pre_heave100_Y8 = Model_LSTM(train_heave100_x8, train_heave100_y8, valid_heave100_x8, valid_heave100_y8, lr=0.1, epochs=60, batch_size=50)
train_heave200_x8, train_heave200_y8, valid_heave200_x8, valid_heave200_y8 = split_train_valid(heave200_x8,heave200_y8, 6900, 7500, 8000)
heave200_model8, heave200_history8, pre_heave200_Y8 = Model_LSTM(train_heave200_x8, train_heave200_y8, valid_heave200_x8, valid_heave200_y8, lr=0.1, epochs=60, batch_size=50)

fan_heave1_real8,fan_heave1_pre8 = FanGuiHua_heave(valid_heave1_y8,pre_heave1_Y8)
fan_heave10_real8,fan_heave10_pre8 = FanGuiHua_heave(valid_heave10_y8,pre_heave10_Y8)
fan_heave30_real8,fan_heave30_pre8 = FanGuiHua_heave(valid_heave30_y8,pre_heave30_Y8)
fan_heave50_real8,fan_heave50_pre8 = FanGuiHua_heave(valid_heave50_y8,pre_heave50_Y8)
fan_heave100_real8,fan_heave100_pre8 = FanGuiHua_heave(valid_heave100_y8,pre_heave100_Y8)
fan_heave200_real8,fan_heave200_pre8 = FanGuiHua_heave(valid_heave200_y8,pre_heave200_Y8)
np.savetxt('heaveN_其他步长8.csv',np.hstack((fan_heave1_real8,fan_heave1_pre8,fan_heave10_pre8,
                                        fan_heave30_pre8,fan_heave50_pre8,fan_heave100_pre8,fan_heave200_pre8)),delimiter=',')


# In[ ]:





# #### surge(偶数)

# In[18]:


# 输入1步长输出2步长
surge1_scaled2 = deal_data1(np.hstack((H,Surge))[398:,:], 2, 2)
surge1_x2,surge1_y2 = split_sequence(surge1_scaled2, 1)
print(surge1_x2.shape)
# 输入10步长输出2步长
surge10_scaled2 = deal_data1(np.hstack((H,Surge))[389:,:], 2, 11)
surge10_x2,surge10_y2 = split_sequence(surge10_scaled2, 10)
print(surge10_x2.shape)
# 输入30步长输出2步长
surge30_scaled2 = deal_data1(np.hstack((H,Surge))[369:,:], 2, 31)
surge30_x2,surge30_y2 = split_sequence(surge30_scaled2, 30)
print(surge30_x2.shape)
# 输入50步长输出2步长
surge50_scaled2 = deal_data1(np.hstack((H,Surge))[349:,:], 2, 51)
surge50_x2,surge50_y2 = split_sequence(surge50_scaled2, 50)
print(surge50_x2.shape)
# 输入100步长输出2步长
surge100_scaled2 = deal_data1(np.hstack((H,Surge))[299:,:], 2, 101)
surge100_x2,surge100_y2 = split_sequence(surge100_scaled2, 100)
print(surge100_x2.shape)
# 输入200步长输出2步长
surge200_scaled2 = deal_data1(np.hstack((H,Surge))[199:,:], 2, 201)
surge200_x2,surge200_y2 = split_sequence(surge200_scaled2, 200)
print(surge200_x2.shape)

# 输入1步长输出4步长
surge1_scaled4 = deal_data1(np.hstack((H,Surge))[396:,:], 2, 4)
surge1_x4,surge1_y4 = split_sequence(surge1_scaled4, 1)
print(surge1_x4.shape)
# 输入10步长输出4步长
surge10_scaled4 = deal_data1(np.hstack((H,Surge))[387:,:], 2, 13)
surge10_x4,surge10_y4 = split_sequence(surge10_scaled4, 10)
print(surge10_x4.shape)
# 输入30步长输出4步长
surge30_scaled4 = deal_data1(np.hstack((H,Surge))[367:,:], 2, 33)
surge30_x4,surge30_y4 = split_sequence(surge30_scaled4, 30)
print(surge30_x4.shape)
# 输入50步长输出4步长
surge50_scaled4 = deal_data1(np.hstack((H,Surge))[347:,:], 2, 53)
surge50_x4,surge50_y4 = split_sequence(surge50_scaled4, 50)
print(surge50_x4.shape)
# 输入100步长输出4步长
surge100_scaled4 = deal_data1(np.hstack((H,Surge))[297:,:], 2, 103)
surge100_x4,surge100_y4 = split_sequence(surge100_scaled4, 100)
print(surge100_x4.shape)
# 输入200步长输出4步长
surge200_scaled4 = deal_data1(np.hstack((H,Surge))[197:,:], 2, 203)
surge200_x4,surge200_y4 = split_sequence(surge200_scaled4, 200)
print(surge200_x4.shape)

# 输入1步长输出6步长
surge1_scaled6 = deal_data1(np.hstack((H,Surge))[394:,:], 2, 6)
surge1_x6,surge1_y6 = split_sequence(surge1_scaled6, 1)
print(surge1_x6.shape)
# 输入10步长输出6步长
surge10_scaled6 = deal_data1(np.hstack((H,Surge))[385:,:], 2, 15)
surge10_x6,surge10_y6 = split_sequence(surge10_scaled6, 10)
print(surge10_x6.shape)
# 输入30步长输出6步长
surge30_scaled6 = deal_data1(np.hstack((H,Surge))[365:,:], 2, 35)
surge30_x6,surge30_y6 = split_sequence(surge30_scaled6, 30)
print(surge30_x6.shape)
# 输入50步长输出6步长
surge50_scaled6 = deal_data1(np.hstack((H,Surge))[345:,:], 2, 55)
surge50_x6,surge50_y6 = split_sequence(surge50_scaled6, 50)
print(surge50_x6.shape)
# 输入100步长输出6步长
surge100_scaled6 = deal_data1(np.hstack((H,Surge))[295:,:], 2, 105)
surge100_x6,surge100_y6 = split_sequence(surge100_scaled6, 100)
print(surge100_x6.shape)
# 输入200步长输出6步长
surge200_scaled6 = deal_data1(np.hstack((H,Surge))[195:,:], 2, 205)
surge200_x6,surge200_y6 = split_sequence(surge200_scaled6, 200)
print(surge200_x6.shape)

# 输入1步长输出8步长
surge1_scaled8 = deal_data1(np.hstack((H,Surge))[392:,:], 2, 8)
surge1_x8,surge1_y8 = split_sequence(surge1_scaled8, 1)
print(surge1_x8.shape)
# 输入10步长输出8步长
surge10_scaled8 = deal_data1(np.hstack((H,Surge))[383:,:], 2, 17)
surge10_x8,surge10_y8 = split_sequence(surge10_scaled8, 10)
print(surge10_x8.shape)
# 输入30步长输出8步长
surge30_scaled8 = deal_data1(np.hstack((H,Surge))[363:,:], 2, 37)
surge30_x8,surge30_y8 = split_sequence(surge30_scaled8, 30)
print(surge30_x8.shape)
# 输入50步长输出8步长
surge50_scaled8 = deal_data1(np.hstack((H,Surge))[343:,:], 2, 57)
surge50_x8,surge50_y8 = split_sequence(surge50_scaled8, 50)
print(surge50_x8.shape)
# 输入100步长输出8步长
surge100_scaled8 = deal_data1(np.hstack((H,Surge))[293:,:], 2, 107)
surge100_x8,surge100_y8 = split_sequence(surge100_scaled8, 100)
print(surge100_x8.shape)
# 输入200步长输出8步长
surge200_scaled8 = deal_data1(np.hstack((H,Surge))[193:,:], 2, 207)
surge200_x8,surge200_y8 = split_sequence(surge200_scaled8, 200)
print(surge200_x8.shape)


# In[19]:


# surge
# 2
train_surge1_x2, train_surge1_y2, valid_surge1_x2, valid_surge1_y2 = split_train_valid(surge1_x2,surge1_y2, 6900, 7500, 8000)
surge1_model2, surge1_history2, pre_surge1_Y2 = Model_LSTM(train_surge1_x2, train_surge1_y2, valid_surge1_x2, valid_surge1_y2, lr=0.1, epochs=60, batch_size=50)
train_surge10_x2, train_surge10_y2, valid_surge10_x2, valid_surge10_y2 = split_train_valid(surge10_x2,surge10_y2, 6900, 7500, 8000)
surge10_model2, surge10_history2, pre_surge10_Y2 = Model_LSTM(train_surge10_x2, train_surge10_y2, valid_surge10_x2, valid_surge10_y2, lr=0.1, epochs=60, batch_size=50)
train_surge30_x2, train_surge30_y2, valid_surge30_x2, valid_surge30_y2 = split_train_valid(surge30_x2,surge30_y2, 6900, 7500, 8000)
surge30_model2, surge30_history2, pre_surge30_Y2 = Model_LSTM(train_surge30_x2, train_surge30_y2, valid_surge30_x2, valid_surge30_y2, lr=0.1, epochs=60, batch_size=50)
train_surge50_x2, train_surge50_y2, valid_surge50_x2, valid_surge50_y2 = split_train_valid(surge50_x2,surge50_y2, 6900, 7500, 8000)
surge50_model2, surge50_history2, pre_surge50_Y2 = Model_LSTM(train_surge50_x2, train_surge50_y2, valid_surge50_x2, valid_surge50_y2, lr=0.1, epochs=60, batch_size=50)
train_surge100_x2, train_surge100_y2, valid_surge100_x2, valid_surge100_y2 = split_train_valid(surge100_x2,surge100_y2, 6900, 7500, 8000)
surge100_model2, surge100_history2, pre_surge100_Y2 = Model_LSTM(train_surge100_x2, train_surge100_y2, valid_surge100_x2, valid_surge100_y2, lr=0.1, epochs=60, batch_size=50)
train_surge200_x2, train_surge200_y2, valid_surge200_x2, valid_surge200_y2 = split_train_valid(surge200_x2,surge200_y2, 6900, 7500, 8000)
surge200_model2, surge200_history2, pre_surge200_Y2 = Model_LSTM(train_surge200_x2, train_surge200_y2, valid_surge200_x2, valid_surge200_y2, lr=0.1, epochs=60, batch_size=50)

fan_surge1_real2,fan_surge1_pre2 = FanGuiHua_surge(valid_surge1_y2,pre_surge1_Y2)
fan_surge10_real2,fan_surge10_pre2 = FanGuiHua_surge(valid_surge10_y2,pre_surge10_Y2)
fan_surge30_real2,fan_surge30_pre2 = FanGuiHua_surge(valid_surge30_y2,pre_surge30_Y2)
fan_surge50_real2,fan_surge50_pre2 = FanGuiHua_surge(valid_surge50_y2,pre_surge50_Y2)
fan_surge100_real2,fan_surge100_pre2 = FanGuiHua_surge(valid_surge100_y2,pre_surge100_Y2)
fan_surge200_real2,fan_surge200_pre2 = FanGuiHua_surge(valid_surge200_y2,pre_surge200_Y2)
np.savetxt('surgeN_其他步长2.csv',np.hstack((fan_surge1_real2,fan_surge1_pre2,fan_surge10_pre2,
                                        fan_surge30_pre2,fan_surge50_pre2,fan_surge100_pre2,fan_surge200_pre2)),delimiter=',')


# In[20]:


# surge
# 4
train_surge1_x4, train_surge1_y4, valid_surge1_x4, valid_surge1_y4 = split_train_valid(surge1_x4,surge1_y4, 6900, 7500, 8000)
surge1_model4, surge1_history4, pre_surge1_Y4 = Model_LSTM(train_surge1_x4, train_surge1_y4, valid_surge1_x4, valid_surge1_y4, lr=0.1, epochs=60, batch_size=50)
train_surge10_x4, train_surge10_y4, valid_surge10_x4, valid_surge10_y4 = split_train_valid(surge10_x4,surge10_y4, 6900, 7500, 8000)
surge10_model4, surge10_history4, pre_surge10_Y4 = Model_LSTM(train_surge10_x4, train_surge10_y4, valid_surge10_x4, valid_surge10_y4, lr=0.1, epochs=60, batch_size=50)
train_surge30_x4, train_surge30_y4, valid_surge30_x4, valid_surge30_y4 = split_train_valid(surge30_x4,surge30_y4, 6900, 7500, 8000)
surge30_model4, surge30_history4, pre_surge30_Y4 = Model_LSTM(train_surge30_x4, train_surge30_y4, valid_surge30_x4, valid_surge30_y4, lr=0.1, epochs=60, batch_size=50)
train_surge50_x4, train_surge50_y4, valid_surge50_x4, valid_surge50_y4 = split_train_valid(surge50_x4,surge50_y4, 6900, 7500, 8000)
surge50_model4, surge50_history4, pre_surge50_Y4 = Model_LSTM(train_surge50_x4, train_surge50_y4, valid_surge50_x4, valid_surge50_y4, lr=0.1, epochs=60, batch_size=50)
train_surge100_x4, train_surge100_y4, valid_surge100_x4, valid_surge100_y4 = split_train_valid(surge100_x4,surge100_y4, 6900, 7500, 8000)
surge100_model4, surge100_history4, pre_surge100_Y4 = Model_LSTM(train_surge100_x4, train_surge100_y4, valid_surge100_x4, valid_surge100_y4, lr=0.1, epochs=60, batch_size=50)
train_surge200_x4, train_surge200_y4, valid_surge200_x4, valid_surge200_y4 = split_train_valid(surge200_x4,surge200_y4, 6900, 7500, 8000)
surge200_model4, surge200_history4, pre_surge200_Y4 = Model_LSTM(train_surge200_x4, train_surge200_y4, valid_surge200_x4, valid_surge200_y4, lr=0.1, epochs=60, batch_size=50)

fan_surge1_real4,fan_surge1_pre4 = FanGuiHua_surge(valid_surge1_y4,pre_surge1_Y4)
fan_surge10_real4,fan_surge10_pre4 = FanGuiHua_surge(valid_surge10_y4,pre_surge10_Y4)
fan_surge30_real4,fan_surge30_pre4 = FanGuiHua_surge(valid_surge30_y4,pre_surge30_Y4)
fan_surge50_real4,fan_surge50_pre4 = FanGuiHua_surge(valid_surge50_y4,pre_surge50_Y4)
fan_surge100_real4,fan_surge100_pre4 = FanGuiHua_surge(valid_surge100_y4,pre_surge100_Y4)
fan_surge200_real4,fan_surge200_pre4 = FanGuiHua_surge(valid_surge200_y4,pre_surge200_Y4)
np.savetxt('surgeN_其他步长4.csv',np.hstack((fan_surge1_real4,fan_surge1_pre4,fan_surge10_pre4,
                                        fan_surge30_pre4,fan_surge50_pre4,fan_surge100_pre4,fan_surge200_pre4)),delimiter=',')


# In[29]:


# surge
# 6
train_surge1_x6, train_surge1_y6, valid_surge1_x6, valid_surge1_y6 = split_train_valid(surge1_x6,surge1_y6, 6900, 7500, 8000)
surge1_model6, surge1_history6, pre_surge1_Y6 = Model_LSTM(train_surge1_x6, train_surge1_y6, valid_surge1_x6, valid_surge1_y6, lr=0.1, epochs=60, batch_size=50)
train_surge10_x6, train_surge10_y6, valid_surge10_x6, valid_surge10_y6 = split_train_valid(surge10_x6,surge10_y6, 6900, 7500, 8000)
surge10_model6, surge10_history6, pre_surge10_Y6 = Model_LSTM(train_surge10_x6, train_surge10_y6, valid_surge10_x6, valid_surge10_y6, lr=0.1, epochs=60, batch_size=50)
train_surge30_x6, train_surge30_y6, valid_surge30_x6, valid_surge30_y6 = split_train_valid(surge30_x6,surge30_y6, 6900, 7500, 8000)
surge30_model6, surge30_history6, pre_surge30_Y6 = Model_LSTM(train_surge30_x6, train_surge30_y6, valid_surge30_x6, valid_surge30_y6, lr=0.1, epochs=60, batch_size=50)
train_surge50_x6, train_surge50_y6, valid_surge50_x6, valid_surge50_y6 = split_train_valid(surge50_x6,surge50_y6, 6900, 7500, 8000)
surge50_model6, surge50_history6, pre_surge50_Y6 = Model_LSTM(train_surge50_x6, train_surge50_y6, valid_surge50_x6, valid_surge50_y6, lr=0.1, epochs=60, batch_size=50)
train_surge100_x6, train_surge100_y6, valid_surge100_x6, valid_surge100_y6 = split_train_valid(surge100_x6,surge100_y6, 6900, 7500, 8000)
surge100_model6, surge100_history6, pre_surge100_Y6 = Model_LSTM(train_surge100_x6, train_surge100_y6, valid_surge100_x6, valid_surge100_y6, lr=0.1, epochs=60, batch_size=50)
train_surge200_x6, train_surge200_y6, valid_surge200_x6, valid_surge200_y6 = split_train_valid(surge200_x6,surge200_y6, 6900, 7500, 8000)
surge200_model6, surge200_history6, pre_surge200_Y6 = Model_LSTM(train_surge200_x6, train_surge200_y6, valid_surge200_x6, valid_surge200_y6, lr=0.1, epochs=60, batch_size=50)

fan_surge1_real6,fan_surge1_pre6 = FanGuiHua_surge(valid_surge1_y6,pre_surge1_Y6)
fan_surge10_real6,fan_surge10_pre6 = FanGuiHua_surge(valid_surge10_y6,pre_surge10_Y6)
fan_surge30_real6,fan_surge30_pre6 = FanGuiHua_surge(valid_surge30_y6,pre_surge30_Y6)
fan_surge50_real6,fan_surge50_pre6 = FanGuiHua_surge(valid_surge50_y6,pre_surge50_Y6)
fan_surge100_real6,fan_surge100_pre6 = FanGuiHua_surge(valid_surge100_y6,pre_surge100_Y6)
fan_surge200_real6,fan_surge200_pre6 = FanGuiHua_surge(valid_surge200_y6,pre_surge200_Y6)
np.savetxt('surgeN_其他步长6.csv',np.hstack((fan_surge1_real6,fan_surge1_pre6,fan_surge10_pre6,
                                        fan_surge30_pre6,fan_surge50_pre6,fan_surge100_pre6,fan_surge200_pre6)),delimiter=',')


# In[30]:


# surge
# 8
train_surge1_x8, train_surge1_y8, valid_surge1_x8, valid_surge1_y8 = split_train_valid(surge1_x8,surge1_y8, 6900, 7500, 8000)
surge1_model8, surge1_history8, pre_surge1_Y8 = Model_LSTM(train_surge1_x8, train_surge1_y8, valid_surge1_x8, valid_surge1_y8, lr=0.1, epochs=60, batch_size=50)
train_surge10_x8, train_surge10_y8, valid_surge10_x8, valid_surge10_y8 = split_train_valid(surge10_x8,surge10_y8, 6900, 7500, 8000)
surge10_model8, surge10_history8, pre_surge10_Y8 = Model_LSTM(train_surge10_x8, train_surge10_y8, valid_surge10_x8, valid_surge10_y8, lr=0.1, epochs=60, batch_size=50)
train_surge30_x8, train_surge30_y8, valid_surge30_x8, valid_surge30_y8 = split_train_valid(surge30_x8,surge30_y8, 6900, 7500, 8000)
surge30_model8, surge30_history8, pre_surge30_Y8 = Model_LSTM(train_surge30_x8, train_surge30_y8, valid_surge30_x8, valid_surge30_y8, lr=0.1, epochs=60, batch_size=50)
train_surge50_x8, train_surge50_y8, valid_surge50_x8, valid_surge50_y8 = split_train_valid(surge50_x8,surge50_y8, 6900, 7500, 8000)
surge50_model8, surge50_history8, pre_surge50_Y8 = Model_LSTM(train_surge50_x8, train_surge50_y8, valid_surge50_x8, valid_surge50_y8, lr=0.1, epochs=60, batch_size=50)
train_surge100_x8, train_surge100_y8, valid_surge100_x8, valid_surge100_y8 = split_train_valid(surge100_x8,surge100_y8, 6900, 7500, 8000)
surge100_model8, surge100_history8, pre_surge100_Y8 = Model_LSTM(train_surge100_x8, train_surge100_y8, valid_surge100_x8, valid_surge100_y8, lr=0.1, epochs=60, batch_size=50)
train_surge200_x8, train_surge200_y8, valid_surge200_x8, valid_surge200_y8 = split_train_valid(surge200_x8,surge200_y8, 6900, 7500, 8000)
surge200_model8, surge200_history8, pre_surge200_Y8 = Model_LSTM(train_surge200_x8, train_surge200_y8, valid_surge200_x8, valid_surge200_y8, lr=0.1, epochs=60, batch_size=50)

fan_surge1_real8,fan_surge1_pre8 = FanGuiHua_surge(valid_surge1_y8,pre_surge1_Y8)
fan_surge10_real8,fan_surge10_pre8 = FanGuiHua_surge(valid_surge10_y8,pre_surge10_Y8)
fan_surge30_real8,fan_surge30_pre8 = FanGuiHua_surge(valid_surge30_y8,pre_surge30_Y8)
fan_surge50_real8,fan_surge50_pre8 = FanGuiHua_surge(valid_surge50_y8,pre_surge50_Y8)
fan_surge100_real8,fan_surge100_pre8 = FanGuiHua_surge(valid_surge100_y8,pre_surge100_Y8)
fan_surge200_real8,fan_surge200_pre8 = FanGuiHua_surge(valid_surge200_y8,pre_surge200_Y8)
np.savetxt('surgeN_其他步长8.csv',np.hstack((fan_surge1_real8,fan_surge1_pre8,fan_surge10_pre8,
                                        fan_surge30_pre8,fan_surge50_pre8,fan_surge100_pre8,fan_surge200_pre8)),delimiter=',')


# In[ ]:





# #### pitch(偶数)

# In[23]:


# 输入1步长输出2步长
pitch1_scaled2 = deal_data1(np.hstack((H,Pitch))[398:,:], 2, 2)
pitch1_x2,pitch1_y2 = split_sequence(pitch1_scaled2, 1)
print(pitch1_x2.shape)
# 输入10步长输出2步长
pitch10_scaled2 = deal_data1(np.hstack((H,Pitch))[389:,:], 2, 11)
pitch10_x2,pitch10_y2 = split_sequence(pitch10_scaled2, 10)
print(pitch10_x2.shape)
# 输入30步长输出2步长
pitch30_scaled2 = deal_data1(np.hstack((H,Pitch))[369:,:], 2, 31)
pitch30_x2,pitch30_y2 = split_sequence(pitch30_scaled2, 30)
print(pitch30_x2.shape)
# 输入50步长输出2步长
pitch50_scaled2 = deal_data1(np.hstack((H,Pitch))[349:,:], 2, 51)
pitch50_x2,pitch50_y2 = split_sequence(pitch50_scaled2, 50)
print(pitch50_x2.shape)
# 输入100步长输出2步长
pitch100_scaled2 = deal_data1(np.hstack((H,Pitch))[299:,:], 2, 101)
pitch100_x2,pitch100_y2 = split_sequence(pitch100_scaled2, 100)
print(pitch100_x2.shape)
# 输入200步长输出2步长
pitch200_scaled2 = deal_data1(np.hstack((H,Pitch))[199:,:], 2, 201)
pitch200_x2,pitch200_y2 = split_sequence(pitch200_scaled2, 200)
print(pitch200_x2.shape)

# 输入1步长输出4步长
pitch1_scaled4 = deal_data1(np.hstack((H,Pitch))[396:,:], 2, 4)
pitch1_x4,pitch1_y4 = split_sequence(pitch1_scaled4, 1)
print(pitch1_x4.shape)
# 输入10步长输出4步长
pitch10_scaled4 = deal_data1(np.hstack((H,Pitch))[387:,:], 2, 13)
pitch10_x4,pitch10_y4 = split_sequence(pitch10_scaled4, 10)
print(pitch10_x4.shape)
# 输入30步长输出4步长
pitch30_scaled4 = deal_data1(np.hstack((H,Pitch))[367:,:], 2, 33)
pitch30_x4,pitch30_y4 = split_sequence(pitch30_scaled4, 30)
print(pitch30_x4.shape)
# 输入50步长输出4步长
pitch50_scaled4 = deal_data1(np.hstack((H,Pitch))[347:,:], 2, 53)
pitch50_x4,pitch50_y4 = split_sequence(pitch50_scaled4, 50)
print(pitch50_x4.shape)
# 输入100步长输出4步长
pitch100_scaled4 = deal_data1(np.hstack((H,Pitch))[297:,:], 2, 103)
pitch100_x4,pitch100_y4 = split_sequence(pitch100_scaled4, 100)
print(pitch100_x4.shape)
# 输入200步长输出4步长
pitch200_scaled4 = deal_data1(np.hstack((H,Pitch))[197:,:], 2, 203)
pitch200_x4,pitch200_y4 = split_sequence(pitch200_scaled4, 200)
print(pitch200_x4.shape)

# 输入1步长输出6步长
pitch1_scaled6 = deal_data1(np.hstack((H,Pitch))[394:,:], 2, 6)
pitch1_x6,pitch1_y6 = split_sequence(pitch1_scaled6, 1)
print(pitch1_x6.shape)
# 输入10步长输出6步长
pitch10_scaled6 = deal_data1(np.hstack((H,Pitch))[385:,:], 2, 15)
pitch10_x6,pitch10_y6 = split_sequence(pitch10_scaled6, 10)
print(pitch10_x6.shape)
# 输入30步长输出6步长
pitch30_scaled6 = deal_data1(np.hstack((H,Pitch))[365:,:], 2, 35)
pitch30_x6,pitch30_y6 = split_sequence(pitch30_scaled6, 30)
print(pitch30_x6.shape)
# 输入50步长输出6步长
pitch50_scaled6 = deal_data1(np.hstack((H,Pitch))[345:,:], 2, 55)
pitch50_x6,pitch50_y6 = split_sequence(pitch50_scaled6, 50)
print(pitch50_x6.shape)
# 输入100步长输出6步长
pitch100_scaled6 = deal_data1(np.hstack((H,Pitch))[295:,:], 2, 105)
pitch100_x6,pitch100_y6 = split_sequence(pitch100_scaled6, 100)
print(pitch100_x6.shape)
# 输入200步长输出6步长
pitch200_scaled6 = deal_data1(np.hstack((H,Pitch))[195:,:], 2, 205)
pitch200_x6,pitch200_y6 = split_sequence(pitch200_scaled6, 200)
print(pitch200_x6.shape)

# 输入1步长输出8步长
pitch1_scaled8 = deal_data1(np.hstack((H,Pitch))[392:,:], 2, 8)
pitch1_x8,pitch1_y8 = split_sequence(pitch1_scaled8, 1)
print(pitch1_x8.shape)
# 输入10步长输出8步长
pitch10_scaled8 = deal_data1(np.hstack((H,Pitch))[383:,:], 2, 17)
pitch10_x8,pitch10_y8 = split_sequence(pitch10_scaled8, 10)
print(pitch10_x8.shape)
# 输入30步长输出8步长
pitch30_scaled8 = deal_data1(np.hstack((H,Pitch))[363:,:], 2, 37)
pitch30_x8,pitch30_y8 = split_sequence(pitch30_scaled8, 30)
print(pitch30_x8.shape)
# 输入50步长输出8步长
pitch50_scaled8 = deal_data1(np.hstack((H,Pitch))[343:,:], 2, 57)
pitch50_x8,pitch50_y8 = split_sequence(pitch50_scaled8, 50)
print(pitch50_x8.shape)
# 输入100步长输出8步长
pitch100_scaled8 = deal_data1(np.hstack((H,Pitch))[293:,:], 2, 107)
pitch100_x8,pitch100_y8 = split_sequence(pitch100_scaled8, 100)
print(pitch100_x8.shape)
# 输入200步长输出8步长
pitch200_scaled8 = deal_data1(np.hstack((H,Pitch))[193:,:], 2, 207)
pitch200_x8,pitch200_y8 = split_sequence(pitch200_scaled8, 200)
print(pitch200_x8.shape)


# In[24]:


# pitch
# 2
train_pitch1_x2, train_pitch1_y2, valid_pitch1_x2, valid_pitch1_y2 = split_train_valid(pitch1_x2,pitch1_y2, 6900, 7500, 8000)
pitch1_model2, pitch1_history2, pre_pitch1_Y2 = Model_LSTM(train_pitch1_x2, train_pitch1_y2, valid_pitch1_x2, valid_pitch1_y2, lr=0.1, epochs=60, batch_size=50)
train_pitch10_x2, train_pitch10_y2, valid_pitch10_x2, valid_pitch10_y2 = split_train_valid(pitch10_x2,pitch10_y2, 6900, 7500, 8000)
pitch10_model2, pitch10_history2, pre_pitch10_Y2 = Model_LSTM(train_pitch10_x2, train_pitch10_y2, valid_pitch10_x2, valid_pitch10_y2, lr=0.1, epochs=60, batch_size=50)
train_pitch30_x2, train_pitch30_y2, valid_pitch30_x2, valid_pitch30_y2 = split_train_valid(pitch30_x2,pitch30_y2, 6900, 7500, 8000)
pitch30_model2, pitch30_history2, pre_pitch30_Y2 = Model_LSTM(train_pitch30_x2, train_pitch30_y2, valid_pitch30_x2, valid_pitch30_y2, lr=0.1, epochs=60, batch_size=50)
train_pitch50_x2, train_pitch50_y2, valid_pitch50_x2, valid_pitch50_y2 = split_train_valid(pitch50_x2,pitch50_y2, 6900, 7500, 8000)
pitch50_model2, pitch50_history2, pre_pitch50_Y2 = Model_LSTM(train_pitch50_x2, train_pitch50_y2, valid_pitch50_x2, valid_pitch50_y2, lr=0.1, epochs=60, batch_size=50)
train_pitch100_x2, train_pitch100_y2, valid_pitch100_x2, valid_pitch100_y2 = split_train_valid(pitch100_x2,pitch100_y2, 6900, 7500, 8000)
pitch100_model2, pitch100_history2, pre_pitch100_Y2 = Model_LSTM(train_pitch100_x2, train_pitch100_y2, valid_pitch100_x2, valid_pitch100_y2, lr=0.1, epochs=60, batch_size=50)


# In[25]:


train_pitch200_x2, train_pitch200_y2, valid_pitch200_x2, valid_pitch200_y2 = split_train_valid(pitch200_x2,pitch200_y2, 6900, 7500, 8000)
pitch200_model2, pitch200_history2, pre_pitch200_Y2 = Model_LSTM(train_pitch200_x2, train_pitch200_y2, valid_pitch200_x2, valid_pitch200_y2, lr=0.1, epochs=60, batch_size=50)

fan_pitch1_real2,fan_pitch1_pre2 = FanGuiHua_pitch(valid_pitch1_y2,pre_pitch1_Y2)
fan_pitch10_real2,fan_pitch10_pre2 = FanGuiHua_pitch(valid_pitch10_y2,pre_pitch10_Y2)
fan_pitch30_real2,fan_pitch30_pre2 = FanGuiHua_pitch(valid_pitch30_y2,pre_pitch30_Y2)
fan_pitch50_real2,fan_pitch50_pre2 = FanGuiHua_pitch(valid_pitch50_y2,pre_pitch50_Y2)
fan_pitch100_real2,fan_pitch100_pre2 = FanGuiHua_pitch(valid_pitch100_y2,pre_pitch100_Y2)
fan_pitch200_real2,fan_pitch200_pre2 = FanGuiHua_pitch(valid_pitch200_y2,pre_pitch200_Y2)
np.savetxt('pitchN_其他步长2.csv',np.hstack((fan_pitch1_real2,fan_pitch1_pre2,fan_pitch10_pre2,
                                        fan_pitch30_pre2,fan_pitch50_pre2,fan_pitch100_pre2,fan_pitch200_pre2)),delimiter=',')


# In[26]:


# pitch
# 4
train_pitch1_x4, train_pitch1_y4, valid_pitch1_x4, valid_pitch1_y4 = split_train_valid(pitch1_x4,pitch1_y4, 6900, 7500, 8000)
pitch1_model4, pitch1_history4, pre_pitch1_Y4 = Model_LSTM(train_pitch1_x4, train_pitch1_y4, valid_pitch1_x4, valid_pitch1_y4, lr=0.1, epochs=60, batch_size=50)
train_pitch10_x4, train_pitch10_y4, valid_pitch10_x4, valid_pitch10_y4 = split_train_valid(pitch10_x4,pitch10_y4, 6900, 7500, 8000)
pitch10_model4, pitch10_history4, pre_pitch10_Y4 = Model_LSTM(train_pitch10_x4, train_pitch10_y4, valid_pitch10_x4, valid_pitch10_y4, lr=0.1, epochs=60, batch_size=50)
train_pitch30_x4, train_pitch30_y4, valid_pitch30_x4, valid_pitch30_y4 = split_train_valid(pitch30_x4,pitch30_y4, 6900, 7500, 8000)
pitch30_model4, pitch30_history4, pre_pitch30_Y4 = Model_LSTM(train_pitch30_x4, train_pitch30_y4, valid_pitch30_x4, valid_pitch30_y4, lr=0.1, epochs=60, batch_size=50)
train_pitch50_x4, train_pitch50_y4, valid_pitch50_x4, valid_pitch50_y4 = split_train_valid(pitch50_x4,pitch50_y4, 6900, 7500, 8000)
pitch50_model4, pitch50_history4, pre_pitch50_Y4 = Model_LSTM(train_pitch50_x4, train_pitch50_y4, valid_pitch50_x4, valid_pitch50_y4, lr=0.1, epochs=60, batch_size=50)
train_pitch100_x4, train_pitch100_y4, valid_pitch100_x4, valid_pitch100_y4 = split_train_valid(pitch100_x4,pitch100_y4, 6900, 7500, 8000)
pitch100_model4, pitch100_history4, pre_pitch100_Y4 = Model_LSTM(train_pitch100_x4, train_pitch100_y4, valid_pitch100_x4, valid_pitch100_y4, lr=0.1, epochs=60, batch_size=50)
train_pitch200_x4, train_pitch200_y4, valid_pitch200_x4, valid_pitch200_y4 = split_train_valid(pitch200_x4,pitch200_y4, 6900, 7500, 8000)
pitch200_model4, pitch200_history4, pre_pitch200_Y4 = Model_LSTM(train_pitch200_x4, train_pitch200_y4, valid_pitch200_x4, valid_pitch200_y4, lr=0.1, epochs=60, batch_size=50)

fan_pitch1_real4,fan_pitch1_pre4 = FanGuiHua_pitch(valid_pitch1_y4,pre_pitch1_Y4)
fan_pitch10_real4,fan_pitch10_pre4 = FanGuiHua_pitch(valid_pitch10_y4,pre_pitch10_Y4)
fan_pitch30_real4,fan_pitch30_pre4 = FanGuiHua_pitch(valid_pitch30_y4,pre_pitch30_Y4)
fan_pitch50_real4,fan_pitch50_pre4 = FanGuiHua_pitch(valid_pitch50_y4,pre_pitch50_Y4)
fan_pitch100_real4,fan_pitch100_pre4 = FanGuiHua_pitch(valid_pitch100_y4,pre_pitch100_Y4)
fan_pitch200_real4,fan_pitch200_pre4 = FanGuiHua_pitch(valid_pitch200_y4,pre_pitch200_Y4)
np.savetxt('pitchN_其他步长4.csv',np.hstack((fan_pitch1_real4,fan_pitch1_pre4,fan_pitch10_pre4,
                                        fan_pitch30_pre4,fan_pitch50_pre4,fan_pitch100_pre4,fan_pitch200_pre4)),delimiter=',')


# In[31]:


# pitch
# 6
train_pitch1_x6, train_pitch1_y6, valid_pitch1_x6, valid_pitch1_y6 = split_train_valid(pitch1_x6,pitch1_y6, 6900, 7500, 8000)
pitch1_model6, pitch1_history6, pre_pitch1_Y6 = Model_LSTM(train_pitch1_x6, train_pitch1_y6, valid_pitch1_x6, valid_pitch1_y6, lr=0.1, epochs=60, batch_size=50)
train_pitch10_x6, train_pitch10_y6, valid_pitch10_x6, valid_pitch10_y6 = split_train_valid(pitch10_x6,pitch10_y6, 6900, 7500, 8000)
pitch10_model6, pitch10_history6, pre_pitch10_Y6 = Model_LSTM(train_pitch10_x6, train_pitch10_y6, valid_pitch10_x6, valid_pitch10_y6, lr=0.1, epochs=60, batch_size=50)
train_pitch30_x6, train_pitch30_y6, valid_pitch30_x6, valid_pitch30_y6 = split_train_valid(pitch30_x6,pitch30_y6, 6900, 7500, 8000)
pitch30_model6, pitch30_history6, pre_pitch30_Y6 = Model_LSTM(train_pitch30_x6, train_pitch30_y6, valid_pitch30_x6, valid_pitch30_y6, lr=0.1, epochs=60, batch_size=50)
train_pitch50_x6, train_pitch50_y6, valid_pitch50_x6, valid_pitch50_y6 = split_train_valid(pitch50_x6,pitch50_y6, 6900, 7500, 8000)
pitch50_model6, pitch50_history6, pre_pitch50_Y6 = Model_LSTM(train_pitch50_x6, train_pitch50_y6, valid_pitch50_x6, valid_pitch50_y6, lr=0.1, epochs=60, batch_size=50)
train_pitch100_x6, train_pitch100_y6, valid_pitch100_x6, valid_pitch100_y6 = split_train_valid(pitch100_x6,pitch100_y6, 6900, 7500, 8000)
pitch100_model6, pitch100_history6, pre_pitch100_Y6 = Model_LSTM(train_pitch100_x6, train_pitch100_y6, valid_pitch100_x6, valid_pitch100_y6, lr=0.1, epochs=60, batch_size=50)
train_pitch200_x6, train_pitch200_y6, valid_pitch200_x6, valid_pitch200_y6 = split_train_valid(pitch200_x6,pitch200_y6, 6900, 7500, 8000)
pitch200_model6, pitch200_history6, pre_pitch200_Y6 = Model_LSTM(train_pitch200_x6, train_pitch200_y6, valid_pitch200_x6, valid_pitch200_y6, lr=0.1, epochs=60, batch_size=50)

fan_pitch1_real6,fan_pitch1_pre6 = FanGuiHua_pitch(valid_pitch1_y6,pre_pitch1_Y6)
fan_pitch10_real6,fan_pitch10_pre6 = FanGuiHua_pitch(valid_pitch10_y6,pre_pitch10_Y6)
fan_pitch30_real6,fan_pitch30_pre6 = FanGuiHua_pitch(valid_pitch30_y6,pre_pitch30_Y6)
fan_pitch50_real6,fan_pitch50_pre6 = FanGuiHua_pitch(valid_pitch50_y6,pre_pitch50_Y6)
fan_pitch100_real6,fan_pitch100_pre6 = FanGuiHua_pitch(valid_pitch100_y6,pre_pitch100_Y6)
fan_pitch200_real6,fan_pitch200_pre6 = FanGuiHua_pitch(valid_pitch200_y6,pre_pitch200_Y6)
np.savetxt('pitchN_其他步长6.csv',np.hstack((fan_pitch1_real6,fan_pitch1_pre6,fan_pitch10_pre6,
                                        fan_pitch30_pre6,fan_pitch50_pre6,fan_pitch100_pre6,fan_pitch200_pre6)),delimiter=',')


# In[32]:


# pitch
# 8
train_pitch1_x8, train_pitch1_y8, valid_pitch1_x8, valid_pitch1_y8 = split_train_valid(pitch1_x8,pitch1_y8, 6900, 7500, 8000)
pitch1_model8, pitch1_history8, pre_pitch1_Y8 = Model_LSTM(train_pitch1_x8, train_pitch1_y8, valid_pitch1_x8, valid_pitch1_y8, lr=0.1, epochs=60, batch_size=50)
train_pitch10_x8, train_pitch10_y8, valid_pitch10_x8, valid_pitch10_y8 = split_train_valid(pitch10_x8,pitch10_y8, 6900, 7500, 8000)
pitch10_model8, pitch10_history8, pre_pitch10_Y8 = Model_LSTM(train_pitch10_x8, train_pitch10_y8, valid_pitch10_x8, valid_pitch10_y8, lr=0.1, epochs=60, batch_size=50)
train_pitch30_x8, train_pitch30_y8, valid_pitch30_x8, valid_pitch30_y8 = split_train_valid(pitch30_x8,pitch30_y8, 6900, 7500, 8000)
pitch30_model8, pitch30_history8, pre_pitch30_Y8 = Model_LSTM(train_pitch30_x8, train_pitch30_y8, valid_pitch30_x8, valid_pitch30_y8, lr=0.1, epochs=60, batch_size=50)
train_pitch50_x8, train_pitch50_y8, valid_pitch50_x8, valid_pitch50_y8 = split_train_valid(pitch50_x8,pitch50_y8, 6900, 7500, 8000)
pitch50_model8, pitch50_history8, pre_pitch50_Y8 = Model_LSTM(train_pitch50_x8, train_pitch50_y8, valid_pitch50_x8, valid_pitch50_y8, lr=0.1, epochs=60, batch_size=50)
train_pitch100_x8, train_pitch100_y8, valid_pitch100_x8, valid_pitch100_y8 = split_train_valid(pitch100_x8,pitch100_y8, 6900, 7500, 8000)
pitch100_model8, pitch100_history8, pre_pitch100_Y8 = Model_LSTM(train_pitch100_x8, train_pitch100_y8, valid_pitch100_x8, valid_pitch100_y8, lr=0.1, epochs=60, batch_size=50)
train_pitch200_x8, train_pitch200_y8, valid_pitch200_x8, valid_pitch200_y8 = split_train_valid(pitch200_x8,pitch200_y8, 6900, 7500, 8000)
pitch200_model8, pitch200_history8, pre_pitch200_Y8 = Model_LSTM(train_pitch200_x8, train_pitch200_y8, valid_pitch200_x8, valid_pitch200_y8, lr=0.1, epochs=60, batch_size=50)

fan_pitch1_real8,fan_pitch1_pre8 = FanGuiHua_pitch(valid_pitch1_y8,pre_pitch1_Y8)
fan_pitch10_real8,fan_pitch10_pre8 = FanGuiHua_pitch(valid_pitch10_y8,pre_pitch10_Y8)
fan_pitch30_real8,fan_pitch30_pre8 = FanGuiHua_pitch(valid_pitch30_y8,pre_pitch30_Y8)
fan_pitch50_real8,fan_pitch50_pre8 = FanGuiHua_pitch(valid_pitch50_y8,pre_pitch50_Y8)
fan_pitch100_real8,fan_pitch100_pre8 = FanGuiHua_pitch(valid_pitch100_y8,pre_pitch100_Y8)
fan_pitch200_real8,fan_pitch200_pre8 = FanGuiHua_pitch(valid_pitch200_y8,pre_pitch200_Y8)
np.savetxt('pitchN_其他步长8.csv',np.hstack((fan_pitch1_real8,fan_pitch1_pre8,fan_pitch10_pre8,
                                        fan_pitch30_pre8,fan_pitch50_pre8,fan_pitch100_pre8,fan_pitch200_pre8)),delimiter=',')


# In[36]:


plt.figure()
# 将x周的刻度线方向设置向内
plt.rcParams['xtick.direction'] = 'in'  
# 将y轴的刻度方向设置向内
plt.rcParams['ytick.direction'] = 'in'  
#设置字体以便支持中文
plt.rcParams['font.sans-serif']=['SimHei']
#为正常显示负号
plt.rcParams['axes.unicode_minus'] = False 
plt.plot(H[:12,], label = 'train loss')
plt.plot(H[:12,], '.')
plt.legend()
plt.show()


# In[37]:


from rembg import remove


# In[ ]:


input_path = 'input.png'
output_path = 'output.png'

with open(input_path, 'rb') as i:
    with open(output_path, 'wb') as o:
        input = i.read()
        output = remove(input)
        o.write(output)


# In[46]:


import paddle 
import paddlehub as hub


# In[ ]:


import os
import paddlehub as hub

# 加载模型
humanseg = hub.Module(name='deeplabv3p_xception65_humanseg')  
base_dir = os.path.abspath(os.path.dirname(__file__))

# 获取当前文件目录
path = os.path.join(base_dir, 'images/')
# 获取文件列表
files = [path + i for i in os.listdir(path)]  
print(files)
# 抠图
results = humanseg.segmentation(data={'image': files})  
for result in results:
    print(result)

