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

import tensorflow.keras.backend as K
from keras.layers import Multiply
from keras.models import Sequential ,Model
from keras.layers import Dense , Input , Reshape , Flatten ,Permute , Lambda , RepeatVector ,Conv1D , MaxPooling1D , Dropout, Bidirectional, Activation
from keras.layers import GRU, LSTM
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD,Adam
# from keras.utils import np_utils   #np_utils
from tensorflow.keras.callbacks import TensorBoard  #TensorBoard可视化


# In[2]:


data_1 = pd.read_csv('t_2_11.2_50.csv')
data_1.head()


# In[3]:


#数目count、均值mean、标准差std、最小值min、下四分位数25%、中位数50%、上四分位数75%、最大值max
data_1.describe().transpose()


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
Force2 = Force2_scaler.fit_transform(data_distance[:,8:9])
# zong_data = np.hstack(())


# In[5]:


print(np.hstack((H,Surge,Sway,Heave,Roll,Pitch,Yaw,Force1,Force2)))


# In[6]:


from scipy.stats import pearsonr

# 数据准备
# x = [1, 2, 3, 4, 5]
# y = [6, 7, 8, 9, 10]

# 计算相关系数和p值
corr, p_value = pearsonr(Pitch.reshape(40000,), Force1.reshape(40000,))
print("Pitch与Force1的相关系数：", corr)
print("Pitch与Force1的p值：", p_value)



corr, p_value = pearsonr(Heave.reshape(40000,), Force1.reshape(40000,))
print("Heave与Force1的相关系数：", corr)
print("Heave与Force1的p值：", p_value)


corr, p_value = pearsonr(Surge.reshape(40000,), Force1.reshape(40000,))
print("Surge与Force1的相关系数：", corr)
print("Surge与Force1的p值：", p_value)


corr, p_value = pearsonr(H.reshape(40000,), Force1.reshape(40000,))
print("H与Force1的相关系数：", corr)
print("H与Force1的p值：", p_value)


# In[7]:


corr, p_value = pearsonr(Sway.reshape(40000,), Force1.reshape(40000,))
print("Sway与Force1的相关系数：", corr)
print("Sway与Force1的p值：", p_value)


corr, p_value = pearsonr(Roll.reshape(40000,), Force1.reshape(40000,))
print("Roll与Force1的相关系数：", corr)
print("Roll与Force1的p值：", p_value)


corr, p_value = pearsonr(Yaw.reshape(40000,), Force1.reshape(40000,))
print("Yaw与Force1的相关系数：", corr)
print("Yaw与Force1的p值：", p_value)


# In[8]:


corr, p_value = pearsonr(Pitch.reshape(40000,), Force2.reshape(40000,))
print("Pitch与Force2的相关系数：", corr)
print("Pitch与Force2的p值：", p_value)
corr, p_value = pearsonr(Heave.reshape(40000,), Force2.reshape(40000,))
print("Heave与Force2的相关系数：", corr)
print("Heave与Force2的p值：", p_value)
corr, p_value = pearsonr(Surge.reshape(40000,), Force2.reshape(40000,))
print("Surge与Force2的相关系数：", corr)
print("Surge与Force2的p值：", p_value)
corr, p_value = pearsonr(H.reshape(40000,), Force2.reshape(40000,))
print("H与Force2的相关系数：", corr)
print("H与Force2的p值：", p_value)
corr, p_value = pearsonr(Sway.reshape(40000,), Force2.reshape(40000,))
print("Sway与Force2的相关系数：", corr)
print("Sway与Force2的p值：", p_value)
corr, p_value = pearsonr(Roll.reshape(40000,), Force2.reshape(40000,))
print("Roll与Force2的相关系数：", corr)
print("Roll与Force2的p值：", p_value)
corr, p_value = pearsonr(Yaw.reshape(40000,), Force2.reshape(40000,))
print("Yaw与Force2的相关系数：", corr)
print("Yaw与Force2的p值：", p_value)


# In[9]:


# corr_pearson1 = np.corrcoef(np.hstack((H,Surge,Sway,Heave,Roll,Pitch,Yaw,Force1,Force2)), rowvar = False) #np.corrcoef指Pearson皮尔逊相关系数
# print(corr_pearson1)
# # # corr_pearson1 = scaled_data.iloc[:, 1:10].corr('pearson')
# # # print(corr_pearson1)

# import seaborn as sns
# figure, ax = plt.subplots(figsize = (10, 10))
# # figsize = (5.67, 5.67)
# #设置字体以便支持中文
# plt.rcParams['font.sans-serif']=['SimHei']
# #为正常显示负号
# plt.rcParams['axes.unicode_minus'] = False 
# #设置字体以便支持中文
# # plt.rcParams['font.sans-serif']=['SongNTR']
# sns.heatmap(corr_pearson1, square=True, annot=True, ax=ax, xticklabels = ['波高', '纵荡', '横荡', '垂荡', '横摇', '纵摇', '艏摇', '迎浪系泊', '背浪系泊'], yticklabels = ['波高', '纵荡', '横荡', '垂荡', '横摇', '纵摇', '艏摇', '迎浪系泊', '背浪系泊'], cmap='RdYlGn')  #cmap调整热力图颜色(Purples)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()
# plt.savefig('corr_features.tif')


# In[29]:


# 训练量=n_train_times_end-n_train_times_start
def split_train_valid(data_X, data_Y, n_train_times_start, n_train_times_end, n_valid_times):
    '''
    说明：将数据集划分为训练集和验证集
    '''
    train_x, valid_x = data_X[n_train_times_start:n_train_times_end, :], data_X[n_train_times_end:n_valid_times, :]
    train_y, valid_y = data_Y[n_train_times_start:n_train_times_end], data_Y[n_train_times_end:n_valid_times]
    train_y = train_y.reshape((n_train_times_end-n_train_times_start, 1))
    valid_y = valid_y.reshape((n_valid_times-n_train_times_end, 1))
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
#     print('MAE:',MAE1)
#     print('MAE(sklearn):', mean_absolute_error(XP1, XA1))
#     print('MAPE(sklearn):', mean_absolute_percentage_error(XP1, XA1))

    #MSE
    r1 = ((XP1-XA1)*(XP1-XA1)).sum()/(XP1.shape[0])
    #RMSE = np.sqrt(r)
    MSE1 = r1
#     print('MSE:', MSE1)
#     print('MSE(sklearn):', mean_squared_error(XP1, XA1))
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
#     evaluate(fan_force1_real,fan_force1_pre)
    return fan_force1_real,fan_force1_pre
def FanGuiHua_force2(valid_for2_y,for2_pre_Y):
    fan_force2_real=Force2_scaler.inverse_transform(valid_for2_y)
    fan_force2_pre=Force2_scaler.inverse_transform(for2_pre_Y)
    plt.figure(figsize=(30,4), dpi=100)
    plt.plot(fan_force2_real,color='blue',label='real')   #真实曲线
    plt.plot(fan_force2_pre,color='orange',label='prediction') #预测曲线
    plt.legend()
    plt.show()
#     evaluate(fan_force2_real,fan_force2_pre)
    return fan_force2_real,fan_force2_pre


# In[11]:


# class InitCentersRandom(Initializer):
#     """ Initializer for initialization of centers of RBF network
#         as random samples from the given data set.

#     # Arguments
#         X: matrix, dataset to choose the centers from (random rows
#           are taken as centers)
#     """

#     def __init__(self, X):
#         self.X = X

#     def __call__(self, shape, dtype=None):
#         assert shape[1] == self.X.shape[1]
#         idx = np.random.randint(self.X.shape[0], size=shape[0])

# 	# type checking to access elements of data correctly
# #         if type(self.X) == np.ndarray:
# #             return self.X[idx, :]
# #         elif type(self.X) == pd.core.frame.DataFrame:
# #             return self.X.iloc[idx, :]
# class RBFLayer(Layer):
#     """ Layer of Gaussian RBF units.
#     # Example

#     ```python
#         model = Sequential()
#         model.add(RBFLayer(10,
#                            initializer=InitCentersRandom(X),
#                            betas=1.0,
#                            input_shape=(1,)))
#         model.add(Dense(1))
#     ```


#     # Arguments
#         output_dim: number of hidden units (i.e. number of outputs of the
#                     layer)
#         initializer: instance of initiliazer to initialize centers
#         betas: float, initial value for betas

#     """
#     def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
#         self.output_dim = output_dim
#         self.init_betas = betas
#         if not initializer:
#             self.initializer = RandomUniform(0.0, 1.0)
#         else:
#             self.initializer = initializer
#         super(RBFLayer, self).__init__(**kwargs)

#     def build(self, input_shape):

#         self.centers = self.add_weight(name='centers',
#                                        shape=(self.output_dim, input_shape[1]),
#                                        initializer=self.initializer,
#                                        trainable=True)
#         self.betas = self.add_weight(name='betas',
#                                      shape=(self.output_dim,),
#                                      initializer=Constant(
#                                          value=self.init_betas),
#                                      # initializer='ones',
#                                      trainable=True)

#         super(RBFLayer, self).build(input_shape)
#     def call(self, x):

#         C = K.expand_dims(self.centers)
#         H = K.transpose(C-K.transpose(x))
#         return K.exp(-self.betas * K.sum(H**2, axis=1))

#         # C = self.centers[np.newaxis, :, :]
#         # X = x[:, np.newaxis, :]

#         # diffnorm = K.sum((C-X)**2, axis=-1)
#         # ret = K.exp( - self.betas * diffnorm)
#         # return ret

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.output_dim)

#     def get_config(self):
#         # have to define get_config to be able to use model_from_json
#         config = {
#             'output_dim': self.output_dim
#         }
#         base_config = super(RBFLayer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))        
# def Model_RBF(train_X, train_Y, valid_X, valid_Y, feature_number, lr=0.01, epochs=20, batch_size=256):
#     model1 = Sequential()
#     model1.add(RBFLayer(30, initializer=InitCentersRandom(train_x), betas=2.0, input_shape=(feature_number,)))
#     model1.add(Dense(units = Dense_units, activation = 'relu', kernel_regularizer = l2(0.001)))
#     model1.add(Dense(1))
#     #model.compile()用来配置学习过程 #优化器选用adam，损失函数选用均方误差
#     adam = Adam(lr=0.001)
#     model1.compile(optimizer = 'adam' , loss = 'mse') 
#     history = model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, validation_data=(valid_X, valid_Y), verbose=2, shuffle=False)
#     model_structure = model.summary()
#     pre_train_Y = model.predict(train_X)
#     pre_Y = model.predict(valid_X)
#     return history, pre_train_Y, pre_Y


def Model_NN(train_X, train_Y, valid_X, valid_Y, feature_number, lr=0.01, epochs=20, batch_size=256):
    input_layer = Input(shape=(feature_number,)) 
    D_layer1 = Dense(units = 150, activation = 'tanh')(input_layer)
    D_layer2 = Dense(units = 150, activation = 'tanh')(D_layer1)
    D_layer3 = Dense(units = 150, activation = 'tanh')(D_layer2)
    output_layer = Dense(1, activation = 'linear')(D_layer2)
    model = Model(inputs = input_layer, outputs = output_layer)
    adam = Adam(lr = lr)
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, validation_data=(valid_X, valid_Y), verbose=2, shuffle=False)
    model_structure = model.summary()
    pre_train_Y = model.predict(train_X)
    pre_Y = model.predict(valid_X)
    return history, pre_train_Y, pre_Y


# 基于直方图的梯度增强回归树（Histogram-based Gradient Boosting Regression Tree, HGBRT）
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingRegressor
max_depth=[3, 5, 6, 7, 9, 12, 15, 17, 25]
# max_leaf_nodes=[29, 30, 31, 32, 33]
# min_samples_leaf=[18, 19, 20, 21, 22]
max_leaf_nodes=[i for i in range(25, 45)]
min_samples_leaf=[i for i in range(10, 25)]
def HGBRT(train_x, train_y,valid_x,max_depth,max_leaf_nodes,min_samples_leaf):
    param_grid = dict(max_depth=max_depth,
                max_leaf_nodes=max_leaf_nodes,
                min_samples_leaf=min_samples_leaf)
    grid_search = GridSearchCV(HistGradientBoostingRegressor(), param_grid, scoring="neg_mean_absolute_error", n_jobs=-1)
    grid_result = grid_search.fit(train_x, train_y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    best_parameters = grid_search.best_estimator_.get_params()
    best_model = HistGradientBoostingRegressor(learning_rate=0.1,
                                               max_depth=best_parameters['max_depth'],
                                               max_leaf_nodes=best_parameters['max_leaf_nodes'],
                                               min_samples_leaf=best_parameters['min_samples_leaf'],
                                               loss='least_squares')
    best_model.fit(train_x, train_y)
    pre_y = best_model.predict(valid_x)
    pre_y = best_model.predict(valid_x)
    return pre_y, best_parameters


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
max_depth=[3, 5, 7, 9, 11, 13, 15]
min_child_weight=[1, 3]
gamma=[0, 0.05, 0.1]
subsample=[0.6, 0.7, 0.8, 0.9, 1]
colsample_bytree=[0.6, 0.7, 0.8, 0.9, 1]
reg_alpha=[0, 0.001, 0.005, 0.025, 0.05]
learning_rate=[0.025, 0.05, 0.1, 0.2]
n_estimators=[50, 100, 200, 400]
# 分布式梯度增强树
def XGBoost(train_x,train_y,valid_x,max_depth,min_child_weight,gamma,subsample,colsample_bytree,reg_alpha,learning_rate,n_estimators):
#     max_depth=3,min_child_weight=1,gamma=0,subsample=0.6,colsample_bytree=0.6,reg_alpha=0.001,learning_rate=0.001,n_estimators=100
    # 参数调优部分(网格调优)
    param_grid = dict(max_depth=max_depth,
                min_child_weight=min_child_weight,
                gamma=gamma,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                learning_rate=learning_rate,
                n_estimators=n_estimators)
    grid_search = GridSearchCV(XGBRegressor(), param_grid, scoring="neg_mean_absolute_error", n_jobs=-1)
    grid_result = grid_search.fit(train_x, train_y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    best_parameters = grid_search.best_estimator_.get_params()
    # 使用模型对测试集数据进行预测
    best_model = XGBRegressor(max_depth=best_parameters['max_depth'],
                              min_child_weight=best_parameters['min_child_weight'],
                              gamma=best_parameters['gamma'],
                              subsample=best_parameters['subsample'],
                              colsample_bytree=best_parameters['colsample_bytree'],
                              reg_alpha=best_parameters['reg_alpha'],
                              learning_rate=best_parameters['learning_rate'],
                              n_estimators=best_parameters['n_estimators'])
    best_model.fit(train_x,train_y)
    pre_y = best_model.predict(valid_x)
    return pre_y, best_parameters


# In[12]:


# 不同非线性模型下的效果
train_x, train_y, valid_x, valid_y= split_train_valid(np.hstack((Surge,Pitch,Heave)), Force1, 20000, 30000, 31000)

# R_history, R_pre_train_Y, R_pre_Y = Model_RBF(train_x, train_y, valid_x, valid_y, feature_number, lr=0.01, epochs=1000, batch_size=256)
N_history, N_pre_train_Y, N_pre_Y = Model_NN(train_x, train_y, valid_x, valid_y, 3, lr=0.01, epochs=1000, batch_size=256)
# X_pre_y, X_best_parameters = XGBoost(train_x, train_y, valid_x,max_depth,min_child_weight,gamma,subsample,colsample_bytree,reg_alpha,learning_rate,n_estimators)
H_pre_y, H_best_parameters = HGBRT(train_x, train_y, valid_x,max_depth,max_leaf_nodes,min_samples_leaf)


# In[13]:


# loss_plot(R_history, epo=1000, length=10, width=6)
loss_plot(N_history, epo=1000, length=10, width=6)


# In[14]:


print(N_pre_Y.shape)
print(H_pre_y.shape)


# In[30]:


# evaluate(valid_y,pre_Y)
# evaluate(valid_y,R_pre_Y)
evaluate(valid_y,N_pre_Y)
evaluate(valid_y,H_pre_y)
# evaluate(valid_y,X_pre_y)
# print(valid_y)

# print(R_pre_Y)

# print(N_pre_Y)

# print(H_pre_y)

# print(H_pre_y)
fan1_force1_real,fan1_force1_pre = FanGuiHua_force1(valid_y,N_pre_Y)
np.savetxt('Force1_非线性_NN.csv',np.hstack((fan1_force1_real,fan1_force1_pre)),delimiter=',')
fan2_force1_real,fan2_force1_pre = FanGuiHua_force1(valid_y,H_pre_y.reshape(1000,1))
np.savetxt('Force1_非线性_HGBRT.csv',np.hstack((fan2_force1_real,fan2_force1_pre)),delimiter=',')


# In[34]:


def campare(first,end):
    value = 100*(end-first)/first
    return print(value,'%')

campare(0.007244716700045015,0.004069999741823512)
campare(0.9963980882262893,0.9974081638469094)


# In[17]:


# plt.figure(figsize=(60, 4), dpi=100)
# plt.plot(valid_y, color='blue', label = 'real')   #真实曲线
# plt.plot(R_pre_Y, color='orange', label = 'prediction') #预测曲线
# plt.legend()
# plt.show()

# plt.figure(figsize=(60, 4), dpi=100)
# plt.plot(valid_y, color='blue', label = 'real')   #真实曲线
# plt.plot(N_pre_Y, color='orange', label = 'prediction') #预测曲线
# plt.legend()
# plt.show()

# plt.figure(figsize=(60, 4), dpi=100)
# plt.plot(valid_y, color='blue', label = 'real')   #真实曲线
# plt.plot(H_pre_y, color='orange', label = 'prediction') #预测曲线
# plt.legend()
# plt.show()

# plt.figure(figsize=(60, 4), dpi=100)
# plt.plot(valid_y, color='blue', label = 'real')   #真实曲线
# plt.plot(X_pre_Y, color='orange', label = 'prediction') #预测曲线
# plt.legend()
# plt.show()


# In[18]:


# 不同非线性模型下的效果
train_x_Force2, train_y_Force2, valid_x_Force2, valid_y_Force2= split_train_valid(np.hstack((Surge,Pitch,Heave)), Force2, 20000, 30000, 31000)

# R_history_Force2, R_pre_train_Y_Force2, R_pre_Y_Force2 = Model_RBF(train_x_Force2, train_y_Force2, valid_x_Force2, valid_y_Force2, feature_number, lr=0.01, epochs=1000, batch_size=256)
N_history_Force2, N_pre_train_Y_Force2, N_pre_Y_Force2 = Model_NN(train_x_Force2, train_y_Force2, valid_x_Force2, valid_y_Force2, 3, lr=0.01, epochs=1000, batch_size=256)
# X_pre_y_Force2 X_best_parameters_Force2= XGBoost(train_x_Force2, train_y_Force2, valid_x_Force2,max_depth,min_child_weight,gamma,subsample,colsample_bytree,reg_alpha,learning_rate,n_estimators)
H_pre_y_Force2, H_best_parameters_Force2= HGBRT(train_x_Force2, train_y_Force2, valid_x_Force2,max_depth,max_leaf_nodes,min_samples_leaf)


# In[19]:


# loss_plot(R_history_Force2, epo=1000, length=10, width=6)
loss_plot(N_history_Force2, epo=1000, length=10, width=6)


# In[31]:


evaluate(valid_y_Force2,N_pre_Y_Force2)
evaluate(valid_y_Force2,H_pre_y_Force2)
print(valid_y_Force2)


print(N_pre_Y_Force2)


print(H_pre_y_Force2)

fan1_force2_real,fan1_force2_pre = FanGuiHua_force2(valid_y_Force2,N_pre_Y_Force2)
np.savetxt('Force2_非线性_NN.csv',np.hstack((fan1_force2_real,fan1_force2_pre)),delimiter=',')
fan2_force2_real,fan2_force2_pre = FanGuiHua_force2(valid_y_Force2,H_pre_y_Force2.reshape(1000,1))
np.savetxt('Force2_非线性_HGBRT.csv',np.hstack((fan2_force2_real,fan2_force2_pre)),delimiter=',')


# In[35]:


def campare(first,end):
    value = 100*(end-first)/first
    return print(value,'%')

campare(0.014168143972618888,0.028121138182992864)
campare(0.9975636351902422,0.9957648332844505)


# In[ ]:





# In[22]:


# plt.figure(figsize=(60, 4), dpi=100)
# plt.plot(valid_y_Force2, color='blue', label = 'real')   #真实曲线
# plt.plot(R_pre_Y_Force2, color='orange', label = 'prediction') #预测曲线
# plt.legend()
# plt.show()

# plt.figure(figsize=(60, 4), dpi=100)
# plt.plot(valid_y_Force2, color='blue', label = 'real')   #真实曲线
# plt.plot(R_pre_Y_Force2, color='orange', label = 'prediction') #预测曲线
# plt.legend()
# plt.show()

plt.figure(figsize=(60, 4), dpi=100)
plt.plot(valid_y_Force2, color='blue', label = 'real')   #真实曲线
plt.plot(N_pre_Y_Force2, color='orange', label = 'prediction') #预测曲线
plt.legend()
plt.show()

plt.figure(figsize=(60, 4), dpi=100)
plt.plot(valid_y_Force2, color='blue', label = 'real')   #真实曲线
plt.plot(H_pre_y_Force2, color='orange', label = 'prediction') #预测曲线
plt.legend()
plt.show()


# In[23]:


# 不同输入特征下的效果H,Surge,Pitch,Heave
train_x1, train_y1, valid_x1, valid_y1= split_train_valid(np.hstack((H,Surge,Pitch,Heave)), Force1, 20000, 30000, 31000)

# R_history1, R_pre_train_Y1, R_pre_Y1 = Model_RBF(train_x1, train_y1, valid_x1, valid_y1, feature_number, lr=0.01, epochs=1000, batch_size=256)
N_history1, N_pre_train_Y1, N_pre_Y1 = Model_NN(train_x1, train_y1, valid_x1, valid_y1, 4, lr=0.01, epochs=1000, batch_size=256)
# X_pre_y1, X_best_parameters1 = XGBoost(train_x1, train_y1, valid_x1,max_depth,min_child_weight,gamma,subsample,colsample_bytree,reg_alpha,learning_rate,n_estimators)
H_pre_y1, H_best_parameters1 = HGBRT(train_x1, train_y1, valid_x1,max_depth,max_leaf_nodes,min_samples_leaf)


# In[32]:


evaluate(valid_y1,N_pre_Y1)
evaluate(valid_y1,H_pre_y1)


# In[25]:


plt.figure(figsize=(60, 4), dpi=100)
plt.plot(valid_y1, color='blue', label = 'real')   #真实曲线
plt.plot(N_pre_Y1, color='orange', label = 'prediction') #预测曲线
plt.legend()
plt.show()

plt.figure(figsize=(60, 4), dpi=100)
plt.plot(valid_y1, color='blue', label = 'real')   #真实曲线
plt.plot(H_pre_y1, color='orange', label = 'prediction') #预测曲线
plt.legend()
plt.show()


# In[26]:


# 不同输入特征下的效果H,Surge,Pitch,Heave,Sway,Roll,Yaw
train_x2, train_y2, valid_x2, valid_y2= split_train_valid(np.hstack((H,Surge,Pitch,Heave,Sway,Roll,Yaw)), Force1, 20000, 30000, 31000)

# R_history2, R_pre_train_Y2, R_pre_Y2 = Model_RBF(train_x2, train_y2, valid_x2, valid_y2, feature_number, lr=0.01, epochs=1000, batch_size=256)
N_history2, N_pre_train_Y2, N_pre_Y2 = Model_NN(train_x2, train_y2, valid_x2, valid_y2, 7, lr=0.01, epochs=1000, batch_size=256)
# X_pre_y2, X_best_parameters2 = XGBoost(train_x2, train_y2, valid_x2,max_depth,min_child_weight,gamma,subsample,colsample_bytree,reg_alpha,learning_rate,n_estimators)
H_pre_y2, H_best_parameters2 = HGBRT(train_x2, train_y2, valid_x2,max_depth,max_leaf_nodes,min_samples_leaf)


# In[33]:


evaluate(valid_y2,N_pre_Y2)
evaluate(valid_y2,H_pre_y2)


# In[28]:


plt.figure(figsize=(60, 4), dpi=100)
plt.plot(valid_y2, color='blue', label = 'real')   #真实曲线
plt.plot(N_pre_Y2, color='orange', label = 'prediction') #预测曲线
plt.legend()
plt.show()

plt.figure(figsize=(60, 4), dpi=100)
plt.plot(valid_y2, color='blue', label = 'real')   #真实曲线
plt.plot(H_pre_y2, color='orange', label = 'prediction') #预测曲线
plt.legend()
plt.show()

