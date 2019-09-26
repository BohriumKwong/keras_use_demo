# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:53:06 2019

@author: Bohrium.Kwong
"""

from keras.applications.xception import Xception
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
#这个是keras自带的数据生成器
from generators import DataGenerator
#这个是经我改善过的第三方数据生成器 By Bohrium.Kwong 
from keras.models import Sequential,load_model,Model
from keras.layers import Dropout, Flatten, Dense,BatchNormalization
from keras.optimizers import *
from keras.callbacks import CSVLogger, Callback, EarlyStopping
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras import initializers
from keras import regularizers
from keras.layers import advanced_activations
from keras.utils import multi_gpu_model
from keras.initializers import Orthogonal,he_normal
from keras import backend as K
import numpy as np
import os,sys
import random
import time
import metrics
#这个是第三方的acc函数定义脚本，提供许多keras所没有的指标统计，如recall 和F-Score

class Mycbk(ModelCheckpoint):
    def __init__(self, model, filepath ,monitor = 'val_loss',mode='min', save_best_only=True):
        self.single_model = model
        super(Mycbk,self).__init__(filepath, monitor, save_best_only, mode)
    def set_model(self,model):
        super(Mycbk,self).set_model(self.single_model)

def get_callbacks(filepath,save_base_name,model,patience=4):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = Mycbk(model, './' + save_base_name + '/'+filepath)
    file_dir = './' + save_base_name + '/log/'+ time.strftime('%Y_%m_%d',time.localtime(time.time()))
    if not os.path.exists(file_dir): os.makedirs(file_dir)
    tb_log = TensorBoard(log_dir=file_dir)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                  patience=4, verbose=0, mode='min', epsilon=-0.95, cooldown=0, min_lr=1e-8)
    log_cv = CSVLogger('./' + save_base_name + '/' + time.strftime('%Y_%m_%d',time.localtime(time.time())) +'_log.csv', separator=',', append=True)
    return [es, msave,reduce_lr,tb_log,log_cv]

os.environ["CUDA_VISIBLE_DEVICES"] ='0,1'
# 按需进行设置,如果只用一张显卡，可以忽略下面的multi_gpu_model(model, gpus=n)及其相关语句


if __name__ == '__main__':
    Xception_model = Xception(weights=None, include_top=False, input_shape=(299,299,3),pooling='Average')
    #实际上也可以设置weights='imagenet',不过这样做的话框架判断用户目录下，即~/.keras/models判断是否有对应的权重文件，
    # 没有的话会执行下载，建议自己另外加载就好。具体参看http://192.168.3.126/Kwong/totem_begin/blob/master/README.md
    # Xception_model = load_model('xception_weights_tf_dim_ordering_tf_kernels_notop.h5') 
    # 如果加载的是imageNet预训练权重，需要添加另外的预处理方法(去imageNet数据集的均值和标准差)
    # 详见：http://192.168.3.126/Kwong/totem_begin/blob/master/tricks_in_processing_and_training/README.md
    top_model_avg = Sequential()
    top_model_avg.add(Flatten(input_shape=Xception_model.output_shape[1:],name='flatten_Average'))
    #top_model_avg.add(Dropout(0.5))
    top_model_avg.add(BatchNormalization())
    top_model_avg.add(Dense(48, name='dense_1', kernel_initializer=initializers.Orthogonal()))
    top_model_avg.add(advanced_activations.PReLU(alpha_initializer='zeros'))
    #top_model_avg.add(Dropout(0.5))
    top_model_avg.add(BatchNormalization())
    top_model_avg.add(Dense(2, name='dense_2', activation="softmax", kernel_initializer=initializers.Orthogonal()))
    
    model = Sequential()
    model.add(Xception_model)
    model.add(top_model_avg)
    # 以上是使用序列式定义模型的方法
    
    #Xception_model2 = Xception(weights=None, include_top=False, input_shape=(299,299,3),pooling='Average')
    #base_out = Flatten()(Xception_model2.output)
    #drout1= Dropout(0.5)(base_out)
    #dense1= Dense(64,activation='relu',kernel_initializer=initializers.he_normal(seed=None))(drout1)
    #drout2= Dropout(0.5)(dense1)
    #dense2= Dense(2, activation='softmax',kernel_initializer=initializers.he_normal(seed=None))(drout2)
    #model=Model(inputs=Xception_model2.input,outputs=dense2)
    # 以上是使用函数式定义模型的方法
    
    parallel_model = multi_gpu_model(model, gpus=2)
    #这是使用多张显卡进行训练的方法，如果只有一张显卡，请忽略这句，下面的compile和fit_generator改为model
    parallel_model.compile(loss='binary_crossentropy',#超过2类的话loss应为'categorical_crossentropy'
                  optimizer = SGD(lr=1e-3, decay=0.0001,momentum=0.7,
                                nesterov=True,clipvalue=0.7,clipnorm=1),
                  metrics=[metrics.precision,metrics.recall,metrics.fscore])
    #我在这里使用了自定义的指标(metrics.*)代替默认的'accuracy',不需要的话可以将其改回来
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip = True,
        vertical_flip = True
        #samplewise_center= True
        #samplewise_std_normalization=True
    )
    val_datagen = ImageDataGenerator(
            rescale=1./255
            #samplewise_center= True
            #samplewise_std_normalization=True
    )
    # 这里使用keras自带的图像生成器
    
#    train_datagen = DataGenerator(horizontal_flip = True, vertical_flip = True, stain_transformation = True)
#    valid_datagen = DataGenerator()
#    # 这里使用的是改善后的第三方图像生成器，对比自带的生成器多了颜色转换增强的方法stain_transformation,按需使用
    
    batch_size = 32
    # 生成训练数据
    train_generator = train_datagen.flow_from_directory(
            '/train',  # 训练数据路径
            target_size=(299, 299),  # 设置图片大小
            batch_size=batch_size # 批次大小
            )
    val_generator = val_datagen.flow_from_directory(
            '/val',  # 训练数据路径
            target_size=(299, 299),  # 设置图片大小
            batch_size=batch_size # 批次大小
            )

    print(train_generator.class_indices)
    # 可通过该语句获取标签字典 ，返回如{'label_0': 0, 'label_1': 1}
    
    totalFileCount = sum([len(files) for root, dirs, files in os.walk('/train')])
    totalFileCount_test = sum([len(files) for root, dirs, files in os.walk('/test')])
    
    save_base_name = 'Xception_299'
    model_save_name = save_base_name + '_' + time.strftime('%Y_%m_%d',time.localtime(time.time())) + '.hdf5'
    callbacks_s = get_callbacks(model_save_name,save_base_name,parallel_model,patience=5)
    #调用get_callbacks，定义early_stopping保存的文件名和日志名
    parallel_model.fit_generator (                                                                                                                                                  
        train_generator,
        steps_per_epoch = totalFileCount//batch_size,  
        epochs=30,
        validation_data = val_generator,
        validation_steps = totalFileCount_test //batch_size,
        callbacks = callbacks_s
        )
#"""
#Epoch 9/10
#33497/33497 [==============================] - 15986s 477ms/step - loss: 0.6058 - precision: 0.7975 - recall: 0.7975 - fmeasure: 0.7975 - val_loss: 0.5397 - val_precision: 0.8067 - val_recall: 0.8067 - val_fmeasure: 0.8067
#Epoch 10/10
#33497/33497 [==============================] - 15992s 477ms/step - loss: 0.6079 - precision: 0.7972 - recall: 0.7972 - fmeasure: 0.7972 - val_loss: 0.5196 - val_precision: 0.8090 - val_recall: 0.8090 - val_fmeasure: 0.8090
#"""
