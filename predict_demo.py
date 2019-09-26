# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:52:48 2019

@author: Bohrium.Kwong
"""

from keras.models import load_model
from keras.optimizers import SGD
from keras.utils import to_categorical
from tqdm._tqdm import trange
import numpy as np
from skimage import io
from scipy.misc import imresize
import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import gc
import glob
import os

class Test_epoch_from_array:
    r"""
    :param model: a model that loaded with keras
    :param input_array: a 4D numpy array [n,h,w,c] which includes n images transform to numpy array(RGB mode)
    :param batch_size: the value of batch_size in model's predicting
    :param verbose: show the information of tdqm method with dataloaders in processing or not
    """      
    def __init__(self, model, input_array,batch_size=256,mean = [0,0,0], std = [1,1,1],verbose=False):      
        self.model = model
        self.input_array = input_array
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.verbose = verbose  
        
        
    def predict(self): 
        self.input_array /= 255
        for i in range(3):
            self.input_array[:,:,:,i] -= self.mean[i]
            self.input_array[:,:,:,i] /= self.std[i]
        if self.input_array.shape[0] <= self.batch_size :
            output_predict =  self.model.predict(self.input_array,batch_size = self.batch_size) 

        else:
            batch_count = self.input_array.shape[0]// self.batch_size
#            with tqdm(dataloaders[phase], desc = phase, file=sys.stdout, disable=not (self.verbose)) as iterator: 
            for i in trange(batch_count + 1,disable=not (self.verbose)):
                top = i * self.batch_size
                bottom = min(self.input_array.shape[0], (i+1) * self.batch_size)
                if top < self.input_array.shape[0]:
                    output_predict_batch =  self.model.predict(self.input_array[top:bottom,:,:,:].copy(),batch_size = self.batch_size) 
                    if i ==0 :
                        output_predict = output_predict_batch.copy()
                    else:
                        output_predict = np.row_stack((output_predict,output_predict_batch.copy()))
                    del image_tensor,output_predict_batch,output
                    gc.collect()
        return output_predict


def get_model_loaded(file_path,out_objects = False,objects_dict = None):
    """
    load the model trained by Keras base on the file name
    :param file_path: the saved model's file_path
    :param out_objects: True/False ,adds custom_objects or not in loadding model
    :param objects_dict: a dict that matches the model which you want to load
    :return: keras model object
    """
    if out_objects and objects_dict is not None:
        model = load_model(file_path,custom_objects=objects_dict)
    else:
        model = load_model(file_path)
    
    model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-2, momentum=0.1,decay=0.001, nesterov=True,clipvalue=0.7),
              metrics=['accuracy'])
    return model
    

def plot_confusion_matrix(test_labels, predictions,label_list):
    """
    the method of draw confusion_matrix    
    """    
    cm = confusion_matrix(test_labels, predictions)
    plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
    plt.title('Confusion matrix', size=15)
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, label_list,rotation=45, size=10)
    plt.yticks(tick_marks, label_list,size=10)
    plt.tight_layout()
    plt.ylabel('Actual label',size=15)
    plt.xlabel('Predicted label',size=15)
    width, height=cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]),xy=(y,x),horizontalalignment='center',verticalalignment='center')
    plt.show()    






if __name__ == '__main__':
    
    np.set_printoptions(suppress=True)
    
    objects_dict = {'precision':metrics.precision,'recall':metrics.recall,'fmeasure':metrics.fscore}
    #如果模型有使用自定义的指标(metrics.*)，就需要定义objects_dict字典，以便进行加载
    model = get_model_loaded('Xception_299_.hdf5',out_objects = True,objects_dict = objects_dict)
    
    #以下是预测单张图片的方法
    I = io.imread('./test_png')
    I = np.expand_dims(I, axis=0)
    preds = model.predict_classes(I)
    print('Predicted:', preds)
    prob = model.predict_proba(I)
    print('Probality:', prob)
#    """
#    Predicted: [1]
#    Probality: [[0.00000055 0.9999994 ]]
#    """
    print(prob[0][1])
    # 0.9999994
    
    
    #以下是预测指定目录下多张图片的方法
    test_dataset_dir = '/test/'
    
    
    test_folders = ["/test/label_0/", "/test/label_1/"]

    data_set_name = test_folders[0].split("/")[-3]
    
    print("\nImages for Testing")
    print("=" * 30)
    
    test_images = []
    test_labels = []
    
    for index, folder in enumerate(test_folders):
        files = glob.glob(folder + "*.png")
        images = io.imread_collection(files)
#        images = [imresize(image, (299, 299)) for image in images] ### Reshape to (299, 299, 3) ###
        labels = [index] * len(images)
        test_images = test_images + images
        test_labels = test_labels + labels
        print("Class: %s. Size: %d" %(folder.split("/")[-2], len(images)))
        
    test_images = test_images.astype(np.float32) ### Standardise

    test_labels = np.array(test_labels).astype(np.int32)
    Y_test = to_categorical(test_labels, num_classes = np.unique(test_labels).shape[0])
    
    test_loss, test_accuracy = model.evaluate(test_images/255, Y_test, batch_size = 50)
    # 这里调用的是keras模型自带的evaluate方法进行评分

    print("\n" + test_folders[0].split("/")[1] + " Loss: %.3f" %(test_loss))
    print("\n" + test_folders[0].split("/")[1] + " Accuracy: %.3f" %(test_accuracy))
    print("=" * 30, "\n")
    
    posteriors = Test_epoch_from_array(model,test_images,256)

    predictions = np.argmax(posteriors, axis = 1)
    
    
    
    print(classification_report(test_labels, predictions, target_names = os.listdir(test_dataset_dir),
                                digits = len(os.listdir(test_dataset_dir))))
    # 打印classification_report
    
    plot_confusion_matrix(test_labels, predictions,os.listdir(test_dataset_dir))
    # 画混淆矩阵
    
    
    