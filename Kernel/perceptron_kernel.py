from sklearn.model_selection import train_test_split
import os
import struct
import math
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
from multiprocessing import Manager
from sklearn.decomposition import PCA


def load_dataset(path):  # 读取数据集
    train_image_path = os.path.join(path, 'train-images.idx3-ubyte')
    train_label_path = os.path.join(path, 'train-labels.idx1-ubyte')
    test_image_path = os.path.join(path, 't10k-images.idx3-ubyte')
    test_label_path = os.path.join(path, 't10k-labels.idx1-ubyte')

    with open(train_image_path, 'rb') as image_path:
        magic_train_number, num_train_images, num_train_rows, num_train_columns = struct.unpack('>IIII',
                                                                                                image_path.read(16))
        # print(magic_train_number,num_train_images,num_train_rows,num_train_columns)
        train_images = np.fromfile(image_path, dtype=np.uint8).reshape(60000, 784)

    with open(train_label_path, 'rb') as label_path:
        magic_train_number2, num_train_items = struct.unpack('>II', label_path.read(8))
        # print(magic_train_number2,num_train_items)
        train_labels = np.fromfile(label_path, dtype=np.uint8)

    with open(test_image_path, 'rb') as image_path2:
        magic_test_number, num_test_images, num_test_rows, num_test_columns = struct.unpack('>IIII',
                                                                                            image_path2.read(16))
        # print(magic_test_number, num_test_images, num_test_rows, num_test_columns)
        test_images = np.fromfile(image_path2, dtype=np.uint8).reshape(10000, 784)

    with open(test_label_path, 'rb') as label_path2:
        magic_test_number2, num_test_items = struct.unpack('>II', label_path2.read(8))
        # print(magic_test_number2, num_test_items)
        test_labels = np.fromfile(label_path2, dtype=np.uint8)

    train_images = (train_images-127.5) / 127.5
    test_images = (test_images-127.5) / 127.5
    return train_images, train_labels, test_images, test_labels
def rbf(x,y):
    return np.exp(-1 * (np.linalg.norm(x - y) ** 2))
def predict(X_train,alphas,x):
    scores=np.zeros((10,1))
    for i in range(10):
        for j in range(X_train.shape[0]):
            scores[i]+=alphas[i][j]*rbf(x,X_train[j])

    return np.argmax(scores)

def train(alphas,X_train,y_train,epochs):
    for epoch in range(1,epochs+1):
        for i in range(X_train.shape[0]):
            x=X_train[i]
            y=y_train[i]
            y_predict=predict(X_train,alphas,x)
            if(y_predict!=y):
                alphas[y][i]+=1
                alphas[y_predict][i]-=1
                # print(alphas)
            print('正在训练第{}轮中第{}个数据'.format(epoch,i+1))

    return alphas

def predict_count(X_train,alphas,X_test,y_test):
    cnt=0
    for i in range(X_test.shape[0]):
        y_predict=predict(X_train,alphas,X_test[i])
        if(y_predict!=y_test[i]):
            cnt+=1
        print('第{}个预测，预测值：{},实际值：{}'.format(i+1,y_predict,y_test[i]))
        print('错误次数：{}'.format(cnt))
    return cnt

if __name__ == '__main__':
    X_train,y_train,X_test,y_test=load_dataset('Mnist')
    X_train=X_train[:1000]
    y_train=y_train[:1000]
    X_test=X_test[:200]
    y_test=y_test[:200]
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    alphas=np.zeros((10,X_train.shape[0]))
    alphas=train(alphas,X_train,y_train,10)
    # for i in range(600):
    #     print('第{}批训练，训练从{}到{}的数据'.format(i,i*100,i*100+99))
    #     alphas=train(alphas,X_train[i*100:i*100+100],y_train[i*100:i*100+100],10)
    cnt=predict_count(X_train,alphas,X_test,y_test)
    print('总错误次数：{}'.format(cnt))