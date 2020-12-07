import math
import os
import struct
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

    train_images = train_images / 127.0
    test_images = test_images / 127.0
    return train_images, train_labels, test_images, test_labels


def one_hot(y):
    ans = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        ans[i][y[i]] = 1
    return ans

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def propagate(X, Y, w, b):
    # print(X.shape,Y.shape,w.shape)
    num = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    # print(A.shape)
    cost = (-1 / num) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
    dw = (1 / num) * np.dot(X, (A - Y).T)
    # print(dw.shape)
    db = (1 / num) * np.sum(A - Y)
    return cost, dw, db

def train(X, y, iterations, learningRate):
    X = np.array(X)
    y = np.array(y)
    X = X.T
    y = y.T
    w = np.zeros((X.shape[0], 1))
    b = 0
    for i in range(iterations):
        cost, dw, db = propagate(X, y, w, b)
        w -= learningRate * dw
        b -= learningRate * db
    return w, b


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_dataset('Mnist')
    y_train = one_hot(train_labels)
    error_rates=[]
    iterations=[10,100,500,1000]
    learningRates=[0.01,0.1,1,10]
    for iteration in iterations:
        for learningRate in learningRates:
            W = []
            B = []
            for i in range(10):
                X = train_images
                y = []
                for j in range(train_images.shape[0]):
                    if (train_labels[j] == i):
                        y.append(1)
                    else:
                        y.append(0)
                w, b = train(X, y, iterations=iteration, learningRate=learningRate)
                # print(w.shape)
                W.append(w)
                B.append(b)
                print('第{}轮已训练完成'.format(i))
            W = np.array(W)
            B = np.array(B)
            # print(W.shape, B.shape)
            cnt=0
            for i in range(test_images.shape[0]):
                res = []
                for j in range(10):
                    res.append(sigmoid(np.dot(test_images[i], W[j]) + B[j]))
                y_predict = np.argmax(res)
                # print(res)
                # print('预测值:{}，实际值:{}'.format(y_predict, test_labels[i]))
                if(y_predict!=test_labels[i]):
                    cnt+=1
            print('错误率为：',cnt/100.0)
            error_rates.append(cnt/100.0)
    print(error_rates)
#24.95 16.7 36.22 31.22
#16.73 11.26 17.23 16.06
#12.64 9.23 11.85 24.23
#11.29 8.74 11.37 22.29

