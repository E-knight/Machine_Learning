from sklearn import svm
from sklearn.model_selection import train_test_split
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

    train_images = (train_images-127.5) / 127.5
    test_images = (test_images-127.5) / 127.5
    return train_images, train_labels, test_images, test_labels

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
    svm_linear=svm.LinearSVC()
    svm_linear.fit(X_train,y_train)
    y_predict=svm_linear.predict(X_test)
    cnt=0
    for i in range(y_test.shape[0]):
        print('第{}条数据，预测值:{},实际值:{}'.format(i+1,y_predict[i],y_test[i]))
        if(y_predict[i]!=y_test[i]):
            cnt+=1
        print('线性SVM错误次数:{}'.format(cnt))
    svm_nonlinear=svm.SVC(kernel='rbf')
    svm_nonlinear.fit(X_train,y_train)
    y_predict = svm_nonlinear.predict(X_test)
    cnt = 0
    for i in range(y_predict.shape[0]):
        print('第{}条数据，预测值:{},实际值:{}'.format(i + 1, y_predict[i], y_test[i]))
        if (y_predict[i] != y_test[i]):
            cnt += 1
        print('非线性SVM错误次数:{}'.format(cnt))