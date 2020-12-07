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


def feature_extraction(train_images, test_images):  # PCA??
    pca = PCA(0.9)
    pca.fit(train_images)
    return pca.transform(train_images), pca.transform(test_images)


def train(train_images, train_labels,iteration):
    w = np.zeros((1, train_images.shape[1]))  # 1*784
    b = 0
    cnt1 = 0
    hasError = True
    learningRate = 1
    while (hasError == True and cnt1 < iteration):
        cnt1 += 1
        hasError = False
        cnt = 0
        for i in range(train_images.shape[0]):
            # if(i%10000==0):
            #     print('拟合第{}条数据'.format(i))
            X = train_images[i]  # 1*784
            y = train_labels[i]
            if ((w.dot(X.T) + b) * y <= 0):
                hasError = True
                w += learningRate * y * X
                b += learningRate * y
                cnt += 1
        print('第{}轮共修正w和b {}次'.format(cnt1, cnt))
    return w, b


def predict(test_images, test_labels, w, b):
    cnt = 0
    for i in range(test_labels.shape[0]):
        X = test_images[i]
        y = test_labels[i]
        y_predict = w.dot(X) + b
        # print('预测值:{},真实值:{}'.format(y_predict, y))
        if (y_predict * y <= 0 and y == 1):
            cnt += 1
            # print('cnt: ', cnt)
    return cnt


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_dataset('Mnist')
    # train_images,test_images=feature_extraction(train_images,test_images)
    # train_images=[train_images[i] for i in range(train_images.shape[0]) if train_labels[i]<2]
    # train_labels=[label for label in train_labels if label<2]
    # test_images = [test_images[i] for i in range(test_images.shape[0]) if test_labels[i] < 2]
    # test_labels = [label for label in test_labels if label < 2]
    # print(train_images.shape,train_labels.shape)
    # train_images=np.array(train_images)
    # train_labels=np.array(train_labels)
    # test_images=np.array(test_images)
    # test_labels=np.array(test_labels)
    # print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
    temp_train_images=train_images
    temp_train_labels=train_labels
    temp_test_images=test_images
    temp_test_labels=test_labels
    iterations=[1,10,100,1000]
    error_rates=[]
    for iteration in iterations:
        train_images=temp_train_images
        train_labels=temp_train_labels
        test_images=temp_test_images
        test_labels=temp_test_labels
        error_cnt=0
        for i in range(9):
            train_labels1 = []
            for j in range(train_labels.shape[0]):
                if (int(train_labels[j]) == i):
                    train_labels1.append(1)
                else:
                    train_labels1.append(-1)
                # print(train_labels[j],train_labels1[j])
            w, b = train(train_images, train_labels1,iteration)
            print('第{}轮训练完成'.format(i))
            test_labels1 = []
            for j in range(test_labels.shape[0]):
                if (int(test_labels[j]) == i):
                    test_labels1.append(1)
                else:
                    test_labels1.append(-1)
                # print(test_labels[j], test_labels1[j])
            test_labels1 = np.array(test_labels1)
            cnt = predict(test_images, test_labels1, w, b)
            error_cnt += cnt
            print('第{}轮错误次数:'.format(i), cnt)
            train_images = [train_images[j] for j in range(train_images.shape[0]) if train_labels1[j] != 1]
            train_labels1 = [label for label in train_labels1 if label != 1]
            train_labels = [label for label in train_labels if label != i]
            train_images = np.array(train_images)
            train_labels1 = np.array(train_labels1)
            train_labels = np.array(train_labels)
            # print(train_images.shape, train_labels1.shape)
            test_images = [test_images[j] for j in range(test_images.shape[0]) if test_labels1[j] != 1]
            test_labels1 = [label for label in test_labels1 if label != 1]
            test_labels = [label for label in test_labels if label != i]
            test_images = np.array(test_images)
            test_labels1 = np.array(test_labels1)
            test_labels = np.array(test_labels)
            # print(test_images.shape, test_labels1.shape)
        print('总错误次数:', error_cnt)
        error_rates.append(error_cnt/100.0)
    print(iterations)
    print(error_rates)
#994 1006 880 806