import math
import os
import struct
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
from multiprocessing import Manager
from sklearn.decomposition import PCA

def load_dataset(path):  # ?????????
    train_image_path = os.path.join(path, 'train-images.idx3-ubyte')
    train_label_path = os.path.join(path, 'train-labels.idx1-ubyte')
    test_image_path = os.path.join(path, 't10k-images.idx3-ubyte')
    test_label_path = os.path.join(path, 't10k-labels.idx1-ubyte')

    with open(train_image_path, 'rb') as image_path:
        magic_train_number, num_train_images, num_train_rows, num_train_columns = struct.unpack('>IIII', image_path.read(16))
        # print(magic_train_number,num_train_images,num_train_rows,num_train_columns)
        train_images = np.fromfile(image_path, dtype=np.uint8).reshape(60000, 784)

    with open(train_label_path, 'rb') as label_path:
        magic_train_number2, num_train_items = struct.unpack('>II', label_path.read(8))
        # print(magic_train_number2,num_train_items)
        train_labels = np.fromfile(label_path, dtype=np.uint8)

    with open(test_image_path, 'rb') as image_path2:
        magic_test_number, num_test_images, num_test_rows, num_test_columns = struct.unpack('>IIII',image_path2.read(16))
        # print(magic_test_number, num_test_images, num_test_rows, num_test_columns)
        test_images = np.fromfile(image_path2, dtype=np.uint8).reshape(10000, 784)

    with open(test_label_path, 'rb') as label_path2:
        magic_test_number2, num_test_items = struct.unpack('>II', label_path2.read(8))
        # print(magic_test_number2, num_test_items)
        test_labels = np.fromfile(label_path2, dtype=np.uint8)

    train_images = train_images / 127.0
    test_images = test_images / 127.0
    return train_images, train_labels, test_images, test_labels


def display_image(image):  # ????
    fig = plt.figure()
    fig.add_subplot(111)
    plt.imshow(image, cmap='gray')
    plt.show()



def error_rate_show(data,error_rate):#?????????????????????
    plt.figure()
    plt.plot(data,error_rate)
    plt.xlabel("lambda")
    plt.ylabel("Test Accuracy")
    plt.show()

def feature_extraction(train_images,test_images):#PCA??
    pca=PCA(0.9)
    pca.fit(train_images)
    return pca.transform(train_images),pca.transform(test_images)

def one_hot(y):
    ans=np.zeros((y.shape[0],10))
    for i in range(y.shape[0]):
        ans[i][y[i]]=1
    return ans

def train(X,y,reg):
    y=one_hot(y)
    left=np.zeros((X.shape[1],X.shape[1]))
    right=np.zeros((X.shape[1],y.shape[1]))
    for i in range(X.shape[0]):
        if(i%100==0):
            print('已训练{}条数据'.format(i))
        left+=np.outer(X[i],X[i].T)
        right+=np.outer(X[i],y[i].T)
    left=np.linalg.inv(left+reg*np.identity(X.shape[1]))
    return np.dot(left,right)

def predict(model, X):
    ans = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        ans[i] = np.argmax(np.dot(model.T, X[i]))
    return ans


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_dataset('Mnist')
    reg=0.001
    regs=[]
    accuracys=[]
    for i in range(12):
        model = train(train_images, train_labels,reg)
        y_train = one_hot(train_labels)
        y_test = one_hot(test_labels)
        pred_labels_test = predict(model, test_images)
        print("Test accuracy: {0}".format(metrics.accuracy_score(test_labels, pred_labels_test)))
        regs.append(reg)
        accuracys.append(metrics.accuracy_score(test_labels, pred_labels_test))
        reg*=10
    error_rate_show(regs,accuracys)
#0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 10000000 100000000
#0.8534 0.8532 0.8533 0.8533 0.8539 0.8551 0.8576 0.8552 0.8389 0.7929 0.7438 0.6919



