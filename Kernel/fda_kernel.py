from kfda import Kfda
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
    train_images,train_labels,test_images,test_labels=load_dataset('Mnist')
    print('数据加载完成')
    X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, train_size=10000, stratify=train_labels)
    print('数据划分完成')
    cls=Kfda(kernel='rbf',n_components=9)
    cls.fit_transform(X_train,y_train)
    print('数据拟合完成')
    y_predict=cls.predict(X_test)
    cnt=0
    for i in range(len(y_predict)):
        if(y_predict[i]!=y_test[i]):
            cnt+=1
        print('预测值：{},实际值:{}'.format(y_predict[i],y_test[i]))
        print('错误次数:',cnt)
    print(len(y_test))
    print(cls.score(X_test,y_test))