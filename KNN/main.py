import math
import os
import struct

import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
from multiprocessing import Manager
from sklearn.decomposition import PCA

def load_dataset(path):  # 从文件中读取数据集
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

    train_images = np.round(train_images / 127)
    test_images = np.round(test_images / 127)
    return train_images, train_labels, test_images, test_labels


def display_image(image):  # 展示图片
    fig = plt.figure()
    fig.add_subplot(111)
    plt.imshow(image, cmap='gray')
    plt.show()


def calc_distance(img1, img2):  # 计算两张图片之间的曼哈顿距离
    dis = np.sum(np.abs(img1 - img2))
    return dis

def calc_distance2(img1,img2):#计算两张图片之间的欧式距离
    dis=np.sum(np.square(img1-img2))
    return math.sqrt(dis)


def KNN(K, train_images, train_labels, test_image):#KNN算法，返回预测值
    label=[0,0,0,0,0,0,0,0,0,0]
    distances = []
    for i in range(len(train_images)):
        dis = calc_distance2(train_images[i], test_image)
        distances.append(dis)
    # print(distances)
    dis_index = np.argsort(distances)#排序后数组元素在原数组的下标值
    list=[]
    for i in range(K):
        label[train_labels[dis_index[i]]]+=1
        list.append(train_labels[dis_index[i]])
    # print(label.index(max(label)))
    # print(list)#展示获取到的前K大元素的标签值
    return label.index(max(label))

def error_rate_show(K,error_rate):#绘制折线图表示K和错误率的关系
    plt.figure()
    plt.plot(K,error_rate)
    plt.xlabel("K values")
    plt.ylabel("Error rates")
    # plt.title("K-Error_rates")
    plt.show()

def error_rate_show2(data,error_rate):#绘制折线图表示测试数据集大小和错误率的关系
    plt.figure()
    plt.plot(data,error_rate)
    plt.xlabel("Test Data")
    plt.ylabel("Error rates")
    plt.show()

def get_error_rates(k,train_images,train_labels,test_images,test_labels,error_rates):#获取错误率
    cnt=0
    # print(len(test_images))
    error_rates.append(0)
    for i in range(len(test_images)):
        predict=KNN(k,train_images,train_labels,test_images[i])
        if predict!=test_labels[i]:
            cnt+=1
        print('第'+str(k)+'轮，第'+str(i)+'个预测：'+str(predict)+'，实际值：'+str(test_labels[i]))
        print('错误次数：'+str(cnt))
    # print((k-1)/2)
    error_rates[int((k-1)/2)]+=cnt/100
    # print(error_rates)
    return error_rates

def feature_extraction(train_images,test_images):#PCA降维
    pca=PCA(0.9)
    pca.fit(train_images)
    return pca.transform(train_images),pca.transform(test_images)

if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_dataset('Mnist')
    train_images,test_images=feature_extraction(train_images,test_images)
    print(train_images.shape)
    print(test_images.shape)
    # for i in range(6):
    #     get_error_rates(2*i+1, train_images, train_labels, test_images[0:100], test_labels)
    # with ThreadPoolExecutor(10) as executor:
    #     for i in range(1,6):
    #        executor.submit(get_error_rates,2*i-1,train_images,train_labels,test_images[0:100],test_labels)
    # print(error_rates)
    manager=Manager()#多进程
    return_list=manager.list()
    process=[]
    K=[]
    # for i in range(1,6):
    #     K.append(2*i-1)
    #     p=multiprocessing.Process(target=get_error_rates,args=(2*i-1,train_images,train_labels,test_images,test_labels,return_list))
    #     process.append(p)
    #     p.start()
    #
    # for p in process:
    #     p.join()
    # print(K)
    # print(return_list)
    # error_rate_show(K,return_list)
    data=[60,600,6000,60000]
    return_list=[41,17.81,6.2,3.25]
    # num=6
    # for i in range(0,4):
    #     num=num*10
    #     data.append(num)
    #     p = multiprocessing.Process(target=get_error_rates,args=(3, train_images[0:num], train_labels[0:num], test_images, test_labels, return_list))
    #     process.append(p)
    #     p.start()
    # for p in process:
    #     p.join()
    # print(data)
    # print(return_list)
    error_rate_show2(data,return_list)


