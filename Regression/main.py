import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
from pylab import mpl

def Linear_Regression(X_train,X_test,y_train,y_test):
    liner_regression=LinearRegression()
    liner_regression.fit(X_train,y_train)
    y_predict=liner_regression.predict(X_test)
    print('一元线性回归MSE：', mean_squared_error(y_test, y_predict))
    print('一元线性回归MAE：',mean_absolute_error(y_test,y_predict))
    print('一元线性回归EVS：',explained_variance_score(y_test,y_predict))
    print('一元线性回归r2_score：',r2_score(y_test,y_predict))
    print('******************')
    plt.plot(y_predict,label='线性回归预测值')
    quadratic=PolynomialFeatures(degree=2)
    X_train_quadratic = quadratic.fit_transform(X_train)
    liner_regression = liner_regression.fit(X_train_quadratic,y_train)
    y_predict = liner_regression.predict(quadratic.fit_transform(X_test))
    print('二元多项式回归MSE：',mean_squared_error(y_test, y_predict))
    print('二元多项式线性回归MAE：', mean_absolute_error(y_test, y_predict))
    print('二元多项式回归EVS：', explained_variance_score(y_test, y_predict))
    print('二元多项式回归r2_score：',r2_score(y_test, y_predict))
    plt.plot(y_predict, label='二元多项式回归预测值')

def Ridge_Regression(X_train,X_test,y_train,y_test):
    ridge=Ridge()
    ridge.fit(X_train,y_train)
    y_predict=ridge.predict(X_test)
    print('Ridge回归MSE：', mean_squared_error(y_test, y_predict))
    print('Ridge回归MAE：', mean_absolute_error(y_test, y_predict))
    print('Ridge回归EVS：', explained_variance_score(y_test, y_predict))
    print('Ridge回归r2_score：',r2_score(y_test,y_predict))
    plt.plot(y_predict, label='Ridge回归预测值')

def Lasso_Regression(X_train,X_test,y_train,y_test):
    lasso = Lasso()
    lasso.fit(X_train, y_train)
    y_predict = lasso.predict(X_test)
    print('Lasso回归MSE：', mean_squared_error(y_test, y_predict))
    print('Lasso回归MAE：', mean_absolute_error(y_test, y_predict))
    print('Lasso回归EVS：', explained_variance_score(y_test, y_predict))
    print('Lasso回归r2_score：', r2_score(y_test, y_predict))
    plt.plot(y_predict, label='Lasso回归预测值')

def Decision_Tree(X_train,X_test,y_train,y_test):
    decesion_tree=DecisionTreeRegressor()
    decesion_tree.fit(X_train, y_train)
    export_graphviz(decesion_tree,out_file='tree.dot',feature_names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],class_names=['MEDV'],filled=True, rounded=True,special_characters=True)
    graph=pydotplus.graph_from_dot_file('tree.dot')
    graph.write_pdf('tree.pdf')
    y_predict = decesion_tree.predict(X_test)
    print('决策树MSE：', mean_squared_error(y_test, y_predict))
    print('决策树MAE：', mean_absolute_error(y_test, y_predict))
    print('决策树EVS：', explained_variance_score(y_test, y_predict))
    print('决策树r2_score：', r2_score(y_test, y_predict))
    plt.plot(y_predict, label='决策树预测值')

def SVM(X_train,X_test,y_train,y_test):
    svr=SVR()
    svr.fit(X_train, y_train)
    y_predict = svr.predict(X_test)
    print('支持向量机MSE：', mean_squared_error(y_test, y_predict))
    print('支持向量机MAE：', mean_absolute_error(y_test, y_predict))
    print('支持向量机EVS：', explained_variance_score(y_test, y_predict))
    print('支持向量机r2_score：', r2_score(y_test, y_predict))
    plt.plot(y_predict, label='支持向量机预测值')

if __name__ == '__main__':
    os.environ['PATH']+=os.pathsep+'D:/Graphviz 2.44.1/bin/'#配置环境变量
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    boston_data=pd.read_csv('./housing.csv', header=None, delimiter=r"\s+",names=columns)
    print(boston_data)
    print(boston_data.describe())
    y=np.array(boston_data['MEDV'])
    X=np.array(boston_data.drop('MEDV',axis=1))
    X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8)
    plt.plot(y_test, label='测试数据实际值')
    Linear_Regression(X_train,X_test,y_train,y_test)
    print('******************')
    Ridge_Regression(X_train,X_test,y_train,y_test)
    print('******************')
    Lasso_Regression(X_train,X_test,y_train,y_test)
    print('******************')
    Decision_Tree(X_train,X_test,y_train,y_test)
    print('******************')
    SVM(X_train,X_test,y_train,y_test)
    plt.legend()
    plt.show()

