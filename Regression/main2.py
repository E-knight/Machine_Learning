import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn import metrics
from IPython.display import Image
import pydotplus
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from pylab import mpl




def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

def append_results(model_name, model, results_df, y_test, pred):
    results_append_df = pd.DataFrame(data=[[model_name, *evaluate(y_test, pred) , cross_val_score(model, X, y, cv=10).mean()]], columns=columns)
    results_df = results_df.append(results_append_df, ignore_index = True)
    return results_df

def Linear_Regression(X_train,X_test,y_train,y_test,results_df):
    liner_regression=LinearRegression(normalize=True)
    liner_regression.fit(X_train,y_train)
    y_predict=liner_regression.predict(X_test)
    results_df=append_results("Linear Regression", LinearRegression(),results_df,y_test,y_predict)
    return results_df

def Ridge_Regression(X_train,X_test,y_train,y_test,results_df):
    ridge=Ridge()
    ridge.fit(X_train,y_train)
    y_predict=ridge.predict(X_test)
    results_df = append_results("Ridge Regression", Ridge(), results_df, y_test, y_predict)
    return results_df

def Lasso_Regression(X_train,X_test,y_train,y_test,results_df):
    lasso = Lasso()
    lasso.fit(X_train, y_train)
    y_predict = lasso.predict(X_test)
    results_df = append_results("Lasso Regression", Lasso(), results_df, y_test, y_predict)
    return results_df

def Decision_Tree(X_train,X_test,y_train,y_test,results_df):
    decesion_tree=DecisionTreeRegressor()
    decesion_tree.fit(X_train, y_train)
    # export_graphviz(decesion_tree,out_file='tree.dot',feature_names=['square','renovationCondition','subway','communityAverage'],class_names=['totalPrice'],filled=True, rounded=True,special_characters=True)
    # graph=pydotplus.graph_from_dot_file('tree.dot')
    # graph.write_pdf('tree.pdf')
    y_predict = decesion_tree.predict(X_test)
    results_df = append_results("Decision_Tree Regression", DecisionTreeRegressor(), results_df, y_test, y_predict)
    return results_df

def SVM(X_train,X_test,y_train,y_test,results_df):
    svr=SVR()
    svr.fit(X_train, y_train)
    y_predict = svr.predict(X_test)
    results_df = append_results("SVM Regression",SVR(), results_df, y_test, y_predict)
    return results_df


if __name__ == '__main__':
    house_data=pd.read_csv('new.csv',encoding='iso-8859-1')
    print(house_data.describe())
    print(house_data.corr())
    house_data=house_data[['square','renovationCondition','subway','communityAverage','totalPrice']]
    house_data=house_data.fillna(house_data.mean())
    y=np.array(house_data['totalPrice'])
    X=np.array(house_data.drop(['totalPrice'],axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    results_df = pd.DataFrame()
    columns = ["Model", "Cross Val Score", "MAE", "MSE", "RMSE", "R2"]
    results_df=Linear_Regression(X_train, X_test, y_train, y_test,results_df)
    print(results_df)
    results_df = Ridge_Regression(X_train, X_test, y_train, y_test, results_df)
    print(results_df)
    results_df = Lasso_Regression(X_train, X_test, y_train, y_test, results_df)
    print(results_df)
    results_df = Decision_Tree(X_train, X_test, y_train, y_test, results_df)
    print(results_df)
    results_df = SVM(X_train, X_test, y_train, y_test, results_df)
    print(results_df)

