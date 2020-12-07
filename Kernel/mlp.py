import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop

if __name__ == '__main__':
    (X_train,y_train),(X_test,y_test)=mnist.load_data()
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    X_train=X_train.reshape(-1,784)
    X_test=X_test.reshape(-1,784)
    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    X_train=X_train/127
    X_test=X_test/127
    y_train=to_categorical(y_train,10)
    y_test=to_categorical(y_test,10)
    print(y_train.shape,y_test.shape)
    model=Sequential()
    model.add(Dense(784,activation='relu',input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(784,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation='softmax'))
    # model.summary()
    model.compile(loss='mse',optimizer=RMSprop(),metrics=['accuracy'])
    model.fit(X_train,y_train,batch_size=60,epochs=2,verbose=1,validation_data=(X_test,y_test))
    score=model.evaluate(X_test,y_test,verbose=1)
    print('测试数据损失值(均方误差):',score[0])
    print('测试数据准确率(均方误差):',score[1])
    model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=60, epochs=2, verbose=1, validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=1)
    print('测试数据损失值(交叉熵):', score[0])
    print('测试数据准确率(交叉熵):', score[1])

    