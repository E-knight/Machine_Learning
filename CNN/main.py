from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((60000,28,28,1))/255.0
    X_test = X_test.reshape((10000,28,28,1))/255.0
    # y_train=to_categorical(y_train,10)
    # y_test=to_categorical(y_test,10)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
    model=Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, batch_size=60, epochs=5)
    test_loss,test_acc=model.evaluate(X_test,y_test)
    print('交叉熵损失：',test_loss)
    print('交叉熵准确率：',test_acc)






