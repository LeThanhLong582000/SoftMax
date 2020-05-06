import numpy as np
import random
from scipy import sparse

N = 800
C = 2

TrainSet = []
TestSet = []
LabelTrain = []
LabelTest = []

pathtrain = 'E:\SoftMax\Train\\'
pathtest = 'E:\SoftMax\Test\\'

#Load TrainSet
print('Loading TrainSet')
listtrain = [_ for _ in range(1, N + 1)]
random.shuffle(listtrain) # Lay ngau nhien thu tu cac anh
for i in listtrain:
    path = ''
    if i <= int(N / 2):
        path = pathtrain + 'hanquoc\\' + str(i) + '.txt'
        LabelTrain.append(0)
    else:
        path = pathtrain + 'ngoclinh\\' + str(i - int(N / 2)) + '.txt'
        LabelTrain.append(1)
    File = open(path, 'r')
    TrainSet.append(np.array([1] + list(map(float, File.read().split()))))
    File.close()
TrainSet = np.array(TrainSet)
print('So phan tu trong Tep Train: ' ,len(TrainSet))

#Load TestSet
print('Loading Test Set')
listtest = [_ for _ in range(1, int(N / 4) + 1)] 
random.shuffle(listtest) # Lay ngau nhien thu tu cac anh
for i in listtest:
    path = ''
    if i <= int(N / 8):
        path = pathtest + 'hanquoc\\' + str(i) + '.txt'
        LabelTest.append(0)
    else:
        path = pathtest + 'ngoclinh\\' + str(i - int(N / 8)) + '.txt'
        LabelTest.append(1)
    File = open(path, 'r')
    TestSet.append(np.array([1] + list(map(float, File.read().split()))))
    File.close()
TestSet = np.array(TestSet)
print('So phan tu trong Tep Test: ' ,len(TestSet))

# SoftMax Function
def SoftMax(Z):
    e_Z = np.exp(Z)
    A = e_Z / (e_Z.sum(axis = 0))
    return A

# One-hot encoding
def convert_labels(y, C = C):
    Y = sparse.coo_matrix((np.ones_like(y), 
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

#predict
def Predict(W, X):
    A = SoftMax(W.T.dot(X.T))
    return np.argmax(A, axis= 0)

# Training
def Training(X, y, W_init, learning_rate, epochs = 200):
    print('Training')
    W = W_init    
    C = W_init.shape[1]
    Y = convert_labels(y, C)
    it = 0
    N = X.shape[0]
    d = X.shape[1]
    for epoch in range(1, epochs + 1):
        mix_id = np.random.permutation(N)
        for i in mix_id:
            x_i = X[i, :].reshape(d, -1)
            y_i = Y[ :, i].reshape(C, 1)
            a_i = SoftMax(W.T.dot(x_i))
            W = W + learning_rate * x_i.dot((y_i - a_i).T)
        if epoch % 10 == 0:
                predict_class = Predict(W, TestSet)
                print('Model Accuracy After {} epoch: '.format(epoch), 100 * np.mean(predict_class == LabelTest))
    return W

learning_rate = 0.05
d = TrainSet.shape[1]
W_init = np.random.rand(d, C)

W = Training(TrainSet, LabelTrain, W_init, learning_rate, 200)

predict_class = Predict(W, TestSet)
print('Model Accuracy: ', 100 * np.mean(predict_class == LabelTest))