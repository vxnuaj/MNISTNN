import numpy as np 
import pandas as pd 
import pickle

#Saving / Loading a model
def save_model(w1, b1, w2, b2, filename):
    with open(filename, 'wb') as f:
        pickle.dump((w1, b1, w2, b2), f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

#Loading Data
train_data = pd.read_csv("MNIST CSV archive/mnist_train.csv")
train_data = np.array(train_data)
m, n = train_data.shape
np.random.shuffle(train_data)


train_data = train_data[0:m].T
Y_train = train_data[0] 
X_train = train_data[1:n]
X_train = X_train / 255

#Initializing model params
def init_params():
    w1 = np.random.rand(32, 784) - .5
    b1 = np.zeros((32,1)) - .5
    w2 = np.random.rand(10, 32) - .5
    b2 = np.zeros((10,1)) - .5
    return w1, b1, w2, b2

# Defining activation functions
def relu(z): #ReLU
    return np.maximum(z,0) # np.maximum(z,0) is ReLU

def softmax(z): #Softmax
    return np.exp(z)/ sum(np.exp(z))

#Forward function
def forward(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1  # calculates weighted sum.
    a1 = relu(z1) # non-linearity per relu function
    z2 = w2.dot(a1) + b2 # calculates weighted sum
    a2 = softmax(z2) # non-linearity per softmax function, for final output
    return z1, a1, z2, a2

def relu_deriv(z):
    return z > 0

#One hot encoding of our data
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def cat_cross_entropy(one_hot_Y, a2 ):
    CCE = -np.sum(one_hot_Y * np.log(a2)) * 1/m
    return CCE


def back_prop(z1, a1, z2, a2, w1, w2, X, Y):
    one_hot_Y = one_hot(Y)
    dz2 = a2 - one_hot_Y
    dw2 = dz2.dot(a1.T) * 1/m
    db2 = np.sum(dz2) * 1/m
    dz1 = relu_deriv(z1) * w2.T.dot(dz2)
    dw1 = dz1.dot(X.T) * 1/m
    db1 = np.sum(dz1) * 1/m
    return dw1, db1, dw2, db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def get_pred(a2):
    pred = np.argmax(a2, axis = 0)
    return pred

def accuracy(predictions, Y):
    acc = np.sum(predictions == Y) / Y.size
    return acc

def gradient_descent(X, Y, alpha, iterations):
# loading model
    model_filename = 'models/mnistnn.pkl'
    try: 
        w1, b1, w2, b2 = load_model(model_filename)
        print("Loaded model from:", model_filename)
    except FileNotFoundError:
        print("Model not found. Initializing new model!")
        w1, b1, w2, b2 = init_params()
#gradient descent
    for i in range(iterations):
        z1, a1, z2, a2 = forward(w1, b1, w2, b2, X)
        dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w1, w2, X, Y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha )
        if i % 10 == 0:
            loss = cat_cross_entropy(one_hot(Y), a2)
            predictions = get_pred(a2)
            acc = accuracy(predictions, Y)
            print("Iteration: ", i)
            print("Accuracy:", acc)
            print("Loss:", loss)
            print(predictions, Y)
    return w1, b1, w2, b2

if __name__ == "__main__":
    w1, b1, w2, b2 = gradient_descent(X_train, Y_train ,.1, 500)
    save_model(w1, b1, w2, b2, 'models/mnistnn.pkl')