## TO-DO | 1. Rather than printing out the digit out the prediction and true digit in the terminal, print it out alongside the matplotlib plot

import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt

from mnistmodel import load_model, forward, get_pred

test_data = pd.read_csv("MNIST CSV archive/mnist_test.csv")
test_data = np.array(test_data)
m, n = test_data.shape # 10000, 784

test_data = test_data[0:m].T # Now in form where each column == individual digits | 784, 10000
Y_test = test_data[0] # Taking the entire first row, holding labels of all 10k digits | 10000,
X_test = test_data[1:n] # Taking the rows from index 1 to the 10000, holding all the data | 9999, 784
X_test = X_test / 255

nn = 'models/mnistnn.pkl'
w1, b1, w2, b2 = load_model(nn)

def make_pred(w1, b1, w2, b2, X, i):
    _, _, _, a2 = forward(w1, b1, w2, b2, X)
    prediction = get_pred(a2)
    prediction = prediction[i]
    true_digit = Y_test[i]
    print(f'Prediction: {prediction}')
    print(f"True Digit: {true_digit}")

    digit = X_test[:, i]
    digit = digit.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(digit)
    plt.show()
    return


prediction = make_pred(w1, b1, w2, b2, X_test, 0)