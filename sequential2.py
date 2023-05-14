import numpy as np
import time
from keras.datasets import mnist
from keras.utils import np_utils
from convolutional_lib import *
from convolutional_nolib import * 
from sigmoid import Sigmoid
from maxpool2d import MaxPoolingLayer
from reshape import Reshape
from fclayer import FCLayer
from softmax import Softmax
from network import train, predict
from loss import *
def preprocess_data(x, y, limit): 
    # Lấy index của các mẫu có nhãn 0, 1, 2 và giới hạn số lượng mẫu cho mỗi nhãn bằng tham số limit
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    two_index = np.where(y == 2)[0][:limit]
    
    # Ghép các index lại thành một array
    all_indices = np.hstack((zero_index, one_index, two_index))
    
    # Xáo trộn thứ tự các index
    np.random.shuffle(all_indices)
    
    # Lấy các mẫu tương ứng với các index đã được xáo trộn
    x = x[all_indices]
    y = y[all_indices]
    
    # Chuẩn hóa dữ liệu x về dạng 4D tensor với kích thước (samples, channels, rows, cols)
    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    
    # One-hot encode nhãn y và chuyển về dạng tensor có kích thước (samples, classes, 1)
    y = np.eye(3)[y].reshape(y.shape[0], 3, 1)
    
    return x, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network1 = [
    CNNLayer((1, 28, 28), 3, 5),
    Sigmoid(),
    MaxPoolingLayer((5,26,26),2, 2),
    CNNLayer((5,13,13), 3, 5),
    Sigmoid(),
    Reshape((5, 11, 11), (5 * 11 * 11, 1)),
    FCLayer(5 * 11 * 11, 100),
    Sigmoid(),
    FCLayer(100, 3),
    Softmax()
]
start = time.time()
# train with CNN with rewritten correlate function
train(
    network1,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)
end = time.time()
print(f'Processing time: {end - start} s')

# test with network1
true_label = 0
for x, y in zip(x_test, y_test):
    output = predict(network1, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    if np.argmax(output) == np.argmax(y):
      true_label += 1

# Score:
print(f"Accuracy: {true_label * 100/len(x_test)}% on predict true: {true_label} vs true: {len(x_test)}")