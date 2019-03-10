#!/usr/bin/env python

"""
没有对矩阵进行求逆的DetNet,属于在作者的基础上取消求逆的基础版本
"""
import time as tm

import numpy as np
import tensorflow as tf

###start here
"""
Parameters
K - size of x
N - size of y
snrdb_low - the lower bound of noise db used during training
snr_high - the higher bound of noise db used during training
L - number of layers in DetNet
v_size = size of auxiliary variable at each layer
hl_size - size of hidden layer at each DetNet layer (the dimention the layers input are increased to
startingLearningRate - the initial step size of the gradient descent algorithm
decay_factor & decay_step_size - each decay_step_size steps the learning rate decay by decay_factor
train_iter - number of train iterations
train_batch_size - batch size during training phase
test_iter - number of test iterations
test_batch_size  - batch size during testing phase
LOG_LOSS - equal 1 if loss of each layer should be sumed in proportion to the layer depth, otherwise all losses have the same weight 
res_alpha- the proportion of the previuos layer output to be added to the current layers output (view ResNet article)
snrdb_low_test & snrdb_high_test & num_snr - when testing, num_snr different SNR values will be tested, uniformly spread between snrdb_low_test and snrdb_high_test 
"""

sess = tf.InteractiveSession()

# parameters
K = 30
N = 60
snrdb_low = 7.0
snrdb_high = 14.0
snr_low = 10.0 ** (snrdb_low / 10.0)
snr_high = 10.0 ** (snrdb_high / 10.0)
L = 90
hl_size = 8 * K * N
startingLearningRate = 0.0001
decay_factor = 0.97
decay_step_size = 1000
train_iter = 500
train_batch_size = 5000
test_iter = 200
test_batch_size = 1000
LOG_LOSS = 1  # 是否采用LOG方式
res_alpha = 0.9
num_snr = 6
snrdb_low_test = snrdb_low + 1
snrdb_high_test = snrdb_high - 1

"""Data generation for train and test phases
In this example, both functions are the same.
This duplication is in order to easily allow testing cases where the test is over different distributions of data than in the training phase.
e.g. training over gaussian i.i.d. channels and testing over a specific constant channel.
currently both test and train are over i.i.d gaussian channel.
"""


def generate_data_iid_test(B, K, N, snr_low, snr_high):
    H_tmp = 2 * (np.random.rand(N, K) - 0.5)
    H_ = np.zeros([B, N, K])
    for i in range(B):
        H_[i, :, :] = H_tmp
    x_ = np.sign(np.random.rand(B, K) - 0.5)
    y_ = np.zeros([B, N])
    w = np.random.randn(B, N)
    SNR_ = np.zeros([B])
    xy_t = np.zeros([B, K * N])
    h_t = np.zeros([B, K * N])
    xx_t = np.zeros([B, K, K])
    for i in range(B):
        SNR = np.random.uniform(low=snr_low, high=snr_high)
        H = H_[i, :, :]

        tmp_snr = (H.T.dot(H)).trace() / K
        H = H / np.sqrt(tmp_snr) * np.sqrt(SNR)
        y_[i, :] = (H.dot(x_[i, :]) + w[i, :])

        temp = np.expand_dims(x_[i, :], 1).dot(np.expand_dims(y_[i, :], 0))

        xy_t[i, :] = np.reshape(temp, N * K)
        h_t[i, :] = np.reshape(H_[i, :, :], K * N)
        xx_t[i, :, :] = x_[i, :].T.dot(x_[i, :])
        SNR_[i] = SNR
    return xy_t, h_t, xx_t, SNR_


def generate_data_train(B, K, N, snr_low, snr_high):
    # H_ = 2 * (np.random.rand(B, N, K) - 0.5)
    H_tmp = 2 * (np.random.rand(N, K) - 0.5)
    H_ = np.zeros([B, N, K])
    for i in range(B):
        H_[i, :, :] = H_tmp
    x_ = np.sign(np.random.rand(B, K) - 0.5)
    y_ = np.zeros([B, N])
    w = np.random.randn(B, N)
    SNR_ = np.zeros([B])
    xy_t = np.zeros([B, K * N])
    h_t = np.zeros([B, K * N])
    xx_t = np.zeros([B, K, K])
    for i in range(B):
        SNR = np.random.uniform(low=snr_low, high=snr_high)
        H = H_[i, :, :]

        tmp_snr = (H.T.dot(H)).trace() / K
        H = H / np.sqrt(tmp_snr) * np.sqrt(SNR)
        y_[i, :] = (H.dot(x_[i, :]) + w[i, :])

        temp = np.expand_dims(x_[i, :], 1).dot(np.expand_dims(y_[i, :], 0))

        xy_t[i, :] = np.reshape(temp, N * K)
        h_t[i, :] = np.reshape(H_[i, :, :], K * N)
        xx_t[i, :, :] = x_[i, :].T.dot(x_[i, :])
        SNR_[i] = SNR
    return xy_t, h_t, xx_t, SNR_


def piecewise_linear_soft_sign(x):
    t = tf.Variable(0.1)
    y = -1 + tf.nn.relu(x + t) / (tf.abs(t) + 0.00001) - tf.nn.relu(x - t) / (tf.abs(t) + 0.00001)
    return y


def affine_layer(x, input_size, output_size, Layer_num):
    W = tf.Variable(tf.random_normal([input_size, output_size], stddev=0.01))
    w = tf.Variable(tf.random_normal([1, output_size], stddev=0.01))
    y = tf.matmul(x, W) + w
    return y


def relu_layer(x, input_size, output_size, Layer_num):
    y = tf.nn.relu(affine_layer(x, input_size, output_size, Layer_num))
    return y


def sign_layer(x, input_size, output_size, Layer_num):
    # y = 2 * (tf.nn.softmax(affine_layer(x, input_size, output_size, Layer_num)) - 0.5)
    # y = piecewise_linear_soft_sign(affine_layer(x, input_size, output_size, Layer_num))
    y = affine_layer(x, input_size, output_size, Layer_num)
    # y = tf.nn.tanh(affine_layer(x, input_size, output_size, Layer_num))
    return y

# tensorflow placeholders, the input given to the model in order to train and test the network
XY_t = tf.placeholder(tf.float32, shape=[None, K * N])
H_t = tf.placeholder(tf.float32, shape=[None, K * N])
XX_t = tf.placeholder(tf.float32, shape=[None, K, K])

batch_size = tf.shape(XY_t)[0]

S = []
S.append(tf.zeros([batch_size, K * N]))
LOSS = []  # 每一层的代价函数
LOSS.append(tf.zeros([]))
MSE = []
MSE.append(tf.zeros([]))
# The architecture of DetNet
for i in range(1, L):
    temp1 = tf.matmul(XX_t, tf.reshape(S[-1], [-1, K, N]))
    temp1 = tf.reshape(temp1, [-1, K * N])
    Z = tf.concat([XY_t, S[-1], temp1], 1)
    ZZ = relu_layer(Z, 3 * K * N, hl_size, 'relu' + str(i))
    S.append(sign_layer(ZZ, hl_size, K * N, 'sign' + str(i)))
    S[i] = (1 - res_alpha) * S[i] + res_alpha * S[i - 1]
    if LOG_LOSS == 1:
        LOSS.append(np.log(i) * tf.reduce_mean(tf.reduce_mean(tf.square(H_t - S[-1]), 1)))
    else:
        LOSS.append(tf.reduce_mean(tf.reduce_sum(tf.square(H_t - S[-1]), 1)))
    MSE.append(tf.reduce_mean(tf.reduce_mean(tf.square(H_t - S[-1]), 1)))

TOTAL_LOSS = tf.add_n(LOSS)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, decay_step_size, decay_factor,
                                           staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(TOTAL_LOSS)
init_op = tf.global_variables_initializer()

sess.run(init_op)
# Training DetNet
for i in range(train_iter):  # num of train iter
    xy_t, h_t, xx_t, SNR = generate_data_train(train_batch_size, K, N, snr_low, snr_high)
    train_step.run(feed_dict={XY_t: xy_t, H_t: h_t, XX_t: xx_t})
    if i % 100 == 0:
        xy_t, h_t, xx_t, SNR = generate_data_iid_test(train_batch_size, K, N, snr_low, snr_high)
        results = sess.run([MSE[L - 1]], {XY_t: xy_t, H_t: h_t, XX_t: xx_t})
        print_string = [i] + results
        print(' '.join('%s' % x for x in print_string))

# Testing the trained model
snrdb_list = np.linspace(snrdb_low_test, snrdb_high_test, num_snr)
snr_list = 10.0 ** (snrdb_list / 10.0)
mses = np.zeros((1, num_snr))
times = np.zeros((1, num_snr))
tmp_mse = np.zeros((1, test_iter))
tmp_times = np.zeros((1, test_iter))
for j in range(num_snr):
    print('snr: ', snrdb_list[j])
    for jj in range(test_iter):
        xy_t, h_t, xx_t, SNR = generate_data_iid_test(test_batch_size, K, N, snr_list[j], snr_list[j])
        tic = tm.time()
        tmp_mse[:, jj] = np.array(sess.run(MSE[L - 1], {XY_t: xy_t, H_t: h_t, XX_t: xx_t}))
        toc = tm.time()
        tmp_times[0][jj] = toc - tic
    times[0][j] = np.mean(tmp_times[0]) / test_batch_size
    mses[0][j] = np.mean(tmp_mse, 1)

print('snrdb_list')
print(snrdb_list)
print('bers')
print(mses)
print('times')
print(times)
