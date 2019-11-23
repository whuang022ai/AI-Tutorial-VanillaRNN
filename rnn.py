# -*- coding: UTF-8 -*-
#
# Vanilla Recurrent Neural Network
#
# This is a break down / hard coding version of an RNN with three timesteps
# and solving the binary addition problem.
# W : weights input to hidden layer
# U : weights hidden to hidden layer
# V : weights hidden to output layer
# 

import numpy as np
import math as mh
import matplotlib.pyplot as plt


def bin_padding(bin, bin_size):
    pedding_size = bin_size - len(bin)
    for i in range(pedding_size):
        bin.append(0)
    return bin


def num2bin(num):
    bin = []
    while num != 0:
        bit = num % 2
        bin.append(int(bit))
        num = num // 2
    return bin


def bin_adding_gen():
    max_num = 3  # 11
    bin_size = 3  # 11 + 11 = 011
    input1 = np.random.randint(low=0, high=max_num)
    input2 = np.random.randint(low=0, high=max_num)
    answer = input1+input2
    a = bin_padding(num2bin(input1), bin_size)
    b = bin_padding(num2bin(input2), bin_size)
    c = bin_padding(num2bin(answer), bin_size)
    mat = []
    ans = []
    for i in range(bin_size):
        tmp = []
        tmp2 = []
        tmp.append(float(a[i]))
        tmp.append(float(b[i]))
        mat.append(tmp)
        tmp2.append(float(c[i]))
        ans.append(tmp2)
    mat_out = np.array(mat)
    ans_out = np.array(ans)
    return mat_out, ans_out


class RNN():
    def __init__(self, input_size, hidden_size, output_size , learing_rate):
        
        self.input_layer_size = input_size
        self.hidden_layer_size = hidden_size
        self.output_layer_size = output_size

        self.W = np.random.random(
            (self.hidden_layer_size, self.input_layer_size))
        self.U = np.random.random(
            (self.hidden_layer_size, self.hidden_layer_size))
        self.V = np.random.random(
            (self.output_layer_size, self.hidden_layer_size))
        self.sigmoid_vec = np.vectorize(self.sigmoid)
        self.sigmoid_div_vec = np.vectorize(self.sigmoid_div)
        self.lr = learing_rate

    def sigmoid(self, x):

        return 1 / (1 + mh.exp(-x))

    def sigmoid_div(self, y):

        return y * (1 - y)

    def double2bin(self, x):
        if x >= 0.5:
            return 1.0
        else:
            return 0.0

    def forward(self, X0, X1, X2):
        self.X0 = X0
        self.X1 = X1
        self.X2 = X2

        self.H0 = self.sigmoid_vec(np.dot(self.W, X0))
        self.O0 = self.sigmoid_vec(np.dot(self.V, self.H0))

        self.H1 = self.sigmoid_vec(np.dot(self.W, X1)+np.dot(self.U, self.H0))
        self.O1 = self.sigmoid_vec(np.dot(self.V, self.H1))

        self.H2 = self.sigmoid_vec(np.dot(self.W, X2)+np.dot(self.U, self.H1))
        self.O2 = self.sigmoid_vec(np.dot(self.V, self.H2))

        return (self.double2bin(self.O0[0]), self.double2bin(self.O1[0]), self.double2bin(self.O2[0]))

    def meansure_error_mse(self, D0, D1, D2):
        self.D0 = D0
        self.D1 = D1
        self.D2 = D2
        E0 = np.square(self.D0 - self.O0).mean()
        E1 = np.square(self.D1 - self.O1).mean()
        E2 = np.square(self.D2 - self.O2).mean()
        ESum = E0 + E1 + E2
        Eavg = ESum / 3
        return Eavg

    def backward(self):
        self.dO2 = -(self.D2-self.O2)*self.sigmoid_div_vec(self.O2)
        self.dH2 = np.dot(self.V.T, self.dO2)*self.sigmoid_div_vec(self.H2)

        self.dO1 = -(self.D1-self.O1)*self.sigmoid_div_vec(self.O1)
        self.dH1 = (np.dot(self.V.T, self.dO1)+np.dot(self.U.T,
                                                      self.dH2))*self.sigmoid_div_vec(self.H1)

        self.dO0 = -(self.D0-self.O0)*self.sigmoid_div_vec(self.O0)
        self.dH0 = (np.dot(self.V.T, self.dO0)+np.dot(self.U.T,
                                                      self.dH1))*self.sigmoid_div_vec(self.H0)

    def update_value_calculation(self):
        # dV
        tmp_H0 = np.reshape(self.H0, (-1, len(self.H0)))
        tmp_dO0 = np.reshape(self.dO0, (-1, len(self.dO0)))
        tmp_dV0 = np.dot(tmp_dO0, tmp_H0)
        #
        tmp_H1 = np.reshape(self.H1, (-1, len(self.H1)))
        tmp_dO1 = np.reshape(self.dO1, (-1, len(self.dO1)))
        tmp_dV1 = np.dot(tmp_dO1, tmp_H1)
        #
        tmp_H2 = np.reshape(self.H2, (-1, len(self.H2)))
        tmp_dO2 = np.reshape(self.dO2, (-1, len(self.dO2)))
        tmp_dV2 = np.dot(tmp_dO2, tmp_H2)
        self.dV = self.lr*(tmp_dV0[0]+tmp_dV1[0]+tmp_dV2[0])
        # dW
        tmp_dH0 = np.reshape(self.dH0, (len(self.dH0), -1))
        tmp_X0 = np.reshape(self.X0, (-1, len(self.X0)))
        tmp_dW0 = np.dot(tmp_dH0, tmp_X0)
        #
        tmp_dH1 = np.reshape(self.dH1, (len(self.dH1), -1))
        tmp_X1 = np.reshape(self.X1, (-1, len(self.X1)))
        tmp_dW1 = np.dot(tmp_dH1, tmp_X1)
        #
        tmp_dH2 = np.reshape(self.dH2, (len(self.dH2), -1))
        tmp_X2 = np.reshape(self.X2, (-1, len(self.X2)))
        tmp_dW2 = np.dot(tmp_dH2, tmp_X2)
        self.dW = self.lr*(tmp_dW0+tmp_dW1+tmp_dW2)
        # dU
        tmp_dH2 = np.reshape(self.dH2, (len(self.dH2), -1))
        tmp_H1 = np.reshape(self.H1, (-1, len(self.H1)))
        tmp_dU2 = np.dot(tmp_dH2, tmp_H1)
        #
        tmp_dH1 = np.reshape(self.dH1, (len(self.dH1), -1))
        tmp_H0 = np.reshape(self.H0, (-1, len(self.H0)))
        tmp_dU1 = np.dot(tmp_dH1, tmp_H0)
        self.dU = self.lr*(tmp_dU2+tmp_dU1)

    def update(self):
        self.V -= self.dV
        self.W -= self.dW
        self.U -= self.dU


if __name__ == "__main__":

    rnn = RNN(2,8,1,0.8)
    dataset = []
    f = open("output.txt", "w")
    plt.figure('Recurrent Neural Network MSE Error Monitor')
    plt.axis([0, 500, 0, 0.0001])
    plt.draw()
    plt.ion()
    plt.autoscale(enable=True, axis='both')

    for i in range(100):
        x, d = bin_adding_gen()
        dataset.append((x, d))
    # print(dataset)
    for j in range(800):
        amse = 0.0
        for i in range(len(dataset)):
            x, d = dataset[i]
            X0 = np.array(x[0])
            X1 = np.array(x[1])
            X2 = np.array(x[2])
            D0 = np.array(d[0])
            D1 = np.array(d[1])
            D2 = np.array(d[2])
            out = rnn.forward(X0, X1, X2)
            # print(out)
            D = (D0[0], D1[0], D2[0])
            f.write(str(D)+"\n")
            f.write(str(out)+"\n")
            f.write("---\n")
            mse = rnn.meansure_error_mse(D0, D1, D2)
            # print(mse)
            amse += mse
            rnn.backward()
            rnn.update_value_calculation()
            rnn.update()
        amse /= len(dataset)
        print(amse)
        plt.plot(j, amse, 'b*-', label="MSE")
        plt.pause(0.01)
        f.write(str(amse))
        f.write("---\n")
    f.close()
    plt.ioff()
    plt.show(block=True)
