#!/usr/bin/env python
from __future__ import print_function
# import argparse

import chainer
# import chainer.functions as F
# import chainer.links as L
import numpy as np

def main():
    # parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    # parser.add_argument('--batchsize', '-b', type=int, default=100,
    #                     help='Number of images in each mini-batch')
    # parser.add_argument('--epoch', '-e', type=int, default=20,
    #                     help='Number of sweeps over the dataset to train')
    # parser.add_argument('--gpu', '-g', type=int, default=-1,
    #                     help='GPU ID (negative value indicates CPU)')
    # parser.add_argument('--out', '-o', default='result',
    #                     help='Directory to output the result')
    # parser.add_argument('--resume', '-r', default='',
    #                     help='Resume the training from snapshot')
    # parser.add_argument('--unit', '-u', type=int, default=1000,
    #                     help='Number of units')
    # args = parser.parse_args()


    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()
    print("type(train) =", type(train))
    print("len(train)",len(train))
    print("len(train[0])", len(train[0]))
    print("len(train[0][0])", len(train[0][0]))
    print("train[0][1]", train[0][1])
    # train_img = train[:][ 0]
    # train_label = train[:][1]

    train_img = []
    train_label = []
    for num, data in enumerate(train):
        train_img.append(data[0])
        train_label.append(data[1])
        
    test_img = []
    test_label = []
    for num, data in enumerate(test):
        test_img.append(data[0])
        test_label.append(data[1])


    # print(train_img)
    # print(train_label)
    print("len(train_img)", len(train_img))
    print("len(train_label)", len(train_label))
    print("len(train_img[0])", len(train_img[0]))
    print("train_label[0]", train_label[0])

    print("len(test_img)", len(test_img))
    print("len(test_label)", len(test_label))
    print("len(test_img[0])", len(test_img[0]))
    print("test_label[0]", test_label[0])

    train_img_np = np.array(train_img, dtype=np.float32)
    train_label_np = np.array(train_label, dtype=np.float32)
    test_img_np = np.array(test_img, dtype=np.float32)
    test_label_np = np.array(test_label, dtype=np.float32)
    print("train_img_np.shape =", train_img_np.shape)
    print("train_label_np.shape =", train_label_np.shape)
    print("test_img_np.shape =", test_img_np.shape)
    print("test_label_np.shape =", test_label_np.shape)

    np.save('mnist_train_img.npy', train_img_np)
    np.save('mnist_train_label.npy', train_label_np)
    np.save('mnist_test_img.npy', test_img_np)
    np.save('mnist_test_label.npy', test_label_np)


if __name__ == '__main__':
    main()
