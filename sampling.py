#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import copy
import itertools
from random import shuffle
from torchvision import datasets, transforms


# np.random.seed(1)
def unique_index(L, f):
    return [i for (i, value) in enumerate(L) if value == f]


def mnist_iid(args, dataset, num_users, num_items):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    if len(dataset) == 60000:
        if args.iid == True:
            num_digits = int(num_items / 10)
            labels = dataset.train_labels.numpy()
            classes = np.unique(labels)
            classes_index = []
            for i in range(len(classes)):
                classes_index.append(unique_index(labels, classes[i]))
            for i in range(num_users):
                c = []
                for j in range(10):
                    b = (np.random.choice(classes_index[j], num_digits, \
                                          replace=False))
                    for m in range(num_digits):
                        c.append(b[m])
                # print(c)
                dict_users[i] = set(c)
        else:
            dict_users, all_idxs = {}, [i for i in range(len(dataset))]
            for i in range(num_users):
                dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
                if num_users * num_items <= len(dataset):
                    all_idxs = list(set(all_idxs) - dict_users[i])
    else:
        c = set(np.random.choice(all_idxs, num_items, replace=False))
        for i in range(num_users):
            dict_users[i] = copy.deepcopy(c)
            # print("\nDivide", len(all_idxs))                      
    return dict_users


def mnist_noniid(args, dataset, num_users, num_items):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    if args.dataset == 'mnist':
        # divide and assign
        num_digit_noniid = 4
        dict_users = {}
        labels = dataset.train_labels.numpy()
        classes = np.unique(labels)
        classes_index = []
        for i in range(len(classes)):
            classes_index.append(unique_index(labels, classes[i]))

        digit_ch_list = list(itertools.combinations(range(len(classes)), num_digit_noniid))
        digit_ch_idx = [i for i in range(len(digit_ch_list))]
        shuffle(digit_ch_idx)
        digit_ch_list_stor = copy.deepcopy(digit_ch_list)

        num_items_iid = int(np.ceil((1 - args.degree_noniid) * num_items / len(classes)))
        num_items_noniid = int(np.ceil(args.degree_noniid * num_items / num_digit_noniid))

        k = 0
        for i in digit_ch_idx:
            digit_ch_list[i] = copy.deepcopy(digit_ch_list_stor[k])
            k += 1

        for i in range(num_users):
            c = []
            for j in range(len(classes)):
                b = (np.random.choice(classes_index[j], int(num_items_iid), \
                                      replace=False))
                classes_index[j] = list(set(classes_index[j]) - set(b))
                for m in range(num_items_iid):
                    c.append(b[m])
            for j in list(digit_ch_list[i]):
                b = (np.random.choice(classes_index[j], int(num_items_noniid), \
                                      replace=False))
                classes_index[j] = list(set(classes_index[j]) - set(b))
                for m in range(num_items_noniid):
                    c.append(b[m])
            dict_users[i] = set(c)

    else:
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        c = set(np.random.choice(all_idxs, num_items, replace=False))
        for i in range(num_users):
            dict_users[i] = copy.deepcopy(c)
    #            if num_users*num_items <= len(dataset):
    #                all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def FashionMNIST_iid(args, dataset, num_users, num_items):
    """
    Sample I.I.D. client data from FashionMNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    if len(dataset) == 60000:
        if args.strict_iid == True:
            num_digits = int(num_items / 10)
            labels = dataset.train_labels.numpy()
            classes = np.unique(labels)
            classes_index = []
            for i in range(len(classes)):
                classes_index.append(unique_index(labels, classes[i]))
            for i in range(num_users):
                c = []
                for j in range(10):
                    b = (np.random.choice(classes_index[j], num_digits, \
                                          replace=False))
                    for m in range(num_digits):
                        c.append(b[m])
                # print(c)
                dict_users[i] = set(c)
        else:
            dict_users, all_idxs = {}, [i for i in range(len(dataset))]
            for i in range(num_users):
                dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
                if num_users * num_items <= len(dataset):
                    all_idxs = list(set(all_idxs) - dict_users[i])
    else:
        c = set(np.random.choice(all_idxs, num_items, replace=False))
        for i in range(num_users):
            dict_users[i] = copy.deepcopy(c)
            # print("\nDivide", len(all_idxs))
    return dict_users

def FashionMNIST_noniid(args, dataset, num_users, num_items):
    """
    Sample non-I.I.D client data from FashionMNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    if args.dataset == 'FashionMNIST':
        # divide and assign
        num_digit_noniid = 4
        dict_users = {}
        labels = dataset.train_labels.numpy()
        classes = np.unique(labels)
        classes_index = []
        for i in range(len(classes)):
            classes_index.append(unique_index(labels, classes[i]))

        digit_ch_list = list(itertools.combinations(range(len(classes)), num_digit_noniid))
        digit_ch_idx = [i for i in range(len(digit_ch_list))]
        shuffle(digit_ch_idx)
        digit_ch_list_stor = copy.deepcopy(digit_ch_list)

        num_items_iid = int(np.ceil((1 - args.degree_noniid) * num_items / len(classes)))
        num_items_noniid = int(np.ceil(args.degree_noniid * num_items / num_digit_noniid))

        k = 0
        for i in digit_ch_idx:
            digit_ch_list[i] = copy.deepcopy(digit_ch_list_stor[k])
            k += 1

        for i in range(num_users):
            c = []
            for j in range(len(classes)):
                b = (np.random.choice(classes_index[j], int(num_items_iid), \
                                      replace=False))
                classes_index[j] = list(set(classes_index[j]) - set(b))
                for m in range(num_items_iid):
                    c.append(b[m])
            for j in list(digit_ch_list[i]):
                b = (np.random.choice(classes_index[j], int(num_items_noniid), \
                                      replace=False))
                classes_index[j] = list(set(classes_index[j]) - set(b))
                for m in range(num_items_noniid):
                    c.append(b[m])
            dict_users[i] = set(c)

    else:
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        c = set(np.random.choice(all_idxs, num_items, replace=False))
        for i in range(num_users):
            dict_users[i] = copy.deepcopy(c)
    #            if num_users*num_items <= len(dataset):
    #                all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# Divide into 100 portions of total data. Allocate 2 random portions for each user
def cifar_noniid(dataset, num_users):
    num_shards, num_imgs = 100, 500
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.train_labels)  # .numpy()
    print(len(idxs))
    print(len(labels))
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 4, replace=False))
        # idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            # np.random.shuffle(dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/FashionMNIST/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)