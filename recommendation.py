import pandas as pd
import pickle
import os
import copy
import math
import numpy as np
import csv
from numpy import seterr
import surprise

seterr(all='raise')


def load_data(path, pickle_path='data.pkl'):
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            return data
    else:
        data = dict()
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                user, num = line.split('|')
                user = int(user)
                num = int(num)
                tmp = dict()
                for _ in range(num):
                    line = f.readline()
                    item, score = line.split('  ')
                    item = int(item)
                    score = int(score)
                    tmp[item] = score
                data[user] = tmp
                line = f.readline()
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
    return data


def split(train_data, rate=0.2, seed=100):
    train_path = 'train_{}.pkl'.format(seed)
    test_path = 'test_{}.pkl'.format(seed)
    if os.path.exists(train_path) and os.path.exists(test_path):
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        return train_data, test_data
    np.random.seed(seed)
    test_data = dict()
    for user, items in train_data.items():
        item_ids = np.random.choice(list(items.keys()),
                                    int(len(items) * rate),
                                    replace=False)
        tmp = dict()
        for id in item_ids:
            tmp[id] = items[id]
            train_data[user].pop(id)
        test_data[user] = tmp
    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test_data, f)
    return train_data, test_data


def get_train_mean(train):
    sum = 0
    num = 0
    for _, items in train.items():
        for item in items.keys():
            sum += items[item]
            num += 1
    return sum / num


def get_train_csv(train):
    with open("train.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user', 'item', 'score'])
        for user, items in train.items():
            for item in items.keys():
                writer.writerow([user, item, items[item]])


def svd_train(train, test, user_num, item_num, epoch, lr, dim, lambda_):
    np.random.seed(1)
    bu = np.zeros(user_num)
    bi = np.zeros(item_num)
    pu = np.random.rand(user_num, dim)
    qi = np.random.rand(item_num, dim)
    mean = get_train_mean(train)
    for epoch_ in range(epoch):
        for user, items in train.items():
            for item in items.keys():
                score = items[item]
                dot = np.dot(pu[user], qi[item])
                error = score - mean - bu[user] - bi[item] - dot
                bu[user] += lr * (error - lambda_ * bu[user])
                bi[item] += lr * (error - lambda_ * bi[item])
                puu = pu[user]
                qii = qi[item]
                pu[user] += lr * (error * qii - lambda_ * puu)
                qi[item] += lr * (error * puu - lambda_ * qii)
        print('epoch {}: RMSE: {}'.format(epoch_,
                                          svd_eval(mean, test, pu, qi, bu,
                                                   bi)))
    d = dict()
    d['bu'] = bu
    d['bi'] = bi
    d['pu'] = pu
    d['qi'] = qi
    with open('svd.pkl', 'wb') as f:
        pickle.dump(d, f)


def svd_eval(mean, test, pu, qi, bu, bi):
    sum = 0
    num = 0
    for user, items in test.items():
        for item in items.keys():
            dot = np.dot(pu[user], qi[item])
            score = dot + bi[item] + bu[user] + mean
            sum += (score - items[item])**2
            num += 1
    return np.sqrt(sum / num)


def surprise_test(test, epoch):
    algo = surprise.SVD(n_epochs=epoch,
                        lr_all=0.005,
                        reg_all=0.02,
                        n_factors=5)
    reader = surprise.Reader(line_format='user item rating',
                             sep=',',
                             skip_lines=1,
                             rating_scale=(0, 100))
    train = surprise.Dataset.load_from_file('train.csv', reader=reader)
    train = train.build_full_trainset()
    algo.fit(train)
    sum = 0
    num = 0
    for user, items in test.items():
        for item in items.keys():
            score = algo.predict(str(user), str(item)).est
            sum += (score - items[item])**2
            num += 1
    print(epoch, np.sqrt(sum / num))


if __name__ == '__main__':
    data = load_data('./data-202205/train.txt')
    train = copy.deepcopy(data)
    train, test = split(train)
    # get_train_csv(train)
    # for i in range(10):
    #     surprise_test(test, i)
    user_num = 19835
    item_num = 624961
    # train = {0:{0:1, 1:2, 2:3, 3:4, 4:5},1:{0:1, 1:2, 2:3, 3:4, 4:5}, 2:{0:1, 1:2, 2:3, 3:4, 4:5}, 3:{0:1, 1:2, 2:3, 3:4, 4:5}, 4:{1:2, 2:3}}
    # test = {4:{0:1,3:4,4:5}}
    # user_num = 5
    # item_num = 5
    svd_train(train,
              test,
              user_num,
              item_num,
              epoch=20,
              lr=0.005,
              dim=5,
              lambda_=0.02)
