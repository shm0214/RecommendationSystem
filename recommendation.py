import pandas as pd
import pickle
import os
import copy
import math
import numpy as np
import csv
import time
from numpy import seterr
import surprise
import time
from neighbour import read_attribute,k_neighbour
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
    best = 100000
    bu = np.zeros(user_num)
    bi = np.zeros(item_num)
    pu = np.random.rand(user_num, dim)
    qi = np.random.rand(item_num, dim)
    mean = get_train_mean(train)
    start = time.time()
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
        rmse = svd_eval(mean, test, pu, qi, bu, bi)
        print('epoch {}: RMSE: {}'.format(epoch_, rmse))
        if rmse < best:
            best = rmse
            d = dict()
            d['bu'] = bu
            d['bi'] = bi
            d['pu'] = pu
            d['qi'] = qi
            with open('svd_{}.pkl'.format(dim), 'wb') as f:
                pickle.dump(d, f)
        Tend= time.time()
        print('训练总用时:%s毫秒' % ((Tend - Tstart)))


def svd_eval(mean, test, pu, qi, bu, bi):
    sum = 0
    num = 0
    for user, items in test.items():
        for item in items.keys():
            dot = np.dot(pu[user], qi[item])
            score = dot + bi[item] + bu[user] + mean
            sum += (score - items[item])**2
            num += 1
            if num>5000:
                np.sqrt(sum / num)
    return np.sqrt(sum / num)

def svd_per_eval(mean,user,item,pu,qi,bu,bi):
    dot = np.dot(pu[user],qi[item])
    score = dot + bi[item] + bu[user]+mean
    return score

def svd_test(train, test_path, pkl_path):
    mean = get_train_mean(train)
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    bu = d['bu']
    bi = d['bi']
    pu = d['pu']
    qi = d['qi']
    with open(test_path, 'r') as f1:
        with open('result.txt', 'w') as f2:
            line = f1.readline()
            while line:
                user, num = line.split('|')
                user = int(user)
                num = int(num)
                f2.write(line)
                for _ in range(num):
                    line = f1.readline()
                    item = int(line)
                    dot = np.dot(pu[user], qi[item])
                    score = dot + bi[item] + bu[user] + mean
                    f2.write("{}  {}\n".format(item, score))
                line = f1.readline()

def svd_eval_with_kneighbour(mean, test, pu, qi, bu, bi,attribute_dict):
    sum = 0
    num = 0
    print("n:user",len(test.items()))
    for user, items in test.items():
        for item in items.keys():
            dot = np.dot(pu[user], qi[item])
            diff_list = k_neighbour(attribute_dict,item)
            for diff in diff_list:
                neighbour_item = diff[0]
                dot += np.dot(pu[user],qi[neighbour_item])
            score = dot/(len(diff_list)+1)+ bi[item] + bu[user] + mean
            sum += (score - items[item])**2
            num += 1
            if num%100==0:
                print(num,time.time()-start)
            if num>5000:
                return np.sqrt(sum / num)
    return np.sqrt(sum / num)



def svdpp_train(train, test, user_num, item_num, epoch, lr, dim, lambda_):
    best = 100000
    bu = np.zeros(user_num)
    bi = np.zeros(item_num)
    pu = np.random.rand(user_num, dim)
    qi = np.random.rand(item_num, dim)
    yj = np.random.rand(item_num, dim)
    mean = get_train_mean(train)
    Tstart= time.time()
    for epoch_ in range(epoch):
        for user, items in train.items():
            print(user)
            sqrt_len = np.sqrt(len(items))
            for item in items.keys():
                # print(user, item)
                im_ratings = np.sum(yj[list(items.keys())], axis=0)
                im_ratings /= sqrt_len
                score = items[item]
                temp = pu[user] + im_ratings
                dot = np.dot(temp, qi[item])
                error = score - mean - bu[user] - bi[item] - dot
                bu[user] += lr * (error - lambda_ * bu[user])
                bi[item] += lr * (error - lambda_ * bi[item])
                puu = pu[user]
                qii = qi[item]
                pu[user] += lr * (error * qii - lambda_ * puu)
                qi[item] += lr * (error * temp - lambda_ * qii)
                yj[item] += lr * (error * qii / sqrt_len - lambda_ * yj[item])
        rmse = svdpp_eval(mean, test, pu, qi, bu, bi, yj)
        print('epoch {}: RMSE: {}'.format(epoch_, rmse))
        if rmse < best:
            best = rmse
            d = dict()
            d['bu'] = bu
            d['bi'] = bi
            d['pu'] = pu
            d['qi'] = qi
            d['yj'] = yj
            with open('svdpp_{}.pkl'.format(dim), 'wb') as f:
                pickle.dump(d, f)


def svdpp_eval(mean, test, pu, qi, bu, bi, yj):
    sum = 0
    num = 0
    for user, items in test.items():
        sqrt_len = np.sqrt(len(items))
        for item in items.keys():
            im_ratings = np.zeros(pu.shape[1])
            for i in items.keys():
                im_ratings += yj[i]
            im_ratings /= sqrt_len
            dot = np.dot(pu[user] + im_ratings, qi[item])
            score = dot + bi[item] + bu[user] + mean
            sum += (score - items[item])**2
            num += 1
    return np.sqrt(sum / num)

def surprise_test(test, epoch):
    algo = surprise.SVDpp(n_epochs=epoch,
                        lr_all=0.002,
                        reg_all=0.02,
                        n_factors=100)
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
    user_num = 19835
    item_num = 624961
    # for i in range(10):
    #     surprise_test(test, i)
    # svd_test(train, "./data-202205/test.txt", './svd_100.pkl')
    svd_train(train,
              test,
              user_num,
              item_num,
              epoch=20,
              lr=0.0005,
              dim=100,
              lambda_=0.02)
    # pkl_path = "svd_100.pkl"
    # mean = get_train_mean(train)
    # with open(pkl_path, 'rb') as f:
    #     d = pickle.load(f)
    # bu = d['bu']
    # bi = d['bi']
    # pu = d['pu']
    # qi = d['qi']
    # attribute_dict = read_attribute("data-202205/itemAttribute.txt")
    # print("eval with kneighbour...")
    # print(svd_eval_with_kneighbour(mean,test,pu,qi,bu,bi,attribute_dict))
    # print("eval...")
    # print(svd_eval(mean,test,pu,qi,bu,bi))

    # svdpp_train(train,
    #           test,
    #           user_num,
    #           item_num,
    #           epoch=20,
    #           lr=0.000005,
    #           dim=10,
    #           lambda_=0.02)
