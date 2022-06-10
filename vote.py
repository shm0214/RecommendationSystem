from recommendation import *
from collections import defaultdict
import scipy.spatial as spt

def get_knn_mean():
    return 50

def get_svd_score(svd,user,item):
    dot = np.dot(svd['pu'][user], svd['qi'][item])
    score = dot + svd['bi'][item] + svd['bu'][user] + d['mean']
    return score

def train_vote(train_set,svd,model,lr):
    for user, items in train_set.items():
        for item in items.keys():
            score = items[item]
            svd_score=get_svd_score(svd,user,item)
            knn_score=get_knn_mean(user,item)
            w=model[user]
            score_diff=score-(w*knn_score+(1-w)*svd_score)
            model[user] += lr*score_diff/(knn_score-svd_score)

def eval_vote(model,svd,test_set):
    sum = 0
    num = 0
    for user, items in test_set.items():
        for item in items.keys():
            svd_score=get_svd_score(svd,user,item)
            knn_score=get_knn_mean(user,item)
            w=model[user]
            predict=w*knn_score+(1-w)*svd_score
            sum +=(items[item]- predict)**2
            num+=1
    return np.sqrt(sum / num)

if __name__ == '__main__':
    data = load_data('./data-202205/train.txt')
    train = copy.deepcopy(data)
    train, test = split(train)

    with open('svd_100.pkl','rb') as f:
        d=pickle.load(f)
    d['mean']=get_train_mean(train)

    with open('attr.pkl','rb') as f:
        item_attr=pickle.load(f)

    vote_weight = defaultdict(float)
    for epoch in range(20):
        train_vote(train,d,vote_weight,0.005)
        rmse=eval_vote(vote_weight,d,test)
        print('epoch {}: RMSE: {}'.format(epoch, rmse))

    



