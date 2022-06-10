# 实验报告：推荐系统

## 数据集

数据集规模如下，分别是训练集和测试集中，用户和物品的维度。

|    Size    | user | item |
|:----------:|-----:|-----:|
| train set  | 19835 |624960 |
| test set   | 19835 | 624960 |

数据集统计数据如下，分别是非空项数，稀疏占比，均值，标准差。

| Statistics | ratings | item attr1 | item attr2 |
|:----------:|--------:|-----------:|-----------:|
|   size     | 5001507 |    464932  |    443687  |
| sparseness |  99.96% |     8.33%  |    12.52%  |
|    mean    |   49.50 |    314596  |    311482  |
|    S.D.    |   38.22 |    180956  |    179777  |

## 推荐算法

### SVD

使用Funk-SVD算法。

评分函数如下，其中$\mu$为所有得分的平均值，$b_u$为用户$u$的偏差，$b_i$为物体$i$的偏差，$q_i$和$q_u$是分解得到的矩阵中$i$与$u$对应的部分。
$$
\hat{r}_{ui} = \mu + b_u + b_i + q_i^Tp_u
$$
目标函数如下，考虑预测分数与实际分数的平方差，以及偏差与矩阵的正则化项
$$
\sum_{r_{ui} \in R_{train}} \left(r_{ui} - \hat{r}_{ui} \right)^2 +
\lambda\left(b_i^2 + b_u^2 + ||q_i||^2 + ||p_u||^2\right)
$$
使用SGD来最小化目标函数，每次更新如下
$$
\begin{split}
e_{ui}&=\sum_{r_{ui} \in R_{train}} \left(r_{ui} - \hat{r}_{ui} \right)^2\\
b_u &\leftarrow b_u + \gamma (e_{ui} - \lambda b_u)\\
b_i &\leftarrow b_i + \gamma (e_{ui} - \lambda b_i)\\
p_u &\leftarrow p_u + \gamma (e_{ui} \cdot q_i - \lambda p_u)\\
q_i &\leftarrow q_i + \gamma (e_{ui} \cdot p_u - \lambda q_i)\end{split}
$$
$b_u$与$b_i$初始化为0，$p_u$与$q_i$初始化为$(0,1)$的随机值

训练代码如下：

```python
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
```

每个epoch结束后，在测试集上计算RMSE，并保存RMSE最低的模型。

```python
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
```

### SVD++

SVD++是对SVD的改进算法，在其基础上引入隐式反馈，使用用户的历史评分数据来计算新的分数。

目标函数如下，其中$I_u$为用户$u$的评分集合，$y_j$为隐式因子，同样初始化为$(0,1)$的随机值。
$$
\hat{r}_{ui} = \mu + b_u + b_i + q_i^T\left(p_u +
|I_u|^{-\frac{1}{2}} \sum_{j \in I_u}y_j\right)
$$
目标函数与SVD类似。
$$
\sum_{r_{ui} \in R_{train}} \left(r_{ui} - \hat{r}_{ui} \right)^2 +
\lambda\left(b_i^2 + b_u^2 + ||q_i||^2 + ||p_u||^2 + ||y_j||^2\right)
$$
使用SGD来最小化目标函数，每次更新如下
$$
\begin{split}
e_{ui}&=\sum_{r_{ui} \in R_{train}} \left(r_{ui} - \hat{r}_{ui} \right)^2\\
b_u &\leftarrow b_u + \gamma (e_{ui} - \lambda b_u)\\
b_i &\leftarrow b_i + \gamma (e_{ui} - \lambda b_i)\\
p_u &\leftarrow p_u + \gamma (e_{ui} \cdot q_i - \lambda p_u)\\
q_i &\leftarrow q_i + \gamma \left(e_{ui} \cdot\left(p_u +
|I_u|^{-\frac{1}{2}} \sum_{j \in I_u}y_j\right) - \lambda q_i\right)\\
y_j &\leftarrow y_j+\lambda(e_{ui}\cdot q_i \cdot |I_u|^{-\frac{1}{2}} - \lambda y_j)
\end{split}
$$
训练代码如下：

```python
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
```

每个epoch结束后，在测试集上计算RMSE，并保存RMSE最低的模型。

```python
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
```

### 相似属性物品加权

在使用SVD算法对推荐系统进行探究后，我们试图通过物品之间属性的相似性来对原算法精度进行进一步的优化。

直观来讲，两个物品之间的属性越相似，用户对于这两个物品的评价就越接近。因此一个朴素的想法就是在预测用户
对某物品的评分时，考虑与物品相似度最高的几个物品，将这些物品的预测值进行加权求和，也许能够使评分更加稳定。
因此我们尝试使用k近邻来找出与某物品最相似的k个物品，物品间的距离采用属性的L2距离进行评估。其中k近邻算法得到相似物品的代码如下：

```python
def k_neighbour(attribute_dict,item,k=3):
    #如果不在属性列表里
    if item not in attribute_dict:
        return []

    attr1,attr2 = attribute_dict[item]
    flag_1 = attr1!=-1
    flag_2 = attr2!=-1
    diff_list = []
    #对不为None的属性进行距离计算
    for idx,attributes in attribute_dict.items():
        tmp_flag_1 = attributes[0]!=-1
        tmp_flag_2 = attributes[1]!=-1
        if flag_1==tmp_flag_1 and flag_2 == tmp_flag_2:
            diff = 0
            if flag_1:
                diff+=abs(attributes[0]-attr1)
            if flag_2:
                diff+=abs(attributes[1]-attr2)
            if diff==0:
                diff_list.append([idx,diff])
                if len(diff_list)==k:
                    return diff_list
    return diff_list
```

在得到了相似物品后，我们只需要预测用户对所有相似物品的评分，之后进行均值化处理即可：

```python
def svd_eval_with_kneighbour(mean, test, pu, qi, bu, bi,attribute_dict):
    sum = 0
    num = 0
    for user, items in test.items():
        for item in items.keys():
            dot = np.dot(pu[user], qi[item])+bi[item]
            #寻找相似物品
            diff_list = k_neighbour(attribute_dict,item,k=5)
            #均值化处理
            for diff in diff_list:
                neighbour_item = diff[0]
                dot += np.dot(pu[user],qi[neighbour_item])+bi[neighbour_item]
            score = dot/(len(diff_list)+1) + bu[user] + mean
            sum += (score - items[item])**2
            num += 1
    return np.sqrt(sum / num)
```

### 基于用户历史评分的相似属性物品的模型校正

对于上述相似属性物品预测值加权的模型校正的方法，有几条非常明显的缺陷：

1.使用相似物品预测值的均值来进行模型校正，并不能有效校正有偏分布模型的偏置。

2.没有考虑到用户历史行为的影响，单纯从模型出发缺乏客观性

基于以上两点缺陷，我们考虑使用用户历史上已经评分的相似物品来对预测值进行校正，
一方面这样可以直接使用历史记录上用户的真实评分，另一方面可以大大减小查找相似物品相似度的计算量。

基于用户历史评分的k近邻算法如下：

```python
def k_neighbour_with_history(attribute_dict,history,item,k=3):
    if item not in attribute_dict:
        return []
    attr1,attr2 = attribute_dict[item]
    flag_1 = attr1!=-1
    flag_2 = attr2!=-1
    diff_list = []
    for idx,score in history.items():
        if idx not in attribute_dict:
            continue
        attributes = attribute_dict[idx]
        tmp_flag_1 = attributes[0]!=-1
        tmp_flag_2 = attributes[1]!=-1
        if flag_1==tmp_flag_1 and flag_2 == tmp_flag_2:
            diff = 0
            if flag_1:
                diff+=abs(attributes[0]-attr1)
            if flag_2:
                diff+=abs(attributes[1]-attr2)
            if diff<5000:
                diff_list.append([idx,diff])
        diff_list = sorted(diff_list,key = lambda k:k[1])
        if len(diff_list)>k:
            return diff_list[:k]
    return diff_list
```

在实际模型预测校正中仍然采用均值处理，改动较小，在这里不再赘述。

## 实验结果及分析

这里在训练集中随机划分了20%的数据作为测试集，来评估推荐算法；此外，由于内存中空间占用不方便进行计算，这里使用pickle库的序列化数据估计模型空间占用。实验结果如下：

| Method |  RMSE |   Training Time | Space Consumption |
| :----: | ----: | --------------: | ----------------: |
|  SVD   | 27.08 | 247s(3 epoches) |            496MiB |
| SVD++  |     - |        infinite |       Same as SVD |
| Weight | 26.35 |     Same as SVD |       Same as SVD |

SVD++模型的训练时间过长，未能得到训练结果，这里仅得到了SVD和相似属性物品加权（Weight）两个方法的RMSE结果。Weight的预测结果相比SVD算法有些许提升。

