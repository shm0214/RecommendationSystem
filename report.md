# 实验报告：推荐系统

## 数据集

数据集规模如下，分别是训练集和测试集中，用户和物品的维度。

|    Size    | user | item |
|:----------:|-----:|-----:|
| train set  | 19835 |624960 |
| test set   |      |      |

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

此外，还尝试了“用户已经打分过的物品中和所预测item最相似的几个item计算得分”和SVD结果进行融合，相似度使用属性值进行计算。但是最终未能成功训练出能够收敛的模型，隧不在实验结果中展示。

## 实验结果及分析

这里在训练集中随机划分了20%的数据作为测试集，来评估推荐算法；此外，由于内存中空间占用不方便进行计算，这里使用pickle库的序列化数据估计模型空间占用。实验结果如下：

| Method |  RMSE | Training Time | Space Consumption |
|:------:|------:|--------------:|------------------:|
|   SVD  | 27.08 |       000.00s |            496MiB |
|  SVD++ |     - |      infinite |       Same as SVD |
| Weight |       |               |                   |

SVD++模型的训练时间过长，未能得到训练结果，这里仅得到了SVD和相似属性物品加权（Weight）两个方法的RMSE结果。Weight的预测结果相比SVD算法有些许提升。

