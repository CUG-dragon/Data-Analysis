
## 数据分析--Task4：数据可视化


```python
# 加载所需的库
# 如果出现 ModuleNotFoundError: No module named 'xxxx'
# 你只需要在终端/cmd下 pip install xxxx 即可
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib具体作用是当你调用matplotlib.pyplot的绘图函数plot()进行绘图的或者生成一个figure画布的时候，可以直接在python console里面生成图像。
#而我们在spyder或者pycharm实际运行代码的时候，可以直接注释掉这一句，也是可以运行成功的。

```


```python
text = pd.read_csv(r'result.csv')
text.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



### 参考：《Python for Data Analysis》第九章

#### 任务一：跟着书本第九章，了解matplotlib，自己创建一个数据项，对其进行基本可视化
【思考】最基本的可视化图案有哪些？分别适用于那些场景？（比如折线图适合可视化某个属性值随时间变化的走势）


```python
data = np.arange(10)
data
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
plt.plot(data)
plt.show()
```


![png](output_5_0.png)



```python
fig = plt.figure()
# 定义figgure对象，增加2*2 4个子图，赋值1-3个子图给变量ax1,ax2,ax3。。。。
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)


# 接着直接调用AxexSubplot对象的实例方法，在空着的ax1,ax2格子里面画图
ax1.hist(np.random.randn(100),bins = 20, color = 'k', alpha = 0.3)
ax2.scatter(np.arange(30),np.arange(30) + 3 * np.random.randn(30))

ax3 = fig.add_subplot(2,2,3)

plt.plot(np.random.randn(50).cumsum(),'k--')
# 默认在最后一个用过的subplot上绘制



```




    [<matplotlib.lines.Line2D at 0xe13b32b630>]




![png](output_6_1.png)



```python
# 接着直接调用AxexSubplot对象的实例方法，在空着的格子里面画图
ax1.hist(np.random.randn(100),bins = 20, color = 'k', alpha = 0.3)
ax2.scatter(np.arange(30),np.arange(30) + 3 * np.random.randn(30))

```




    <matplotlib.collections.PathCollection at 0xe13b2f77f0>



#### 调整 subplot 间距


```python
fig, axes = plt.subplots(2,2,sharex = True, sharey = True)
for i in range(2):
    for j in range(2):
        axes[i,j].hist(np.random.randn(500),bins = 50, color = 'k', alpha = 0.5)
plt.subplots_adjust(wspace = 0,hspace = 0)

# wspace 和 hspace 用于控制宽度和高度的百分比，用作subplot之间的间距，示例间距缩小为0.
# 问题： 轴标签重叠了，需要自己设定刻度位置和刻度标签
```


![png](output_9_0.png)


#### 颜色，标记和线型


```python
from numpy.random import randn
plt.plot(randn(30).cumsum(),'ko--')

# 注意：标记类型（强调数据点）和线型，必须放在颜色参数后面
```




    [<matplotlib.lines.Line2D at 0xe13d2a7d68>]




![png](output_11_1.png)



```python
plt.plot(randn(30).cumsum(),color = 'b', linestyle = 'dashed', marker = '^')
```




    [<matplotlib.lines.Line2D at 0xe13c0db0f0>]




![png](output_12_1.png)



```python
data = randn(30).cumsum()
plt.plot(data, 'k--',label='Dafault')
plt.plot(data,'k-',drawstyle = 'steps-post', label = 'steps-post')
plt.legend(loc = 'best')
```




    <matplotlib.legend.Legend at 0xe13c3472b0>




![png](output_13_1.png)



```python

```


```python

```


```python

```

#思考回答

#这一部分需要了解可视化图案的的逻辑，知道什么样的图案可以表达什么样的信号

#### 任务二、 可视化展示泰坦尼克号数据集中男女中生存人数分布情况（用柱状图试试）。


```python
sex = text.groupby('Sex')['Survived'].sum()
sex.plot.bar()
plt.title('survived_count')
plt.show()
```


![png](output_19_0.png)


【思考】计算出泰坦尼克号数据集中男女中死亡人数，并可视化展示？  
如何和男女生存人数可视化柱状图结合到一起？  
看到你的数据可视化，说说你的第一感受（比如：你一眼看出男生存活人数更多，那么性别可能会影响存活率）。

【回答】： 堆积图，如下任务三。

#### 任务三：可视化展示泰坦尼克号数据集中男女中生存人与死亡人数的比例图（用柱状图试试）。


```python
# 提示：计算男女中死亡人数 1表示生存，0表示死亡
text.groupby(['Sex','Survived'])['Survived'].count().unstack().plot(kind='bar',stacked='True')
plt.title('survived_count')
plt.ylabel('count')
```




    Text(0, 0.5, 'count')




![png](output_21_1.png)


【提示】男女这两个数据轴，存活和死亡人数按比例用柱状图表示

#### 任务四：可视化展示泰坦尼克号数据集中不同票价的人生存和死亡人数分布情况。（用折线图试试）（横轴是不同票价，纵轴是存活人数）

【提示】对于这种统计性质的且用折线表示的数据，你可以考虑将数据排序或者不排序来分别表示。看看你能发现什么？


```python
# 计算不同票价中生存与死亡人数 1表示生存，0表示死亡
fare_sur = text.groupby(['Fare'])['Survived'].value_counts().sort_values(ascending=False)
fare_sur
```




    Fare     Survived
    8.0500   0           38
    7.8958   0           37
    13.0000  0           26
    7.7500   0           22
    26.0000  0           16
    13.0000  1           16
    26.0000  1           15
    10.5000  0           15
    0.0000   0           14
    7.7750   0           13
    7.2500   0           12
    7.7500   1           12
    8.6625   0           12
    7.2292   0           11
    7.8542   0           10
    7.9250   0           10
    10.5000  1            9
    7.2250   0            9
    26.5500  1            8
    7.9250   1            8
    24.1500  0            7
    9.5000   0            7
    16.1000  0            7
    26.5500  0            7
    69.5500  0            7
    31.2750  0            7
    7.0500   0            7
    14.4542  0            6
    27.9000  0            6
    39.6875  0            6
                         ..
    90.0000  0            1
    12.8750  0            1
    12.6500  1            1
    12.5250  0            1
    16.0000  1            1
    81.8583  1            1
    17.4000  1            1
    79.6500  0            1
    75.2500  1            1
    26.3875  1            1
    76.2917  1            1
    26.2833  1            1
    25.9250  0            1
    25.5875  0            1
    24.1500  1            1
    78.8500  0            1
    24.0000  1            1
             0            1
    78.8500  1            1
    22.5250  0            1
    22.0250  1            1
    21.6792  0            1
    20.5750  1            1
             0            1
    20.5250  0            1
    20.2500  1            1
             0            1
    18.7875  1            1
             0            1
    15.0500  0            1
    Name: Survived, Length: 330, dtype: int64




```python
# 排序后绘折线图
fig = plt.figure(figsize=(20, 18))
fare_sur.plot(grid=True)
plt.legend()
plt.show()
```


![png](output_25_0.png)



```python
# 排序前绘折线图
fare_sur1 = text.groupby(['Fare'])['Survived'].value_counts()
fare_sur1
```




    Fare      Survived
    0.0000    0           14
              1            1
    4.0125    0            1
    5.0000    0            1
    6.2375    0            1
    6.4375    0            1
    6.4500    0            1
    6.4958    0            2
    6.7500    0            2
    6.8583    0            1
    6.9500    0            1
    6.9750    0            1
              1            1
    7.0458    0            1
    7.0500    0            7
    7.0542    0            2
    7.1250    0            4
    7.1417    1            1
    7.2250    0            9
              1            3
    7.2292    0           11
              1            4
    7.2500    0           12
              1            1
    7.3125    0            1
    7.4958    0            2
              1            1
    7.5208    0            1
    7.5500    0            3
              1            1
                          ..
    106.4250  0            1
              1            1
    108.9000  0            1
              1            1
    110.8833  1            3
              0            1
    113.2750  1            2
              0            1
    120.0000  1            4
    133.6500  1            2
    134.5000  1            2
    135.6333  1            2
              0            1
    146.5208  1            2
    151.5500  0            2
              1            2
    153.4625  1            2
              0            1
    164.8667  1            2
    211.3375  1            3
    211.5000  0            1
    221.7792  0            1
    227.5250  1            3
              0            1
    247.5208  0            1
              1            1
    262.3750  1            2
    263.0000  0            2
              1            2
    512.3292  1            3
    Name: Survived, Length: 330, dtype: int64




```python
fig = plt.figure(figsize=(20, 18))
fare_sur1.plot(grid=True)
plt.legend()
plt.show()
```


![png](output_27_0.png)


#### 任务五：可视化展示泰坦尼克号数据集中不同仓位等级的人生存和死亡人员的分布情况。（用柱状图试试）


```python
# 1表示生存，0表示死亡
pclass_sur = text.groupby(['Pclass'])['Survived'].value_counts()
pclass_sur
```




    Pclass  Survived
    1       1           136
            0            80
    2       0            97
            1            87
    3       0           372
            1           119
    Name: Survived, dtype: int64




```python
import seaborn as sns
sns.countplot(x="Pclass", hue="Survived", data=text)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xe13f936898>




![png](output_30_1.png)


【思考】看到这个前面几个数据可视化，说说你的第一感受和你的总结

#### 任务六：可视化展示泰坦尼克号数据集中不同年龄的人生存与死亡人数分布情况。(不限表达方式)


```python
facet = sns.FacetGrid(text, hue="Survived",aspect=3)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, text['Age'].max()))
facet.add_legend()
```

    D:\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    




    <seaborn.axisgrid.FacetGrid at 0xe13f9396d8>




![png](output_33_2.png)


 #### 任务七：可视化展示泰坦尼克号数据集中不同仓位等级的人年龄分布情况。（用折线图试试）


```python
text.Age[text.Pclass == 1].plot(kind='kde')
text.Age[text.Pclass == 2].plot(kind='kde')
text.Age[text.Pclass == 3].plot(kind='kde')
plt.xlabel("age")
plt.legend((1,2,3),loc="best")
```




    <matplotlib.legend.Legend at 0xe13fa475c0>




![png](output_35_1.png)


【思考】上面所有可视化的例子做一个总体的分析，你看看你能不能有自己发现

【总结】到这里，我们的可视化就告一段落啦，如果你对数据可视化极其感兴趣，你还可以了解一下其他可视化模块，如：pyecharts，bokeh等。

如果你在工作中使用数据可视化，你必须知道数据可视化最大的作用不是炫酷，而是最快最直观的理解数据要表达什么，你觉得呢？
