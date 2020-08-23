
复习：在前面我们已经学习了Pandas基础，第二章我们开始进入数据分析的业务部分，在第二章第一节的内容中，我们学习了数据的清洗，这一部分十分重要，只有数据变得相对干净，我们之后对数据的分析才可以更有力。而这一节，我们要做的是数据重构，数据重构依旧属于数据理解（准备）的范围。


```python
# 导入基本库
import numpy as np
import pandas as pd
```


```python
# 载入data文件中的:train-left-up.csv
text = pd.read_csv('D:\\DataWhale\\Titanic\\train-left-up.csv')
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
    </tr>
  </tbody>
</table>
</div>



## 2 第二章：数据重构(上)
### 2.4 数据的合并
#### 2.4.1 任务一：将data文件夹里面的所有数据都载入，与之前的原始数据相比，观察他们的之间的关系


```python
text_left_up = pd.read_csv("D:\\DataWhale\\Titanic\\train-left-up.csv")
text_left_down = pd.read_csv("D:\\DataWhale\\Titanic\\train-left-down.csv")
text_right_up = pd.read_csv("D:\\DataWhale\\Titanic\\train-right-up.csv")
text_right_down = pd.read_csv("D:\\DataWhale\\Titanic\\train-right-down.csv")
```


```python
text_left_up.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
    </tr>
  </tbody>
</table>
</div>




```python
text_left_down.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>440</td>
      <td>0</td>
      <td>2</td>
      <td>Kvillner, Mr. Johan Henrik Johannesson</td>
    </tr>
    <tr>
      <th>1</th>
      <td>441</td>
      <td>1</td>
      <td>2</td>
      <td>Hart, Mrs. Benjamin (Esther Ada Bloomfield)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>442</td>
      <td>0</td>
      <td>3</td>
      <td>Hampe, Mr. Leon</td>
    </tr>
    <tr>
      <th>3</th>
      <td>443</td>
      <td>0</td>
      <td>3</td>
      <td>Petterson, Mr. Johan Emil</td>
    </tr>
    <tr>
      <th>4</th>
      <td>444</td>
      <td>1</td>
      <td>2</td>
      <td>Reynaldo, Ms. Encarnacion</td>
    </tr>
  </tbody>
</table>
</div>




```python
text_right_down.head()
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
      <td>male</td>
      <td>31.0</td>
      <td>0</td>
      <td>0</td>
      <td>C.A. 18723</td>
      <td>10.500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>45.0</td>
      <td>1</td>
      <td>1</td>
      <td>F.C.C. 13529</td>
      <td>26.250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>345769</td>
      <td>9.500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>25.0</td>
      <td>1</td>
      <td>0</td>
      <td>347076</td>
      <td>7.775</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>female</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>230434</td>
      <td>13.000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
text_right_up.head()
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



【提示】结合之前我们加载的train.csv数据，大致预测一下上面的数据是什么?

#### 2.4.2 任务二：使用concat方法：将数据train-left-up.csv和train-right-up.csv横向合并为一张表，并保存这张表为result_up


```python
list_up = [text_left_up,text_right_up]
result_up = pd.concat(list_up,axis=1)
result_up.head()
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
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
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
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
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
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>5.0</td>
      <td>0.0</td>
      <td>3.0</td>
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



#### 2.4.3 任务三：使用concat方法：将train-left-down和train-right-down横向合并为一张表，并保存这张表为result_down。然后将上边的result_up和result_down纵向合并为result。


```python
list_down=[text_left_down,text_right_down]
result_down = pd.concat(list_down,axis=1)
result = pd.concat([result_up,result_down])
result.head()
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
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
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
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
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
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>5.0</td>
      <td>0.0</td>
      <td>3.0</td>
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



#### 2.4.4 任务四：使用DataFrame自带的方法join方法和append：完成任务二和任务三的任务


```python
resul_up = text_left_up.join(text_right_up)
result_down = text_left_down.join(text_right_down)
result = result_up.append(result_down)
result.head()
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
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
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
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
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
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>5.0</td>
      <td>0.0</td>
      <td>3.0</td>
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



#### 2.4.5 任务五：使用Panads的merge方法和DataFrame的append方法：完成任务二和任务三的任务


```python
result_up = pd.merge(text_left_up,text_right_up,left_index=True,right_index=True)
result_down = pd.merge(text_left_down,text_right_down,left_index=True,right_index=True)
result = resul_up.append(result_down)
result.head()
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



【思考】对比merge、join以及concat的方法的不同以及相同。思考一下在任务四和任务五的情况下，为什么都要求使用DataFrame的append方法，如何只要求使用merge或者join可不可以完成任务四和任务五呢？

#### 2.4.6 任务六：完成的数据保存为result.csv


```python
result.to_csv('result.csv')
```

### 2.5 换一种角度看数据
#### 2.5.1 任务一：将我们的数据变为Series类型的数据
这个stack函数是干什么的？


```python
# 将完整的数据加载出来
text = pd.read_csv('result.csv')
text.head()
# 代码写在这里
unit_result=text.stack().head(20)
unit_result.head()
```




    0  Unnamed: 0                           0
       PassengerId                          1
       Survived                             0
       Pclass                               3
       Name           Braund, Mr. Owen Harris
    dtype: object




```python
#将代码保存为unit_result,csv
unit_result.to_csv('unit_result.csv')
```


```python
test = pd.read_csv('unit_result.csv')
```


```python
print(text)
```

         Unnamed: 0  PassengerId  Survived  Pclass  \
    0             0            1         0       3   
    1             1            2         1       1   
    2             2            3         1       3   
    3             3            4         1       1   
    4             4            5         0       3   
    5             5            6         0       3   
    6             6            7         0       1   
    7             7            8         0       3   
    8             8            9         1       3   
    9             9           10         1       2   
    10           10           11         1       3   
    11           11           12         1       1   
    12           12           13         0       3   
    13           13           14         0       3   
    14           14           15         0       3   
    15           15           16         1       2   
    16           16           17         0       3   
    17           17           18         1       2   
    18           18           19         0       3   
    19           19           20         1       3   
    20           20           21         0       2   
    21           21           22         1       2   
    22           22           23         1       3   
    23           23           24         1       1   
    24           24           25         0       3   
    25           25           26         1       3   
    26           26           27         0       3   
    27           27           28         0       1   
    28           28           29         1       3   
    29           29           30         0       3   
    ..          ...          ...       ...     ...   
    861         422          862         0       2   
    862         423          863         1       1   
    863         424          864         0       3   
    864         425          865         0       2   
    865         426          866         1       2   
    866         427          867         1       2   
    867         428          868         0       1   
    868         429          869         0       3   
    869         430          870         1       3   
    870         431          871         0       3   
    871         432          872         1       1   
    872         433          873         0       1   
    873         434          874         0       3   
    874         435          875         1       2   
    875         436          876         1       3   
    876         437          877         0       3   
    877         438          878         0       3   
    878         439          879         0       3   
    879         440          880         1       1   
    880         441          881         1       2   
    881         442          882         0       3   
    882         443          883         0       3   
    883         444          884         0       2   
    884         445          885         0       3   
    885         446          886         0       3   
    886         447          887         0       2   
    887         448          888         1       1   
    888         449          889         0       3   
    889         450          890         1       1   
    890         451          891         0       3   
    
                                                      Name     Sex   Age  SibSp  \
    0                              Braund, Mr. Owen Harris    male  22.0    1.0   
    1    Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0    1.0   
    2                               Heikkinen, Miss. Laina  female  26.0    0.0   
    3         Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0    1.0   
    4                             Allen, Mr. William Henry    male  35.0    0.0   
    5                                     Moran, Mr. James    male   NaN    0.0   
    6                              McCarthy, Mr. Timothy J    male  54.0    0.0   
    7                       Palsson, Master. Gosta Leonard    male   2.0    3.0   
    8    Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)  female  27.0    0.0   
    9                  Nasser, Mrs. Nicholas (Adele Achem)  female  14.0    1.0   
    10                     Sandstrom, Miss. Marguerite Rut  female   4.0    1.0   
    11                            Bonnell, Miss. Elizabeth  female  58.0    0.0   
    12                      Saundercock, Mr. William Henry    male  20.0    0.0   
    13                         Andersson, Mr. Anders Johan    male  39.0    1.0   
    14                Vestrom, Miss. Hulda Amanda Adolfina  female  14.0    0.0   
    15                    Hewlett, Mrs. (Mary D Kingcome)   female  55.0    0.0   
    16                                Rice, Master. Eugene    male   2.0    4.0   
    17                        Williams, Mr. Charles Eugene    male   NaN    0.0   
    18   Vander Planke, Mrs. Julius (Emelia Maria Vande...  female  31.0    1.0   
    19                             Masselmani, Mrs. Fatima  female   NaN    0.0   
    20                                Fynney, Mr. Joseph J    male  35.0    0.0   
    21                               Beesley, Mr. Lawrence    male  34.0    0.0   
    22                         McGowan, Miss. Anna "Annie"  female  15.0    0.0   
    23                        Sloper, Mr. William Thompson    male  28.0    0.0   
    24                       Palsson, Miss. Torborg Danira  female   8.0    3.0   
    25   Asplund, Mrs. Carl Oscar (Selma Augusta Emilia...  female  38.0    1.0   
    26                             Emir, Mr. Farred Chehab    male   NaN    0.0   
    27                      Fortune, Mr. Charles Alexander    male  19.0    3.0   
    28                       O'Dwyer, Miss. Ellen "Nellie"  female   NaN    0.0   
    29                                 Todoroff, Mr. Lalio    male   NaN    0.0   
    ..                                                 ...     ...   ...    ...   
    861                        Giles, Mr. Frederick Edward    male  21.0    1.0   
    862  Swift, Mrs. Frederick Joel (Margaret Welles Ba...  female  48.0    0.0   
    863                  Sage, Miss. Dorothy Edith "Dolly"  female   NaN    8.0   
    864                             Gill, Mr. John William    male  24.0    0.0   
    865                           Bystrom, Mrs. (Karolina)  female  42.0    0.0   
    866                       Duran y More, Miss. Asuncion  female  27.0    1.0   
    867               Roebling, Mr. Washington Augustus II    male  31.0    0.0   
    868                        van Melkebeke, Mr. Philemon    male   NaN    0.0   
    869                    Johnson, Master. Harold Theodor    male   4.0    1.0   
    870                                  Balkic, Mr. Cerin    male  26.0    0.0   
    871   Beckwith, Mrs. Richard Leonard (Sallie Monypeny)  female  47.0    1.0   
    872                           Carlsson, Mr. Frans Olof    male  33.0    0.0   
    873                        Vander Cruyssen, Mr. Victor    male  47.0    0.0   
    874              Abelson, Mrs. Samuel (Hannah Wizosky)  female  28.0    1.0   
    875                   Najib, Miss. Adele Kiamie "Jane"  female  15.0    0.0   
    876                      Gustafsson, Mr. Alfred Ossian    male  20.0    0.0   
    877                               Petroff, Mr. Nedelio    male  19.0    0.0   
    878                                 Laleff, Mr. Kristo    male   NaN    0.0   
    879      Potter, Mrs. Thomas Jr (Lily Alexenia Wilson)  female  56.0    0.0   
    880       Shelley, Mrs. William (Imanita Parrish Hall)  female  25.0    0.0   
    881                                 Markun, Mr. Johann    male  33.0    0.0   
    882                       Dahlberg, Miss. Gerda Ulrika  female  22.0    0.0   
    883                      Banfield, Mr. Frederick James    male  28.0    0.0   
    884                             Sutehall, Mr. Henry Jr    male  25.0    0.0   
    885               Rice, Mrs. William (Margaret Norton)  female  39.0    0.0   
    886                              Montvila, Rev. Juozas    male  27.0    0.0   
    887                       Graham, Miss. Margaret Edith  female  19.0    0.0   
    888           Johnston, Miss. Catherine Helen "Carrie"  female   NaN    1.0   
    889                              Behr, Mr. Karl Howell    male  26.0    0.0   
    890                                Dooley, Mr. Patrick    male  32.0    0.0   
    
         Parch            Ticket      Fare        Cabin Embarked  
    0      0.0         A/5 21171    7.2500          NaN        S  
    1      0.0          PC 17599   71.2833          C85        C  
    2      0.0  STON/O2. 3101282    7.9250          NaN        S  
    3      0.0            113803   53.1000         C123        S  
    4      0.0            373450    8.0500          NaN        S  
    5      0.0            330877    8.4583          NaN        Q  
    6      0.0             17463   51.8625          E46        S  
    7      1.0            349909   21.0750          NaN        S  
    8      2.0            347742   11.1333          NaN        S  
    9      0.0            237736   30.0708          NaN        C  
    10     1.0           PP 9549   16.7000           G6        S  
    11     0.0            113783   26.5500         C103        S  
    12     0.0         A/5. 2151    8.0500          NaN        S  
    13     5.0            347082   31.2750          NaN        S  
    14     0.0            350406    7.8542          NaN        S  
    15     0.0            248706   16.0000          NaN        S  
    16     1.0            382652   29.1250          NaN        Q  
    17     0.0            244373   13.0000          NaN        S  
    18     0.0            345763   18.0000          NaN        S  
    19     0.0              2649    7.2250          NaN        C  
    20     0.0            239865   26.0000          NaN        S  
    21     0.0            248698   13.0000          D56        S  
    22     0.0            330923    8.0292          NaN        Q  
    23     0.0            113788   35.5000           A6        S  
    24     1.0            349909   21.0750          NaN        S  
    25     5.0            347077   31.3875          NaN        S  
    26     0.0              2631    7.2250          NaN        C  
    27     2.0             19950  263.0000  C23 C25 C27        S  
    28     0.0            330959    7.8792          NaN        Q  
    29     0.0            349216    7.8958          NaN        S  
    ..     ...               ...       ...          ...      ...  
    861    0.0             28134   11.5000          NaN        S  
    862    0.0             17466   25.9292          D17        S  
    863    2.0          CA. 2343   69.5500          NaN        S  
    864    0.0            233866   13.0000          NaN        S  
    865    0.0            236852   13.0000          NaN        S  
    866    0.0     SC/PARIS 2149   13.8583          NaN        C  
    867    0.0          PC 17590   50.4958          A24        S  
    868    0.0            345777    9.5000          NaN        S  
    869    1.0            347742   11.1333          NaN        S  
    870    0.0            349248    7.8958          NaN        S  
    871    1.0             11751   52.5542          D35        S  
    872    0.0               695    5.0000  B51 B53 B55        S  
    873    0.0            345765    9.0000          NaN        S  
    874    0.0         P/PP 3381   24.0000          NaN        C  
    875    0.0              2667    7.2250          NaN        C  
    876    0.0              7534    9.8458          NaN        S  
    877    0.0            349212    7.8958          NaN        S  
    878    0.0            349217    7.8958          NaN        S  
    879    1.0             11767   83.1583          C50        C  
    880    1.0            230433   26.0000          NaN        S  
    881    0.0            349257    7.8958          NaN        S  
    882    0.0              7552   10.5167          NaN        S  
    883    0.0  C.A./SOTON 34068   10.5000          NaN        S  
    884    0.0   SOTON/OQ 392076    7.0500          NaN        S  
    885    5.0            382652   29.1250          NaN        Q  
    886    0.0            211536   13.0000          NaN        S  
    887    0.0            112053   30.0000          B42        S  
    888    2.0        W./C. 6607   23.4500          NaN        S  
    889    0.0            111369   30.0000         C148        C  
    890    0.0            370376    7.7500          NaN        Q  
    
    [891 rows x 13 columns]
    


```python
test.head()
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
      <th>0</th>
      <th>Unnamed: 0</th>
      <th>0.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>PassengerId</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Survived</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>Pclass</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>Name</td>
      <td>Braund, Mr. Owen Harris</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Sex</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>



## 2第二章：数据重构 (下)




```python
# 导入基本库
import numpy as np
import pandas as pd
```


```python
# 载入data文件中的:result.csv
text = pd.read_csv('result.csv')
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



## 第一部分：数据聚合与运算
### 2.6 数据运用
#### 2.6.1 任务一：通过《Python for Data Analysis》P303、Google or Baidu来学习了解GroupBy机制


### 心得
groupby().size()方法返回已给含有分组大小的Series  
df.groupby() 支持迭代产生一组二元元祖  
DataFrame产生的GroupBy对象，可以对选取列进行聚合  
[利用Python进行数据分析 10.1 GroupBy机制](https://www.jianshu.com/p/512273858104)


在了解GroupBy机制之后，运用这个机制完成一系列的操作，来达到我们的目的。

下面通过几个任务来熟悉GroupBy机制

#### 2.6.2：任务二：计算泰坦尼克号男性与女性的平均票价


```python
df  = text['Fare'].groupby(text['Sex'])
means = df.mean()
means
```




    Sex
    female    44.479818
    male      25.523893
    Name: Fare, dtype: float64



#### 2.6.3：任务三：统计泰坦尼克号中男女的存活人数


```python
survived_sex = text['Survived'].groupby(text['Sex']).sum()
survived_sex.head()
```




    Sex
    female    233
    male      109
    Name: Survived, dtype: int64



#### 2.6.4：任务四：计算客舱不同等级的存活人数


```python
survived_pclass = text['Survived'].groupby(text['Pclass'])
survived_pclass.sum()
```




    Pclass
    1    136
    2     87
    3    119
    Name: Survived, dtype: int64



【提示：】表中的存活那一栏，可以发现如果还活着记为1，死亡记为0

【思考：】从数据分析的角度，上面的统计结果可以得出那些结论


```python
#例子： 修正参考答案的例子，其中df替换为text, 原来的Sex, Pclass不是数字类型，测试替换为Age,Survived
# 验证了groupby的方法
text.groupby('Survived').agg({'Age': 'mean', 'Survived': 'sum'}).rename(columns = {'Age': 'mean_Age', 'Survived': 'sum_Survived'})
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
      <th>mean_Age</th>
      <th>sum_Survived</th>
    </tr>
    <tr>
      <th>Survived</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30.626179</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28.343690</td>
      <td>342</td>
    </tr>
  </tbody>
</table>
</div>



#### 2.6.5：任务五：统计在不同等级的票中的不同年龄的船票花费的平均值


```python
text.groupby(['Pclass','Age'])['Fare'].mean().head()
```




    Pclass  Age  
    1       0.92     151.5500
            2.00     151.5500
            4.00      81.8583
            11.00    120.0000
            14.00    120.0000
    Name: Fare, dtype: float64



#### 2.6.6：任务六：将任务二和任务三的数据合并，并保存到sex_fare_survived.csv


```python
##result1 = pd.merge(means,survived_sex,on = 'Sex')
##result1
##ValueError: can not merge DataFrame with instance of type <class 'pandas.core.series.Series'>

```

错误分析暂放下：  
How to fix ValueError: can not merge DataFrame with instance of type <class 'pandas.core.series.Series'> ?  

df_mean = df.groupby('id').col.mean().rename('mean_col')  
df_min = df.groupby('id').col.min().rename('min_col')  
df_result = pd.concat([df_mean, df_min], axis=1).reset_index()  
python - Merging two DataFrames - Stack Overflow  
https://stackoverflow.com/questions/37968785/merging-two-dataframes  
df1.merge(df2.to_frame(), left_on='id', right_index=True)  
python - Combining two Series into a DataFrame in pandas - Stack Overflow  
https://stackoverflow.com/questions/18062135/combining-two-series-into-a-dataframe-in-pandas  
pd.concat([s1, s2], axis=1).reset_index()  


```python
result.to_csv('sex_fare_survived.csv')
```

#### 2.6.7：任务七：得出不同年龄的总的存活人数，然后找出存活人数的最高的年龄，最后计算存活人数最高的存活率（存活人数/总人数）


```python
#不同年龄的存活人数 
survived_age = text['Survived'].groupby(text['Age']).sum() 
survived_age.head()
```




    Age
    0.42    1
    0.67    1
    0.75    2
    0.83    2
    0.92    1
    Name: Survived, dtype: int64




```python
#找出最大值的年龄段
survived_age[survived_age.values==survived_age.max()]
```




    Age
    24.0    15
    Name: Survived, dtype: int64




```python
_sum = text['Survived'].sum()
print(_sum)
```

    342
    


```python
#首先计算总人数
_sum = text['Survived'].sum()

print("sum of person:"+str(_sum))

precetn =survived_age.max()/_sum

print("最大存活率："+str(precetn))
```

    sum of person:342
    最大存活率：0.043859649122807015
    


```python

```
