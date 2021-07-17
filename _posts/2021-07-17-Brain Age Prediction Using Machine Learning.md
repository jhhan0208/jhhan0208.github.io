---
tags: [Data Science]

excerpt: "Data Science"
---

## 프로젝트 소개

### 목표

brain-age 정확하게 예측하는 모델 개발

### 상세 목표

brain predicted age difference (brain-predicted age–actual age) 최소화

### 의의

1. brain age와 actual age의 비교 통해 뇌 건강 진단 가능
2. 뇌 관련 질병의 발병 시기를 예측 가능

### 제공되는 data

68개 regional cortical thickness, intracranial volume



### 필요 라이브러리들 불러오기, train data 불러오기


```python
import pandas as pd
import numpy as np

train_data = pd.read_csv('IXI_train.csv')
train_data.head()
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
      <th>ID</th>
      <th>Sex</th>
      <th>Age</th>
      <th>lbankssts</th>
      <th>rbankssts</th>
      <th>lcaudalanteriorcingulate</th>
      <th>rcaudalanteriorcingulate</th>
      <th>lcaudalmiddlefrontal</th>
      <th>rcaudalmiddlefrontal</th>
      <th>lcuneus</th>
      <th>...</th>
      <th>rsupramarginal</th>
      <th>lfrontalpole</th>
      <th>rfrontalpole</th>
      <th>ltemporalpole</th>
      <th>rtemporalpole</th>
      <th>ltransversetemporal</th>
      <th>rtransversetemporal</th>
      <th>linsula</th>
      <th>rinsula</th>
      <th>ICV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>IXI002</td>
      <td>2.0</td>
      <td>36.0</td>
      <td>2.314134</td>
      <td>2.445358</td>
      <td>2.271341</td>
      <td>2.295045</td>
      <td>2.641145</td>
      <td>2.789410</td>
      <td>1.867423</td>
      <td>...</td>
      <td>2.417642</td>
      <td>2.989790</td>
      <td>2.802352</td>
      <td>3.629097</td>
      <td>3.922553</td>
      <td>2.653073</td>
      <td>2.690095</td>
      <td>2.975480</td>
      <td>2.922020</td>
      <td>1393.44</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IXI012</td>
      <td>1.0</td>
      <td>39.0</td>
      <td>2.256589</td>
      <td>2.679695</td>
      <td>2.225613</td>
      <td>2.416877</td>
      <td>2.454876</td>
      <td>2.438577</td>
      <td>1.907090</td>
      <td>...</td>
      <td>2.375922</td>
      <td>2.599895</td>
      <td>2.467085</td>
      <td>3.669829</td>
      <td>3.288221</td>
      <td>2.241236</td>
      <td>2.109445</td>
      <td>2.755640</td>
      <td>2.801413</td>
      <td>1622.12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IXI013</td>
      <td>1.0</td>
      <td>47.0</td>
      <td>2.268161</td>
      <td>2.233568</td>
      <td>2.260727</td>
      <td>2.013187</td>
      <td>2.271975</td>
      <td>2.470277</td>
      <td>1.626596</td>
      <td>...</td>
      <td>2.435376</td>
      <td>2.837961</td>
      <td>2.213487</td>
      <td>2.857899</td>
      <td>3.033377</td>
      <td>1.995272</td>
      <td>2.270841</td>
      <td>2.832390</td>
      <td>2.885394</td>
      <td>1541.86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IXI014</td>
      <td>2.0</td>
      <td>34.0</td>
      <td>2.426588</td>
      <td>2.517577</td>
      <td>2.345924</td>
      <td>2.179330</td>
      <td>2.292475</td>
      <td>2.520211</td>
      <td>1.700885</td>
      <td>...</td>
      <td>2.410944</td>
      <td>2.562584</td>
      <td>2.628849</td>
      <td>3.533523</td>
      <td>3.024386</td>
      <td>2.187879</td>
      <td>2.511657</td>
      <td>2.828906</td>
      <td>2.789627</td>
      <td>1327.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>IXI015</td>
      <td>1.0</td>
      <td>24.0</td>
      <td>2.583845</td>
      <td>2.716359</td>
      <td>2.337998</td>
      <td>2.287648</td>
      <td>2.632270</td>
      <td>2.574164</td>
      <td>1.822473</td>
      <td>...</td>
      <td>2.602356</td>
      <td>2.720133</td>
      <td>2.393970</td>
      <td>3.349275</td>
      <td>3.239221</td>
      <td>2.619177</td>
      <td>2.593085</td>
      <td>2.951370</td>
      <td>2.846432</td>
      <td>1499.27</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 72 columns</p>
</div>



### train data에서 Nan 값 제거


```python
train_data = train_data.dropna(axis=0)
train_data.head()
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
      <th>ID</th>
      <th>Sex</th>
      <th>Age</th>
      <th>lbankssts</th>
      <th>rbankssts</th>
      <th>lcaudalanteriorcingulate</th>
      <th>rcaudalanteriorcingulate</th>
      <th>lcaudalmiddlefrontal</th>
      <th>rcaudalmiddlefrontal</th>
      <th>lcuneus</th>
      <th>...</th>
      <th>rsupramarginal</th>
      <th>lfrontalpole</th>
      <th>rfrontalpole</th>
      <th>ltemporalpole</th>
      <th>rtemporalpole</th>
      <th>ltransversetemporal</th>
      <th>rtransversetemporal</th>
      <th>linsula</th>
      <th>rinsula</th>
      <th>ICV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>IXI002</td>
      <td>2.0</td>
      <td>36.0</td>
      <td>2.314134</td>
      <td>2.445358</td>
      <td>2.271341</td>
      <td>2.295045</td>
      <td>2.641145</td>
      <td>2.789410</td>
      <td>1.867423</td>
      <td>...</td>
      <td>2.417642</td>
      <td>2.989790</td>
      <td>2.802352</td>
      <td>3.629097</td>
      <td>3.922553</td>
      <td>2.653073</td>
      <td>2.690095</td>
      <td>2.975480</td>
      <td>2.922020</td>
      <td>1393.44</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IXI012</td>
      <td>1.0</td>
      <td>39.0</td>
      <td>2.256589</td>
      <td>2.679695</td>
      <td>2.225613</td>
      <td>2.416877</td>
      <td>2.454876</td>
      <td>2.438577</td>
      <td>1.907090</td>
      <td>...</td>
      <td>2.375922</td>
      <td>2.599895</td>
      <td>2.467085</td>
      <td>3.669829</td>
      <td>3.288221</td>
      <td>2.241236</td>
      <td>2.109445</td>
      <td>2.755640</td>
      <td>2.801413</td>
      <td>1622.12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IXI013</td>
      <td>1.0</td>
      <td>47.0</td>
      <td>2.268161</td>
      <td>2.233568</td>
      <td>2.260727</td>
      <td>2.013187</td>
      <td>2.271975</td>
      <td>2.470277</td>
      <td>1.626596</td>
      <td>...</td>
      <td>2.435376</td>
      <td>2.837961</td>
      <td>2.213487</td>
      <td>2.857899</td>
      <td>3.033377</td>
      <td>1.995272</td>
      <td>2.270841</td>
      <td>2.832390</td>
      <td>2.885394</td>
      <td>1541.86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IXI014</td>
      <td>2.0</td>
      <td>34.0</td>
      <td>2.426588</td>
      <td>2.517577</td>
      <td>2.345924</td>
      <td>2.179330</td>
      <td>2.292475</td>
      <td>2.520211</td>
      <td>1.700885</td>
      <td>...</td>
      <td>2.410944</td>
      <td>2.562584</td>
      <td>2.628849</td>
      <td>3.533523</td>
      <td>3.024386</td>
      <td>2.187879</td>
      <td>2.511657</td>
      <td>2.828906</td>
      <td>2.789627</td>
      <td>1327.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>IXI015</td>
      <td>1.0</td>
      <td>24.0</td>
      <td>2.583845</td>
      <td>2.716359</td>
      <td>2.337998</td>
      <td>2.287648</td>
      <td>2.632270</td>
      <td>2.574164</td>
      <td>1.822473</td>
      <td>...</td>
      <td>2.602356</td>
      <td>2.720133</td>
      <td>2.393970</td>
      <td>3.349275</td>
      <td>3.239221</td>
      <td>2.619177</td>
      <td>2.593085</td>
      <td>2.951370</td>
      <td>2.846432</td>
      <td>1499.27</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 72 columns</p>
</div>



### train data preprocessing


```python
import math

# ID 제거
Train = train_data.values
Train_1 = Train[:, 1]
Train_1 = Train_1.reshape(463, 1)

Train = Train[:, 3:]
Train = np.concatenate([Train_1, Train], axis=1)

# scaling data
for i in range(0, 463):
    Train[:, -1:][i] = Train[:, -1:][i]**(1.0/4.0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(Train)
Train_scale = pd.DataFrame(scaler.transform(Train))
Train_scale = Train_scale.values

# target 분리
X = Train_scale 
y = train_data['Age'].values
```

### Target 분포 확인하기


```python
train_data['Age'].plot(kind = 'density')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22924717248>




![png](/images/기계학습_term_project/output_10_1.png)


### 필요한 library들 import


```python
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

import matplotlib.pyplot as plt
```

### train data split


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 123)
```

### fit gaussian process regression model using train data


```python
kernel = ConstantKernel() + ConstantKernel() * RBF() + WhiteKernel()
model = GaussianProcessRegressor(alpha= 1e-5,kernel=kernel, n_restarts_optimizer = 30)
model.fit(X_train, y_train)
y_pred_tr, y_pred_tr_std = model.predict(X_train, return_std=True)
y_pred_te, y_pred_te_std = model.predict(X_test, return_std=True)
```

### r2_score


```python
print('R2_score = %.2f' % r2_score(y_test, y_pred_te))
```

    R2_score = 0.72
    

### visualize results


```python
plt.figure(figsize = (7, 7))

for i in range(93):
    if(y_test[i] > y_pred_te[i]):
        young = plt.errorbar(y_test[i], y_pred_te[i], fmt='co')
    else:
        old = plt.errorbar(y_test[i], y_pred_te[i], fmt='ro')

plt.title('Gaussian Process Regression\n R2_score=%.2f' % r2_score(y_test, y_pred_te))
plt.xlabel('chronological age')
plt.ylabel('predicted age')

x_graph = np.linspace(20, 80, 100)
y_graph = x_graph
plt.plot(x_graph, y_graph, 'k')


plt.legend([old, young], ['older brain', 'younger brain',], fontsize=16)

plt.show
```




    <function matplotlib.pyplot.show(*args, **kw)>




![png](/images/기계학습_term_project/output_20_1.png)



```python
plt.figure(figsize = (7, 7))

for i in range(93):
    if(y_test[i] >= y_pred_te[i]):
        young = plt.errorbar(y_test[i], y_pred_te[i], yerr = [(0, ), (y_test[i] - y_pred_te[i],)], fmt='bo')
    else:
        old = plt.errorbar(y_test[i], y_pred_te[i], yerr = [(y_pred_te[i] - y_test[i],), (0, )], fmt='bo')
        

plt.title('Gaussian Process Regression\n R2_score=%.2f' % r2_score(y_test, y_pred_te))
plt.xlabel('chronological age')
plt.ylabel('predicted age')

x_graph = np.linspace(15, 90, 100)
y_graph = x_graph
plt.plot(x_graph, y_graph, 'r')


plt.legend([old], ['brain-PAD'], fontsize=16)

plt.show
```




    <function matplotlib.pyplot.show(*args, **kw)>




![png](/images/기계학습_term_project/output_21_1.png)


### IXI_test data 불러오기


```python
IXI_test_data = pd.read_csv('IXI_test.csv')
IXI_test_data.head()
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
      <th>ID</th>
      <th>Sex</th>
      <th>Age</th>
      <th>lbankssts</th>
      <th>rbankssts</th>
      <th>lcaudalanteriorcingulate</th>
      <th>rcaudalanteriorcingulate</th>
      <th>lcaudalmiddlefrontal</th>
      <th>rcaudalmiddlefrontal</th>
      <th>lcuneus</th>
      <th>...</th>
      <th>rsupramarginal</th>
      <th>lfrontalpole</th>
      <th>rfrontalpole</th>
      <th>ltemporalpole</th>
      <th>rtemporalpole</th>
      <th>ltransversetemporal</th>
      <th>rtransversetemporal</th>
      <th>linsula</th>
      <th>rinsula</th>
      <th>ICV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>IXI026</td>
      <td>2</td>
      <td>NaN</td>
      <td>2.790918</td>
      <td>2.860664</td>
      <td>2.377023</td>
      <td>2.694348</td>
      <td>2.767666</td>
      <td>2.746961</td>
      <td>1.887981</td>
      <td>...</td>
      <td>2.563580</td>
      <td>2.934798</td>
      <td>2.925411</td>
      <td>3.349163</td>
      <td>3.106885</td>
      <td>2.481392</td>
      <td>2.391845</td>
      <td>3.349330</td>
      <td>3.176278</td>
      <td>1257.57</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IXI027</td>
      <td>1</td>
      <td>NaN</td>
      <td>2.594282</td>
      <td>2.700979</td>
      <td>2.458627</td>
      <td>2.423396</td>
      <td>2.742329</td>
      <td>2.817457</td>
      <td>1.737984</td>
      <td>...</td>
      <td>2.613335</td>
      <td>2.751007</td>
      <td>2.959820</td>
      <td>3.667810</td>
      <td>3.428951</td>
      <td>2.370657</td>
      <td>2.254736</td>
      <td>3.067785</td>
      <td>3.160975</td>
      <td>1554.63</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IXI028</td>
      <td>1</td>
      <td>NaN</td>
      <td>2.597014</td>
      <td>2.674919</td>
      <td>2.258676</td>
      <td>2.016510</td>
      <td>2.460518</td>
      <td>2.367396</td>
      <td>1.595580</td>
      <td>...</td>
      <td>2.332489</td>
      <td>2.843313</td>
      <td>2.465880</td>
      <td>3.841134</td>
      <td>3.152598</td>
      <td>1.926546</td>
      <td>1.782548</td>
      <td>3.100923</td>
      <td>3.101951</td>
      <td>1468.54</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IXI040</td>
      <td>2</td>
      <td>NaN</td>
      <td>2.520381</td>
      <td>2.443344</td>
      <td>2.236262</td>
      <td>2.550456</td>
      <td>2.453389</td>
      <td>2.395759</td>
      <td>1.772909</td>
      <td>...</td>
      <td>2.332510</td>
      <td>2.426691</td>
      <td>2.395207</td>
      <td>3.659196</td>
      <td>3.312159</td>
      <td>2.337989</td>
      <td>2.234992</td>
      <td>2.824175</td>
      <td>2.936322</td>
      <td>1280.93</td>
    </tr>
    <tr>
      <th>4</th>
      <td>IXI041</td>
      <td>1</td>
      <td>NaN</td>
      <td>2.669905</td>
      <td>2.510474</td>
      <td>2.209484</td>
      <td>2.374890</td>
      <td>2.539132</td>
      <td>2.601794</td>
      <td>1.935135</td>
      <td>...</td>
      <td>2.611295</td>
      <td>2.469409</td>
      <td>2.648380</td>
      <td>3.893395</td>
      <td>3.600993</td>
      <td>2.437859</td>
      <td>2.511397</td>
      <td>2.859236</td>
      <td>2.982760</td>
      <td>1708.56</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 72 columns</p>
</div>



### IXI_test data preprocessing


```python
IXI_Test = IXI_test_data.values

IXI_Test_1 = IXI_Test[:, 1]
IXI_Test_1 = IXI_Test_1.reshape(100, 1)

IXI_Test = IXI_Test[:, 3:]

IXI_Test = np.concatenate([IXI_Test_1, IXI_Test], axis=1)

for i in range(0, 100):
    IXI_Test[:, -1:][i] = IXI_Test[:, -1:][i]**(1.0/4.0)
    

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(IXI_Test)
IXI_Test_scale = pd.DataFrame(scaler.transform(IXI_Test))
IXI_Test_scale = IXI_Test_scale.values

X_IXI_test = IXI_Test_scale 
y_IXI_test = IXI_test_data['Age'].values

print(X_IXI_test)
```

    [[ 0.86855395  2.06022336  1.81267592 ...  2.31552946  1.30926576
      -1.17315757]
     [-1.15133896  0.80925806  0.8915895  ...  0.83304858  1.22831896
       0.8994097 ]
     [-1.15133896  0.82663601  0.74127454 ...  1.0075364   0.91610098
       0.33177745]
     ...
     [-1.15133896 -0.19536707  0.99396575 ...  0.03515842 -1.06804538
       1.14140249]
     [-1.15133896 -1.73497532 -1.37640543 ... -1.59298208 -1.52470966
       1.57447924]
     [-1.15133896  1.1875096   1.01930532 ...  0.6225658   0.1787746
       1.98883328]]
    

### predict  age of IXI_test data using trained model


```python
y_pred_IXI_test, y_pred_IXI_test_std = model.predict(X_IXI_test, return_std=True)

for i in range(100):
    y_pred_IXI_test[i] = round(y_pred_IXI_test[i])
    
print(y_pred_IXI_test)
```

    [40. 41. 69. 58. 33. 42. 28. 28. 29. 34. 63. 36. 35. 29. 60. 42. 57. 47.
     40. 30. 39. 18. 57. 66. 37. 50. 62. 34. 25. 39. 33. 56. 73. 40. 49. 76.
     27. 50. 34. 34. 61. 61. 65. 62. 47. 61. 49. 49. 42. 47. 39. 29. 51. 62.
     41. 32. 33. 61. 47. 46. 52. 59. 53. 58. 61. 68. 70. 58. 33. 70. 40. 71.
     31. 64. 50. 40. 62. 76. 39. 76. 73. 64. 62. 51. 71. 42. 32. 56. 25. 38.
     43. 29. 53. 40. 42. 66. 44. 41. 57. 43.]
    

### export predict age of IXI_test data  to xlsx


```python
IXI_test_age = pd.DataFrame({'ID': IXI_test_data["ID"], 'Age': y_pred_IXI_test})

#IXI_test_age.to_excel('IXI_test_brain_age_submission_소프트웨어융합학과_2020105742_한지훈_최종.xlsx', index = False)
```

### COBRE_test data 불러오기


```python
COBRE_test_data = pd.read_csv('COBRE_test_scz.csv')
COBRE_test_data.head()
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
      <th>ID</th>
      <th>Age</th>
      <th>Sex</th>
      <th>lh_bankssts_thickness</th>
      <th>lh_caudalanteriorcingulate_thickness</th>
      <th>lh_caudalmiddlefrontal_thickness</th>
      <th>lh_cuneus_thickness</th>
      <th>lh_entorhinal_thickness</th>
      <th>lh_fusiform_thickness</th>
      <th>lh_inferiorparietal_thickness</th>
      <th>...</th>
      <th>rh_rostralmiddlefrontal_thickness</th>
      <th>rh_superiorfrontal_thickness</th>
      <th>rh_superiorparietal_thickness</th>
      <th>rh_superiortemporal_thickness</th>
      <th>rh_supramarginal_thickness</th>
      <th>rh_frontalpole_thickness</th>
      <th>rh_temporalpole_thickness</th>
      <th>rh_transversetemporal_thickness</th>
      <th>rh_insula_thickness</th>
      <th>ICV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40000</td>
      <td>NaN</td>
      <td>2</td>
      <td>2.637</td>
      <td>2.462</td>
      <td>2.755</td>
      <td>2.037</td>
      <td>3.870</td>
      <td>2.830</td>
      <td>2.566</td>
      <td>...</td>
      <td>2.555</td>
      <td>2.809</td>
      <td>2.202</td>
      <td>2.834</td>
      <td>2.518</td>
      <td>2.949</td>
      <td>3.401</td>
      <td>2.411</td>
      <td>2.908</td>
      <td>1376372.945</td>
    </tr>
    <tr>
      <th>1</th>
      <td>40001</td>
      <td>NaN</td>
      <td>1</td>
      <td>2.449</td>
      <td>2.379</td>
      <td>2.553</td>
      <td>1.969</td>
      <td>3.567</td>
      <td>2.590</td>
      <td>2.463</td>
      <td>...</td>
      <td>2.450</td>
      <td>2.657</td>
      <td>2.142</td>
      <td>2.688</td>
      <td>2.476</td>
      <td>2.730</td>
      <td>3.519</td>
      <td>2.147</td>
      <td>2.737</td>
      <td>1450970.104</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40002</td>
      <td>NaN</td>
      <td>1</td>
      <td>2.516</td>
      <td>2.593</td>
      <td>2.586</td>
      <td>1.875</td>
      <td>3.516</td>
      <td>2.573</td>
      <td>2.376</td>
      <td>...</td>
      <td>2.268</td>
      <td>2.726</td>
      <td>2.096</td>
      <td>2.637</td>
      <td>2.483</td>
      <td>2.627</td>
      <td>3.653</td>
      <td>2.435</td>
      <td>3.018</td>
      <td>1455186.664</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40003</td>
      <td>NaN</td>
      <td>1</td>
      <td>2.458</td>
      <td>2.695</td>
      <td>2.654</td>
      <td>1.779</td>
      <td>3.270</td>
      <td>2.605</td>
      <td>2.458</td>
      <td>...</td>
      <td>2.256</td>
      <td>2.598</td>
      <td>2.164</td>
      <td>2.783</td>
      <td>2.544</td>
      <td>2.616</td>
      <td>3.925</td>
      <td>2.018</td>
      <td>3.217</td>
      <td>1798564.922</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40004</td>
      <td>NaN</td>
      <td>1</td>
      <td>2.422</td>
      <td>2.531</td>
      <td>2.515</td>
      <td>1.710</td>
      <td>3.750</td>
      <td>2.662</td>
      <td>2.303</td>
      <td>...</td>
      <td>2.282</td>
      <td>2.626</td>
      <td>2.117</td>
      <td>2.618</td>
      <td>2.464</td>
      <td>2.878</td>
      <td>3.598</td>
      <td>2.330</td>
      <td>2.808</td>
      <td>1678245.894</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 72 columns</p>
</div>



### COBRE_test data preprocessing


```python
COBRE_Test = COBRE_test_data.values    

for i in range(72):
    COBRE_Test[i][71] = 0.001 * COBRE_Test[i][71]

COBRE_Test_1 = COBRE_Test[:, 2]
COBRE_Test_1 = COBRE_Test_1.reshape(72, 1)

COBRE_Test = COBRE_Test[:, 3:]

COBRE_Test = np.concatenate([COBRE_Test_1, COBRE_Test], axis=1)

np.set_printoptions(suppress=True)

for i in range(0, 72):
    COBRE_Test[:, -1:][i] = COBRE_Test[:, -1:][i]**(1.0/4.0)
    
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(COBRE_Test)
COBRE_Test_scale = pd.DataFrame(scaler.transform(COBRE_Test))
COBRE_Test_scale = COBRE_Test_scale.values

X_COBRE_test = COBRE_Test_scale 
y_COBRE_test = COBRE_test_data['Age'].values

print(X_COBRE_test)
```

    [[ 2.03540098  1.10644625 -0.42796554 ...  0.3215593   0.15829241
      -1.1574305 ]
     [-0.49130368 -0.20157615 -0.83723456 ... -0.86573658 -0.69686166
      -0.74851488]
     [-0.49130368  0.26458077  0.21798917 ...  0.42949529  0.70839152
      -0.72587634]
     ...
     [-0.49130368  1.53781534 -0.85202741 ...  0.74430859 -0.62684905
      -0.17524084]
     [-0.49130368  0.43851992  1.48031288 ...  0.50145262 -0.20177246
      -0.47141783]
     [-0.49130368  1.38474889  0.69629151 ...  2.81757905  2.40869786
       0.33547578]]
    

### predict age of COBRE_test data using trained model


```python
y_pred_COBRE_test, y_pred_COBRE_test_std = model.predict(X_COBRE_test, return_std=True)

for i in range(72):
    y_pred_COBRE_test[i] = round(y_pred_COBRE_test[i])
    
print(y_pred_COBRE_test)
```

    [43. 34. 47. 46. 49. 61. 58. 56. 38. 37. 63. 52. 42. 45. 69. 45. 45. 42.
     42. 37. 45. 54. 50. 68. 59. 57. 50. 59. 34. 47. 43. 49. 45. 66. 42. 36.
     43. 73. 42. 59. 55. 40. 50. 44. 59. 54. 38. 58. 47. 58. 37. 42. 41. 44.
     48. 52. 45. 58. 49. 46. 42. 44. 60. 50. 49. 68. 64. 58. 34. 34. 48. 37.]
    

### export predict age of COBRE_test data to xlsx


```python
COBRE_test_age = pd.DataFrame({'ID': COBRE_test_data["ID"], 'Age': y_pred_COBRE_test})

#COBRE_test_age.to_excel('COBRE_test_brain_age_submission_소프트웨어융합학과_2020105742_한지훈_최종.xlsx', index = False)
```


### 프로젝트 요약
1. 주어진 feature들로 뇌 나이 예측하는 Gaussian Process Regression 모델 train
2. 두 관점의 시각화 통해 예측 결과 분석
3. train된 모델로 test data(IXI test data set, COBRE test data set)의 뇌 나이 예측  

### 느낀 점
1. 수업에서 배운 machine learning 내용들을 적용할 수 있어서 뿌듯했다.
2. data science의 한 cycle을 경험해볼 수 있었던 좋은 기회

### 프로젝트 보고서
<a href="/images/files/프로젝트 최종 보고서.doc">style.doc</a>