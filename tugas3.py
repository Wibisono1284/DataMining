import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset =  pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)
[['France' 44.0 72000.0]
 ['Spain' 27.0 48000.0]
 ['Germany' 30.0 54000.0]
 ['Spain' 38.0 61000.0]
 ['Germany' 40.0 nan]
 ['France' 35.0 58000.0]
 ['Spain' nan 52000.0]
 ['France' 48.0 79000.0]
 ['Germany' 50.0 83000.0]
 ['France' 37.0 67000.0]]

 print(y)
 ['NO' 'YES' 'NO' 'NO' 'YES' 'YES' 'NO' 'YES' 'NO' 'YES']

 from sklearn.impute import SimpleImputer
 imputer = SimpleImputer (missing_values=np.nan, stategy='mean')
 imputer.fit(x[:, 1:3])
 x[:, 1:3] = imputer.transform(x[:, 1:3])

 print(x)
 [['France' 44.0 72000.0]
  ['Spain' 27.0 48000.0]
  ['Germany' 30.0 54000.0]
  ['Spain' 38.0 61000.0]
  ['Germany' 40.0 63777.77777777778]
  ['France' 35.0 58000.0]
  ['Spain' 38.77777777778 52000.0]
  ['France' 48.0 79000.0]
  ['Germany' 50.0 83000.0]
  ['France' 37.0 67000.0]]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncorder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

 print(x)
 [[1.0 0.0 0.0 44.0 72000.0]
  [0.0 0.0 1.0 27.0 48000.0]
  [0.0 1.0 0.0 30.0 54000.0]
  [0.0 0.0 1.0 38.0 61000.0]
  [0.0 1.0 0.0 40.0 63777.77777777778]
  [1.0 0.0 0.0 35.0 58000.0]
  [0.0 0.0 1.0 38.77777777778 52000.0]
  [1.0 0.0 0.0 48.0 79000.0]
  [0.0 1.0 0.0 50.0 83000.0]
  [1.0 0.0 0.0 37.0 67000.0]]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)
[0 1 0 0 1 1 0 1 0 1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

print(x_train)
[[0.0 0.0 1.0 38.77777777778 52000.0]
 [0.0 1.0 0.0 40.0 63777.777777777778]
 [1.0 0.0 0.0 44.0 72000.0]
 [0.0 0.0 1.0 38.0 61000.0]
 [0.0 0.0 1.0 27.0 48000.0]
 [1.0 0.0 0.0 48.0 79000.0]
 [0.0 1.0 0.0 50.0 83000.0]
 [1.0 0.0 0.0 35.0 58000.0]]

print(x_test)
[[0.0 1.0 0.0 30.0 54000.0]
 [1.0 0.0 0.0 37.0 67000.0]]

print(y_train)
[0 1 0 0 1 1 0 1]

print(y_test)
[0 1]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)
[[0.0 0.0 1.0 -0.19159184384578545 -1.0781259408412425]
 [0.0 1.0 0.0 -0.014117293757657777 -0.07013167641635372]
 [1.0 0.0 0. 0.566708506533324 0.633562432710455]
 [0.0 0.0 1.0 -0.30453019390224867 -0.30786617274297867]
 [0.0 0.0 1.0 -1.9018011447007988 -1.420463615551582]
 [1.0 0.0 0.0 1.1475343068237058 1.232653363453549]
 [0.0 1.0 0.0 1.4379472069688968 1.5749910381638885]
 [1.0 0.0 0.0 -0.7401495441200351 -0.5646194287757332]]
print(x_test)
[[0.0 1.0 0.0 -14661817944830124 -0.9069571034860727]
 [1.0 0.0 0.0 -0.44973664397484414 0.2056403393225306]]