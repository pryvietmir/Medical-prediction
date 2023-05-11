import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier #
import io
from sklearn.metrics import accuracy_score, precision_score,recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
f = pd.read_csv("C:/Users/se-sc/OneDrive/Рабочий стол/stroke_data.csv")
f = f.dropna()
print(f.columns)
df = f.dropna(axis=0)
X = df.iloc[:, 0:10]
Y = df['avg_glucose_level']
# bfeatures = SelectKBest(k=3)
# fit = bfeatures.fit(X,Y)
# ans = pd.DataFrame({"scores": fit.scores_,"name":X.columns}).sort_values(by="scores",ascending=False)
# print(ans)
nwd = pd.DataFrame()
lst = []
#lst = ['heart_disease',"hypertension","smoking_status"]
for i in range(len(lst)):
    nwd[lst[i]] = f[lst[i]]
nwd = nwd.dropna()
# print(f.iloc[5])
# lstq = ['heart_disease: ',"hypertension: ","smoking_status: "]
# lst = []
# dff = pd.DataFrame()
# print("Здравствуйте! Этот код замеряет ваш уровень средний глюкозы уровень ")
# for i in range(0,3):
#     lst.append(float(input(lstq[i])))
# dff = pd.DataFrame([lst])
# model = LinearRegression()
# regression = model.fit(X_train, Y_train)
# predict = regression.predict(X_test)
# dff = pd.DataFrame({'heart_disease': [lst[0]], 'hypertension': [lst[1]],"smoking_status":[lst[2]]})
# vall = model.predict(dff)
# #
from sklearn import preprocessing
from sklearn import utils


X = nwd.iloc[:,1:]
Y = nwd["hypertension"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=15)
knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train, Y_train)
ypred = knn.predict(X_test)

lsti = ["0","1","2"]
lst = []
lstq = ["Ваш средний уровень глюкозы?: ","Есть болезни сердца?: ","Женат/Замужем?: "]
dff = pd.DataFrame()

print("Здравствуйте! Этот код проверяет на наличие гипертонии, пожалуйста, введите следующие данные: ")
for i in range(0,3):
    lst.append(float(input(lstq[i])))
dff = pd.DataFrame([lst])

dff = pd.DataFrame({'avg_glucose_level': [lst[0]], 'heart_disease': [lst[1]],"ever_married":[lst[2]]})
vall = knn.predict(dff)

if vall == 0:
     print("Вы не страдаете гипертонией.")
else:
     print("Вам стоит обратиться к врачу.")
# регрессия по глюкозе в крови