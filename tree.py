import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree, model_selection
import pandas as pd


names = ["Liczba","Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","Species"]
data = pd.read_csv("Graduate - IRISES dataset (2019-06).csv", skiprows=1, names = names, delimiter = "|" )
df = data.drop(['Liczba'], axis=1)
print(df)

# print(data.isnull().any())
len = len(data["Petal.Length"])

for i in range(1,len):
    df["Sepal.Length"][i] = float((df["Sepal.Length"][i]))
    df["Sepal.Width"][i] = float((df["Sepal.Width"][i]))
    df["Petal.Length"][i] = float((df["Petal.Length"][i]))
    df["Petal.Width"][i] = float((df["Petal.Width"][i]).replace(",","."))

    if df["Sepal.Length"][i]=="NA":
        df["Sepal.Length"][i]= (df["Sepal.Length"]).mean()

X = df[["Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"]]
Y = df[['Species']]

#lista parametrów do drzewa decyzyjnego może być dowolna, u mnie zamyka się w przedziale od 1 do 5
param_grid = {'criterion': ['gini', 'entropy'],
    'max_depth': np.arange(1,5),
              'max_features': np.arange(1,5),
              }

classifier = tree.DecisionTreeClassifier()

grid_classifier = model_selection.GridSearchCV(classifier, param_grid, return_train_score=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=100)

X_walidacja, X_koncowe, Y_walidacja, Y_koncowe = train_test_split(X_test, Y_test, test_size = 0.5, random_state=10)

grid_classifier.fit(X_train,Y_train)

print(grid_classifier.best_estimator_)
print("best :", grid_classifier.best_params_)



