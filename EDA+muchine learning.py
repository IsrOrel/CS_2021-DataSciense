###Imports we use on EDA and muchine learning
import datetime
from pprint import pprint
from time import sleep
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import statistics
import seaborn as sns
from dateutil.relativedelta import relativedelta
import sklearn
from sklearn import preprocessing, linear_model, model_selection
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import neighbors, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,make_scorer
###Here we read the file and convert the wind columns to numreic one
###It help us in the EDA procces 
df=pd.read_csv("AfterCleaning_V2.csv")
df1=df.copy()
df1['WindDirection'] = pd.to_numeric(df1['WindDirection'], errors='coerce')
df1['WindSpeed'] = pd.to_numeric(df1['WindSpeed'], errors='coerce')
###In here we are define a figure and insert 8 subplot 
###Each subplot is a graph of a histogram or bar plot of the columns
###That help us to see the distribution of each column
fig=plt.figure(figsize=(12,8))
fig1=fig.add_subplot(3,3,1)
fig2=fig.add_subplot(3,3,2)
fig3=fig.add_subplot(3,3,3)
fig4=fig.add_subplot(3,3,4)
fig5=fig.add_subplot(3,3,5)
fig6=fig.add_subplot(3,3,6)
fig7=fig.add_subplot(3,3,7)
fig8=fig.add_subplot(3,3,8)

fig1.hist(df1.Temperature,bins=50)
fig1.set_title("Temperature")
fig1.set_xlabel("Temperature")
fig1.set_ylabel("Frequency")

fig2.hist(df1.WetTemperature,bins=50)
fig2.set_title("WetTemperature")
fig2.set_xlabel("WetTemperature")
fig2.set_ylabel("Frequency")

fig3.hist(df1.DewPointTemperature,bins=50)
fig3.set_title("DewPointTemperature")
fig3.set_xlabel("DewPointTemperature")
fig3.set_ylabel("Frequency")

fig4.hist(df1.RelativeHumidity,bins=50)
fig4.set_title("RelativeHumidity")
fig4.set_xlabel("RelativeHumidity")
fig4.set_ylabel("Frequency")
 
sns.kdeplot(df1['WindDirection'], color='#016604', shade=True, ax=fig5)
fig5.set_title('WindDirection')

sns.kdeplot(df1['WindSpeed'], color='red', shade=True, ax=fig6)
fig6.set_title('WindSpeed')

fig6.hist(df1.WindSpeed,bins=50)
fig6.set_title("WindSpeed")
fig6.set_xlabel("WindSpeed")
fig6.set_ylabel("Frequency")

sns.countplot(x='OpenOrClose', data=df1, ax=fig7, color='#3A9CFD')

fig7.set_title('OpenOrClose')
fig7.set_xlabel('OpenOrClose')
fig7.set_ylabel('Count')


fig8.hist(df1.CmSnow,bins=50)
fig8.set_title("CmSnow")
fig8.set_xlabel("CmSnow")
fig8.set_ylabel("Frequency")
plt.tight_layout()
fig.show()
###Here we show a Boxplot of the column "OpenOrClose" with the column "Temperature"
sns.boxplot(x="OpenOrClose", y="Temperature", data=df1,whis=3)
plt.show()
###Here we show a Boxplot of the column "OpenOrClose" with the column "CmSnow"
sns.boxplot(x="OpenOrClose", y="CmSnow", data=df1)
plt.show()
###Here we show a Boxplot of the column "OpenOrClose" with the column "WindDirection"
sns.boxplot(x="OpenOrClose", y="WindDirection", data=df1)
plt.show()
###Here we show a Boxplot of the column "OpenOrClose" with the column "WindSpeed"
sns.boxplot(x="OpenOrClose", y="WindSpeed", data=df1, whis=15)
plt.show()
###Pair plot pf the columns "Temperature", "CmSnow", "WindSpeed", "WindDirection", "OpenOrClose" with the column "OpenOrClose"
cols = ["Temperature", "CmSnow", "WindSpeed", "WindDirection", "OpenOrClose"]
df = df1[cols]
sns.pairplot(df, hue="OpenOrClose")
plt.show()

###Heat map of "WindSpeed", "WindDirection" with "OpenOrClose"
df1["OpenOrClose"] = df1["OpenOrClose"].map({"Open": 1, "Close": 0})
cols = ["WindSpeed", "WindDirection"]
heatmap_data = df1.pivot_table(index=pd.cut(df1["WindSpeed"], bins=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, np.inf], include_lowest=True),
                               columns=pd.cut(df1["WindDirection"], bins=[0, 45, 90, 135, 180, 225, 270, 315, 360], include_lowest=True),
                               values="OpenOrClose", aggfunc="mean")
heatmap_data = heatmap_data.apply(lambda x: np.where(x >= 0.5, 1, 0))
sns.heatmap(heatmap_data, cmap="Spectral",annot=True, fmt=".2f")
plt.title("Heatmap of WindSpeed and WindDirection with Binary Color Scale")
plt.show()
###Heat map of "WindSpeed", "CmSnow" with "OpenOrClose"
cols = ["WindSpeed", "CmSnow"]
heatmap_data = df1.pivot_table(index=pd.cut(df1["WindSpeed"], bins=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, np.inf], include_lowest=True),
                               columns=pd.cut(df1["CmSnow"], bins=[0, 1, 2, 3,4, np.inf], include_lowest=True),
                               values="OpenOrClose", aggfunc="mean")
heatmap_data = heatmap_data.apply(lambda x: np.where(x >= 0.5, 1, 0))
sns.heatmap(heatmap_data, cmap="Spectral",annot=True, fmt=".2f")
plt.title("Heatmap of WindSpeed and CmSnow with Binary Color Scale")
plt.show()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
xs = df1['Temperature']
ys = df1['CmSnow']
zs = df1['OpenOrClose']
colors = ['r' if x == 1 else 'b' for x in zs]
ax.scatter(xs, ys, zs, c=colors)
ax.set_xlabel('Temperature')
ax.set_ylabel('CmSnow')
ax.set_zlabel('OpenOrClose')
plt.show()
###In the code below we split the "DateandTime" column to a 3 column:"Years","Month","Day","Hour"
###we do this because the muchine learining models cant work with a strings
df1['DateandTime'] = pd.to_datetime(df1['DateandTime'], format="%Y-%m-%d %H:%M:%S")
years = []
months = []
days = []
hours = []
for timestamp in df1["DateandTime"]:
    years.append(timestamp.year)
    months.append(timestamp.month)
    days.append(timestamp.dayofweek)
    hours.append(timestamp.hour)
df1['Year'] = years
df1['Month'] = months
df1['Day'] = days
df1['Hour'] = hours
###Here we are crate an logistic regression model to predict the opening days of Mount Hermon 
df1 = df1.drop(['DateandTime'], axis=1)
train=df1.columns[df1.columns != "OpenOrClose"]
target="OpenOrClose"
X = df1[train]
y = df1[target]
X_train,X_test,y_train,y_test=train_test_split(X ,y ,test_size=0.2 ,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
df2=pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
print(df2)
###Here we are crate an knn model to predict the opening days of Mount Hermon and see if the accuracy batter then the logistic regression model
param_grid = {'n_neighbors': [3, 7, 9, 11]}
model = KNeighborsClassifier()
scoring = make_scorer(f1_score)
grid = GridSearchCV(model, param_grid, scoring=scoring, refit=True, verbose=2)
grid.fit(X_train, y_train)
best_K = grid.best_params_['n_neighbors']
best_f1_val = grid.best_score_
print(best_f1_val)
###Here we are crate an random forest model to predict the opening days of Mount Hermon and see if the accuracy batter then the others models
param_grid = {'n_estimators': [11, 51, 71]}
model = RandomForestClassifier()
scoring = make_scorer(f1_score, average='micro')
grid = GridSearchCV(model, param_grid, scoring=scoring, refit=True, verbose=2)
grid.fit(X_train, y_train)
best_num_estimators = grid.best_params_['n_estimators']
best_f1_val_rf = grid.best_score_
print(best_f1_val_rf)