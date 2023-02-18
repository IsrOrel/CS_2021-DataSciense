
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


# df1.to_csv('AfterCleaning.csv', index=False) 
df=pd.read_csv("AfterCleaning1.csv")
df1=df.copy()
df1['WindDirection'] = pd.to_numeric(df1['WindDirection'], errors='coerce')
df1['WindSpeed'] = pd.to_numeric(df1['WindSpeed'], errors='coerce')

df1['DateandTime'] = pd.to_datetime(df1['DateandTime'], format='%d/%m/%Y %H:%M')

median_values = []

def fill_wind_direction_missing_values(column_to_fill,year_to_fill, years_to_take, start_month, end_month):
    format = '%m/%d/%Y'
    end_month =  end_month + 12 if  start_month > end_month else end_month
    for m in range(start_month, end_month + 1):
        effective_month = m if m <= 12 else (m % 12)
        median_values = []
        for y in years_to_take:
            start = datetime.datetime(y, effective_month, 1)
            end = start + relativedelta(months=+1)
            print(f"(df1['DateandTime'] >= {start.strftime(format)}) & (df1['DateandTime'] < {end.strftime(format)})")
            mask = (df1['DateandTime'] >= start.strftime(format)) & (df1['DateandTime'] < end.strftime(format))
            median_values.append(df1[mask][column_to_fill].median())
        median = statistics.median(median_values)
        fill_start = datetime.datetime(year_to_fill if effective_month <= 12 else year_to_fill + 1, effective_month, 1)
        fill_end = fill_start + relativedelta(months=+1)
        mask = (df1['DateandTime'] >= fill_start.strftime(format)) & (df1['DateandTime'] < fill_end.strftime(format))
        print(f"(df1['DateandTime'] >= {fill_start.strftime(format)}) & (df1['DateandTime'] < {fill_end.strftime(format)}")
        df1.loc[mask, column_to_fill] = df1[mask][column_to_fill].fillna(median)
        df1[mask].info()


fill_wind_direction_missing_values('WindDirection',2018, [2020,2021], 11, 12)
fill_wind_direction_missing_values('WindDirection',2019, [2021,2022], 1, 3)
fill_wind_direction_missing_values('WindSpeed',2018, [2020,2021], 11, 12)
fill_wind_direction_missing_values('WindSpeed',2019, [2021,2022], 1, 3)
df1.info()
df1.WindDirection.shape
df1.to_csv('AfterCleaning_V2.csv', index=False)

df=pd.read_csv("AfterCleaning_V2.csv")
df1=df.copy()
df=pd.read_csv("AfterCleaning_V2.csv")
df1=df.copy()
df1['WindDirection'] = pd.to_numeric(df1['WindDirection'], errors='coerce')
df1['WindSpeed'] = pd.to_numeric(df1['WindSpeed'], errors='coerce')
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

sns.boxplot(x="OpenOrClose", y="Temperature", data=df1,whis=3)
plt.show()

sns.boxplot(x="OpenOrClose", y="CmSnow", data=df1)
plt.show()

sns.boxplot(x="OpenOrClose", y="WindDirection", data=df1)
plt.show()

sns.boxplot(x="OpenOrClose", y="WindSpeed", data=df1, whis=15)
plt.show()

cols = ["Temperature", "CmSnow", "WindSpeed", "WindDirection", "OpenOrClose"]
df = df1[cols]

sns.pairplot(df, hue="OpenOrClose")
plt.show()

df1["OpenOrClose"] = df1["OpenOrClose"].map({"Open": 1, "Close": 0})
cols = ["WindSpeed", "WindDirection"]
heatmap_data = df1.pivot_table(index=pd.cut(df1["WindSpeed"], bins=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, np.inf], include_lowest=True),
                               columns=pd.cut(df1["WindDirection"], bins=[0, 45, 90, 135, 180, 225, 270, 315, 360], include_lowest=True),
                               values="OpenOrClose", aggfunc="mean")
heatmap_data = heatmap_data.apply(lambda x: np.where(x >= 0.5, 1, 0))
sns.heatmap(heatmap_data, cmap="Spectral",annot=True, fmt=".2f")
plt.title("Heatmap of WindSpeed and WindDirection with Binary Color Scale")
plt.show()

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
param_grid = {'n_neighbors': [3, 7, 9, 11]}
model = KNeighborsClassifier()
scoring = make_scorer(f1_score)
grid = GridSearchCV(model, param_grid, scoring=scoring, refit=True, verbose=2)
grid.fit(X_train, y_train)
best_K = grid.best_params_['n_neighbors']
best_f1_val = grid.best_score_
print(best_f1_val)

param_grid = {'n_estimators': [11, 51, 71]}
model = RandomForestClassifier()
scoring = make_scorer(f1_score, average='micro')
grid = GridSearchCV(model, param_grid, scoring=scoring, refit=True, verbose=2)
grid.fit(X_train, y_train)
best_num_estimators = grid.best_params_['n_estimators']
best_f1_val_rf = grid.best_score_
print(best_f1_val_rf)



