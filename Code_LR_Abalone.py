#import libraries
from os import system
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns

#clear screen
system ("clear")

#load data
df = pd.read_csv('abalone.data')
df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']

#insert a bias term & remove categorical feature
df.insert(0, "Bias", 1)
df = df[['Bias', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']]

#manually split data into training & testing sets
train_size = int(0.8 * len(df))
train_dataset = df[:train_size]
test_dataset = df[train_size:]

#segregate datasets into input and output
X_train, Y_train = train_dataset.iloc[:, :-1], train_dataset['Rings']
X_test, Y_test = test_dataset.iloc[:, :-1], test_dataset['Rings']

#define function for metrics
def metric(Y_true, Y_pred):
    print('MSE: %.2f' % metrics.mean_squared_error(Y_true, Y_pred))
    print('MAE: %.2f' % metrics.mean_absolute_error(Y_true, Y_pred))
    print('R_Squared: %.2f' % metrics.r2_score(Y_true, Y_pred))

#define function for plots
def plots(Y_true, Y_pred, dataset):

    fig1 = plt.figure(figsize=(12,5))
    
    ax1 = fig1.add_subplot(121)
    sns.scatterplot(x=Y_true, y=Y_pred, ax=ax1, color='g')
    ax1.plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()], 'r-')
    ax1.set_title(f'Actual Vs Predicted ({dataset})')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    ax2 = fig1.add_subplot(122)
    sns.histplot((Y_true - Y_pred), ax=ax2, color='b')
    ax2.axvline((Y_true - Y_pred).mean(), color='r', linestyle='--')
    ax2.set_title(f'Residual Plot ({dataset})')
    plt.xlabel("Rings")
    plt.show()

#OLS method
w_ols = np.matmul(np.linalg.inv(np.matmul(X_train.T, X_train)), np.matmul(X_train.T, Y_train))
print("OLS Parameters:", w_ols)

#make predictions
Y_pred_train = np.matmul(X_train, w_ols)
Y_pred_test = np.matmul(X_test, w_ols)

#print metrics
metric(Y_train, Y_pred_train)
metric(Y_test, Y_pred_test)

#plots
plots(Y_train, Y_pred_train, 'OLS - Train')
plots(Y_test, Y_pred_test, 'OLS - Test')

#SGDRegressor
#remove the bias term from the train & test features
X_train_SGD = train_dataset.iloc[:, 1:-1]
X_test_SGD = test_dataset.iloc[:, 1:-1]

#define model
model = SGDRegressor(random_state=42)

#train the model
model.fit(X_train_SGD, Y_train)

#print parameters
print('SGDRegressor parameters:', model.intercept_, model.coef_)

#make predictions
Y_pred_train_SGD = model.predict(X_train_SGD)
Y_pred_test_SGD = model.predict(X_test_SGD)

#print metrics
metric(Y_train, Y_pred_train_SGD)
metric(Y_test, Y_pred_test_SGD)

#plots
plots(Y_train, Y_pred_train_SGD, 'SGD - Train')
plots(Y_test, Y_pred_test_SGD, 'SGD - Test')
