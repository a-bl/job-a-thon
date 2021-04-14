import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, SGDRegressor, ARDRegression, ElasticNet, LassoCV, HuberRegressor
from sklearn.linear_model import PoissonRegressor, TweedieRegressor, GammaRegressor, PassiveAggressiveRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import xgboost
import catboost
from lightgbm import LGBMRegressor

import warnings

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# Data Inspection
train = pd.read_csv("train_v9rqX0R.csv")
test = pd.read_csv("test_AbJTz2l.csv")

print('Shape of Train Set: {}\nShape of Test Set: {}'.format(train.shape, test.shape))

# ratio of null values
print(train.isnull().sum() / train.shape[0] * 100)
print(test.isnull().sum() / test.shape[0] * 100)

# categorical features (train)
categorical = train.select_dtypes(include=[np.object])
print("Categorical Features in Train Set: {}".format(categorical.shape[1]))
print('Categorical Features:', categorical.columns)

# numerical features (train)
numerical = train.select_dtypes(include=[np.float64, np.int64])
print("Numerical Features in Train Set: {}".format(numerical.shape[1]))

# categorical features (test)
categorical = test.select_dtypes(include=[np.object])
print("Categorical Features in Test Set: {}".format(categorical.shape[1]))

# numerical features (test)
numerical = test.select_dtypes(include=[np.float64, np.int64])
print("Numerical Features in Test Set: {}".format(numerical.shape[1]))

print(train.describe())
print('**************************************')
print(test.describe())

# Data Cleaning
print(train.isnull().sum())
print(test.isnull().sum())

# Item Weight
plt.figure(figsize=(8, 5))
sns.boxplot('Item_Weight', data=train)
plt.gcf()
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot('Item_Weight', data=test)
plt.gcf()
plt.show()


# Imputing with Mean
train['Item_Weight'] = train['Item_Weight'].fillna(train['Item_Weight'].mean())
test['Item_Weight'] = test['Item_Weight'].fillna(test['Item_Weight'].mean())

print('{} missing values in Train'.format(train['Item_Weight'].isnull().sum()))
print('{} missing values in Test'.format(test['Item_Weight'].isnull().sum()))

print('{} missing values in Test'.format(train['Outlet_Size'].isnull().sum()))
print('{} missing values in Test'.format(test['Outlet_Size'].isnull().sum()))

print(train['Outlet_Size'].value_counts())
print('******************************************')
print(test['Outlet_Size'].value_counts())

# Imputing with Mode
train['Outlet_Size'] = train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0])
test['Outlet_Size'] = test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0])

print('{} missing values in Test'.format(train['Outlet_Size'].isnull().sum()))
print('{} missing values in Test'.format(test['Outlet_Size'].isnull().sum()))

print(train.isnull().sum())
print(test.isnull().sum())

# Exploratory Data Analysis

print(train.columns)
print(train.head())

print(train['Item_Fat_Content'].value_counts())

train['Item_Fat_Content'].replace(['low fat', 'LF', 'reg'], ['Low Fat', 'Low Fat', 'Regular'], inplace=True)
test['Item_Fat_Content'].replace(['low fat', 'LF', 'reg'], ['Low Fat', 'Low Fat', 'Regular'], inplace=True)

train['Item_Fat_Content'] = train['Item_Fat_Content'].astype(str)

train['Years_Established'] = train['Outlet_Establishment_Year'].apply(lambda x: 2021 - x)
test['Years_Established'] = test['Outlet_Establishment_Year'].apply(lambda x: 2021 - x)

train['Item_Type_Combined'] = train['Item_Identifier'].apply(lambda x: x[0:2])
train['Item_Type_Combined'] = train['Item_Type_Combined'].map({'FD': 'Food',
                                                               'NC': 'Non-Consumable',
                                                               'DR': 'Drinks'})

test['Item_Type_Combined'] = test['Item_Identifier'].apply(lambda x: x[0:2])
test['Item_Type_Combined'] = test['Item_Type_Combined'].map({'FD': 'Food',
                                                             'NC': 'Non-Consumable',
                                                             'DR': 'Drinks'})
print(train['Item_Type_Combined'].value_counts())

plt.figure(figsize=(8, 5))
sns.countplot('Item_Fat_Content', data=train, palette='summer')
plt.gcf()
plt.show()

plt.figure(figsize=(25, 7))
sns.countplot('Item_Type', data=train, palette='spring')
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot('Outlet_Size', data=train, palette='autumn')
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot('Outlet_Location_Type', data=train, palette='winter')
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot('Outlet_Type', data=train, palette='twilight')
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot('Years_Established', data=train, palette='mako')
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot('Item_Fat_Content', 'Item_Outlet_Sales', data=train, palette='mako')
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter('Item_Visibility', 'Item_Outlet_Sales', data=train)
plt.gcf()
plt.show()

train['Item_Visibility'] = train['Item_Visibility'].replace(0, train['Item_Visibility'].mean())
test['Item_Visibility'] = test['Item_Visibility'].replace(0, test['Item_Visibility'].mean())

plt.figure(figsize=(8,5))
plt.scatter('Item_Visibility', 'Item_Outlet_Sales',data=train)
plt.xlabel('Item Outlet Sales')
plt.ylabel('Item Visibility')
plt.gcf()
plt.show()

plt.figure(figsize=(10, 8))
sns.barplot(y='Item_Type', x='Item_Outlet_Sales', data=train, palette='flag')
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter('Item_Outlet_Sales', 'Item_MRP', data=train)
plt.xlabel('Item MRP')
plt.ylabel('Item Outlet Sales')
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot('Outlet_Size', 'Item_Outlet_Sales', data=train, palette='winter')
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot('Outlet_Location_Type', 'Item_Outlet_Sales', data=train, palette='plasma')
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot('Years_Established', 'Item_Outlet_Sales', data=train, palette='viridis')
plt.gcf()
plt.show()

# Multivariable Analysis
plt.figure(figsize=(25, 5))
sns.barplot('Item_Type', 'Item_Outlet_Sales', hue='Item_Fat_Content', data=train, palette='mako')
plt.legend()
plt.gcf()
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot('Outlet_Location_Type', 'Item_Outlet_Sales', hue='Outlet_Type', data=train, palette='magma')
plt.legend()
plt.gcf()
plt.show()


# Feature Engineering
print(train.head())

# Encoding our labels
le = LabelEncoder()
train['Outlet'] = le.fit_transform(train['Outlet_Identifier'])
test['Outlet'] = le.fit_transform(test['Outlet_Identifier'])
var_mod = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Item_Type_Combined', 'Outlet_Type', 'Outlet']
le = LabelEncoder()
for i in var_mod:
    train[i] = le.fit_transform(train[i])
    test[i] = le.fit_transform(test[i])

train = pd.get_dummies(train, columns=['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type',
                                       'Item_Type_Combined', 'Outlet'])
test = pd.get_dummies(test, columns=['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type',
                                     'Item_Type_Combined', 'Outlet'])

print(train.dtypes)
# print(train.columns)
# print(test.columns)
print(train.head())

warnings.filterwarnings('ignore')
train.drop(columns=['Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)
test.drop(columns=['Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)

print('train shape:', train.shape)
print('test shape:', test.shape)

# Seperate Features and Target
X = train.drop(columns=['Item_Outlet_Sales', 'Item_Identifier', 'Outlet_Identifier'], axis=1)
y = train['Item_Outlet_Sales']
test = test.drop(columns=['Item_Identifier', 'Outlet_Identifier'], axis=1)

# 20% data as validation set
seed = 7
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=seed)

# Model Building

####################################################################################

# features = X.columns
# LR = LinearRegression(normalize=True)
# LR.fit(X_train, y_train)
# y_pred = LR.predict(X_valid)
# coef = pd.Series(LR.coef_, features).sort_values()

# Barplot for coefficients
# plt.figure(figsize=(8, 5))
# sns.barplot(LR.coef_, features)
# plt.gcf()
# plt.show()

# MSE = metrics.mean_squared_error(y_valid, y_pred)
# rmse = np.sqrt(MSE)
# print("Root Mean Squared Error:", rmse)

# Submission
#
# submission = pd.read_csv('sample_submission_8RXa3c6.csv')
# final_predictions = LR.predict(test)
# submission['Item_Outlet_Sales'] = final_predictions
# #only positive predictions for the target variable
# submission['Item_Outlet_Sales'] = submission['Item_Outlet_Sales'].apply(lambda x: 0 if x < 0 else x)
# submission.to_csv('my_submission.csv', index=False)

#####################################################################################################

# algos = [
#          # LinearRegression(normalize=True),
#          # Ridge(alpha=0.05, normalize=True),
#          # Lasso(),
#          # LassoCV(alphas=[1, 0.1, 0.001, 0.0005]),
#          # SGDRegressor(max_iter=1000, loss='huber', tol=1e-3, average=True),
#          # ARDRegression(),
#          # ElasticNet(),
#          # HuberRegressor(),
#          # PoissonRegressor(),
#          # TweedieRegressor(),
#          # GammaRegressor(),
#          # PassiveAggressiveRegressor(),
#          # SVR(),
#          # RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=43, verbose=0, warm_start=False),
#          LGBMRegressor(num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True, importance_type='split'),
#          XGBRegressor(),
#          # GradientBoostingRegressor(alpha=0.999, criterion='friedman_mse', init=None, learning_rate=0.005, loss='huber', max_depth=8, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=60, min_samples_split=1200, min_weight_fraction_leaf=0.0, n_estimators=1200, n_iter_no_change=None, random_state=10, subsample=0.8, tol=0.0001, validation_fraction=0.1, verbose=0, warm_start=True),
#          GradientBoostingRegressor(alpha=0.9, learning_rate=0.05, max_depth=5, min_samples_leaf=3, min_samples_split=2, n_estimators=80, random_state=30),
#          # KNeighborsRegressor(),
#          # DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=0, splitter='best'),
#          # ExtraTreesRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None),
#          # MLPRegressor(),
#          # KernelRidge()]
#          ]
# # names = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'LassoCV', 'SGD Regression', 'ARDRegression', 'ElasticNet',
# #          'Huber Regression', 'PoissonRegressor', 'TweedieRegressor', 'GammaRegressor', 'PassiveAgressiveRegressor',
# #          'SVR', 'RandomForestRegressor', 'LGBMRegressor', 'XGBRegressor', 'GradientBoostingRegressor',
# #          'K Neighbors Regressor', 'Decision Tree Regressor', 'ExtraTreesRegressor', 'MLPRegressor', 'KernelRidge']
# names = ['LGBM Regressor', 'XGB Regressor', 'Gradient Boosting Regressor']
# rmse_list = []
#
# for name in algos:
#     model = name
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_valid)
#     MSE = metrics.mean_squared_error(y_valid, y_pred)
#     rmse = np.sqrt(MSE)
#     rmse_list.append(rmse)
#
# evaluation = pd.DataFrame({'Model': names,
#                            'RMSE': rmse_list})
# print(evaluation)
# print(evaluation['RMSE'].min())
#
# submission = pd.read_csv('sample_submission_8RXa3c6.csv')
# # model = GradientBoostingRegressor(alpha=0.9,learning_rate=0.05, max_depth=5, min_samples_leaf=3, min_samples_split=2, n_estimators=80, random_state=30)
# for name in algos:
#     model = name
#     model.fit(X, y)
#     final_predictions = model.predict(test)
#     submission['Item_Outlet_Sales'] = final_predictions
#     #only positive predictions for the target variable
#     submission['Item_Outlet_Sales'] = submission['Item_Outlet_Sales'].apply(lambda x: 0 if x < 0 else x)
#     name = str(name)
#     submission.to_csv('my_submission_test_{}.csv'.format(name.split('(')[0]), index=False)

########################################################################################################

######## Gradient Boosting Regressor ########
# print('GradientBoostingRegressor')
#
# # Standard scaling
# scaler = preprocessing.StandardScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)
# # param_grid = dict(n_estimators=np.array([50, 60, 70, 80, 100]))
# param_grid = {'n_estimators': [50, 60, 70, 80, 100],
#               'alpha': [0.9],
#               'learning_rate': [0.03, 0.05],
#               'max_depth': [5, 6],
#               'min_samples_leaf': [3],
#               'min_samples_split': [2]
# }
#
# # Use GridSearchCV to find best estimator & RMS (root mean square)to calculate score
# num_folds = 10
# RMS = 'neg_mean_squared_error'
# model = GradientBoostingRegressor(random_state=seed)
# kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=RMS, cv=kfold, verbose=2)
# grid_result = grid.fit(rescaledX, y_train)
#
# print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print('{} ({}) with: {}'.format(mean, stdev, param))
#
# # Make predictions on validation dataset using best estimstor = 60
# # prepare the model
# scaler = preprocessing.StandardScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)
# model = GradientBoostingRegressor(alpha=0.9, learning_rate=0.05, max_depth=5, min_samples_leaf=3, min_samples_split=2,
#                                   n_estimators=80, random_state=seed)
# model.fit(rescaledX, y_train)
# # transform the validation dataset
# rescaledValidationX = scaler.transform(X_valid)
# predictions = model.predict(rescaledValidationX)
# print('RMSE = ', mean_squared_error(y_valid, predictions))
#
# Stanrdard scaling of the test dataset
# rescaled_test = scaler.transform(test)
# # Predict on the test dataset
# predicted_prices = model.predict(rescaled_test)
#
# # PRepare submission file
# submission = pd.read_csv('sample_submission_8RXa3c6.csv')
# submission['Item_Outlet_Sales'] = predicted_prices
# submission['Item_Outlet_Sales'] = submission['Item_Outlet_Sales'].apply(lambda x: 0 if x < 0 else x)
# submission.to_csv('submission_GBR1.csv', index=False)
# submission.head()

##################

######## XGBRegressor ########
# print('XGBRegressor')
#
# # Standard scaling
# scaler = preprocessing.StandardScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)
# param_grid = {'n_estimators': [70, 100, 120, 150, 200],
#               'nthread': [4],  # when use hyperthread, xgboost may become slower
#               'objective': ['reg:squarederror'],
#               'learning_rate': [0.03, 0.05],  # so called `eta` value
#               'max_depth': [5, 6],
#               'min_child_weight': [4],
#               'subsample': [0.7],
#               'colsample_bytree': [0.7]
#               }
#
# # Use GridSearchCV to find best estimator & RMS (root mean square)to calculate score
# num_folds = 10
# RMS = 'neg_root_mean_squared_error'
# model = xgboost.XGBRegressor(random_state=seed)
# kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=RMS, cv=kfold, verbose=2)
# grid_result = grid.fit(rescaledX, y_train)
#
# print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print('{} ({}) with: {}'.format(mean, stdev, param))
# 'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': 5, 'min_child_weight': 4, 'n_estimators': 150, 'nthread': 4, 'objective': 'reg:squarederror', 'subsample': 0.7
#
# # Make predictions on validation dataset using best estimstor = 80
# # prepare the model
# scaler = preprocessing.StandardScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)
# model = xgboost.XGBRegressor(colsample_bytree=0.7, learning_rate=0.03, max_depth=5, min_child_weight=4,
#                              n_estimators=150, nthread=4, objective='reg:squarederror', subsample=0.7)
# model.fit(rescaledX, y_train, eval_metric='rmse')
# # transform the validation dataset
# rescaledValidationX = scaler.transform(X_valid)
# predictions = model.predict(rescaledValidationX)
# print('RMSE = ', mean_squared_error(y_valid, predictions))
#
# # Stanrdard scaling of the test dataset
# rescaled_test = scaler.transform(test)
# # Predict on the test dataset
# predicted_prices = model.predict(rescaled_test)
#
# # PRepare submission file
# submission = pd.read_csv('sample_submission_8RXa3c6.csv')
# submission['Item_Outlet_Sales'] = predicted_prices
# submission['Item_Outlet_Sales'] = submission['Item_Outlet_Sales'].apply(lambda x: 0 if x < 0 else x)
# submission.to_csv('submission_XGBR1.csv', index=False)
# submission.head()

#############################################

######## LGBMRegressor ########
# print('LGBMRegressor')
#
# Standard scaling
# scaler = preprocessing.StandardScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)
# param_grid = {'n_estimators': [50, 60, 70],
#               'colsample_bytree': [0.7],
#               'max_depth': [5, 6],
#               'num_leaves': [50, 55, 60],
#               'reg_alpha': [1.1, 1.2],
#               'reg_lambda': [1.1, 1.2],
#               'min_split_gain': [0.4],
#               'subsample': [0.7, 0.85],
#               'subsample_freq': [20]
#               }
# # Use GridSearchCV to find best estimator & RMS (root mean square)to calculate score
# num_folds = 10
# RMS = 'neg_mean_squared_error'
# model = LGBMRegressor(random_state=seed, silent=True, metric='None', n_jobs=4)
# kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=RMS, cv=kfold, verbose=2)
# grid_result = grid.fit(rescaledX, y_train)
#
# print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print('{} ({}) with: {}'.format(mean, stdev, param))
#
# # Make predictions on validation dataset using best estimstor = 80
# # prepare the model
# scaler = MinMaxScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)
# model = LGBMRegressor(colsample_bytree=0.7,
#                       max_depth=5,
#                       min_split_gain=0.4,
#                       n_estimators=50,
#                       num_leaves=50,
#                       reg_alpha=1.2,
#                       reg_lambda=1.1,
#                       subsample=0.7,
#                       subsample_freq=20)
# model.fit(rescaledX, y_train, eval_metric='rmse')
# # transform the validation dataset
# rescaledValidationX = scaler.transform(X_valid)
# predictions = model.predict(rescaledValidationX)
# print('RMSE = ', mean_squared_error(y_valid, predictions))
#
# # Stanrdard scaling of the test dataset
# rescaled_test = scaler.transform(test)
# # Predict on the test dataset
# predicted_prices = model.predict(rescaled_test)
#
# # PRepare submission file
# submission = pd.read_csv('sample_submission_8RXa3c6.csv')
# submission['Item_Outlet_Sales'] = predicted_prices
# submission['Item_Outlet_Sales'] = submission['Item_Outlet_Sales'].apply(lambda x: 0 if x < 0 else x)
# submission.to_csv('submission_LGBMR1.csv', index=False)
# submission.head()
################################################
#######CatBoost#######
print('CatBoost')

# # Standard scaling
# scaler = MinMaxScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)
# param_grid = {'depth': [6, 8, 10],
#               'learning_rate': [0.01, 0.05, 0.1],
#               'l2_leaf_reg': [2, 4, 8],
#               'iterations': [30, 50, 100, 150]
#               }
# # Use GridSearchCV to find best estimator & RMS (root mean square)to calculate score
# num_folds = 2
# RMS = 'neg_mean_squared_error'
# model = catboost.CatBoostRegressor(random_state=seed, silent=True, loss_function='RMSE')
# kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=RMS, cv=kfold, verbose=2)
# grid_result = grid.fit(rescaledX, y_train)
#
# print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print('{} ({}) with: {}'.format(mean, stdev, param))
#
# # Make predictions on validation dataset using best estimstor = 80
# # prepare the model
# scaler = MinMaxScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)
# model = catboost.CatBoostRegressor(depth=6, iterations=100, learning_rate=0.05, l2_leaf_reg=4, loss_function='RMSE')
# model.fit(rescaledX, y_train)
# # transform the validation dataset
# rescaledValidationX = scaler.transform(X_valid)
# predictions = model.predict(rescaledValidationX)
# print('RMSE = ', mean_squared_error(y_valid, predictions))
#
# # Stanrdard scaling of the test dataset
# rescaled_test = scaler.transform(test)
# # Predict on the test dataset
# predicted_prices = model.predict(rescaled_test)
#
# # PRepare submission file
# submission = pd.read_csv('sample_submission_8RXa3c6.csv')
# submission['Item_Outlet_Sales'] = predicted_prices
# submission['Item_Outlet_Sales'] = submission['Item_Outlet_Sales'].apply(lambda x: 0 if x < 0 else x)
# submission.to_csv('submission_CBR1.csv', index=False)
# submission.head()
#######################################################################
#######Random_Forest#######
print('RandomRorest')

# # Standard scaling
# scaler = MinMaxScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)
# param_grid = {'n_estimators': [3, 4, 5, 7, 9, 10, 12],
#               'max_depth': [5, 7, 9, 10, 12]
#               }
# # Use GridSearchCV to find best estimator & RMS (root mean square)to calculate score
# num_folds = 2
# RMS = 'neg_mean_squared_error'
# model = RandomForestRegressor(random_state=seed)
# kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=RMS, cv=kfold, verbose=2)
# grid_result = grid.fit(rescaledX, y_train)
#
# print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print('{} ({}) with: {}'.format(mean, stdev, param))

# Make predictions on validation dataset using best estimstor = 80
# prepare the model
scaler = MinMaxScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = RandomForestRegressor(max_depth=5, n_estimators=7, random_state=seed)
model.fit(rescaledX, y_train)
# transform the validation dataset
rescaledValidationX = scaler.transform(X_valid)
predictions = model.predict(rescaledValidationX)
print('RMSE = ', mean_squared_error(y_valid, predictions))

# Stanrdard scaling of the test dataset
rescaled_test = scaler.transform(test)
# Predict on the test dataset
predicted_prices = model.predict(rescaled_test)

# PRepare submission file
submission = pd.read_csv('sample_submission_8RXa3c6.csv')
submission['Item_Outlet_Sales'] = predicted_prices
submission['Item_Outlet_Sales'] = submission['Item_Outlet_Sales'].apply(lambda x: 0 if x < 0 else x)
submission.to_csv('submission_RFR1.csv', index=False)
submission.head()
################################################################
# scaler = MinMaxScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)
# algos = [
#          LinearRegression(fit_intercept=True, normalize=True, copy_X=True),
#          Ridge(alpha=0.05, normalize=True),
#          Lasso(),
#          LassoCV(alphas=[1, 0.1, 0.001, 0.0005]),
#          SGDRegressor(max_iter=1000, loss='huber', tol=1e-3, average=True),
#          ARDRegression(),
#          ElasticNet(),
#          HuberRegressor(),
#          PoissonRegressor(),
#          TweedieRegressor(),
#          PassiveAggressiveRegressor(),
#          SVR(),
#          RandomForestRegressor(criterion='mse', max_depth=7, min_samples_leaf=9, min_samples_split=2, n_estimators=100, random_state=seed),
#          LGBMRegressor(colsample_bytree=0.7, max_depth=5, min_split_gain=0.4, n_estimators=50, num_leaves=50, reg_alpha=1.2, reg_lambda=1.1,subsample=0.7, subsample_freq=20),
#          xgboost.XGBRegressor(colsample_bytree=0.7, learning_rate=0.03, max_depth=5, min_child_weight=4, n_estimators=150, nthread=4, objective='reg:squarederror', subsample=0.7),
#          GradientBoostingRegressor(alpha=0.9, learning_rate=0.05, max_depth=5, min_samples_leaf=3, min_samples_split=2, n_estimators=80, random_state=seed),
#          KNeighborsRegressor(),
#          DecisionTreeRegressor(criterion='mse', max_depth=8, min_samples_leaf=9, min_samples_split=2, random_state=seed, splitter='best'),
#          ExtraTreesRegressor(max_depth=8, min_samples_leaf=8, min_samples_split=2, n_estimators=70, random_state=seed),
#          MLPRegressor()]
#
# names = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'LassoCV', 'SGD Regression', 'ARDRegression', 'ElasticNet',
#          'Huber Regression', 'PoissonRegressor', 'TweedieRegressor', 'PassiveAgressiveRegressor',
#          'SVR', 'RandomForestRegressor', 'LGBMRegressor', 'XGBRegressor', 'GradientBoostingRegressor',
#          'K Neighbors Regressor', 'Decision Tree Regressor', 'ExtraTreesRegressor', 'MLPRegressor']
# # names = ['LGBM Regressor', 'XGB Regressor', 'Gradient Boosting Regressor']
# rmse_list = []
#
# for name in algos:
#     model = name
#     print(model)
#     model.fit(rescaledX, y_train)
#     scaler = MinMaxScaler().fit(X_train)
#     rescaledValidationX = scaler.transform(X_valid)
#     y_pred = model.predict(rescaledValidationX)
#     MSE = metrics.mean_squared_error(y_valid, y_pred)
#     rmse = np.sqrt(MSE)
#     rmse_list.append(rmse)
#
#
# evaluation = pd.DataFrame({'Model': names,
#                            'RMSE': rmse_list})
# print(evaluation)
# print(evaluation['RMSE'].min())
#
# submission = pd.read_csv('sample_submission_8RXa3c6.csv')
# # model = GradientBoostingRegressor(alpha=0.9,learning_rate=0.05, max_depth=5, min_samples_leaf=3, min_samples_split=2, n_estimators=80, random_state=30)
# for name in algos:
#     model = name
#     rescaled_X = scaler.transform(X)
#     model.fit(rescaled_X, y)
#     scaler = MinMaxScaler().fit(X_train)
#     rescaled_test = scaler.transform(test)
#     final_predictions = model.predict(rescaled_test)
#     submission['Item_Outlet_Sales'] = final_predictions
#     #only positive predictions for the target variable
#     submission['Item_Outlet_Sales'] = submission['Item_Outlet_Sales'].apply(lambda x: 0 if x < 0 else x)
#     name = str(name)
#     submission.to_csv('my_submission_{}.csv'.format(name.split('(')[0]), index=False)
