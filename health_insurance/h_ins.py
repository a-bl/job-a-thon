# Importing the Relevant Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor, ARDRegression, ElasticNet, LassoCV, HuberRegressor, PoissonRegressor, TweedieRegressor, GammaRegressor, PassiveAggressiveRegressor
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

import warnings

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# Data Inspection

train = pd.read_csv('train_Df64byy.csv')
test = pd.read_csv('test_YCcRUnU.csv')
print('Shape of Train Set: {}\nShape of Test Set: {}'.format(train.shape, test.shape))

# ratio of null values
print(train.isnull().sum() / train.shape[0] * 100)
print(test.isnull().sum() / test.shape[0] * 100)

# categorical features (train)
categorical = train.select_dtypes(include=[np.object])
print("Categorical Features in Train Set: {}".format(categorical.shape[1]))

# numerical features (train)
numerical = train.select_dtypes(include=[np.float64, np.int64])
print("Numerical Features in Train Set: {}".format(numerical.shape[1]))

# categorical features (test)
categorical = test.select_dtypes(include=[np.object])
print("Categorical Features in Test Set: {}".format(categorical.shape[1]))

# numerical features (test)
numerical = test.select_dtypes(include=[np.float64, np.int64])
print("Numerical Features in Test Set: {}".format(numerical.shape[1]))

print('Categorical Features:', categorical.columns)
print('Numerical Features:', numerical.columns)

print(train.describe())
print('*********************************')
print(test.describe())

# Data Cleaning

print(train.isnull().sum())
print(test.isnull().sum())

# Hoalding_Policy_Type
plt.figure(figsize=(8, 5))
sns.boxplot('Holding_Policy_Type', data=train)
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot('Holding_Policy_Type', data=test)
plt.gcf()
plt.show()

# Imputing with Mean
train['Holding_Policy_Type'] = train['Holding_Policy_Type'].fillna(train['Holding_Policy_Type'].mean())
test['Holding_Policy_Type'] = test['Holding_Policy_Type'].fillna(test['Holding_Policy_Type'].mean())

print('{} missing values in Train'.format(train['Holding_Policy_Type'].isnull().sum()))
print('{} missing values in Test'.format(test['Holding_Policy_Type'].isnull().sum()))

# Health Indicator
print('{} missing values in Test'.format(train['Health Indicator'].isnull().sum()))
print('{} missing values in Test'.format(test['Health Indicator'].isnull().sum()))

print(train['Health Indicator'].value_counts())
print('******************************************')
print(test['Health Indicator'].value_counts())

# Imputing with Mode
train['Health Indicator'] = train['Health Indicator'].fillna(train['Health Indicator'].mode()[0])
test['Health Indicator'] = test['Health Indicator'].fillna(test['Health Indicator'].mode()[0])

print('{} missing values in Test'.format(train['Health Indicator'].isnull().sum()))
print('{} missing values in Test'.format(test['Health Indicator'].isnull().sum()))

print(train.isnull().sum())
print(test.isnull().sum())

# Holding_Policy_Duration
print('{} missing values in Test'.format(train['Holding_Policy_Duration'].isnull().sum()))
print('{} missing values in Test'.format(test['Holding_Policy_Duration'].isnull().sum()))

print(train['Holding_Policy_Duration'].value_counts())
print('******************************************')
print(test['Holding_Policy_Duration'].value_counts())

# Imputing with Mode
train['Holding_Policy_Duration'] = train['Holding_Policy_Duration'].fillna(train['Holding_Policy_Duration'].mode()[0])
test['Holding_Policy_Duration'] = test['Holding_Policy_Duration'].fillna(test['Holding_Policy_Duration'].mode()[0])

print('{} missing values in Test'.format(train['Holding_Policy_Duration'].isnull().sum()))
print('{} missing values in Test'.format(test['Holding_Policy_Duration'].isnull().sum()))

print(train.isnull().sum())
print(test.isnull().sum())

# Exploratory Data Analysis

train['Middle_Age'] = train['Upper_Age'] - train['Lower_Age']
test['Middle_Age'] = test['Upper_Age'] - test['Lower_Age']
train.drop(['Upper_Age', 'Lower_Age'], axis=1, inplace=True)
test.drop(['Upper_Age', 'Lower_Age'], axis=1, inplace=True)

plt.figure(figsize=(8, 5))
sns.countplot('Middle_Age', data=train, palette='summer')
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot('City_Code', data=train, palette='summer')
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot('Accomodation_Type', data=train, palette='summer')
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot('Reco_Insurance_Type', data=train, palette='summer')
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot('Is_Spouse', data=train, palette='summer')
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot('Health Indicator', data=train, palette='summer')
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot('Holding_Policy_Duration', data=train, palette='summer')
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot('Holding_Policy_Type', data=train, palette='summer')
plt.gcf()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot('Reco_Policy_Cat', data=train, palette='summer')
plt.gcf()
plt.show()

# Feature Engineering
print(train.head())

# Encoding our labels
le = LabelEncoder()
var_mod = ['City_Code', 'Accomodation_Type', 'Reco_Insurance_Type', 'Is_Spouse', 'Health Indicator', 'Holding_Policy_Duration']
le = LabelEncoder()
for i in var_mod:
    train[i] = le.fit_transform(train[i])
    test[i] = le.fit_transform(test[i])

train = pd.get_dummies(train, columns=['City_Code', 'Accomodation_Type', 'Reco_Insurance_Type', 'Is_Spouse',
                                       'Health Indicator', 'Holding_Policy_Duration'])
test = pd.get_dummies(test, columns=['City_Code', 'Accomodation_Type', 'Reco_Insurance_Type', 'Is_Spouse',
                                     'Health Indicator', 'Holding_Policy_Duration'])

print(train.dtypes)
# print(train.columns)
# print(test.columns)
print(train.head())

warnings.filterwarnings('ignore')
train.drop(columns=['ID'], axis=1, inplace=True)
test.drop(columns=['ID'], axis=1, inplace=True)

print('train shape:', train.shape)
print('test shape:', test.shape)

# Seperate Features and Target
X = train.drop(columns=['Response'], axis=1)
y = train['Response']

# 20% data as validation set
seed = 7
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=seed)

# Model Building

print('Linear_Regression')
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]}
# 'fit_intercept': True, 'normalize': True, 'copy_X': True
num_folds = 10
RMS = 'neg_mean_squared_error'
model = LinearRegression()
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=RMS, cv=kfold, verbose=2)
grid_result = grid.fit(rescaledX, y_train)

print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{} ({}) with: {}'.format(mean, stdev, param))

# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
model.fit(rescaledX, y_train)
# transform the validation dataset
rescaledValidationX = scaler.transform(X_valid)
predictions = model.predict(rescaledValidationX)
print('RAS = ', roc_auc_score(y_valid, predictions, average='weighted', multi_class='ovr'))

# Stanrdard scaling of the test dataset
rescaled_test = scaler.transform(test)
# Predict on the test dataset
predicted_prices = model.predict(rescaled_test)

# PRepare submission file
submission = pd.read_csv('sample_submission_QrCyCoT.csv')
submission['Response'] = predicted_prices
submission['Response'] = submission['Response'].apply(lambda x: 0 if x < 0 else x)
submission.to_csv('submission_LR.csv', index=False)
submission.head()
# ####################################
print('Gradient_Boosting_Regressor')
scaler = MinMaxScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {'n_estimators': [80, 90, 100, 110],
              'alpha': [0.9, 0.999],
              'learning_rate': [0.03, 0.05],
              'max_depth': [6, 8],
              'min_samples_leaf': [8, 9],
              'min_samples_split': [2, 3]
              }
# 'alpha': 0.9, 'learning_rate': 0.05, 'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100
# 'alpha': 0.9, 'learning_rate': 0.05, 'max_depth': 8, 'min_samples_leaf': 9, 'min_samples_split': 2, 'n_estimators': 100
num_folds = 2
RMS = 'roc_auc'
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=RMS, cv=kfold, verbose=2)
grid_result = grid.fit(rescaledX, y_train)

print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{} ({}) with: {}'.format(mean, stdev, param))
# 'alpha': 0.9, 'learning_rate': 0.03, 'max_depth': 8, 'min_samples_leaf': 8, 'min_samples_split': 2, 'n_estimators': 110
# 'alpha': 0.9, 'learning_rate': 0.03, 'max_depth': 8, 'min_samples_leaf': 8, 'min_samples_split': 2, 'n_estimators': 110
# prepare the model
scaler = MinMaxScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(alpha=0.9, learning_rate=0.03, max_depth=8, min_samples_leaf=8, min_samples_split=2, n_estimators=110, random_state=seed)
model.fit(rescaledX, y_train)
# transform the validation dataset
rescaledValidationX = scaler.transform(X_valid)
predictions = model.predict(rescaledValidationX)
print('RAS = ', roc_auc_score(y_valid, predictions, average='weighted', multi_class='ovo'))

# Stanrdard scaling of the test dataset
rescaled_test = scaler.transform(test)
# Predict on the test dataset
predicted_prices = model.predict(rescaled_test)

# PRepare submission file
submission = pd.read_csv('sample_submission_QrCyCoT.csv')
submission['Response'] = predicted_prices
submission['Response'] = submission['Response'].apply(lambda x: 0 if x < 0 else x)
submission.to_csv('submission_GBR2.csv', index=False)
submission.head()

# ######################
print('XGB_Regressor')
# Standard scaling
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {'n_estimators': [100, 130, 200],
              'nthread': [4],  # when use hyperthread, xgboost may become slower
              'objective': ['reg:squarederror'],
              'learning_rate': [0.03, 0.05],  # so called `eta` value
              'max_depth': [6, 8],
              'min_child_weight': [4],
              'subsample': [0.7, 0.85],
              'colsample_bytree': [0.7]
              }
# 'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 4, 'n_estimators': 130, 'nthread': 4, 'objective': 'reg:squarederror', 'subsample': 0.7
# 'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': 8, 'min_child_weight': 4, 'n_estimators': 200, 'nthread': 4, 'objective': 'reg:squarederror', 'subsample': 0.85
# Use GridSearchCV to find best estimator & RMS (root mean square)to calculate score
num_folds = 2
RMS = 'neg_mean_squared_error'
model = xgb.XGBRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=RMS, cv=kfold, verbose=2)
grid_result = grid.fit(rescaledX, y_train)

print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{} ({}) with: {}'.format(mean, stdev, param))


# Make predictions on validation dataset using best estimstor = 80
# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = xgb.XGBRegressor(colsample_bytree=0.7, learning_rate=0.05, max_depth=6, min_child_weight=4, n_estimators=130,
                         nthread=4, objective='reg:squarederror', subsample=0.7)
model.fit(rescaledX, y_train)
# transform the validation dataset
rescaledValidationX = scaler.transform(X_valid)
predictions = model.predict(rescaledValidationX)
print('RAS = ', roc_auc_score(y_valid, predictions))

# Stanrdard scaling of the test dataset
rescaled_test = scaler.transform(test)
# Predict on the test dataset
predicted_prices = model.predict(rescaled_test)

# PRepare submission file
submission = pd.read_csv('sample_submission_QrCyCoT.csv')
submission['Response'] = predicted_prices
submission['Response'] = submission['Response'].apply(lambda x: 0 if x < 0 else x)
submission.to_csv('submission_XGBR.csv', index=False)
submission.head()

#####################
print('LGBMRegressor')
# Standard scaling
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {'n_estimators': [50, 70],
              'colsample_bytree': [0.7],
              'max_depth': [6, 7],
              'num_leaves': [50, 55],
              'reg_alpha': [1.1, 1.2],
              'reg_lambda': [1.1, 1.2],
              'min_split_gain': [0.4],
              'subsample': [0.7],
              'subsample_freq': [20]
              }
# Use GridSearchCV to find best estimator & RMS (root mean square)to calculate score
num_folds = 2
RMS = 'neg_mean_squared_error'
model = LGBMRegressor(random_state=seed, silent=True, metric='None', n_jobs=4)
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=RMS, cv=kfold, verbose=2)
grid_result = grid.fit(rescaledX, y_train)

print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{} ({}) with: {}'.format(mean, stdev, param))

# Make predictions on validation dataset using best estimstor = 80
# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = LGBMRegressor(colsample_bytree=0.7,
                      max_depth=7,
                      min_split_gain=0.4,
                      n_estimators=50,
                      num_leaves=50,
                      reg_alpha=1.1,
                      reg_lambda=1.1,
                      subsample=0.7,
                      subsample_freq=20)
model.fit(rescaledX, y_train)
# transform the validation dataset
rescaledValidationX = scaler.transform(X_valid)
predictions = model.predict(rescaledValidationX)
print('RAS = ', roc_auc_score(y_valid, predictions))

# Stanrdard scaling of the test dataset
rescaled_test = scaler.transform(test)
# Predict on the test dataset
predicted_prices = model.predict(rescaled_test)

# PRepare submission file
submission = pd.read_csv('sample_submission_QrCyCoT.csv')
submission['Response'] = predicted_prices
submission['Response'] = submission['Response'].apply(lambda x: 0 if x < 0 else x)
submission.to_csv('submission_LGBMR.csv', index=False)
submission.head()

##########################
print('Extra_Trees_Regression')
# Standard scaling
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {'n_estimators': [70, 100, 150],
              'max_depth': [7, 8],
              'min_samples_split': [2],
              'min_samples_leaf': [8, 9],

              }
# Use GridSearchCV to find best estimator & RMS (root mean square)to calculate score
num_folds = 2
RMS = 'neg_mean_squared_error'
model = ExtraTreesRegressor(random_state=seed, n_jobs=4)
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=RMS, cv=kfold, verbose=2)
grid_result = grid.fit(rescaledX, y_train)

print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{} ({}) with: {}'.format(mean, stdev, param))

# Make predictions on validation dataset using best estimstor = 80
# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = ExtraTreesRegressor(max_depth=8, min_samples_leaf=8, min_samples_split=2, n_estimators=70)
model.fit(rescaledX, y_train)
# transform the validation dataset
rescaledValidationX = scaler.transform(X_valid)
predictions = model.predict(rescaledValidationX)
print('RAS = ', roc_auc_score(y_valid, predictions))

# Stanrdard scaling of the test dataset
rescaled_test = scaler.transform(test)
# Predict on the test dataset
predicted_prices = model.predict(rescaled_test)

# PRepare submission file
submission = pd.read_csv('sample_submission_QrCyCoT.csv')
submission['Response'] = predicted_prices
submission['Response'] = submission['Response'].apply(lambda x: 0 if x < 0 else x)
submission.to_csv('submission_ETR.csv', index=False)
submission.head()

######################################################################
scaler = MinMaxScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
algos = [
         LinearRegression(fit_intercept=True, normalize=True, copy_X=True),
         Ridge(alpha=0.05, normalize=True),
         Lasso(),
         LassoCV(alphas=[1, 0.1, 0.001, 0.0005]),
         SGDRegressor(max_iter=1000, loss='huber', tol=1e-3, average=True),
         ARDRegression(),
         ElasticNet(),
         HuberRegressor(),
         PoissonRegressor(),
         TweedieRegressor(),
         PassiveAggressiveRegressor(),
         SVR(),
         RandomForestRegressor(criterion='mse', max_depth=7, min_samples_leaf=9, min_samples_split=2, n_estimators=100, random_state=seed),
         LGBMRegressor(colsample_bytree=0.7, max_depth=7, min_split_gain=0.4, n_estimators=50, num_leaves=50, reg_alpha=1.1, reg_lambda=1.1, subsample=0.7, subsample_freq=20),
         xgb.XGBRegressor(colsample_bytree=0.7, learning_rate=0.05, max_depth=6, min_child_weight=4, n_estimators=130,nthread=4, objective='reg:squarederror', subsample=0.7),
         GradientBoostingRegressor(alpha=0.9, learning_rate=0.03, max_depth=8, min_samples_leaf=8, min_samples_split=2, n_estimators=110, random_state=seed),
         KNeighborsRegressor(),
         DecisionTreeRegressor(criterion='mse', max_depth=8, min_samples_leaf=9, min_samples_split=2, random_state=seed, splitter='best'),
         ExtraTreesRegressor(max_depth=8, min_samples_leaf=8, min_samples_split=2, n_estimators=70, random_state=seed),
         MLPRegressor()]

names = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'LassoCV', 'SGD Regression', 'ARDRegression', 'ElasticNet',
         'Huber Regression', 'PoissonRegressor', 'TweedieRegressor', 'PassiveAgressiveRegressor',
         'SVR', 'RandomForestRegressor', 'LGBMRegressor', 'XGBRegressor', 'GradientBoostingRegressor',
         'K Neighbors Regressor', 'Decision Tree Regressor', 'ExtraTreesRegressor', 'MLPRegressor']
# names = ['LGBM Regressor', 'XGB Regressor', 'Gradient Boosting Regressor']
ras_list = []

for name in algos:
    model = name
    print(model)
    model.fit(rescaledX, y_train)
    rescaledValidationX = scaler.transform(X_valid)
    y_pred = model.predict(rescaledValidationX)
    RAS = roc_auc_score(y_valid, y_pred, average='weighted', multi_class='ovo')
    ras_list.append(RAS)

evaluation = pd.DataFrame({'Model': names,
                           'RAS': ras_list})
print(evaluation)
print(evaluation['RAS'].max())

submission = pd.read_csv('sample_submission_QrCyCoT.csv')
# model = GradientBoostingRegressor(alpha=0.9,learning_rate=0.05, max_depth=5, min_samples_leaf=3, min_samples_split=2, n_estimators=80, random_state=30)
for name in algos:
    model = name
    model.fit(X, y)
    rescaled_test = scaler.transform(test)
    final_predictions = model.predict(rescaled_test)
    submission['Response'] = final_predictions
    #only positive predictions for the target variable
    submission['Response'] = submission['Response'].apply(lambda x: 0 if x < 0 else x)
    name = str(name)
    submission.to_csv('my_submission_{}.csv'.format(name.split('(')[0]), index=False)
