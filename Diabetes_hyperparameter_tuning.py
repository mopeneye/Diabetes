#imports

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def load_Diabetes_data():
    data = pd.read_csv(r"datasets\diabetes.csv")

    return data

df = load_Diabetes_data()

# Noisy data operations

# df.loc[(df['Glucose'] == 0), 'Glucose'] = df['Glucose'].median()
#
# df.loc[(df['BloodPressure'] == 0), 'BloodPressure'] = df['BloodPressure'].median()
#
# df.loc[(df['SkinThickness'] == 0), 'SkinThickness'] = df['SkinThickness'].median()
#
# df.loc[(df['Insulin'] == 0), 'Insulin'] = df['Insulin'].median()
#
# df.loc[(df['BMI'] == 0), 'BMI'] = df['BMI'].median()
#
# df.loc[(df['SkinThickness'] == 0), 'SkinThickness'] = df['SkinThickness'].median()
#
# df.loc[(df['Pregnancies'] >= 12.5), 'Pregnancies'] = df['Pregnancies'].median()

#global

dict_feature_median = {} # pregnancy and age dictionary

# func calculates 0 feature values for ages
def calc_feature_median_func(x,y):
    global dict_feature_median
    if (y == 0):
        if x not in dict_feature_median:
            val = df[(df['BMI'] == x)]['BloodPressure'].mean()
            return dict_feature_median.update({x:val})
        else:
            return dict_feature_median.get(x)
    else:
        return y

# df['BloodPressure'] = df.apply(lambda x: calc_feature_median_func(x['BMI'],  x['BloodPressure']), axis=1)

# MISSING VALUES

df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

def missing_values_table(dataframe):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df)
    return variables_with_na


missing_values_table(df)

def feature_median(var):
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp

#Insulin
feature_median('Insulin')

df.loc[(df['Outcome'] == 0 ) & (df['Insulin'].isnull()), 'Insulin'] = 102.5
df.loc[(df['Outcome'] == 1 ) & (df['Insulin'].isnull()), 'Insulin'] = 169.5

#Glucose
feature_median('Glucose')

df.loc[(df['Outcome'] == 0 ) & (df['Glucose'].isnull()), 'Glucose'] = 107
df.loc[(df['Outcome'] == 1 ) & (df['Glucose'].isnull()), 'Glucose'] = 140

#SkinThickness

feature_median('SkinThickness')

df.loc[(df['Outcome'] == 0 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 27
df.loc[(df['Outcome'] == 1 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 32

#BloodPressure

feature_median('BloodPressure')

df.loc[(df['Outcome'] == 0 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 70
df.loc[(df['Outcome'] == 1 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 74.5

#BMI output

feature_median('BMI')

df.loc[(df['Outcome'] == 0 ) & (df['BMI'].isnull()), 'BMI'] = 30.1
df.loc[(df['Outcome'] == 1 ) & (df['BMI'].isnull()), 'BMI'] = 34.3

def missing_values_table(dataframe):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df)
    return variables_with_na


missing_values_table(df)

# Feature Engineering

# Glucose and Age

df.loc[:,'Gluage']=0
df.loc[(df['Age']<=30) & (df['Glucose']<=120),'Gluage']=1

# BMI

df.loc[:,'BMIcons']=0
df.loc[(df['BMI']<=30),'BMIcons']=1


# Pregnancy and Age

df.loc[:,'Preage']=0
df.loc[(df['Age']<=30) & (df['Pregnancies']<=6),'Preage']=1

# Glucose and blood_pressure

df.loc[:,'Glublood']=0
df.loc[(df['Glucose']<=105) & (df['BloodPressure']<=80),'Glublood']=1

# SkinThickness

df.loc[:,'Sktcons']=0
df.loc[(df['SkinThickness']<=20) ,'Sktcons']=1

# Skinthickness and BMI

df.loc[:,'SktBMI']=0
df.loc[(df['BMI']<30) & (df['SkinThickness']<=20),'SktBMI']=1

# Glucose nd BMI

df.loc[:,'GluBMI']=0
df.loc[(df['Glucose']<=105) & (df['BMI']<=30),'GluBMI']=1

# Insulin

df.loc[:,'Inscons']=0
df.loc[(df['Insulin']<200),'Inscons']=1

# blood_pressure

df.loc[:,'Bloodcons']=0
df.loc[(df['BloodPressure']<80),'Bloodcons']=1

# Pregnancies

df.loc[:,'Pregcons']=0
df.loc[(df['Pregnancies']<4) & (df['Pregnancies']!=0) ,'Pregcons']=1

df.drop(['Preage', 'Bloodcons', 'Pregcons', 'Sktcons', 'BMIcons', 'SktBMI', 'GluBMI', 'Inscons'], axis=1, inplace=True )


# OUTLIERS
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.1)
    quartile3 = dataframe[variable].quantile(0.90)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def has_outliers(dataframe, num_col_names, plot=False):
    variable_names = []
    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col, ":", number_of_outliers)
            variable_names.append(col)
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()
    return variable_names

num_cols = [col for col in df.columns if df[col].dtypes != 'O']

has_outliers(df, num_cols)


#LOF applied

clf = LocalOutlierFactor(n_neighbors = 20, contamination=0.1)

clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_

# np.sort(df_scores)[0:1000]

threshold = np.sort(df_scores)[90]

outlier_tbl = df_scores > threshold

press_value = df[df_scores == threshold]
outliers = df[~outlier_tbl]

res = outliers.to_records(index = False)
res[:] = press_value.to_records(index = False)

df[~outlier_tbl] = pd.DataFrame(res, index = df[~outlier_tbl].index)

has_outliers(df, num_cols)

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(15, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.8, annot_kws={'size': 12}, linecolor='w',
                      cmap='RdBu')
    plt.show()


correlation_matrix(df, df.columns)

# Modeling

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

models = [('RF', RandomForestClassifier()),
          ('XGB', GradientBoostingClassifier()),
          ("LightGBM", LGBMClassifier())]

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=123456)
    cv_results = cross_val_score(model, X, y, cv=10, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print('Base: ', msg)

    # RF Tuned
    if name == 'RF':
        rf_params = {"n_estimators": [200, 500, 1000, 1500],
                     "max_features": [5, 10, 50, 100],
                     "min_samples_split": [5, 10, 20, 50, 100],
                     "max_depth": [5, 10, 20, 50, None]}

        rf_model = RandomForestClassifier(random_state=12345)
        print('RF Baslangic zamani: ', datetime.now())
        gs_cv = GridSearchCV(rf_model,
                             rf_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X, y)
        rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, verbose=2, n_jobs=-1).fit(X, y)  # ???
        print('RF Bitis zamani: ', datetime.now())
        rf_tuned = RandomForestClassifier(**gs_cv.best_params_).fit(X, y)
        cv_results = cross_val_score(rf_tuned, X, y, cv=10, scoring="accuracy").mean()
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print('RF Tuned: ', msg)
        print('RF Best params: ', gs_cv.best_params_)

        # Feature Importance
        feature_imp = pd.Series(rf_tuned.feature_importances_,
                                index=X.columns).sort_values(ascending=False)

        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Değişken Önem Skorları')
        plt.ylabel('Değişkenler')
        plt.title("Değişken Önem Düzeyleri")
        plt.show()
        plt.savefig('rf_importances.png')

    # LGBM Tuned
    elif name == 'LightGBM':
        lgbm_params = {"learning_rate": [0.01, 0.1, 0.5],
        "n_estimators": [500, 1000, 1500],
        "max_depth": [3, 5, 8],
        'num_leaves': [31, 50, 100]}

        lgbm_model = LGBMClassifier(random_state=12345)
        print('LGBM Baslangic zamani: ', datetime.now())
        gs_cv = GridSearchCV(lgbm_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X, y)
        print('LGBM Bitis zamani: ', datetime.now())
        lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X, y)
        cv_results = cross_val_score(lgbm_tuned, X, y, cv=10, scoring="accuracy").mean()
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print('LGBM Tuned: ', msg)
        print('LGBM Best params: ', gs_cv.best_params_)

        # Feature Importance
        feature_imp = pd.Series(lgbm_tuned.feature_importances_,
                                index=X.columns).sort_values(ascending=False)

        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Değişken Önem Skorları')
        plt.ylabel('Değişkenler')
        plt.title("Değişken Önem Düzeyleri")
        plt.show()
        plt.savefig('lgbm_importances.png')

    # XGB Tuned
    elif name == 'XGB':
        xgb_params = {#"colsample_bytree": [0.05, 0.1, 0.5, 1],
                      'max_depth': np.arange(1, 11),
                      'subsample': [0.5, 1, 5, 10, 50],
                      'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.5, 1],
                      'n_estimators': [100, 500, 1000],
                      'loss': ['deviance', 'exponential']}

        xgb_model = GradientBoostingClassifier(random_state=12345)

        print('XGB Baslangic zamani: ', datetime.now())
        gs_cv = GridSearchCV(xgb_model,
                             xgb_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X, y)
        print('XGB Bitis zamani: ', datetime.now())
        xgb_tuned = GradientBoostingClassifier(**gs_cv.best_params_).fit(X, y)
        cv_results = cross_val_score(xgb_tuned, X, y, cv=10, scoring="accuracy").mean()
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print('XGB Tuned: ', msg)
        print('XGB Best params: ', gs_cv.best_params_)


# RF
# Base:  RF: 0.903657 (0.022673)
# RF Baslangic zamani:  2020-10-26 16:19:52.923916
# Fitting 10 folds for each of 400 candidates, totalling 4000 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# [Parallel(n_jobs=-1)]: Done 4000 out of 4000 | elapsed: 21.0min finished
# RF Bitis zamani:  2020-10-26 17:01:49.275376
# RF Tuned:  RF: 0.919310 (0.000000)
# RF Best params:  {'max_depth': 10, 'max_features': 10, 'min_samples_split': 20, 'n_estimators': 1000}
#
#
# XGB
# Base:  XGB: 0.915396 (0.019410)
# XGB Baslangic zamani:  2020-10-26 17:07:23.692784
# Fitting 10 folds for each of 300 candidates, totalling 3000 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# XGB Bitis zamani:  2020-10-26 17:25:57.041236
# XGB Tuned:  XGB: 0.925803 (0.000000)
# XGB Best params:  {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 100, 'subsample': 1}
#
#
# LGBM
# Base:  LightGBM: 0.910202 (0.030352)
# LGBM Baslangic zamani:  2020-10-26 16:15:32.082033
# Fitting 10 folds for each of 81 candidates, totalling 810 fits
# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
# LGBM Bitis zamani:  2020-10-26 16:16:34.979179
# LGBM Tuned:  LightGBM: 0.919293 (0.000000)
# LGBM Best params:  {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 500, 'num_leaves': 31}
# jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.