#!/usr/bin/env python
# coding: utf-8
# https://www.kaggle.com/c/tmdb-box-office-prediction
# **Data Description**
# **id** - Integer unique id of each movie
#
# **belongs_to_collection** - Contains the TMDB Id, Name, Movie Poster and Backdrop URL  of a movie in JSON format. You can see the Poster and Backdrop Image like this: https://image.tmdb.org/t/p/original/<Poster_path_here>. Example: https://image.tmdb.org/t/p/original//iEhb00TGPucF0b4joM1ieyY026U.jpg
#
# **budget**:Budget of a movie in dollars. 0 values mean unknown.
#
# **genres** : Contains all the Genres Name & TMDB Id in JSON Format
#
# **homepage** - Contains the official homepage URL of a movie. Example: http://sonyclassics.com/whiplash/	, this is the homepage of Whiplash movie.
#
# **imdb_id** - IMDB id of a movie (string). You can visit the IMDB Page like this: https://www.imdb.com/title/<imdb_id_here>
#
# **original_language** - Two digit code of the original language, in which the movie was made. Like: en = English, fr = french.
#
# **original_title** - The original title of a movie. Title & Original title may differ, if the original title is not in English.
#
# **overview** - Brief description of the movie.
#
# **popularity** -  Popularity of the movie in float.
#
# **poster_path** - Poster path of a movie. You can see the full image like this: https://image.tmdb.org/t/p/original/<Poster_path_here>
#
# **production_companies** - All production company name and TMDB id in JSON format of a movie.
#
# **production_countries** - Two digit code and full name of the production company in JSON format.
#
# **release_date** - Release date of a movie in mm/dd/yy format.
#
# **runtime** - Total runtime of a movie in minutes (Integer).
#
# **spoken_languages** - Two digit code and full name of the spoken language.
#
# **status** - Is the movie released or rumored?
#
# **tagline** - Tagline of a movie
#
# **title** - English title of a movie
#
# **Keywords** - TMDB Id and name of all the keywords in JSON format.
#
# **cast** - All cast TMDB id, name, character name, gender (1 = Female, 2 = Male) in JSON format
#
# **crew** - Name, TMDB id, profile path of various kind of crew members job like Director, Writer, Art, Sound etc.
#
# **revenue** - Total revenue earned by a movie in dollars.

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # api: https://seaborn.pydata.org/api.html
from datetime import datetime
import json
from scipy.stats import norm
from tqdm import tqdm
import warnings
from datetime import datetime

# %%
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# %%
train.info()
train.shape
test.shape
train.head(3)
train.describe(include='all')
test.describe(include='all')

# %%
## belongs_to_collection and homepage have lots of missing values
train.isna().sum()
test.isna().sum()

# %% DATA ANALYSIS
# %% --- REVENUE
sns.displot(train.revenue, height=8, aspect=2) # distribution plot
sns.displot(train.revenue, height=8, log_scale=True) # log scale shows clearer distribution on skew feature
train.revenue.describe()

# %% --- RELEASE_DATE
release_cols = ['release_month','release_day','release_year']
## pandas.Series.str.split(expand=True) puts result into seperate columns
train[release_cols] = train['release_date'].str.split('/', expand=True).replace(np.nan, 0).astype(int)
train[release_cols].describe()

afterTwoThousands = train['release_year'] <= 19
beforeTwoThousands = (train['release_year'] > 19) & (train['release_year'] < 100)
train.loc[beforeTwoThousands, 'release_year'] += 1900
train.loc[afterTwoThousands, 'release_year'] += 2000
train[release_cols].head()

## Add two more release columns
releaseDate_df = train[release_cols].rename(columns=dict(release_month='month', release_day='day', release_year='year'))
releaseDate = pd.to_datetime(releaseDate_df)
train['release_dayofweek'] = releaseDate.dt.dayofweek
train['release_quarter'] = releaseDate.dt.quarter

test[release_cols] = test['release_date'].str.split('/', expand=True).replace(np.nan, 0).astype(int)
test[test['release_month'] == 0]
# pd.to_datetime(test.iloc[828]['release_date']).dt.dayofweek
## Check all final release columns
release_cols = [col for col in train.columns if 'release' in col]
release_cols.remove('release_date')

for col in release_cols:
    plt.figure(figsize=(20, 10))
    sns.countplot(x=train[col])

# %% mean Revenue by year, by month, by day of week, by quarter.
## only by year show correlation
revByYear = train.groupby('release_year')['revenue'].agg('mean')
revByMonth = train.groupby('release_month')['revenue'].agg('mean')
revByDayOfWeek = train.groupby('release_dayofweek')['revenue'].agg('mean')
revByQuarter = train.groupby('release_quarter')['revenue'].agg('mean')

revByX = [revByYear, revByMonth, revByDayOfWeek, revByQuarter]
for col in revByX:
    plt.figure()
    col.plot(figsize=(10, 5))

# %% --- BUDGET, POPULARITY (numeric), RUNTIME (bye)
## only budget vs revenue seems correlated.
## budget in training data has lots of missing values
print((train.budget > 0).sum())
print((train.budget <= 0).sum())
print((test.budget > 0).sum())
print((test.budget <= 0).sum())

print((train.revenue > 0).sum())
print((train.revenue <= 0).sum())
# print((test.revenue > 0).sum()) is the value to predict
# print((test.revenue <= 0).sum())

sns.displot(data=train, x='budget')
train[(train.revenue < 1e6) & (train.budget <= 0)]
sns.displot(data=train, x='revenue')
np.log1p(train.budget)
sns.displot(data=np.log1p(train.budget))

# %%
for x in ['budget', 'popularity', 'runtime']:

    sns.jointplot(data=train, x=x, y="revenue", height=8)
    plt.title(f'{x} x revenue')
    plt.show()

    col = "mean" + x.title() + "ByYear"
    train[col] = train.groupby("release_year")[x].aggregate('mean')
    train[col].plot(figsize=(15,10),color="g")
    plt.xticks(np.arange(1920,2018,4))
    plt.xlabel("Release Year")
    plt.ylabel(x)
    plt.title(f"Mean {x} by Year",fontsize=20)
    plt.show()

# %%
# first three numerical values, follow with release date, then revenue.
aaa = train[['budget','popularity','release_year','release_month', 'revenue']]
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(aaa.corr(), annot=True)
# x=budget and y=revenue strongly related with 0.75 correlation.

# %% --- GENRES (categorical)
## One hot encoding this feature
def get_dictionary(s): # for processing json strings
    try:    d = eval(s)
    except: d = {}
    return d

train['genres'].head()

def genre_map_func(genre_list):
    updated_genre_list = []
    for genre_dict in get_dictionary(genre_list):
        genre = genre_dict['name']
        genre = str(genre)
        updated_genre_list.append(genre)

    updated_genre_list.sort()
    genre_string = '|'.join(updated_genre_list)
    return genre_string

train_genres_df = train['genres'].map(genre_map_func).str.get_dummies()
test_genres_df = test['genres'].map(genre_map_func).str.get_dummies()
test_genres_df.columns ^ train_genres_df.columns

# %% --- ORIGINAL LANGUAGE (categorical)
## True or False encoding this feature
plt.figure(figsize=(15,15))
train['original_language'].value_counts()
sns.countplot(train['original_language'].sort_values())
plt.title("Original Language Count",fontsize=20)
plt.show()

# Oringal language is English usually has higher revenue
train['isOriginalLanguageEng'] = train['original_language'].map(lambda x: 1 if x == 'en' else 0)
sns.catplot(data=train, x='isOriginalLanguageEng', y='revenue', kind='box')

# %% --- STATUS (bye)
## A strange feature, data point too few, no use.
train['status'].value_counts()
test['status'].value_counts()
train.loc[train.status == 'Rumored']

# %% --- HOMEPAGE (categorical)
## True or False encoding this feature
train.homepage
train['has_homepage'] = train['homepage'].map(lambda x: 0 if pd.isnull(x) else 1)
train.has_homepage.value_counts()

## has_homepage vs revenue
## if a movie has a homepage, it usually has higher revenue.
sns.catplot(data=train, x='has_homepage', y='revenue', kind='box')

# %% --- TAGELINE (categorical)
## True or False encoding this feature
train.tagline
train['has_tagline'] = train['tagline'].map(lambda x: 0 if pd.isnull(x) else 1)
train.has_tagline.value_counts()

## has_tagline vs revenue
## if a movie has a tagline, it usually has higher revenue.
sns.catplot(data=train, x='has_tagline', y='revenue', kind='box')

# %% --- BELONGS_TO_COLLECTION (categorical)
train['belongs_to_collection']
train['isBelongs_to_collection'] = train['belongs_to_collection'].map(lambda x: 0 if pd.isnull(x) else 1)
train.isBelongs_to_collection.value_counts()
sns.catplot(data=train, x='isBelongs_to_collection', y='revenue', kind='box')




# %% FEATURE ENGINEERING
# cols_to_drop = ['id', 'homepage', 'overview', 'tagline', 'original_title', 'title', 'imdb_id', 'poster_path', 'status']
# train = init_train.drop(columns=cols_to_drop)
# test = init_test.drop(columns=cols_to_drop)
# train.head()
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Process columns in json format
json_cols = ['genres', 'production_companies', 'production_countries',
    'spoken_languages', 'Keywords', 'cast', 'crew'] # dropping 'belongs_to_collection'

def get_dictionary(s):
    try:    d = eval(s)
    except: d = {}
    return  d

for col in json_cols:
    train[col] = train[col].apply(lambda x : get_dictionary(x))
    test[col] = test[col].apply(lambda x : get_dictionary(x))

from collections import defaultdict
def get_json_dict(df):
    df_dict = dict()
    for jcol in json_cols: # a json column
        count = defaultdict(int)

        for a_value in df[jcol].values: # for each value in a series (in this case you can think of the row)
            if a_value is None: continue # NaN
            for an_input in a_value: # extracting one of the input
                name = an_input['name'] # we only care about 'name' in the json
                count[name] += 1

        df_dict[jcol] = count

    return df_dict

train_dict = get_json_dict(train)
test_dict = get_json_dict(test)

# remove category with low frequency (bias) or only in train or test.
for col in json_cols:
    train_keys = set(train_dict[col].keys())
    test_keys = set(test_dict[col].keys())
    keys_to_remove = train_keys ^ test_keys

    for key in train_keys & test_keys:
        if train_dict[col][key] < 10:
            keys_to_remove.add(key)

    for key in keys_to_remove:
        train_dict[col].pop(key, None)
        test_dict[col].pop(key, None)

# print(train_dict['genres'].keys())

# %%
from functools import wraps
def prepare(df):
# --- REVENUE
    # Every training data has revenue, even though it is low, it is what it is on IMDB.com as well.

# --- RELEASE_DATE
    release_cols = ['release_month', 'release_day', 'release_year']
    df[release_cols] = df['release_date'].str.split('/', expand=True).replace(np.nan, 0).astype(int)

    afterTwoThousands = df['release_year'] <= 19
    beforeTwoThousands = (df['release_year'] > 19) & (df['release_year'] < 100)
    df.loc[beforeTwoThousands, 'release_year'] += 1900
    df.loc[afterTwoThousands, 'release_year'] += 2000

# --- BUDGET
    # df['originalBudget'] = df['budget']
    # df['infaltedBudget'] = df['budget'] * 1.02**(2018 - df['release_year']) # simple inflation
    df['budget'] = np.log1p(df['budget'])

# --- CAST
    df['cast_count'] = df['cast'].apply(lambda x: len(x) if x is not None else 0)

# --- CREW
    df['crew_count'] = df['crew'].apply(lambda x: len(x) if x is not None else 0)

# --- ORIGINAL LANGUAGE
    df['isOriginalLanguageEng'] = df['original_language'].map(lambda x: 1 if x == 'en' else 0)

# --- HOMEPAGE
    df['has_homepage'] = df['homepage'].map(lambda x: 0 if pd.isnull(x) else 1)

# --- TAGELINE
    df['has_tagline'] = df['tagline'].map(lambda x: 0 if pd.isnull(x) else 1)

# --- BELONGS_TO_COLLECTION
    df['isBelongs_to_collection'] = df['belongs_to_collection'].map(lambda x: 0 if pd.isnull(x) else 1)

# --- PRODUCTION_COMPANIES
    def json_map_func(func):
        @wraps(func)
        def wrapper(value):
            col_name = func()
            output = []
            for d in get_dictionary(value):
                name = str(d['name'])
                if name in train_dict[col_name]:
                    output.append(name)
            output.sort()
            return '|'.join(output)
        return wrapper

    @json_map_func
    def production_companies():
        return 'production_companies'

    df['production_companies_count'] = df['production_companies'].apply(lambda x: len(x) if x is not None else 0)

# --- PRODUCTION_COUNTRIES
    @json_map_func
    def production_countries():
        return 'production_countries'

    df['production_countries_count'] = df['production_countries'].apply(lambda x: len(x) if x is not None else 0)

# --- GENRES
    @json_map_func
    def genres():
        return 'genres'

    # col_df = df['genres'].map(genres).str.get_dummies()
    # train = pd.concat([df, col_df], axis=1)

# --- SPOKEN_LANGUAGES
    @json_map_func
    def spoken_languages():
        return 'spoken_languages'

    for col in (genres, production_countries, spoken_languages, production_companies):
        col_df = df[col.__name__].map(col).str.get_dummies()
        df = pd.concat([df, col_df], axis=1)

# --- drop columns
    columns_to_drop = ['belongs_to_collection', 'genres', 'homepage',
        'production_companies', 'production_countries', 'release_date',
        'spoken_languages', 'cast', 'crew', 'original_language', 'tagline', # features engineered

        'imdb_id', 'overview', 'runtime', 'poster_path', 'status', 'title',
        'Keywords', 'id', 'original_title'] # features not used
    df = df.drop(columns_to_drop, axis=1)

# --- final touch up
    df.fillna(value=0.0, inplace=True)
    return df


# %% Data Preparation
all = pd.concat([train, test]).reset_index(drop=True)
all.shape

all_prepared = prepare(all)
all_prepared.shape

train_prep = all_prepared.loc[:len(train)-1, :] # pd.loc has different indexing scheme
test_prep = all_prepared.loc[len(train):, :]
train_prep.shape
test_prep.shape

#*** Remove training data where budget is 0, see if model performs better.
train_prep = train_prep.loc[train_prep.budget > 0, :]
train_prep.shape

train_y = np.log1p(train_prep.revenue)
train_x = train_prep.drop(['revenue'], axis=1)
test_x = test_prep.drop(['revenue'], axis=1)
train_x.shape
train_y.shape
test_x.shape


# %% Modeling
# %% PCA
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.decomposition import PCA

# 1. Standardize the Data
sc = StandardScaler()
train_sc = sc.fit_transform(train_x)
test_sc  = sc.transform(test_x)

# 2. PCA fit
pca = PCA(n_components=13)
pca.fit(train_sc)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

# 3. PCA transform
train_pca = pca.transform(train_sc)
test_pca = pca.transform(test_sc)
print(train_pca.shape)
print(test_pca.shape)

# %% A helper function
# it does K-fold (5 is the default) cross validation,
# then compute an average score across K-folds in RMSE,
# for an input model.
from sklearn.model_selection import KFold, cross_val_score
def rmse_cv(model):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # there is no pos_mean_squared_err, so we add a - sign.
    rmse = np.sqrt(-cross_val_score(model, train_x, train_y,
                                    scoring="neg_mean_squared_error",
                                    cv = kf))
    return(rmse.mean())

# %% KNN
from sklearn.neighbors import KNeighborsRegressor
rmse_cv(KNeighborsRegressor(n_neighbors=5))

# %% Decision Tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

dt_params = {"max_depth": [2, 5, 10, 20, 50, 100, None],
             "min_samples_split":[2, 5, 10, 20, 50, 100]}
grid_dt = GridSearchCV(DecisionTreeRegressor(),
                       dt_params,
                       scoring="neg_mean_squared_error",
                       cv=3,
                       verbose=3)
grid_dt.fit(train_x, train_y)

print(grid_dt.best_params_)
best_dt = grid_dt.best_estimator_
rmse_cv(best_dt)

# %% Linear Regression
# there is categorical features...
from sklearn.linear_model import ElasticNet
lr_params = {'alpha': [i / 10 for i in range(0, 10)],
             'l1_ratio': [i / 10 for i in range(0, 10)]}
grid_lr = GridSearchCV(ElasticNet(),
                       lr_params,
                       scoring="neg_mean_squared_error",
                       cv=3,
                       verbose=3)
grid_lr.fit(train_x, train_y)
print(grid_lr.best_params_)
best_lr = grid_lr.best_estimator_
rmse_cv(best_lr)

# %% Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=0)
rf_params_grid = {
    'n_estimators': [40, 50, 60],
    'min_samples_split': [5, 10, 15],
    'max_depth': [None, 10, 20],
    }
grid_rf = GridSearchCV(rf, rf_params_grid,
                       scoring="neg_mean_squared_error",
                       cv=5,
                       verbose=1)
grid_rf.fit(train_x, train_y)
print(grid_rf.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_rf.best_score_)))
best_rf = grid_rf.best_estimator_
rmse_cv(best_rf)


# %% XGBboost
import xgboost as xgb

xgb_regressor_params_grid = {
    'max_depth': [i for i in range(1, 5)],
    'learning_rate': [0.1, 0.2],
    'n_estimators': [i for i in range(25, 151, 25)],
    'reg_alpha': [0.25, 0.5],
    'reg_lambda': [0.25, 0.5],
}

# XGBRegressor is sklearn API, so we can use GridSearchCV.
xgb_regressor = xgb.XGBRegressor(seed=2021, learning_rate=0.1, n_jobs=2)

grid_xgb = GridSearchCV(xgb_regressor, xgb_regressor_params_grid,
                        scoring="neg_mean_squared_error",
                        cv=5,
                        verbose=0
                        )

grid_xgb.fit(train_x, train_y)
print(grid_xgb.best_params_)

# 1st try 2.12503
# Without PCA
# Best parameters found:  {'learning_rate': 0.1, 'max_depth': 4,
#                'n_estimators': 125, 'reg_alpha': 0.75, 'reg_lambda': 1}
# 2nd try Lowest RMSE found:  2.0912780707156142
# result isn't better with PCA...
best_xgb_regressor = grid_xgb.best_estimator_
xgb.plot_importance(best_xgb_regressor)
grid_xgb.best_score_
best_xgb_regressor
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_xgb.best_score_)))
rmse_cv(best_xgb_regressor)

# %% Submission
test_y = grid_xgb.predict(test_x)

df_ans = pd.DataFrame()
df_ans['id'] = test['id']
df_ans['revenue'] = np.expm1(test_y)
df_ans.to_csv("submission.csv", index=False)
