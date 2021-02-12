# In[1]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from scipy.stats import norm
from tqdm import tqdm
import warnings
from datetime import datetime


# In[2]:
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
id = test.id
# Example deal with the dict format
# train['collection']=train['belongs_to_collection'].apply(lambda x : {} if pd.isna(x) else eval(x))
# train['collection'][0]
# train['collection_name'] = train['collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
# train['collection_name']

# In[11]:
json_cols = ['belongs_to_collection', 'genres', 'production_companies',
    'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d

for col in tqdm(json_cols) :
    train[col] = train[col].apply(lambda x : get_dictionary(x))
    test[col] = test[col].apply(lambda x : get_dictionary(x))

# Using the directionary transfrom from json format, to calculate the number of appearances of each name tag within each dataframe colume.
def get_json_dict(df):
    global json_cols
    result = dict()
    for e_col in json_cols: # genres
        d = dict()
        rows = df[e_col].values # get all values of that series
        for row in rows: # for each value in a series (in this case you can think of the row)
            if row is None : continue
            for i in row : # there can be serveral categories of genres dict, here it extract each category dict
                if i['name'] not in d : # we only care about the pair that has key "name" in it
                    d[i['name']] = 0
                d[i['name']] += 1
        result[e_col] = d  #
    return result

train_dict = get_json_dict(train)
test_dict = get_json_dict(test)


# Remove name tag in each dataframe column that is low frequency (less than 10), or only appear in train or test.
for col in json_cols:
    train_id = set(list(train_dict[col].keys())) # translate to set so later can minus each other
    test_id = set(list(test_dict[col].keys()))

    remove = list(train_id - test_id) + list(test_id - train_id) # find keys that's only in train or test
    for name in train_id.intersection(test_id): # for key that's in both train and test
        if name == '' or train_dict[col][name] < 10:
            remove.append(name)

    for name in remove:
        if name in train_dict[col]:
            del train_dict[col][name]
        if name in test_dict[col]:
            del test_dict[col][name]

# In[16]:
def prepare(df):
    # budget
    df['budget'] = np.log1p(df['budget'])

    # belongs_to_collection
    df['collection_name'] = df['belongs_to_collection'].apply(lambda x: 1 if x != {} else 0)
    df['has_collection'] = df['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)

    # release_date [month, day, year]
    df['release_year'] = df.release_date.apply(lambda x: int(x.split("/")[2]))
    df.loc[ (df.release_year > 19), "release_year"] += 1900
    df.loc[ (df.release_year <= 19), "release_year"] += 2000
    df['release_month'] = df.release_date.apply(lambda x: int(x.split("/")[0]))
    releaseDate = pd.to_datetime(df.release_date)
    df['release_dayofweek'] = releaseDate.dt.dayofweek
    df['release_quarter'] = releaseDate.dt.quarter

    # various count
    df['genres_count'] = df['genres'].apply(lambda x: len(x))
    df['production_companies_count'] = df['production_companies'].apply(lambda x: len(x))
    df['production_countries_count'] = df['production_countries'].apply(lambda x: len(x))
    df['spoken_languages_count'] = df['spoken_languages'].apply(lambda x: len(x))
    df['Keywords_count'] = df['Keywords'].apply(lambda x: len(x))
    df['cast_count'] = df['cast'].apply(lambda x: len(x))
    df['crew_count'] = df['crew'].apply(lambda x: len(x))

    # runtime
    df.loc[df.runtime.isna(), 'runtime'] = df.runtime.mean()

    # popularity
    # nothing to do.

    # original_language
    original_language_df = pd.get_dummies(df['original_language'])
    df = pd.concat([df, original_language_df], axis=1)

    # spoken_languages
    #df['num_spoken_languages'] = df['spoken_languages'].apply(lambda x : len(x) if x != {} else 0)
    df['list_of_languages'] = list(df['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
    languages_flat_list = [item for sublist in df['list_of_languages'] for item in sublist]

    import collections
    languages_count = collections.Counter(languages_flat_list)
    top9_languages = languages_count.most_common(9)

    for li in top9_languages:
        df['language_' + li[0]] = df['list_of_languages'].apply(lambda x: 1 if li[0] in x else 0)

    # keywords
    df['list_of_keywords'] = list(df['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

    keywords_flat_list = [item for sublist in df['list_of_keywords'] for item in sublist]
    keywords_count = collections.Counter(keywords_flat_list)
    top25_keywords = keywords_count.most_common(25)
    for ki in top25_keywords:
        df['keywords_' + ki[0]] = df['list_of_keywords'].apply(lambda x: 1 if ki[0] in x else 0)

    # cast
    # cast name
    df['list_of_cast'] = list(df['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
    cast_flat_list = [item for sublist in df['list_of_cast'] for item in sublist]
    cast_count = collections.Counter(cast_flat_list)
    top15_cast = cast_count.most_common(15)
    for ki in top15_cast:
        df['cast_name_' + ki[0]] = df['list_of_cast'].apply(lambda x: 1 if ki[0] in x else 0)

    # cast gender
    df['genders0_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]) if x!={} else 0)
    df['genders1_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]) if x!={} else 0)
    df['genders2_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]) if x!={} else 0)

    # crew
    # crew name
    df['list_of_crew'] = list(df['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
    crew_flat_list = [item for sublist in df['list_of_crew'] for item in sublist]
    crew_count = collections.Counter(crew_flat_list)
    top15_crew = crew_count.most_common(15)
    for ki in top15_crew:
        df['crew_name_' + ki[0]] = df['list_of_crew'].apply(lambda x: 1 if ki[0] in x else 0)

    # crew department
    df['list_of_crew_department'] = list(df['crew'].apply(lambda x: [i['department'] for i in x] if x != {} else []).values)
    crew_department_flat_list = [item for sublist in df['list_of_crew_department'] for item in sublist]
    crew_department_count = collections.Counter(crew_department_flat_list)
    top10_crew_department = crew_department_count.most_common(15)
    for ki in top10_crew_department:
        df['crew_department_name_' + ki[0]] = df['crew'].apply(lambda x:  sum([1 for i in x if i['department'] == ki[0]]))

    # crew gender
    df['genders0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]) if x!={} else 0)
    df['genders1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]) if x!={} else 0)
    df['genders2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]) if x!={} else 0)

    # drop columns
    # cols_to_drop = ['id', 'homepage', 'overview', 'tagline', 'original_title', 'title', 'imdb_id', 'poster_path', 'status']
    columns_to_drop = ['belongs_to_collection', 'genres', 'production_companies',
        'production_countries', 'release_date', 'spoken_languages', 'cast', 'crew',
        'runtime', 'Keywords', 'original_language',
         'list_of_languages', 'list_of_keywords', 'list_of_cast', 'list_of_crew',
         'list_of_crew_department', # features engineered

        'id', 'homepage', 'overview', 'tagline', 'original_title', 'title',
        'imdb_id', 'poster_path', 'status'] # features not used
    df = df.drop(columns_to_drop, axis=1)

    df.fillna(value=0.0, inplace=True)
    return df

# In[17]:
test.loc[test.release_date.isna(), 'release_date'] = '1/1/90'
all = pd.concat([train, test]).reset_index(drop=True)

# In[18]:
prepare_all = prepare(all)
prepare_all.head()
train = prepare_all.loc[:len(train)-1, :] # pd.loc has different indexing scheme
test = prepare_all.loc[len(train):, :]

# %%
train_y = np.log1p(train.revenue)
train_x = train.drop(['revenue'], axis=1)
test_x = test.drop(['revenue'], axis=1)

# In[20]:
print(train_y.shape)
print(train_x.shape)
print(test_x.shape)

# # In[22]:
# # Train Test Split
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#
# # In[27]:
#
#
# # Linear Regression
# from sklearn.linear_model import LinearRegression
# LinearRegression().fit(X_train, y_train)
#
#
# # In[28]:
#
#
# from sklearn.linear_model import RidgeCV
# reg_cv = RidgeCV(alphas = [0.001, 0.01, 0.1, 1, 5, 10], cv=10)
# model_cv = reg_cv.fit(X_train, y_train)
#
#
# # In[29]:
#
#
# from sklearn.ensemble import RandomForestRegressor
# rnd_reg = RandomForestRegressor()
# param_grid = {'n_estimators':[100, 150, 200],
#              'max_depth': [5, 15, 20, 30],
#              'max_leaf_nodes': [8, 10, 14, 18, 20]}
# rnd_grid = GridSearchCV(rnd_reg, param_grid, cv=5, n_jobs=-1)
# rnd_grid.fit(X_train, y_train)
#
# y_pred = rnd_grid.predict(X_test)
# print('Training Score: {}'.format(rnd_grid.score(X_train, y_train)))
# print('Test Score: {}'.format(rnd_grid.score(X_test, y_terst)))
# # In[30]:
#
#
# import xgboost as xgb
#
# xgb_train = xgb.DMatrix(X_train, label = y_train)
# xgb_test = xgb.DMatrix(X_test, label= y_test)
#
# params={'objective':'reg:linear',
#         'eval_metric':'rmse',
#         'learning_rate': 0.2,
#         'max_depth': 6,
#         'max_leaves': 0,
#         'n_estimators': 100,
#         'min_child_weight': 1,
#         'gamma': 0,
#         'grow_policy': 'lossguide',
#         'colsample_bytree': 0.85,
#         'min_child_weight': 1,
#         'tree_method': 'gpu_hist'}
#
# watchlist = [(xgb_train, 'xgb_train'), (xgb_test, 'xgb_test')]
# xgbm = xgb.train(params = params,
#                  dtrain = xgb_train,
#                  evals = watchlist,
#                  num_boost_round = 2000,
#                  early_stopping_rounds = 50,
#                  verbose_eval = 100)
#
# y_pred = xgbm.predict(xgb_test)
# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
#

# %%
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

xgb_regressor_params_grid = {
    'max_depth': [i for i in range(1, 5)],
    'learning_rate': [0.1, 0.2],
    'n_estimators': [i for i in range(25, 151, 25)],
    'reg_alpha': [0.25, 0.5],
    'reg_lambda': [0.25, 0.5],
}

# XGBRegressor is sklearn API, so we can use GridSearchCV.
xgb_regressor = xgb.XGBRegressor(seed=2021)

grid_xgb = GridSearchCV(xgb_regressor, xgb_regressor_params_grid,
                        scoring="neg_mean_squared_error",
                        cv=5,
                        verbose=0
                        )

grid_xgb.fit(train_x, train_y)
print(grid_xgb.best_params_)

best_xgb_regressor = grid_xgb.best_estimator_
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_xgb.best_score_)))


# %% Submission 
test_y = grid_xgb.predict(test_x)

df_ans = pd.DataFrame()
df_ans['id'] = id
df_ans['revenue'] = np.expm1(test_y)
df_ans.to_csv("submission.csv", index=False)
