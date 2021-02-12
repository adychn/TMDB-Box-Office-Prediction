# TMDB-Box-Office-Prediction
Kaggle competition website: https://www.kaggle.com/c/tmdb-box-office-prediction

Download training and testing data from Kaggle: https://www.kaggle.com/c/tmdb-box-office-prediction/data

tmdb_a.py did data analysis, heavy feature engineerging, tried out multiple machine learning models, and get rid of some training data that did not have "budget" number.

tmdb_b.py did little feature engineering, the feature dimension is high with lots of one-hot encoding features. It performs better than tmdb_a.py.

Both of them obtained the best RMSE with XGBoost (a GBDT algorithm).
