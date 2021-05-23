# kaggle-TMDB-Box-Office-Prediction
We're going to make you an offer you can't refuse: a Kaggle competition!

In a world… where movies made an estimated $41.7 billion in 2018, the film industry is more popular than ever. But what movies make the most money at the box office? How much does a director matter? Or the budget? For some movies, it's "You had me at 'Hello.'" For others, the trailer falls short of expectations and you think "What we have here is a failure to communicate."

In this competition, you're presented with metadata on over 7,000 past films from The Movie Database to try and predict their overall worldwide box office revenue. Data points provided include cast, crew, plot keywords, budget, posters, release dates, languages, production companies, and countries. You can collect other publicly available data to use in your model predictions, but in the spirit of this competition, use only data that would have been available before a movie's release.

Join in, "make our day", and then "you've got to ask yourself one question: 'Do I feel lucky?'"

Kaggle competition website: https://www.kaggle.com/c/tmdb-box-office-prediction


How to use
=================================================================
1. Install python packages.
  ```  
  pip install -r requirements.txt
  ```
2. Download data
  ```
  kaggle competitions download -c tmdb-box-office-prediction
  ```
3. Run code 
  - You can open tmdb_a.py or tmdb_b.py with Atom, and installed Hydrogen, to run code line by line like a jupter notebook.
   

Conclusion
=================================================================
tmdb_a.py did data analysis, heavy feature engineerging, tried out multiple machine learning models, and get rid of some training data that did not have "budget" number.

tmdb_b.py did little feature engineering, the feature dimension is high with lots of one-hot encoding features. It performs better than tmdb_a.py.

Both of them obtained the best RMSE with XGBoost (a GBDT algorithm).
