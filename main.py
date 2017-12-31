# import zipfile
# zipfile = zipfile.ZipFile('ml-100k.zip', 'r')
# zipfile.extractall()
# zipfile.close()

# Way1 start - ModelBCF

# from surprise import Reader, Dataset
# # Define the format
# reader = Reader(line_format='user item rating timestamp', sep='\t')
# # Load the data from the file using the reader format
# data = Dataset.load_from_file('./ml-100k/u.data', reader=reader)

# print("data:", data)

# data.split(n_folds=5)

# from surprise import SVD, evaluate
# algo = SVD()

# # evaluate(algo, data, measures=['RMSE', 'MAE'])

# # Retrieve the trainset.
# trainset = data.build_full_trainset()
# algo.train(trainset)

# userid = str(196)
# itemid = str(302)
# actual_rating = 4
# print(algo.predict(userid, 302, 4))

# Way1 end

# Way2 start

import numpy as np
import pandas as pd

header = ['user_id', 'item_id', 'rating', 'timestamp']
# u.data contain the whole dataset
df = pd.read_csv('ml-100k/u.data', sep='\t', names=header)

# explore data
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

# split the dataset into testing and training
# cross validation.train_test_split suffles and splits the data into two datasets
# acc. to the % o test examples (test_size)
from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(df, test_size=0.25)

# MemoryBCF

# Way2 end 
