# import zipfile
# zipfile = zipfile.ZipFile('ml-100k.zip', 'r')
# zipfile.extractall()
# zipfile.close()

from surprise import Reader, Dataset
# Define the format
reader = Reader(line_format='user item rating timestamp', sep='\t')
# Load the data from the file using the reader format
data = Dataset.load_from_file('./ml-100k/u.data', reader=reader)

print("data:", data)

data.split(n_folds=5)

from surprise import SVD, evaluate
algo = SVD()

# evaluate(algo, data, measures=['RMSE', 'MAE'])

# Retrieve the trainset.
trainset = data.build_full_trainset()
algo.train(trainset)

userid = str(196)
itemid = str(302)
actual_rating = 4
print(algo.predict(userid, 302, 4))