#model utilities
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def convert_lables(Y_data):
  for i in range(len(Y_data)): 
    if Y_data[i] == 0 : Y_data[i] = -1
  return Y_data


def train_with_crossvalidation(splits, X_data, Y_data, model):

  kf = KFold(n_splits = splits)
  kf.get_n_splits(X_data)
  results = []

  for train_index, test_index in kf.split(X_data): #indexes of splitted data

      X_train, X_test = X_data[train_index], X_data[test_index]
      Y_train, Y_test = Y_data[train_index], Y_data[test_index]

      X_train, Y_train = shuffle(X_train, Y_train)
      model = model.fit(X_train, Y_train)
      predictions = model.predict(X_test)
      results.append(accuracy_score(Y_test, predictions))

  return model, results