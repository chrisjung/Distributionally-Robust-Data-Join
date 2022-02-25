import pandas as pd
import sklearn.preprocessing as preprocessing
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.metrics import accuracy_score
import logisticRegression
import DRODataJoiner

# LOADING THE DATASET
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))



# print(X_train)
# print(y_train)

print(data.shape)

one_or_seven = [1,7]

# print(len(y_train[y_train == 1]))
# print(len(y_train[y_train == 7]))

one_indices = digits.target == 1
seven_indices = digits.target == 8

data_ones = data[one_indices]
data_sevens = data[seven_indices]


y_ones = digits.target[one_indices]
y_sevens = digits.target[seven_indices]


X = np.concatenate((data_ones, data_sevens))
y = np.concatenate((y_ones, y_sevens))


y = [1 if v == 1 else -1 for v in y]


scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)



# PARAMETERS
d_X = 32

n_P = 30
overlapped = 10

kappa_A = 5
kappa_P = 5

r_A = 1.85
r_P = 1.85

n_neighbors = 1

lambda_regularized = 0.08
lambda_overlapped_regularized = 0.08


run_data_join = True


print("**** PARAMETERS ****")
print("kappa_A: {0}".format(kappa_A))
print("kappa_P: {0}".format(kappa_P))
print("r_A: {0}".format(r_A))
print("r_P: {0}".format(r_P))
print("n_neighbors: {0}".format(n_neighbors))


print("lambda_regularized: {0}".format(lambda_regularized))
print("lambda_overlapped_regularized: {0}".format(lambda_overlapped_regularized))


print("d_X: {0}".format(d_X))
print("n_P: {0}".format(n_P))
print("overlapped: {0}".format(overlapped))



def simulate_one_iter(X, y, iter_num):
    iter_num += 42
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, stratify=y, test_size=0.3, random_state=iter_num)

    myLogisticRegression = logisticRegression.LogisticRegression(n_iters = 1500, learning_rate = 7e-2, random_key=iter_num)
    myLogisticRegression.fit(X_train, y_train)
    y_pred_proba = myLogisticRegression.predict(X_test, myLogisticRegression.params)
    y_pred = [1 if val >= 0.5 else -1 for val in y_pred_proba]

    full_training_accuracy = accuracy_score(y_test, y_pred)

    X_A = X_train[n_P - overlapped:, :]
    y_A = y_train[n_P - overlapped:]
    X_P = X_train[:n_P, :d_X]
    y_P = y_train[:n_P]

    y_test = np.array(y_test)


    myLogisticRegression = logisticRegression.LogisticRegression(n_iters = 1500, learning_rate = 7e-2, random_key=iter_num)
    myLogisticRegression.fit(X_P, y_P)
    y_pred_proba = myLogisticRegression.predict(X_test[:, :d_X], myLogisticRegression.params)
    y_pred = [1 if val >= 0.5 else -1 for val in y_pred_proba]
    vanilla_logistic = accuracy_score(y_test, y_pred)

    myLogisticRegression = logisticRegression.LogisticRegression(n_iters = 1500, learning_rate = 7e-2, random_key=iter_num)
    myLogisticRegression.fit_regularized(X_P, y_P, lambda_regularized)
    y_pred_proba = myLogisticRegression.predict(X_test[:, :d_X], myLogisticRegression.params)
    y_pred = [1 if val >= 0.5 else -1 for val in y_pred_proba]
    regularized_logistic = accuracy_score(y_test, y_pred)

    myLogisticRegression = logisticRegression.LogisticRegression(n_iters = 1500, learning_rate = 7e-2, random_key=iter_num)
    myLogisticRegression.fit(X_A[:overlapped], y_A[:overlapped])
    y_pred_proba = myLogisticRegression.predict(X_test, myLogisticRegression.params)
    y_pred = [1 if val >= 0.5 else -1 for val in y_pred_proba]
    overlapped_acc = accuracy_score(y_test, y_pred)

    myLogisticRegression = logisticRegression.LogisticRegression(n_iters=1500, learning_rate=7e-2, random_key=iter_num)
    myLogisticRegression.fit_regularized(X_A[:overlapped], y_A[:overlapped], lambda_overlapped_regularized)
    y_pred_proba = myLogisticRegression.predict(X_test, myLogisticRegression.params)
    y_pred = [1 if val >= 0.5 else -1 for val in y_pred_proba]
    overlapped_regularized = accuracy_score(y_test, y_pred)



    if run_data_join:
        myDRODataJoiner = DRODataJoiner.DRODataJoiner(n_iters=1500, learning_rate=7e-2, random_key=iter_num)
        myDRODataJoiner.initialize(X_A, X_P, y_P, kappa_A, kappa_P, r_A, r_P, n_neighbors)
        print("avg pairwise distance is {0}".format(myDRODataJoiner.pairwise_distance_avg))
        dj_params = myDRODataJoiner.fit(X_A, X_P, y_P, kappa_A, kappa_P, r_A, r_P, num_nearest_neighbors=n_neighbors)

        y_pred_proba = myDRODataJoiner.predict(X_test, dj_params)
        y_pred = [1 if val >= 0.5 else -1 for val in y_pred_proba]
        dj_acc = accuracy_score(y_test, y_pred)
    else:
        dj_acc, acc_P = 0, 0

    return full_training_accuracy, vanilla_logistic, regularized_logistic, overlapped_acc, overlapped_regularized, dj_acc


full_training_accuracys = []
vanilla_logistics = []
regularized_logistics =[]
overlapped_accs = []
overlapped_regularizeds = []
acc_joins = []


total_num_sim = 10
for i in range(total_num_sim):
    full_training_accuracy, vanilla_logistic, regularized_logistic, overlapped_acc, overlapped_regularized, dj_acc = simulate_one_iter(X, y, i)
    full_training_accuracys.append(full_training_accuracy)
    vanilla_logistics.append(vanilla_logistic)
    regularized_logistics.append(regularized_logistic)
    overlapped_accs.append(overlapped_acc)
    overlapped_regularizeds.append(overlapped_regularized)
    acc_joins.append(dj_acc)


print("**** RESULTS over {0} splits ****".format(total_num_sim))
print("full training: {0}".format(np.mean(full_training_accuracys)))
print("vanilla training: {0}".format(np.mean(vanilla_logistics)))
print("regularized training: {0}".format(np.mean(regularized_logistics)))
print("overlapped training: {0}".format(np.mean(overlapped_accs)))
print("overlapped training: {0}".format(np.mean(overlapped_regularizeds)))
print("data join: {0}".format(np.mean(acc_joins)))

print("Standard deviations")
print("full training: {0}".format(np.std(full_training_accuracys)))
print("vanilla training: {0}".format(np.std(vanilla_logistics)))
print("regularized training: {0}".format(np.std(regularized_logistics)))
print("overlapped training: {0}".format(np.std(overlapped_accs)))
print("overlapped training: {0}".format(np.std(overlapped_regularizeds)))
print("data join: {0}".format(np.std(acc_joins)))
print("")
print("")
