import numpy as np
import sklearn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

import logisticRegression
import DRODataJoiner

# LOADING THE DATASET
X, y = datasets.load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
y = [1 if val==1 else -1 for val in y]


# PARAMETERS
kappa_A = 5
kappa_P = 5

r_A = 0.65
r_P = 0.65

n_neighbors = 1

regularized_penalty = 0.07
overlapped_regularized_penalty = 0.04

d_X = 5
n_P = 20
overlapped = 5


print("**** PARAMETERS ****")
print("kappa_A: {0}".format(kappa_A))
print("kappa_P: {0}".format(kappa_P))
print("r_A: {0}".format(r_A))
print("r_P: {0}".format(r_P))
print("n_neighbors: {0}".format(n_neighbors))
print("regularizer penalty: {0}".format(regularized_penalty))
print("overlapped regularizer penalty: {0}".format(overlapped_regularized_penalty))
print("d_X: {0}".format(d_X))
print("n_P: {0}".format(n_P))
print("overlapped: {0}".format(overlapped))

run_data_join = False

def simulate_breast_cancer_one_iter(X, y, iter_num):
    random_key = iter_num
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_key)

    X_A = X_train[n_P - overlapped:, :]
    y_A = y_train[n_P - overlapped:]


    X_P = X_train[:n_P, :d_X]
    y_P = y_train[:n_P]

    y_test = np.array(y_test)

    # I. X_P, y_p
    myLogisticRegression = logisticRegression.LogisticRegression(n_iters=1500, learning_rate=7e-2, random_key=iter_num)
    myLogisticRegression.fit(X_P, y_P)
    y_pred_proba = myLogisticRegression.predict(X_test[:, :d_X], myLogisticRegression.params)
    y_pred = [1 if val >= 0.5 else -1 for val in y_pred_proba]
    vanilla_acc = accuracy_score(y_test, y_pred)


    myLogisticRegression.fit_regularized(X_P, y_P, regularized_penalty)
    y_pred_proba = myLogisticRegression.predict(X_test[:, :d_X], myLogisticRegression.params)
    y_pred = [1 if val >= 0.5 else -1 for val in y_pred_proba]
    regularized_acc = accuracy_score(y_test, y_pred)


    # II. overlapped
    if overlapped > 0:
        myLogisticRegression = logisticRegression.LogisticRegression(n_iters=1500, learning_rate=7e-2, random_key=iter_num)
        myLogisticRegression.fit(X_A[:overlapped, :], y_A[:overlapped])
        y_pred_proba = myLogisticRegression.predict(X_test, myLogisticRegression.params)
        y_pred = [1 if val >= 0.5 else -1 for val in y_pred_proba]
        overlapped_vanilla_acc = accuracy_score(y_test, y_pred)

        myLogisticRegression = logisticRegression.LogisticRegression(n_iters=1500, learning_rate=7e-2, random_key=iter_num)
        myLogisticRegression.fit_regularized(X_A[:overlapped, :], y_A[:overlapped], overlapped_regularized_penalty)
        y_pred_proba = myLogisticRegression.predict(X_test, myLogisticRegression.params)
        y_pred = [1 if val >= 0.5 else -1 for val in y_pred_proba]
        overlapped_regularized_acc = accuracy_score(y_test, y_pred)
    else:
        overlapped_vanilla_acc = 0
        overlapped_regularized_acc = 0

    # III. data join
    if run_data_join:
        myDRODataJoiner = DRODataJoiner.DRODataJoiner(n_iters=1500, learning_rate=7e-2, random_key=iter_num)
        myDRODataJoiner.initialize(X_A, X_P, y_P, kappa_A, kappa_P, r_A, r_P, n_neighbors)
        print("avg pairwise distance is {0}".format(myDRODataJoiner.pairwise_distance_avg))
        dj_params = myDRODataJoiner.fit(X_A, X_P, y_P, kappa_A, kappa_P, r_A, r_P, num_nearest_neighbors=n_neighbors)
        y_pred_proba = myDRODataJoiner.predict(X_test, dj_params)
        y_pred = [1 if val >= 0.5 else -1 for val in y_pred_proba]
        data_join_acc = accuracy_score(y_test, y_pred)
    else:
        data_join_acc = 0

    myLogisticRegression = logisticRegression.LogisticRegression(n_iters=1500, learning_rate=7e-2, random_key=iter_num)
    myLogisticRegression.fit(X_train, y_train)
    y_pred_proba = myLogisticRegression.predict(X_test, myLogisticRegression.params)
    y_pred = [1 if val >= 0.5 else -1 for val in y_pred_proba]

    full_training_acc = accuracy_score(y_test, y_pred)

    return full_training_acc, vanilla_acc, regularized_acc, overlapped_vanilla_acc, overlapped_regularized_acc, data_join_acc


full_training_accs = []
vanilla_accs = []
regularized_accs = []
overlapped_vanilla_accs = []
overlapped_regularized_accs = []

data_join_accs = []


total_num_sim = 10
for i in range(total_num_sim):
    full_training_acc, vanilla_acc, regularized_acc, overlapped_vanilla_acc, overlapped_regularized_acc, data_join_acc = simulate_breast_cancer_one_iter(X, y, i)

    full_training_accs.append(full_training_acc)
    vanilla_accs.append(vanilla_acc)
    regularized_accs.append(regularized_acc)
    overlapped_vanilla_accs.append(overlapped_vanilla_acc)
    overlapped_regularized_accs.append(overlapped_regularized_acc)
    data_join_accs.append(data_join_acc)

print("**** RESULTS over {0} splits ****".format(total_num_sim))
print("full training: {0}".format(np.mean(full_training_accs)))
print("vanilla log: {0}".format(np.mean(vanilla_accs)))
print("reg log: {0}".format(np.mean(regularized_accs)))
print("overlapped vanilla: {0}".format(np.mean(overlapped_vanilla_accs)))
print("overlapped reg: {0}".format(np.mean(overlapped_regularized_accs)))
print("data join: {0}".format(np.mean(data_join_accs)))

print("Standard Deviations")
print("full training: {0}".format(np.std(full_training_accs)))
print("vanilla log: {0}".format(np.std(vanilla_accs)))
print("reg log: {0}".format(np.std(regularized_accs)))
print("overlapped vanilla: {0}".format(np.std(overlapped_vanilla_accs)))
print("overlapped reg: {0}".format(np.std(overlapped_regularized_accs)))
print("data join: {0}".format(np.std(data_join_accs)))
print("")
print("")
