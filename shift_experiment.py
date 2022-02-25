import numpy as np
import sklearn
from sklearn.metrics import accuracy_score

import logisticRegression
import DRODataJoiner
import DRLogistic

import pdb

np.random.seed(41)

### First data set
# np.random.normal(size=(d))
beta_1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
beta_2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

beta_2_negative = -beta_2

d = beta_1.shape[0]
d_X = 2
curr_scale = 0.2

n_group_1 = 2000
n_1_positive_rate = 0.5

n_group_2 = 200
n_2_positive_rate = 0.5

kappa_A = 10
kappa_P = 10

r_A = 0.2
r_P = 0.2

n_neighbors = 1

print("**** PARAMETERS ****")
print("kappa_A: {0}".format(kappa_A))
print("kappa_P: {0}".format(kappa_P))
print("r_A: {0}".format(r_A))
print("r_P: {0}".format(r_P))
print("n_neighbors: {0}".format(n_neighbors))



n_group_1 = 400
n_1_positive_rate = 0.5

n_group_2 = 20
n_2_positive_rate = 0.5


regularized_penalty = 10

print("lambda: {0}".format(regularized_penalty))


r = 100
kappa = 10

print("kappa: {0}".format(kappa))
print("r: {0}".format(r))


run_dros = True

def create_data(n_group_1, n_1_positive_rate, beta_1, beta_2, beta_2_negative, n_group_2, n_2_positive_rate):
    d = beta_1.shape[0]
    n_1_positives = int(n_group_1 * n_1_positive_rate)
    n_1_negatives = n_group_1 - n_1_positives


    group_1_positives = np.random.normal(loc=beta_1, scale=curr_scale, size=(n_1_positives, d))
    group_1_negatives = np.random.normal(loc=-beta_1, scale=curr_scale, size=(n_1_negatives, d))

    n_2_positives = int(n_group_2 * n_2_positive_rate)
    n_2_negatives = n_group_2 - n_2_positives

    group_2_positives = np.random.normal(loc=beta_2_negative, scale=curr_scale, size=(n_2_positives, d))
    group_2_negatives = np.random.normal(loc=beta_2, scale=curr_scale, size=(n_2_negatives, d))

    X = np.concatenate([group_1_positives, group_1_negatives, group_2_positives, group_2_negatives])
    y = np.concatenate(
        [np.ones(n_1_positives), -np.ones(n_1_negatives), np.ones(n_2_positives), -np.ones(n_2_negatives)])

    return X, y


def run_one_experiment(iter_num, n_group_1, n_1_positive_rate, beta_1, beta_2, beta_2_negative, n_group_2, n_2_positive_rate):
    iter_num = iter_num + 100
    X_P_full, y_P = create_data(n_group_1, n_1_positive_rate, beta_1, beta_2, beta_2_negative, n_group_2, n_2_positive_rate)
    X_P = X_P_full[:, :d_X]

    n_group_1 = 200
    n_1_positive_rate = 0.5

    n_group_2 = 2000
    n_2_positive_rate = 0.5
    X_A_all, y_A_all = create_data(n_group_1, n_1_positive_rate, beta_1, beta_2, beta_2_negative, n_group_2, n_2_positive_rate)

    X_A, X_test, y_A, y_test = sklearn.model_selection.train_test_split(X_A_all, y_A_all, test_size=0.3,
                                                                        random_state=iter_num)

    #### LOGISTIC REGRESSION
    myLogisticRegression = logisticRegression.LogisticRegression(n_iters=1500, learning_rate=7e-2, random_key=iter_num)
    myLogisticRegression.fit(X_P, y_P)

    y_pred_proba = myLogisticRegression.predict(X_test[:, :d_X], myLogisticRegression.params)
    y_pred = [1 if val >= 0.5 else -1 for val in y_pred_proba]

    vanilla_logistic_acc = accuracy_score(y_test, y_pred)

    #### REGULARIZED REGRESSION
    myLogisticRegression = logisticRegression.LogisticRegression(n_iters=1500, learning_rate=7e-2, random_key=iter_num)
    myLogisticRegression.fit_regularized(X_P, y_P, regularized_penalty=regularized_penalty)

    y_pred_proba = myLogisticRegression.predict(X_test[:, :d_X], myLogisticRegression.params)
    y_pred = [1 if val >= 0.5 else -1 for val in y_pred_proba]
    regularized_logistic_acc = accuracy_score(y_test, y_pred)

    #### DR LOGISTIC REGRESSION
    droLogisticRegression = DRLogistic.DRLogisticRegression(n_iters=1500, learning_rate=7e-2, random_key=iter_num)

    params, cost = droLogisticRegression.fit(X_P, y_P, kappa=kappa, r=r, show_convergence=False)

    y_pred_proba = droLogisticRegression.predict(X_test[:, :d_X], params)
    y_pred = [1 if val >= 0.5 else -1 for val in y_pred_proba]

    dlogistic_acc = accuracy_score(y_test, y_pred)

    if run_dros:
        #### DATA JOIN
        myDRODataJoiner2 = DRODataJoiner.DRODataJoiner(n_iters=1500, learning_rate=7e-2, random_key=iter_num)
        myDRODataJoiner2.initialize(X_A, X_P, y_P, kappa_A, kappa_P, r_A, r_P, n_neighbors)
        print("avg pairwise distance is {0}".format(myDRODataJoiner2.pairwise_distance_avg))
        params_A = myDRODataJoiner2.fit(X_A, X_P, y_P, kappa_A, kappa_P, r_A, r_P,
                                                                  num_nearest_neighbors=n_neighbors, show_convergence=False)

        y_pred_proba = myDRODataJoiner2.predict(X_test, params_A)
        y_pred = [1 if val >= 0.5 else -1 for val in y_pred_proba]
        data_join_acc = accuracy_score(y_test, y_pred)
    else:
        data_join_acc = 0

    return vanilla_logistic_acc, regularized_logistic_acc, dlogistic_acc, data_join_acc


vanilla_logistic_accs = []
regularized_logistic_accs = []
regularized_logistic_2_accs = []
dlogistic_accs = []
data_join_accs = []

total_iters = 10
for iter_num in range(total_iters):
    vanilla_logistic_acc, regularized_logistic_acc, dlogistic_acc, data_join_acc = run_one_experiment(iter_num, n_group_1, n_1_positive_rate, beta_1, beta_2, beta_2_negative, n_group_2, n_2_positive_rate)
    vanilla_logistic_accs.append(vanilla_logistic_acc)
    regularized_logistic_accs.append(regularized_logistic_acc)
    dlogistic_accs.append(dlogistic_acc)
    data_join_accs.append(data_join_acc)

### RESULTS
print("************ACCURACY************")
print("lr: {0:0.4f}".format(np.mean(vanilla_logistic_accs)))
print("rlr: {0:0.4f}".format(np.mean(regularized_logistic_accs)))
print("drlr: {0:0.4f}".format(np.mean(dlogistic_accs)))
print("dj: {0:0.4f}".format(np.mean(data_join_accs)))


print("************STD************")
print("lr: {0:0.4f}".format(np.std(vanilla_logistic_accs)))
print("rlr: {0:0.4f}".format(np.std(regularized_logistic_accs)))
print("drlr: {0:0.4f}".format(np.std(dlogistic_accs)))
print("dj: {0:0.4f}".format(np.std(data_join_accs)))


