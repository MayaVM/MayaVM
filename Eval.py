import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans




def run_time(func, A, dim):
    res = []
    for j in dim:

        start = timeit.default_timer()
        func(A,j)
        stop = timeit.default_timer()
        res.append(stop - start)

    return res

def generate_x(C, k):
    X = np.zeros(C.shape)
    X0 = KMeans(n_clusters=k).fit(C).labels_

    for i in range(X.shape[0]):
        j = X0[i]
        s_j = np.count_nonzero(X0 == j)
        X[i,j] = 1/math.sqrt(s_j)

    return X

def F(A, C ,k):

    X = generate_x(C, k)
    sketch_matrix =X @ X.T @ A
    F = np.linalg.norm((A-sketch_matrix), 'fro') ** 2
    return F

def objective_val(func, A, dim):
    res = []
    for j in dim:
        X = func(A,j)
        norm_A = np.linalg.norm(A, 'fro') ** 2
        val = F(A,X,j)/norm_A
        res.append(val)
    return res


def vec_to_list(vec):
    vec_set = []
    for i in range(vec.shape[0]):
        vec_set.append(int(vec[i]))
    return vec_set

def accuracy(func,A, b, dim):
    res = []
    for j in dim:
        b_list = vec_to_list(b)
        temp = []
        C = func(A,j)
        centers = KMeans(n_clusters=j).fit(C)
        labels = centers.labels_
        size = C.shape[0]
        for i in range(size):
            if labels[i] == b_list[i]:
                temp.append(1)
            else:
                temp.append(0)
        res.append(sum(temp)/size)

    return res


def numpy_to_list(matrix):
    return matrix.tolist()


def accuracy2(func, A , dim):
    res = []
    for j in dim:
        temp = []
        C = func(A, j)
        clusters = KMeans(n_clusters=j).fit(C).cluster_centers_
        true_clusters = KMeans(n_clusters=j).fit(C).cluster_centers_
        true_clusters_list = numpy_to_list(true_clusters)
        for i in range(j):
            if clusters[i].tolist() in true_clusters_list:
                temp.append(1)
            else:
                temp.append(0)
        res.append(sum(temp)/j)

    return res


def run_alg(alg_set, dims, test_name, A, data_val):
    result = {}
    if test_name == accuracy:
        i = 0
        for alg in alg_set:
            result[i] = test_name(alg, A, data_val, dims)
            i = i + 1
    else:
        i = 0
        for alg in alg_set:
            result[i] = test_name(alg, A, dims)
            i = i + 1
    return result



def plt_data(data_set, data_names, data_val, alg_set, alg_names, dims, test):
    if test == 'time':
        j = 0
        for data in data_set:
            time_result = run_alg(alg_set, dims, run_time, data, data_val)
            for i in range(len(time_result)):
                plt.plot(dims, time_result[i], label=alg_names[i])
            plt.xlabel('# of dimensions (r)')

            plt.ylabel('Running time (seconds)')

            plt.title('Running time vs. number of dimensions (r)')
            plt.legend()

            plt.savefig(f'Running time for {data_names[j]}.eps')
            plt.savefig(f'Running time for {data_names[j]}.png')
            print(f'Running time for {data_names[j]}')
            plt.clf()
            j += 1

    elif test == 'objective value':
        j = 0
        for data in data_set:
            objective_result = run_alg(alg_set, dims, objective_val, data, data_val)
            for i in range(len(objective_result)):
                plt.plot(dims, objective_result[i], label=alg_names[i])
            plt.xlabel('# of dimensions (r)')

            plt.ylabel('Normalize objective value')

            plt.title('Objective value vs. number of dimensions (r)')
            plt.legend()

            plt.savefig(f'Normalize objective value for {data_names[j]}.eps')
            plt.savefig(f'Normalize objective value for {data_names[j]}.png')
            print(f'Normalize objective value for {data_names[j]}')
            plt.clf()
            j += 1

    elif test == 'accuracy':
        j = 0
        for data in data_set:
            Accuracy_result = run_alg(alg_set, dims, accuracy, data, data_val[j])
            for i in range(len(Accuracy_result)):
                plt.plot(dims, Accuracy_result[i], label=alg_names[i])
            plt.xlabel('# of dimensions (r)')

            plt.ylabel('Accuracy')

            plt.title('Accuracy vs. number of dimensions (r)')
            plt.legend()

            plt.savefig(f'Accuracy for {data_names[j]}.eps')
            plt.savefig(f'Accuracy for {data_names[j]}.png')
            print(f'Running for {data_names[j]}')
            plt.clf()
            j += 1

    elif test == 'accuracy2':
        j = 0

        for data in data_set:
            Accuracy_result2 = run_alg(alg_set, dims, accuracy2, data, data_val)
            for i in range(len(Accuracy_result2)):
                plt.plot(dims, Accuracy_result2[i], label=alg_names[i])
            plt.xlabel('# of dimensions (r)')

            plt.ylabel('Accuracy')

            plt.title('Accuracy vs. number of dimensions (r)')
            plt.legend()

            plt.savefig(f'Accuracy for {data_names[j]}2.eps')
            plt.savefig(f'Accuracy for {data_names[j]}2.png')
            print(f'Running time for {data_names[j]}2')
            plt.clf()
            j += 1

    else:
        print('Test is not defined, please pick a test from the test list')
