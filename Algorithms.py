import numpy as np
import math

from sklearn.cluster import KMeans

def FastFrobeniusSVD(A,k,e):
    n = A.shape[1]
    r = k+ math.ceil(k/e +1)
    R = np.random.standard_normal(size = (n,r))
    Y = np.matmul(A,R)
    Q = Y / np.linalg.norm(Y)
    Z0 = np.matmul(np.transpose(Q),A)
    u, s, vh = np.linalg.svd(Z0, full_matrices=True)
    Z = np.argsort(vh[:k])
    return np.transpose(Z)

def RandomizedSampling(X, r):
    n = X.shape[0]
    G = np.zeros((n,r))
    S = np.zeros((r,r))
    prob = []
    for j in range(n):
        prob.append(math.pow(np.linalg.norm(X[j]),2)/math.pow(np.linalg.norm(X, 'fro'),2))
    for t in range(r):
        i = np.random.choice([k for k in range(n)], p=prob)
        G[i,t] = 1
        S[t,t] = 1/math.sqrt(r * prob[i])
    return [G, S]

def SampleSVD(A,k):
    n = A.shape[1]
    r = k
    R = np.random.standard_normal(size=(n, r))
    Y = np.matmul(A, R)
    Q = Y / np.linalg.norm(Y)
    Z0 = np.matmul(np.transpose(Q), A)
    u, s, vh = np.linalg.svd(Z0, full_matrices=True)
    Z = np.argsort(vh[:k]).T

    G, S = RandomizedSampling(Z, r)
    return (A @ G) @ S

def SamplApproxSVD(A,k,e=1/3):
    Z = FastFrobeniusSVD(A, k, e)
    c1 = 16 * 10^6
    r = k
    G, S = RandomizedSampling(Z, r)
    return (A @ G) @ S

def RP(A, k, e = 1/3):
    if (e> 1/3) or (e<0):
        return "e needs to be between 0 and 1/3"
    c2 = 3330 * 15^2
    r = k
    n = A.shape[1]
    prob = [1/math.sqrt(r),-1/math.sqrt(r)]
    R = np.zeros((n,r))
    for i in range(n):
        for j in range(r):
            R[i,j] = np.random.choice(prob, 1, p=[0.5,0.5])
    return A @ R

def SVD(A, k):
    n = A.shape[1]
    R = np.random.standard_normal(size = (n,k))
    Y = np.matmul(A,R)
    Q = Y / np.linalg.norm(Y)
    Z0 = np.matmul(np.transpose(Q),A)
    u, s, vh = np.linalg.svd(Z0, full_matrices=True)
    Z = np.argsort(vh[:k])
    return A @ np.transpose(Z)

def ApprSVD(A, k, e=1/3):
    Z = FastFrobeniusSVD(A, k, e)
    return A @ Z

def k_means(A, k):
    iter_no = 500
    rand_initializations = 5
    kmeans = KMeans(n_clusters=k, n_init=rand_initializations, max_iter= iter_no).fit(A)
    return A @ np.transpose(kmeans.cluster_centers_)

def elkan(A, k):
    iter_no = 500
    rand_initializations = 5
    elkan_means = KMeans(n_clusters=k, n_init=rand_initializations, max_iter= iter_no, algorithm='elkan').fit(A)
    return A @ np.transpose(elkan_means.cluster_centers_)
