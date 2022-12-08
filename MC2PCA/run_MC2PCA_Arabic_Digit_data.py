import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

class MTS:
    def __init__(self, ts):
        self.ts = ts

    def cov_mat(self, centering=True):
        stdsc = StandardScaler()
        X = self.ts
        X = stdsc.fit_transform(X)
        self.ts = X
        return X.transpose() @ X

class CPCA:
    def __init__(self, epsilon=1e-5):
        self.cov = None
        self.epsilon = epsilon
        self.U = None
        self.V = None
        self.S = None

    def fit(self, listMTS):
        if (len(listMTS) > 0):
            P = listMTS[0].cov_mat().shape[1]
            cov_mat = [mat.cov_mat() for mat in listMTS]
            self.cov = sum(cov_mat) / len(cov_mat)
            # Add epsilon Id in order to ensure invertibility
            cov = self.cov + self.epsilon * np.eye(P)
            # Compute SVD
            U, S, V = np.linalg.svd(self.cov)
            # Save SVD
            self.U = U
            self.S = S
            self.V = V

    def pred(self, listMTS, ncp):
        predicted = []
        if (self.U is not None):
            predicted = [elem.ts @ self.U[:, :ncp] for elem in listMTS]
        return predicted

    def reconstitution_error(self, listMTS, ncp):
        mse = np.full(len(listMTS), np.inf)
        if (self.U is not None):
            prediction = self.pred(listMTS, ncp)
            reconstit = [elem @ ((self.U)[:, :ncp].transpose()) for elem in prediction]
            mse = [((listMTS[i].ts - reconstit[i]) ** 2).sum() for i in range(len(prediction))]
        return mse

    def reconstitution_error_distance(self, listMTS, ncp, i):
        if (self.U is not None):
            prediction = self.pred(listMTS, ncp)
            reconstit = [elem @ ((self.U)[:, :ncp].transpose()) for elem in prediction]
            mse_distance = ((listMTS[i].ts - reconstit[i]) ** 2).sum(axis=0)
        return mse_distance

class Mc2PCA:
    def __init__(self, K, ncp, itermax=1000, conv_crit=1e-5):
        self.K = K
        self.N = None
        self.ncp = ncp
        self.iter_max = itermax
        self.converged = False
        self.CPCA_final = None
        self.conv_crit = conv_crit
        self.pred = None

    def fit(self, X):
        N = len(X)
        # initialisation
        index_cluster = np.tile(np.arange(self.K), int(N / self.K) + 1)[:N]
        to_continue = True
        i = 0
        old_error = -1

        while to_continue:
            # Split all MTS according to the cluster
            # we store it in a list of lists of MTS (each list inside the list corresponding to a cluster)
            MTS_by_cluster = [[X[i] for i in list(np.where(index_cluster == j)[0])] for j in range(self.K)]

            CPCA_by_cluster = [CPCA() for i in range(self.K)]

            # fit by cluster
            [CPCA_by_cluster[i].fit(MTS_by_cluster[i]) for i in range(self.K)]

            res = np.array([cpca.reconstitution_error(X, self.ncp) for cpca in CPCA_by_cluster])
            # Update index cluster
            index_cluster = res.argmin(axis=0)     #就在这里可以计算其与质心的距离,但是还是要计算一下每个维度的距离，他这里直接计算范数了

            # new total error
            new_error = res.min(axis=0).sum()
            to_continue = (abs(old_error - new_error) > self.conv_crit) & (self.iter_max > i)
            self.converged = np.abs(old_error - new_error) < self.conv_crit

            # 更新我们需要的每个实例的每个变量至质心对应变量的欧氏距离


            # Updata
            old_error = new_error
            i += 1

        final_error_distance_matrix = np.zeros((N, X[0].ts.shape[1]))
        for i in range(0,N):
            final_error_distance_matrix[i,:]=CPCA_by_cluster[index_cluster[i]].reconstitution_error_distance(X, self.ncp, i)

        self.CPCA_final = CPCA_by_cluster
        self.pred = index_cluster
        return index_cluster, final_error_distance_matrix

    def precision(self, gt_cluster):
        index_cluster = self.pred
        N = gt_cluster.shape[0]
        g = np.unique(gt_cluster)
        nb_g = g.shape[0]

        G = [np.where(gt_cluster == i)[0] for i in range(nb_g)]
        C = [np.where(index_cluster == i)[0] for i in range(self.K)]

        # to handle case where a cluster is empty
        max_part = list()
        for j in range(self.K):
            l = list()
            for i in range(nb_g):
                if len(C[j]) != 0:
                    l.append([np.intersect1d(G[i], C[j]).shape[0] / C[j].shape[0]])
                else:
                    l.append(0)
            max_part.append(np.max(l))
        max_part = np.array(max_part)

        # max_part = np.array([max([np.intersect1d(G[i],C[j]).shape[0]/C[j].shape[0] for i in range(nb_g)]) for j in range(self.K)])
        prop_part = np.array([C[j].shape[0] / N for j in range(self.K)])
        return max_part.dot(prop_part)


def search_ncp(X, K, ncp_list, y_true):
    pres = np.zeros(ncp_list.shape[0])
    for i in range(len(ncp_list)):
        m = Mc2PCA(K, ncp_list[i])
        m.fit(X)
        pres[i] = m.precision(y_true)
    pre = np.max(pres)
    best_ncp = ncp_list[np.argmax(pres)]
    return best_ncp, pre

file_name = "./Test_Arabic_Digit.npy"
ground_truth_N = 2190
ground_truth_cluster_number = 10

res = np.load(file_name, allow_pickle=True)
res = [pd.DataFrame(res[i]) for i in range(res.size)]
res = [MTS(elem.to_numpy()) for elem in res]
gt_nb_cluster = []
for i in range(1,10):
    gt_nb_cluster = gt_nb_cluster+[i for j in range((i-1)*220,i*220)]
gt_nb_cluster = gt_nb_cluster+[10 for j in range(9*220,2190)]
gt_nb_cluster = np.array(gt_nb_cluster)

m = Mc2PCA(K=ground_truth_cluster_number,ncp=3)
(final_index_cluster, final_error_distance_matrix) = m.fit(res)
np.save("test_Arabic_error_distance_by_MC2PCA.npy",final_error_distance_matrix)

print("clustering result is: "+str(final_index_cluster))
print("clustering precision is: "+ str(m.precision(gt_nb_cluster)))

# ncp_list = np.arange(1,8)
# print("search the best p: "+ str(search_ncp(res,ground_truth_cluster_number,ncp_list,gt_nb_cluster)))

