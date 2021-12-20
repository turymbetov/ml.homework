import config
import numpy as np
from scipy import sparse
from scipy.sparse.csr import csr_matrix

def MSE(y, y_pred):
    l, _ = y.shape
    return np.sum(np.square(y - y_pred)) / l

def RMSE(y, y_pred):
    return np.sqrt(MSE(y, y_pred))

def R2(y, y_pred):
    return 1 - (np.sum(np.square(y - y_pred))) / (np.sum(np.square(y - y.mean())))

def prediction(X, w, V):
    a = np.sum(np.square(X.dot(V)), axis=1).reshape(-1, 1)
    b = np.sum(X.power(2).dot(np.square(V)), axis=1).reshape(-1, 1)
    return X.dot(w) + 0.5 * (a - b)

data1: csr_matrix = sparse.load_npz('{0}.npz'.format(config.csr_paths[0]))
data2: csr_matrix = sparse.load_npz('{0}.npz'.format(config.csr_paths[1]))
data3: csr_matrix = sparse.load_npz('{0}.npz'.format(config.csr_paths[2]))
data4: csr_matrix = sparse.load_npz('{0}.npz'.format(config.csr_paths[3]))
r, c = data1.shape

Y1 = data1[:, c - 1]
Y2 = data2[:, c - 1]
Y3 = data3[:, c - 1]
Y4 = data4[:, c - 1]

X1 = data1[:, : c - 1]
X2 = data2[:, : c - 1]
X3 = data3[:, : c - 1]
X4 = data4[:, : c - 1]

X = [X1, X2, X3, X4]
Y = [Y1, Y2, Y3, Y4]

m = 0
for w_path, V_path in config.answer_paths:
    V = np.load('{0}.npy'.format(V_path))
    w = np.load('{0}.npy'.format(w_path))
    predictions = []
    print('Train Fold {0}'.format(m))
    for j in range(len(config.answer_paths)):
        if m != j:
            print('Test fold {0}'.format(j))
            a = prediction(X[j], w, V)
            b = Y[j]
            print('RMSE', RMSE(a, b))
    m += 1


