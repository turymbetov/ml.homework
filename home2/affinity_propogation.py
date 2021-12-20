import numpy as np

def first_formula(i: int, k: int, A: np.ndarray, S: np.ndarray) -> int:
    m = -np.inf
    _, cols = A.shape
    for j in range(cols):
        if j != k:
            temp = A[i][j] + S[i][j]
            if temp > m:
                m = temp
    return S[i][k] - m

def second_formula(i: int, k: int, R: np.ndarray) -> int:
    _, cols = R.shape
    acc = 0
    for j in range(cols):
        if (j != i) and (j != k):
            acc = acc + max(0, R[j][k])
    return min(0, R[k][k] + acc)

def third_formula(k: int, R: np.ndarray) -> int:
    _, cols = R.shape
    acc = 0
    for j in range(cols):
        if j != k:
            acc = acc + max(0, R[j][k])
    return acc

def answer(i: int, A: np.ndarray, R: np.ndarray) -> int:
    c = -np.inf
    arg = 0
    _, cols = A.shape
    for k in range(cols):
        temp = A[i][k] + R[i][k]
        if temp > c:
            c = temp
            arg = k
    return arg

def AP(S: np.ndarray, iterations: int) -> np.ndarray:
    rows, cols = S.shape
    A = np.zeros((rows, cols))
    R = np.zeros((rows, cols))

    it = 0
    while it < iterations:
        print(it, '/', iterations)
        # R
        for i in range(rows):
            for k in range(cols):
                R[i][k] = first_formula(i, k, A, S)

        # A
        for i in range(rows):
            for k in range(cols):
                if i != k:
                    A[i][k] = second_formula(i, k, R)

        for k in range(cols):
            A[k][k] = third_formula(k, R)

        it = it + 1

    ans = []
    for i in range(cols):
        ans.append(answer(i, A, R))

    return np.array(ans)