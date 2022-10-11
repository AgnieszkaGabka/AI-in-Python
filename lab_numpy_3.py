import numpy as np

X = np.arange(1, 26).reshape(5, 5)
X

Ones = np.ones(X.shape)
Ones

np.dot(X, Ones)

diag = np.zeros(X.shape)
np.fill_diagonal(diag, 1)
diag
np.dot(X, diag)

np.where(X > 10, 1, 0)

np.where(X % 2 == 0, 1, 0)

np.where(X % 2 == 0, X, 0)

np.where(X % 2 == 0, X, X + 1)

np.where(X > 10, 2 * X, 0)

X_bis = np.where(X > 10, 2 * X, 0)
X_bis
np.count_nonzero(X_bis)

x = np.array([[10, 20, 30], [40, 50, 60]])
y = np.array([[100], [200]])
print(np.append(x, y, axis=1))

x = np.array([[10, 20, 30], [40, 50, 60]])
y = np.array([[100, 200, 300]])
print(np.append(x, y, axis=0))

x = np.array([[10, 20, 30], [40, 50, 60]])
print(np.append(x, x, axis=0))