import numpy as np

X = np.arange(1, 26).reshape(5, 5)
# tablica dwuwymiarowa z kolejnymi liczbami od 1 do 25
X

Ones = np.ones(X.shape)
Ones
# tablica o takim samym kształcie jak X, ale cała wypełniona tylko jedynkami

np.dot(X, Ones)
# mnożenie przez siebie X i ones

diag = np.zeros(X.shape)
# wypełnienie tablicy zerami
np.fill_diagonal(diag, 1)
diag
# diagonalne wypełnienie tablicy jedynkami
np.dot(X, diag)

np.where(X > 10, 1, 0)
# tablica o wymiarach takich jak X, gdzie występują tylko jedynki i zera
# (jedynki tam, gdzie w X wartośc jest większa od 10)

np.where(X % 2 == 0, 1, 0)
# jedynki tam, gdzie w X jest wartość parzysta, a w pozosotałych przypadkach zero


np.where(X % 2 == 0, X, 0)
# występują liczby parzyste

np.where(X % 2 == 0, X, X + 1)

np.where(X > 10, 2 * X, 0)

X_bis = np.where(X > 10, 2 * X, 0)
X_bis
# Dla wartości w X większych od 10 zwrócona jest wartość 2 razy większa,
# a dla pozostałych 0
np.count_nonzero(X_bis)
# liczenie, ile jest wartości niezerowych


x = np.array([[10, 20, 30], [40, 50, 60]])
y = np.array([[100], [200]])
print(np.append(x, y, axis=1))
# dodanie do tablicy x tablicę y jako nową kolumnę

x = np.array([[10, 20, 30], [40, 50, 60]])
y = np.array([[100, 200, 300]])
print(np.append(x, y, axis=0))
# dodanie do tablicy x tablicę y jako nowy wiersz


x = np.array([[10, 20, 30], [40, 50, 60]])
print(np.append(x, x, axis=0))
# dodanie do tablicy x tablicy x jako nowe wiersze