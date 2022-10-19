import numpy as np

X = np.arange(-25, 25, 1).reshape(10, 5)
# tablica 10 wierszy i 5 kolumn o wartościach od -25 do 24

ones = np.ones((X.shape[0], 1))
# tablica, zawierająca tyle iwerszy co X i jedną kolumnę wypełnioną jedynkami

X_1 = np.append(X.copy(), ones, axis=1)
# zmienna powstała przez dkolejenie do zmiennej X kolumny pochdzącej z ones

w = np.random.rand(X_1.shape[1])


# wektor, który ma tyle samo współrzędnych co ilość kolumn w zmiennaj X_1
# Wartości współrzędnych są wylosowane


def predict(x, w):
    total_stimulation = np.dot(x, w)
    y_pred = 1 if total_stimulation > 0 else -1
    return y_pred


# funkcja mnożaca x (jeden wiersz z tablicy X_1) oraz w
# dla total_stimulation > 0 zwraca 1, w przeciwnym razie -1
# test dla w i X_1[0,]

print(predict(X_1[0,], w))

for x in X_1:
    y_pred = predict(x, w)
    print(y_pred)

# dla każdego wiersza z X_1 pętla wywołuje funkcję predict