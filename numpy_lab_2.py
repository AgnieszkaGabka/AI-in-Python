import numpy as np

arr = np.arange(5, 30, 2)
arr

boolArr = arr < 10
boolArr

newArr = arr[boolArr]  # na odpowiedniej pozycji w boolArr znajduje się wartość True
newArr

newArr = arr[arr < 20]  # wartość pozycji jest < 20
newArr

newArr = arr[arr % 3 == 0]  # wartość pozycji jest podzielna przez 3
newArr

newArr = arr[(arr > 10) & (arr < 20)]  # wartość pozycji jest większa niż 10 i mniejsza niż 20
newArr

arr = np.arange(24).reshape(4, 6)
arr

arr[1]  # pierwszy wiersz.

arr[1][2]  # z pierwszego wiersza drugą pozycję
arr[1, 2]  # z pierwszego wiersza drugą pozycję

arr[1, 2:4]  # z pierwszego wiersza pozycje od 2 do 3
arr[1, 2:5]  # z pierwszego wiersza pozycje od 2 do 4

arr[1, :]  # z pierwszego wiersza wszystkie pozycje

arr[:, 2]  # z drugiej kolumny wszystkie wiersze
arr[0:3, 2]  # z wierszy od 0 do 2 wyświetl tylko 2 kolumnę

arr[:3, 2]  # wiersze do drugiego, druga kolumna

arr[:3, 2:4]  # wiersze do drugiego, kolumny od 2 do 3

arr[:, -1]  # wszystkie wiersze, ostatnia kolumna

arr[:, :-1]  # wszystkie wiersze, wszystkie kolumny oprócz ostatniej

arr = np.arange(50).reshape(10, 5)
arr

# how much data should be "test-data' - here 20%
split_level = 0.2  # podział danych w celu przygotowania częsci do sprawdzenia
# działania modelu
num_rows = arr.shape[0]  # ilość rzędów
split_border = split_level * num_rows  # miejsce podziału danych

arr[:round(split_border), :]  # wyświetlenie wierszy od początku do split_border
# i wszystkich kolumn
arr[round(split_border):, :]  # wyświetlenie wierszy od split border do końca
# i wszystkich kolumn
# wartość split board to liczba float, trzeba ją zaokrąglić

np.random.shuffle(arr)  # mieszamy wiersze, żeby dane pobierane zawsze z początku
# zbioru były bardziej losowe
arr

arr[:round(split_border), :]
arr[round(split_border):, :]

# piszemy kod, który dzieli dane na uczące i testowe oraz rozdziela je na zbiór
# feature i zbiór target

data = np.arange(500).reshape(100, 5)
data
np.random.shuffle(data)
data

split_level = 0.2
num_rows = data.shape[0]
split_border = split_level * num_rows

X_train = data[round(split_border):, :-1]  # 80% danych - wszystkie kolumny oprócz
# ostatniej
X_test = data[:round(split_border), :-1]  # 20% danych - wszystkie kolumny oprócz
# ostatniej
y_train = data[round(split_border):, -1]  # 80% danych - ostatnia kolumna
y_test = data[:round(split_border), -1]  # 20% danych - ostatnia kolumna

data.shape
X_train.shape
X_test.shape
y_train.shape
y_test.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data[:, :-1], data[:, -1], test_size=0.2, shuffle=True)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data[:, :-1], data[:, -1], test_size=0.2, shuffle=True)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

