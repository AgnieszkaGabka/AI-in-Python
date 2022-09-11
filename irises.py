# Zaczynamy od zaÅ‚adowania bibliotek. Te najpopularniejsze to
# pandas - do pracy z danymi
# matplotlib - do rysowania wykresow
# sklearn - zawierajÄ…cy gotowe funkcje modelujÄ…ce dane

import pandas as pd
import matplotlib.pyplot as plt
% matplotlib
inline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# tutaj Å‚adujemy dane do obiektu data frame z biblioteki pandas
# plik CSV nie posiada nagÅ‚Ã³wka dlatego header=None
# kolumnom nadajemy nazwy korzystajÄ…c z parametru names
# W skryptach ML dane trzeba skÄ…dÅ› pobraÄ‡, stad znajomoÅ›Ä‡ polecenia
# read_csv jest super przydatna

iris = pd.read_csv(r"/home/agnieszka/Pobrane/iris.data", #alternatively iris.data file from project
                   header=None,
                   names=['petal length', 'petal width',
                          'sepal length', 'sepal width', 'species'])

iris.head()

# moÅ¼na sprawdzic rozmiar wczytanego zbioru
# jeÅ›li obiekt ma wiÄ™cej wymiarÃ³w, to moÅ¼na niezaleÅ¼nie sprawdzaÄ‡ kaÅ¼dy z nich
# W skryptach ML, czÄ™sto trzeba zainicjowaÄ‡ rozmiary innych obiektÃ³w zaleÅ¼nie od
# rozmiaru danych wejÅ›ciowych. Robi siÄ™ to korzystajÄ…c wÅ‚asnie z wÅ‚aÅ›ciwoÅ›ci shape
iris.shape
iris.shape[0]
iris.shape[1]

# dalej przygotowujemy wykres - tutaj wyznaczenie wartoÅ›ci min i max dla
# 2 wybranych kolumn z rozmiarami kwiatÃ³w. Kiedy chcesz siÄ™ odwoÅ‚aÄ‡ do caÅ‚ej kolumny w data frame,
# to w nawiasie kwadratowym podajesz nazwÄ™ tej kolumny
x_min, x_max = iris['petal length'].min() - .5, iris['petal length'].max() + .5
y_min, y_max = iris['petal width'].min() - .5, iris['petal width'].max() + .5

# kaÅ¼dy gatunek ma byÄ‡ wyÅ›wietlony w innym kolorze - definiujemy sÅ‚ownik
colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}

# tworzymy obiekt odpowiedzialny za rysowany wykres i jego wspÃ³Å‚rzÄ™dne
# instrukcje odtÄ…d aÅ¼ do plt.show() uruchom zaznaczajÄ…c caÅ‚y ten blok kodu
fig, ax = plt.subplots(figsize=(8, 6))

# grupujemy dane ze wzglÄ™du na gatunek i rysujemy dane. Korzystamy tu z metody groupby obiektu data frame
# funkcja zwraca klucz identyfikujÄ…cy nazwÄ™ grupy (tutaj jest to nazwa gatunku kwiatu) oraz
# prÃ³bki wchodzÄ…ce w skÅ‚ad tej grupy. To pozwala rysowaÄ‡ kaÅ¼dÄ… grupÄ™ w innym kolorze
for key, group in iris.groupby(by='species'):
    plt.scatter(group['petal length'], group['petal width'],
                c=colors[key], label=key)

# dodajemy legendÄ™ i opis osi
ax.legend()
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
ax.set_title("IRIS DATASET CATEGORIZED")

plt.show()

# teraz podobny wykres moÅ¼na sporzÄ…dziÄ‡ dla sepal
# pamiÄ™taj o uruchomieniu majÄ…c zaznaczony blok kodu odtÄ…d aÅ¼ do plt.show()
# kroki sÄ… takie same jak w poprzednim przykÅ‚adzie
x_min, x_max = iris['sepal length'].min() - .5, iris['sepal length'].max() + .5
y_min, y_max = iris['sepal width'].min() - .5, iris['sepal width'].max() + .5

colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}

fig, ax = plt.subplots(figsize=(8, 6))

for key, group in iris.groupby(by='species'):
    # funkcja scatter przyjmuje argumenty - wspÃ³Å‚rzÄ™dne X punktÃ³w, wspÃ³Å‚rzÄ™dne Y punktÃ³w,
    # kolor i nazwÄ™ rysowanej grupy
    plt.scatter(group['sepal length'], group['sepal width'],
                c=colors[key], label=key)

ax.legend()
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
ax.set_title("IRIS DATASET CATEGORIZED")

plt.show()

# utwÃ³rz wykres skÅ‚adajÄ…cy siÄ™ z 4 maÅ‚ych wykresÃ³w
fig, ax = plt.subplots(2, 2, figsize=(10, 6))

# aktualnie rysowanie odbÄ™dzie siÄ™ w okreÅ›lonym pod-wykresie
plt_position = 1

# obrazujemy zaleÅ¼noÅ›Ä‡ miedzy tÄ… zmiennÄ…, a pozostaÅ‚ymi cechami prÃ³bek
feature_x = 'petal width'

# dla kaÅ¼dej cechy opisujÄ…cej kwiaty
for feature_y in iris.columns[:4]:

    # wybierz kolejny pod wykres
    plt.subplot(2, 2, plt_position)

    # i rysuj osobne wykresy dla kaÅ¼dego gatunku (te 3 rysowane tu wykresy
    # nakÅ‚adajÄ… sie na siebie, co pozwala automatycznie generowaÄ‡ legendÄ™)
    for species, color in colors.items():
        # podczas rysowanie naleÅ¼y odfiltrowaÄ‡ tylko kwiaty jednego gatunku
        # zobacz jak filtrowaÄ‡ dane. SÅ‚uÅ¼y do tego funkcja loc wywoÅ‚ywana dla data frame
        # wyraÅ¼enie w nawiasie kwadratowym ma zwracaÄ‡ True/False. ZwrÃ³cone bÄ™dÄ… wiersze,
        # gdzie wyraÅ¼enie ma wartoÅ›Ä‡ True. Po przecinku znajduje siÄ™ nazwa kolumny, ktÃ³ra ma byÄ‡ zwrÃ³cona
        plt.scatter(iris.loc[iris['species'] == species, feature_x],
                    iris.loc[iris['species'] == species, feature_y],
                    label=species,
                    alpha=0.45,  # transparency
                    color=color)

    # opisujemy wykres
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend()
    plt_position += 1

plt.show()

# Zamiast analizowaÄ‡ kaÅ¼dÄ… parÄ™ niezaleÅ¼nie moÅ¼na generowaÄ‡ tzw. scatter matrix,
# czyli gotowÄ… macierz z wykresami dla kaÅ¼dej pary wÅ‚aÅ›ciwoÅ›ci
# tutaj wykorzystujemy funkcjÄ™ scatter_matrix zaimplementowanÄ… w pandas...
# Do wyznaczenia koloru skorzystaliÅ›my z funkcji apply. Pozwala ona wywoÅ‚aÄ‡ prostÄ… funkcjÄ™ na rzecz
# kaÅ¼dego wiersza z data frame lub serii danych
pd.plotting.scatter_matrix(iris, figsize=(8, 8),
                           color=iris['species'].apply(lambda x: colors[x]));
plt.show()

# ... a tutaj podobny wykres generowany przez funkcjÄ™ pairplot z moduÅ‚u seaborn
import seaborn as sns

sns.set()
sns.pairplot(iris, hue="species")