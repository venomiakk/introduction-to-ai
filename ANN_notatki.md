# Notatki do odpowiedzi: Sztuczna Sieć Neuronowa (ANN) — Klasyfikacja Iris

## 1. Ogólne założenia i architektura
Zadanie polega na klasyfikacji gatunków irysa (**Iris-setosa**, **Iris-versicolor**, **Iris-virginica**) na podstawie 4 cech numerycznych (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`).

- **Architektura sieci** (`ann/neural_network.py`):
  $$\text{Wejście (4)} \rightarrow \text{Ukryta 1 (16, ReLU)} \rightarrow \text{Ukryta 2 (12, ReLU)} \rightarrow \text{Wyjście (3, Softmax)}$$
- **Rodzaj warstw**: w pełni połączone (`DenseLayer` w `ann/layers.py`) — każdy neuron z poprzedniej warstwy łączy się z każdym w następnej.
- **Uczenie**: klasyczny spadek gradientowy na całym zbiorze treningowym naraz (**Full-Dataset Vanilla Gradient Descent** — bez batchingu, bez optymalizatorów typu Adam/Momentum, bez regularyzacji L2/Dropout).

---

## 2. Przygotowanie i podział danych (`ann/data_loader.py`)
Funkcja `load_iris_data()` wykonuje kolejno 4 kroki:

1. **Wczytanie CSV (`read_iris_csv`)**:
   - Wczytuje 150 próbek → macierz cech `X` o wymiarze `(150, 4)` oraz wektor etykiet `y` o wymiarze `(150,)` z wartościami `0, 1, 2`.
2. **Podział warstwowy (`stratified_train_test_split`)**:
   - Dzieli dane w proporcji 80/20 (**120 treningowych / 30 testowych**) z zachowaniem proporcji klas — z 50 próbek każdej klasy dokładnie 40 trafia do treningu i 10 do testu.
   - Indeksy są tasowane z ustalonym `seed` (powtarzalność wyników).
3. **Standaryzacja cech (`standardize_features`)**:
   - Przekształca każdą cechę według wzoru $x' = \frac{x - \mu}{\sigma}$ (średnia `0`, odchylenie standardowe `1`).
   - Zabezpieczenie przed dzieleniem przez zero: `std = np.where(std == 0.0, 1.0, std)`.
   - **Brak wycieku danych (*Data Leakage*)**: `mean` i `std` są liczone **wyłącznie na `X_train`**, a następnie używane do przeskalowania `X_test`. Gdyby policzyć je przed podziałem, model pośrednio poznałby statystyki zbioru testowego.
4. **Kodowanie One-Hot (`one_hot_encode`)**:
   - Zamienia etykietę klasową (np. `2`) na wektor binarny `[0.0, 0.0, 1.0]` → macierz `(N, 3)`.

---

## 3. Funkcje aktywacji (`ann/activations.py`)

**Po co w ogóle nieliniowa funkcja aktywacji?**  
Bez niej złożenie 3 warstw liniowych ($XW_1W_2W_3$) matematycznie zredukowałoby się do jednej warstwy liniowej i sieć nie potrafiłaby nauczyć się nieliniowych granic decyzyjnych.

### ReLU (warstwy ukryte)
- Wzór: $\text{ReLU}(z) = \max(0, z)$.
- Pochodna (`relu_derivative`): `1.0` dla $z > 0$, `0.0` dla $z \le 0$.
- Zalety: szybka obliczeniowo, nie ma problemu zanikającego gradientu dla wartości dodatnich.

### Softmax (warstwa wyjściowa)
Zamienia surowe wyniki warstwy liniowej (tzw. **logity**, dowolne liczby z $(-\infty, +\infty)$) na **rozkład prawdopodobieństwa** dla 3 klas:

```python
def softmax(z):
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
```

- **Input (`z`)**: macierz `(N, 3)` — każdy wiersz to jedna próbka, każda kolumna to logit dla jednej klasy.
- **Krok 1 (`shifted`)**: odejmujemy maksimum w każdym wierszu osobno (`axis=1, keepdims=True`). Robimy to dla **stabilności numerycznej** — zapobiega przepełnieniu `np.exp(duża_liczba) = inf`. Matematycznie stała się skraca:
  $$\frac{e^{z_j - c}}{\sum_k e^{z_k - c}} = \frac{e^{-c} e^{z_j}}{e^{-c} \sum_k e^{z_k}} = \frac{e^{z_j}}{\sum_k e^{z_k}}$$
- **Krok 2 (`exp_vals`)**: $e^x$ sprawia, że wszystkie wartości są ściśle dodatnie i wzmacnia różnice między klasami.
- **Krok 3 (dzielenie przez sumę w wierszu)**: normalizuje każdy wiersz tak, aby sumował się do `1.0`.
- **Output**: macierz `(N, 3)` z wartościami w $(0, 1)$, gdzie każdy wiersz sumuje się do `1.0` (np. `[0.705, 0.259, 0.035]`). Predykcja klasy to `np.argmax(..., axis=1)`.
- **Dlaczego Softmax, a nie Sigmoid?** W zbiorze Iris klasy się **wzajemnie wykluczają** (kwiatek należy do dokładnie jednego gatunku). Sigmoid liczyłby niezależne prawdopodobieństwa (nie sumowałyby się do 1), a Softmax wymusza konkurencję między klasami.

---

## 4. Budowa warstwy i inicjalizacja wag (`ann/layers.py`)
Klasa `DenseLayer` przechowuje wagi wszystkich neuronów warstwy w postaci macierzy:
- `W` o wymiarze `(input_size, output_size)` — każda kolumna to wagi jednego neuronu.
- `b` o wymiarze `(1, output_size)` — jeden bias na każdy neuron, inicjalizowany zerami.

**Inicjalizacja wag:**
- Nie inicjalizujemy wag zerami, bo wszystkie neurony liczyłyby to samo i dostawały identyczny gradient (problem symetrii).
- Wagi losowane są z rozkładu normalnego $\mathcal{N}(0, \text{scale}^2)$:
  - **Inicjalizacja He** dla ReLU: $\text{scale} = \sqrt{2 / \text{input\_size}}$ (czynnik `2` kompensuje to, że ReLU zeruje ok. połowę sygnałów).
  - **Inicjalizacja Xavier** dla Softmax: $\text{scale} = \sqrt{1 / \text{input\_size}}$.

---

## 5. Funkcja straty, Forward Pass, Backpropagation, Gradient i Bias

### Funkcja straty: Categorical Cross-Entropy (`ann/neural_network.py`)
$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$
- Ponieważ $y$ jest zakodowane one-hot, dla każdej próbki zostaje tylko $-\log(\hat{y}_{\text{poprawna klasa}})$.
- Jeśli sieć jest pewna poprawnej klasy ($\hat{y} \approx 1$), strata $\approx 0$. Jeśli się myli ($\hat{y} \to 0$), kara rośnie do $+\infty$.
- `np.clip(Y_pred, 1e-15, 1.0)` chroni przed liczeniem $\log(0) = -\infty$ (`NaN`).

### Czym jest gradient?
Funkcję straty można wyobrazić sobie jako górzysty teren (wysokość = błąd). **Gradient** $\left(\frac{\partial L}{\partial W}, \frac{\partial L}{\partial b}\right)$ dla każdej wagi mówi:
1. **Znak (`+` / `-`)**: czy zwiększenie wagi zwiększy, czy zmniejszy błąd.
2. **Wartość**: jak silnie błąd zależy od tej wagi.

Gradient wskazuje kierunek **najszybszego wzrostu** błędu, więc aby błąd minimalizować, idziemy w stronę przeciwną (znak minus):
$$W \leftarrow W - \eta \cdot dW, \qquad b \leftarrow b - \eta \cdot db$$
gdzie $\eta$ to `learning_rate` (długość kroku).

### Przepływ w przód (Forward Pass)
Dane `X` płyną od wejścia do wyjścia. W każdej warstwie:
1. `Z = X @ self.W + self.b` — wymiary: `(N, in) @ (in, out) + (1, out) → (N, out)`.
2. `A = activation(Z)` — wymiary: `(N, out)`.
3. Jeśli `cache=True` (podczas treningu), warstwa zapamiętuje `self.X`, `self.Z`, `self.A` na potrzeby propagacji wstecznej. Podczas predykcji (`predict_proba`) ustawiamy `cache=False`, żeby nie nadpisywać cache z treningu.

### Przepływ w tył (Backward Pass / Backpropagation)
Wykorzystuje **regułę łańcuchową**, cofając błąd od warstwy wyjściowej do wejściowej:
1. **Warstwa wyjściowa (Softmax + Cross-Entropy)**:
   ```python
   dZ_out = (Y_pred - Y_true_onehot) / n_samples
   ```
   Połączenie pochodnych Softmaxa i Cross-Entropy upraszcza się do różnicy między predykcją a prawdą (podzielonej przez $N$, dzięki czemu dalsze gradienty są automatycznie uśrednione po próbkach).
2. **Wewnątrz każdej warstwy (`DenseLayer.backward`)**:
   - Warstwy ukryte ReLU: `dZ = d_out * relu_derivative(self.Z)` — neurony nieaktywne ($Z \le 0$) blokują gradient (`0`).
   - Gradient wag: `self.dW = self.X.T @ dZ` — wymiary `(in, N) @ (N, out) → (in, out)`.
   - Gradient biasów: `self.db = np.sum(dZ, axis=0, keepdims=True)` — wymiary `(1, out)`.
   - Gradient dla poprzedniej warstwy: `dX = dZ @ self.W.T` — wymiary `(N, out) @ (out, in) → (N, in)`.
3. **Aktualizacja (`update_parameters`)**: po policzeniu gradientów we wszystkich warstwach aktualizujemy `W` i `b`.

### Co dokładnie dzieje się z biasem (`b`)?
- **Rola**: jak wyraz wolny $b$ w równaniu prostej $y = ax + b$ — przesuwa granicę decyzyjną i pozwala neuronowi ustalić własny próg aktywacji (bez biasu dla wejścia `0` neuron zawsze dawałby `0`).
- **Forward**: `b` ma kształt `(1, out)` i dzięki *broadcastingowi* NumPy jest dodawany do każdego ze 120 wierszy w `X @ W`.
- **Backward**: ponieważ $\frac{\partial (XW + b)}{\partial b} = 1$, a ten sam bias dodano do wszystkich `N` próbek, jego gradient to suma `dZ` po wszystkich próbkach (`np.sum(dZ, axis=0, keepdims=True)`). Używamy `sum`, a nie `mean`, bo podzielenie przez `n_samples` nastąpiło już w `dZ_out`.
- **Update**: `self.b -= learning_rate * self.db`.

---

## 6. Jak przebiega trening vs testowanie?

### Trening (`model.fit()` w `ann/neural_network.py`)
- Zgodnie z wymogiem *„No batching is needed”*, sieć stosuje **Full-Dataset Gradient Descent**.
- Dane treningowe **NIE przechodzą pojedynczo** (to byłoby Stochastic GD — 120 aktualizacji na epokę). Zamiast tego cała macierz `X_train (120, 4)` przechodzi przez sieć **naraz**.
- W **jednej epoce**: 1× forward na 120 próbkach → obliczenie średniej straty → 1× backward (uśredniony gradient ze 120 próbek) → **1 aktualizacja wag**.
- Cykl powtarzany jest przez określoną liczbę epok (domyślnie `1000`).

### Testowanie (`predict_proba` / `predict`)
- Podczas ewaluacji na `X_test (30, 4)` **wagi są zamrożone** — wywoływany jest tylko `forward(..., cache=False)`, bez `backward()` i bez `update_parameters()`.
- W `fit()` ewaluacja na `X_test` po każdej epoce służy wyłącznie do zapisu historii (wykresy `test_loss` i `test_acc`).
- Podczas predykcji to, czy podamy 30 próbek naraz jako macierz `(30, 4)`, czy po jednej `(1, 4)`, daje identyczny wynik dla każdej próbki (wiersze w `forward` nie wpływają na siebie).

---

## 7. Metryki oceny i weryfikacja gradientu (`ann/evaluation.py`)
- **Confusion Matrix (`compute_confusion_matrix`)**: macierz 3×3 (wiersze = prawdziwe klasy, kolumny = predykcje). Pokazuje, które gatunki są mylone (w Iris *Setosa* jest łatwo oddzielalna, błędy zdarzają się głównie między *Versicolor* i *Virginica*).
- **Precision, Recall, F1 (`compute_classification_metrics`)**: wyliczane per klasa i wypisywane w `ann/main.py`. W zbalansowanym zbiorze Iris samo Accuracy jest już miarodajne, ale przy danych niezbalansowanych samo Accuracy bywa mylące i wtedy Precision/Recall/F1 są kluczowe.
- **Numerical Gradient Check (`numerical_gradient_check`)**: przed głównym treningiem w `main.py` sprawdza poprawność wzorów w `backward()`, porównując gradient analityczny z numerycznym przybliżeniem różnicą centralną:
  $$\frac{L(\theta + \varepsilon) - L(\theta - \varepsilon)}{2\varepsilon}$$

---

## 8. Wyniki eksperymentów (`ann/ann_experiment_results.xlsx` i `ann/experiments.py`)

Eksperymenty uruchamiane są w `ann/main.py` (`run_parameter_experiments(num_runs=10)`). W `ann/experiments.py` testowany jest **jeden parametr naraz** (reszta zostaje na wartościach domyślnych: `hidden=(16, 12)`, `lr=0.05`, `epochs=1000`, `standardize=True`). Każda konfiguracja jest uruchamiana **10 razy** (`seed=100..109` — inny podział train/test i inne wagi początkowe), a wyniki są uśredniane.

### Znaczenie kolumn w Excelu
- **Experiment Group / Parameter / Value**: nazwa grupy, nazwa parametru i testowana wartość.
- **Avg Train Acc (%)**: średnia dokładność na zbiorze treningowym (120 próbek).
- **Avg Test Acc (%)**: średnia dokładność na zbiorze testowym (30 próbek) — główna miara generalizacji.
- **Best / Worst Test Acc (%)**: najlepszy i najgorszy wynik na teście z 10 prób — pokazuje **stabilność** modelu.
- **Avg Train Loss / Avg Test Loss**: średnia strata Cross-Entropy. Wzrost `Test Loss` przy spadku `Train Loss` to wyraźny sygnał **overfittingu** (model robi się nadmiernie pewny błędnych predykcji).
- **Avg Execution Time (ms)**: średni czas treningu jednego uruchomienia.

### Wnioski z 4 grup eksperymentów

1. **`hidden_layer_sizes` (architektura warstw ukrytych)**:
   - `(4, 4)`: lekki **underfitting** (`Avg Test Acc 92.00%`, `Worst 80%`, `Test Loss 0.1984`).
   - `(8, 6)` i `(16, 12)`: **złoty środek** (`Avg Test Acc 95.33%` i `94.67%`, najniższy `Test Loss 0.1203` i `0.1367`, `Worst 90%`).
   - `(32, 16)` i `(64, 32)`: początek **overfittingu** (`Train Acc 99.08%`, ale `Test Acc` spada do `94.33%`, `Test Loss` rośnie do `~0.156–0.159`, czas rośnie do `161.79 ms`).

2. **`learning_rate` (współczynnik uczenia $\eta$)**:
   - `0.005` i `0.01`: **za wolne uczenie (underfitting)** (`Test Acc 90%` / `93.67%`, wysoki `Test Loss 0.2937`).
   - `0.05`: **optymalny kompromis** (`Test Acc 94.67%`, najniższy `Test Loss 0.1367`).
   - `0.1` i szczególnie `0.5`: **overfitting / nadmierna pewność siebie** — przy `0.5` mamy `Train Acc 100%`, `Train Loss 0.0022`, `Test Acc 95.33%`, ale **`Test Loss` skacze aż do `0.4224`** (wagi urosły mocno bez regularyzacji i każda pomyłka jest surowo karana przez $-\log$).
   - Czas wykonania jest stały (`~105 ms`), bo `lr` nie zmienia liczby operacji.

3. **`epochs` (liczba epok)**:
   - `100` i `300`: **underfitting** (`Test Acc 89.67%` / `93.33%`, wysoki `Test Loss`).
   - `500` i `1000`: **złoty środek** (`Test Loss 0.1367`, `Test Acc 94.67%` dla 1000 epok).
   - `2000`: początek **overfittingu** — `Train Loss` dalej spada (`0.0235`), `Test Acc` stoi w miejscu (`94.67%`), a **`Test Loss` rośnie (`0.1780`)** przy 2× dłuższym czasie (`196.88 ms`).
   - Czas rośnie liniowo względem liczby epok (`100 → 10 ms`, `1000 → 100 ms`, `2000 → 197 ms`).

4. **`standardize` (standaryzacja cech)**:
   - `True`: `Avg Test Acc 94.67%`, **`Worst Test Acc 90%`**, `Avg Train Loss 0.0364` — stabilne i szybkie uczenie.
   - `False`: `Avg Test Acc 94.00%`, ale **`Worst Test Acc` spada aż do `66.67%`**, a `Avg Train Loss` jest ponad 3× gorszy (`0.114`). Bez standaryzacji różne skale cech zaburzają zbieżność Gradient Descent i w części uruchomień sieć nie rozdziela poprawnie dwóch z trzech klas.
