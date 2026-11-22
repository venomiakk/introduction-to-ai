# GA (Genetic Algorithm) dla JSSP — Teoria, analiza kodu i pytania na obronę

Notatki do **Zadania 1** z [ga_pso/IAI_2.md](ga_pso/IAI_2.md) na podstawie implementacji w katalogu `ga_pso/ga/`.

---

## 1. Cel zadania i teoria JSSP

Rozwiązywanym problemem jest **Job-Shop Scheduling Problem (JSSP)** — szeregowanie zadań (*tasks/jobs*) na dostępnych zasobach/maszynach (*resources*) tak, aby zminimalizować całkowity czas wykonania wszystkich zadań.

### Ograniczenia problemu (*constraints*)
1. **Kolejność technologiczna (*precedence*):** operacja $k+1$ danego zadania może rozpocząć się dopiero po zakończeniu operacji $k$ tego samego zadania.
2. **Wyłączność zasobu (*capacity*):** każda maszyna może wykonywać w danej chwili tylko jedną operację.
3. **Brak wywłaszczania (*no preemption*):** rozpoczętej operacji nie można przerwać przed jej ukończeniem.

### Funkcja celu (*fitness*) i dolne ograniczenie (*Lower Bound*)
Minimalizujemy **makespan ($C_{\max}$)**, czyli czas zakończenia ostatniej operacji ze wszystkich zadań:

$$C_{\max} = \max_{j} C_j$$

W `get_problem_summary` ([ga_pso/ga/data_loader.py](ga_pso/ga/data_loader.py)) obliczane jest **teoretyczne dolne ograniczenie**:

$$\text{Lower Bound} = \max\left(\max_{r} \sum_{o \in r} T_o,\; \max_{j} \sum_{o \in j} T_o\right)$$

Harmonogram nie może skończyć się szybciej niż:
- suma czasów na najbardziej obciążonym zasobie (wąskie gardło — w danych z Excela to **Zasób 7** o sumarycznym czasie **2073**), ani
- łączny czas trwania najdłuższego pojedynczego zadania (jego operacje muszą iść po kolei).

Jeśli GA osiągnie $C_{\max} = 2073$, mamy matematyczny dowód, że znaleziono **optimum globalne** (maszyna-wąskie gardło pracowała od $t = 0$ do $t = 2073$ bez ani chwili przestoju).

### Ważna uwaga na obronę (dwie interpretacje pliku `GA_task.xlsx`)
W [ga_pso/ga/data_loader.py](ga_pso/ga/data_loader.py) (linie 20–38) zawarte są dwie interpretacje arkusza:
- **Aktywna w kodzie (Interpretacja 1):** `11` zadań $\times$ `50` operacji (każdy wiersz = 1 zadanie).
- **Zakomentowana (Interpretacja 2):** `50` zadań $\times$ `11` operacji (każda para kolumn `R, T` = 1 zadanie).

W [ga_pso/ga/GA_explanation.md](ga_pso/ga/GA_explanation.md) opis na początku odnosi się do wariantu `50 × 11`, natomiast tabele w sekcji 6 zawierają wyniki dla **obu** interpretacji.

---

## 2. Struktura projektu

| Plik | Rola |
| :--- | :--- |
| [ga_pso/ga/data_loader.py](ga_pso/ga/data_loader.py) | Wczytywanie zadań z `GA_task.xlsx` oraz obliczanie statystyk i `theoretical_lower_bound` |
| [ga_pso/ga/genetic_algorithm.py](ga_pso/ga/genetic_algorithm.py) | Rdzeń algorytmu genetycznego — klasa `JobSchedulingGA` |
| [ga_pso/ga/experiments.py](ga_pso/ga/experiments.py) | Badanie wpływu 9 grup parametrów (po 5 uruchomień na konfigurację) + eksport do Excela |
| [ga_pso/ga/plotter.py](ga_pso/ga/plotter.py) | Wykres zbieżności, wykres Gantta harmonogramu oraz porównania parametrów |
| [ga_pso/ga/main.py](ga_pso/ga/main.py) | Punkt wejścia: pojedynczy przebieg domyślny, wykresy i (opcjonalnie) seria eksperymentów |

---

## 3. Reprezentacja chromosomu i dekodowanie

### Reprezentacja operacyjna (*Operation-Based Representation*)
W `JobSchedulingGA.__init__` ([ga_pso/ga/genetic_algorithm.py](ga_pso/ga/genetic_algorithm.py), linie 36–39):

```python
self.base_genes = [
    task_id for task_id, task in enumerate(tasks) for _ in range(len(task))
]
```

- **Gen** to numer zadania (`task_id`).
- **Chromosom** to lista długości $550$, w której każde `task_id` występuje dokładnie tyle razy, ile operacji ma to zadanie.
- **Zaleta:** **każda permutacja** tej listy reprezentuje w 100% poprawny harmonogram — złamanie kolejności operacji wewnątrz zadania jest konstrukcyjnie niemożliwe.

### Dekodowanie półaktywne (`evaluate_makespan`, linie 49–65)
Przeglądamy chromosom od lewej do prawej:
- $k$-te wystąpienie danego `task_id` oznacza $k$-tą operację tego zadania.
- Utrzymujemy 3 tablice:
  - `task_op_idx[task_id]` — numer kolejnej operacji do zaplanowania dla zadania,
  - `task_ready[task_id]` — moment zakończenia poprzedniej operacji zadania,
  - `res_ready[resource]` — moment zwolnienia maszyny.

Dla każdego genu wyznaczamy:
```python
start_time = max(task_ready[task_id], res_ready[resource])
end_time = start_time + duration
task_ready[task_id] = end_time
res_ready[resource] = end_time
```
Na końcu zwracamy `max(task_ready)`.

> **Ważne:** Zaimplementowane jest **dekodowanie półaktywne (*semi-active*)** — operacje dokładane są na koniec kolejki maszyny (bez wstawiania w wcześniejsze luki, czyli bez *gap-fillingu*). Zapewnia to bardzo szybką, liniową złożoność **`O(N)`** ($N = 550$). Funkcja `decode_schedule` (linie 67–92) wykonuje ten sam algorytm, ale dodatkowo zapisuje listę słowników ze znacznikami czasu pod wykres Gantta.

### Mini-przykład dekodowania krok po kroku
Załóżmy **3 zadania (0, 1, 2)** po **2 operacje** na **2 maszynach (R1, R2)**:
- Zadanie 0: `(R1, 3)` $\rightarrow$ `(R2, 2)`
- Zadanie 1: `(R2, 4)` $\rightarrow$ `(R1, 2)`
- Zadanie 2: `(R1, 2)` $\rightarrow$ `(R2, 3)`

Szablon: `base_genes = [0, 0, 1, 1, 2, 2]`.
Dekodujemy chromosom **`[0, 2, 1, 0, 1, 2]`**:

| Krok | Gen (`task_id`) | Operacja | `(R, T)` | `start = max(task_ready, res_ready)` | `end` | Nowy stan (`task_ready` / `res_ready`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `0` | 1. op. Z0 | `(R1, 3)` | `max(0, 0) = 0` | `3` | Z0=3, R1=3 |
| 2 | `2` | 1. op. Z2 | `(R1, 2)` | `max(0, 3) = 3` *(czeka na R1)* | `5` | Z2=5, R1=5 |
| 3 | `1` | 1. op. Z1 | `(R2, 4)` | `max(0, 0) = 0` | `4` | Z1=4, R2=4 |
| 4 | `0` | 2. op. Z0 | `(R2, 2)` | `max(3, 4) = 4` *(czeka na R2)* | `6` | Z0=6, R2=6 |
| 5 | `1` | 2. op. Z1 | `(R1, 2)` | `max(4, 5) = 5` *(czeka na R1)* | `7` | Z1=7, R1=7 |
| 6 | `2` | 2. op. Z2 | `(R2, 3)` | `max(5, 6) = 6` *(czeka na R2)* | `9` | Z2=9, R2=9 |

Wynik: `makespan = max(6, 7, 9) = 9`.

---

## 4. Analiza krok po kroku klasy `JobSchedulingGA`

### 4.1. Inicjalizacja (`init_population`, linie 41–47)
Tworzy `pop_size` (domyślnie `100`) kopii `self.base_genes` i tasuje każdą za pomocą `random.shuffle`. Kopiowanie (`.copy()`) jest konieczne, bo `shuffle` działa w miejscu.

### 4.2. Elitaryzm (`run`, linie 173–176)
Sortuje indeksy populacji rosnąco po `makespans` i przepisuje bez zmian `elitism_count` (domyślnie `2`) najlepszych chromosomów (`.copy()`) do nowej populacji.
- **Cel:** gwarantuje, że najlepszy znaleziony dotąd harmonogram nigdy nie zostanie utracony przez losowe krzyżowanie lub mutację.

### 4.3. Selekcja rodziców (`select_parent`, linie 94–100 oraz 178–181)
1. **Turniejowa (`tournament`, domyślna):**
   - Losuje bez zwracania `tournament_size` (domyślnie `3`) kandydatów i zwraca tego o najmniejszym makespanie.
   - Większe `tournament_size` = większa **presja selekcyjna** (szybsza zbieżność, ale wyższe ryzyko przedwczesnej utraty różnorodności).
   - Niezależna od skali wartości funkcji celu.
2. **Ruletka (`roulette`):**
   - Wagi wyliczane raz na pokolenie:
     ```python
     roulette_weights = [(max_c - c + 1.0) / c for c in makespans]
     ```
   - Ponieważ minimalizujemy koszt $c$, musimy zamienić mniejszy koszt na większą wagę.
   - Dodanie `+ 1.0` zapewnia niezerową wagę nawet dla najgorszego osobnika (i chroni przed dzieleniem przez zero przy identycznych makespanach).
   - Przesunięcie `max_c - c + 1.0` wzmacnia różnice między dobrymi a słabymi osobnikami (samo `1 / c` w późnych pokoleniach dawałoby niemal równe szanse wszystkim).

### 4.4. Krzyżowanie (`crossover`, `crossover_pox`, `crossover_ox`, linie 102–144)
Zwykłe krzyżowanie jednopunktowe nie działa, bo zepsułoby liczbę powtórzeń zadań w chromosomie. Z prawdopodobieństwem `crossover_prob` (domyślnie `0.85`) wywoływany jest jeden z dwóch operatorów (w przeciwnym razie zwracane są kopie rodziców):

#### A. POX — Precedence Preservative / Job-Based Order Crossover (domyślny, linie 102–114)
Dla rodziców `P1 = [0, 2, 1, 0, 1, 2]` i `P2 = [1, 0, 2, 1, 2, 0]`:
1. Losujemy połowę zadań do zbioru `group_a` (`self.num_tasks // 2`), np. `group_a = {0}`.
2. Geny z `group_a` przepisujemy z `P1` na ich **dokładnie te same pozycje**:
   `child = [0, -1, -1, 0, -1, -1]`
3. Puste miejsca (`-1`) wypełniamy od lewej do prawej genami spoza `group_a` w kolejności ich występowania w `P2` (`1, 2, 1, 2`):
   `child = [0, 1, 2, 0, 1, 2]`
4. Drugie dziecko powstaje symetrycznie przez `make_child(P2, P1)`.

**Dlaczego POX wygrywa w JSSP?** Zachowuje nienaruszoną strukturę i wzajemną kolejność całych grup zadań na maszynach z obu rodziców. Złożoność: `O(N)`.

#### B. OX — Generalized Order Crossover (linie 116–137)
1. Losujemy przedział `[start, end)`, np. `start = 1, end = 4`, i kopiujemy wycinek z `P1`:
   `child = [-1, 2, 1, 0, -1, -1]`
2. Na bazie tablicy `needed = ops_per_task.copy()` zmniejszamy liczniki dla skopiowanych genów (zostaje `needed = [1, 1, 1]`).
3. Przeglądamy `P2` od lewej do prawej: jeśli `needed[gene] > 0`, wpisujemy gen na najbliższe wolne miejsce (przeskakując wycinek `if pos == start: pos = end`) i zmniejszamy `needed[gene]`.
   Wynik: `child = [1, 2, 1, 0, 0, 2]`.

**Dlaczego OX wypada gorzej?** Cięcie po indeksach przesuwa operacje poza wycinkiem i niszczy dobre podharmonogramy na maszynach.

### 4.5. Mutacja (`mutate`, linie 146–152)
Wywoływana w miejscu (*in-place*) dla każdego dziecka z prawdopodobieństwem `mutation_prob` (domyślnie `0.15`). Losuje dwa różne indeksy `i < j`:
- **`swap`:** zamienia miejscami `chromosome[i]` i `chromosome[j]`.
- **`inversion`:** odwraca cały fragment `chromosome[i : j + 1]` (większa zmiana w przestrzeni rozwiązań niż `swap`).

Obie mutacje jedynie przestawiają istniejące geny, więc **zawsze zachowują poprawność chromosomu**.

### 4.6. Ilu rodziców jest wybieranych i ile powstaje dzieci? (`run`, linie 173–194)
Nowa populacja ma zawsze dokładnie rozmiar `pop_size`:
- **Liczba elit:** `elitism_count` (domyślnie `2`).
- **Liczba dzieci trafiających do nowej generacji:** dokładnie $M = \text{pop\_size} - \text{elitism\_count}$ (domyślnie `100 - 2 = 98`).
- **Liczba losowań rodziców:** w każdej iteracji pętli `while` losowana jest **1 para (2 rodziców)**, która tworzy **2 dzieci** (`c1, c2`):
  - Dla domyślnych ustawień (`pop_size = 100`, `elitism_count = 2`): pętla wykonuje się `49` razy $\rightarrow$ **`98` losowań rodziców** (49 par) $\rightarrow$ **`98` dzieci** + `2` elity = `100` osobników.
  - Gdy $M$ jest **nieparzyste** (np. `pop_size = 100`, `elitism_count = 1`, $M = 99$): pętla wykonuje się $\lceil M / 2 \rceil = 50$ razy (100 losowań rodziców), ale warunek `if len(new_population) < self.pop_size:` (linia 191) w ostatniej iteracji dodaje tylko `c1` i odrzuca `c2`.
- **Uwaga:** Rodzice wybierani są **ze zwracaniem** — ten sam dobry osobnik może być wybrany wielokrotnie w jednym pokoleniu (a nawet trafić się jako `p1` i `p2` w tej samej parze).

### 4.7. Warunek stopu (`run`, linie 171, 218–222)
Algorytm kończy działanie, gdy:
1. Osiągnie maksymalną liczbę pokoleń `generations` (domyślnie `300`), **lub**
2. Zadziała **Early Stopping**: `no_improvement >= max_no_improvement` (domyślnie `60` pokoleń bez poprawy najlepszego wyniku).

---

## 5. Eksperymenty i wnioski ([ga_pso/ga/experiments.py](ga_pso/ga/experiments.py))

Eksperymenty przeprowadzono metodą **OFAT** (*One Factor At a Time*) — zmieniany jest jeden parametr względem `DEFAULT_PARAMS`, a każda konfiguracja uruchamiana jest **`5` razy** ze względu na stochastyczny charakter GA.

### Najważniejsze wnioski:
1. **POX vs OX:** najważniejszy czynnik w całym projekcie — `pox` zdecydowanie przewyższa `ox` w obu interpretacjach (w 50×11 osiąga optimum globalne `2073`, podczas gdy `ox` zatrzymuje się w okolicach `2251`).
2. **Rozmiar populacji (`pop_size`):** zbyt mała (`30`) powoduje przedwczesną zbieżność; większa (`100–200`) daje najlepsze wyniki kosztem liniowego wzrostu czasu.
3. **Prawdopodobieństwa (`crossover_prob`, `mutation_prob`):** wysokie krzyżowanie (`0.85–0.95`) i umiarkowana mutacja (`0.10–0.20`) zapewniają dobry balans **eksploracji** i **eksploatacji**. Zbyt niska mutacja (`0.02`) powoduje szybką stagnację, a zbyt wysoka (`0.40`) utrudnia stabilizację dobrych rozwiązań.
4. **Presja selekcyjna (`tournament_size`):** łagodna presja (`2–3`) sprawdza się lepiej niż wysoka (`5–7`), która zbyt szybko eliminuje różnorodność genetyczną.
5. **Early stopping (`max_no_improvement`):** zbyt małe (`25`) przerywa szukanie przed opuszczeniem plateau; `60–100` daje jakość równą pełnym `300` pokoleniom przy krótszym czasie obliczeń.

---

## 6. Typowe pytania na obronę (ściągawka)

1. **Dlaczego nie zastosowano zwykłego krzyżowania jednopunktowego?**
   > Bo chromosom jest permutacją z powtórzeniami (każde zadanie musi wystąpić dokładnie tyle razy, ile ma operacji). Zwykłe cięcie i sklejenie stworzyłoby niepoprawne osobniki. Dlatego użyto POX i OX.
2. **Jak zagwarantowano poprawność kolejności operacji w zadaniu?**
   > Przez reprezentację operacyjną: w chromosomie są tylko ID zadań, a dekoder przy $k$-tym napotkaniu danego `task_id` zawsze planuje jego $k$-tą operację nie wcześniej niż `task_ready[task_id]`.
3. **Skąd wiadomo, czy znaleziony wynik jest optymalny?**
   > Porównujemy go z teoretycznym dolnym ograniczeniem `max(najbardziej obciążona maszyna, najdłuższe zadanie)`. Dla tego zbioru to `2073` (Zasób 7) — osiągnięcie tej wartości dowodzi globalnej optymalności.
4. **Po co w algorytmie elitaryzm?**
   > Chroni najlepsze znalezione dotąd rozwiązanie przed przypadkowym zniszczeniem przez krzyżowanie lub mutację (zapewnia monotoniczność `best_history`).
5. **Dlaczego każdą konfigurację w eksperymentach uruchamiano 5 razy?**
   > Bo GA jest algorytmem stochastycznym — pojedynczy przebieg zależy od losowań, więc wiarygodna ocena wymaga statystyk (`best`, `avg`, `worst`) z wielu uruchomień.
6. **Co odpowiada za eksplorację, a co za eksploatację?**
   > **Eksploracja:** mutacja (`swap`/`inversion`), większa populacja, mniejszy `tournament_size`. **Eksploatacja:** elitaryzm, krzyżowanie POX, większa presja selekcyjna.
7. **Co można by jeszcze ulepszyć w tym kodzie?**
   > 1. Zastąpić dekoder półaktywny aktywnym (*gap-filling*), wstawiającym operacje w wolne luki na maszynach.
   > 2. Dodać przeszukiwanie lokalne na ścieżce krytycznej (algorytm memetyczny).
   > 3. W mutacji `swap` wymusić losowanie dwóch pozycji o różnych wartościach genów (`chromosome[i] != chromosome[j]`), aby uniknąć „pustych” mutacji.
