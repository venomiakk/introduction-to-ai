# PSO (Particle Swarm Optimization) — Teoria, analiza kodu i pytania na obronę

Notatki do **Zadania 2** z [ga_pso/IAI_2.md](ga_pso/IAI_2.md) na podstawie implementacji w katalogu `ga_pso/pso/`.

---

## 1. Cel zadania i funkcja celu

Celem zadania jest znalezienie minimum globalnego funkcji dwóch zmiennych:

$$f(x, y) = (1.5 - x - xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2, \quad x, y \in [-4.5, 4.5]$$

### Ważna uwaga (częsta pułapka na obronie)
- Funkcja jest sumą trzech kwadratów, więc w całej dziedzinie $f(x, y) \ge 0$.
- Jest to **zmodyfikowana funkcja Beale'a**. W klasycznej funkcji Beale'a pierwszy nawias ma plus: $(1.5 - x + xy)^2$, i wtedy minimum globalne wynosi dokładnie $0$ w punkcie $(3, 0.5)$.
- W treści zadania w pierwszym nawiasie jest **minus**: $(1.5 - x - xy)^2$. Przez to wszystkie trzy nawiasy nie mogą wyzerować się jednocześnie — prawdziwe minimum globalne w zadanej dziedzinie wynosi:
  $$(x^*, y^*) \approx (2.517991,\ -0.374632), \quad f(x^*, y^*) \approx 0.01351449$$

---

## 2. Teoria PSO (Particle Swarm Optimization)

**PSO** (Kennedy & Eberhart, 1995) to **metaheurystyka populacyjna i stochastyczna**, inspirowana zachowaniem stad ptaków lub ławic ryb szukających pożywienia.

W odróżnieniu od algorytmu genetycznego (GA):
- **nie ma** selekcji, krzyżowania ani mutacji,
- cząstki nie „umierają” — ten sam rój porusza się w sposób ciągły po przestrzeni poszukiwań,
- każda cząstka ma **pozycję**, **prędkość** oraz **pamięć**.

### Co pamięta każda cząstka $i$?
1. Aktualną pozycję $\mathbf{x}_i = (x_i, y_i)$ i prędkość $\mathbf{v}_i = (v_{x,i}, v_{y,i})$.
2. **`pbest` (Personal Best)** — najlepszą pozycję, jaką **ta konkretna cząstka** kiedykolwiek odwiedziła.
3. Rój wspólnie zna **`gbest` (Global Best)** — najlepszą pozycję znalezioną przez **jakąkolwiek** cząstkę w całym roju.

### Równania ruchu cząstki
W każdej iteracji $t$ prędkość aktualizowana jest osobno dla każdej współrzędnej ($x$ i $y$):

$$v_{x,i}^{(t+1)} = \underbrace{w \cdot v_{x,i}^{(t)}}_{\text{Bezwładność (inercja)}} + \underbrace{c_1 \cdot r_{1,x} \cdot \left(p_{\text{best}, x, i} - x_i^{(t)}\right)}_{\text{Składnik kognitywny (pamięć własna)}} + \underbrace{c_2 \cdot r_{2,x} \cdot \left(g_{\text{best}, x} - x_i^{(t)}\right)}_{\text{Składnik społeczny (wiedza roju)}}$$

$$v_{y,i}^{(t+1)} = w \cdot v_{y,i}^{(t)} + c_1 \cdot r_{1,y} \cdot \left(p_{\text{best}, y, i} - y_i^{(t)}\right) + c_2 \cdot r_{2,y} \cdot \left(g_{\text{best}, y} - y_i^{(t)}\right)$$

Następnie aktualizowana jest pozycja (z przycięciem do dziedziny):

$$x_i^{(t+1)} = \text{clamp}\left(x_i^{(t)} + v_{x,i}^{(t+1)},\ -4.5,\ 4.5\right)$$

$$y_i^{(t+1)} = \text{clamp}\left(y_i^{(t)} + v_{y,i}^{(t+1)},\ -4.5,\ 4.5\right)$$

### Rola parametrów (balans eksploracja vs eksploatacja)
- **$w$ (waga bezwładności / inertia weight)** — decyduje, jak bardzo cząstka utrzymuje swój dotychczasowy pęd.
  - Duże $w$ → silna **eksploracja** (cząstki przelatują daleko, wolniej hamują). Dla $w > 1$ prędkość może rosnąć bez ograniczeń i rój przestaje zbiegać.
  - Małe $w$ → silna **eksploatacja** (szybkie wygaszanie ruchu i dokładne przeszukiwanie okolicy).
- **$c_1$ (współczynnik kognitywny)** — siła przyciągania do własnego `pbest`. Utrzymuje różnorodność roju i chroni przed zbyt szybkim zbiciem się wszystkich cząstek w jednym punkcie.
- **$c_2$ (współczynnik społeczny)** — siła przyciągania do `gbest`. Przyspiesza zbieżność całego roju ku najlepszemu znalezionemu obszarowi.
- **$r_1, r_2 \sim U(0, 1)$** — liczby losowe z rozkładu jednostajnego, wprowadzające stochastyczność (zapobiegają deterministycznemu utknięciu).

---

## 3. Struktura i analiza kodu

Projekt składa się z 4 plików Pythona i opisu [ga_pso/pso/PSO_explanation.md](ga_pso/pso/PSO_explanation.md):

| Plik | Rola |
| :--- | :--- |
| [ga_pso/pso/pso_algorithm.py](ga_pso/pso/pso_algorithm.py) | Rdzeń algorytmu PSO i funkcja celu |
| [ga_pso/pso/experiments.py](ga_pso/pso/experiments.py) | Badanie wpływu parametrów (10 uruchomień na konfigurację) + eksport do Excela |
| [ga_pso/pso/plotter.py](ga_pso/pso/plotter.py) | Wykresy zbieżności, mapa konturowa 2D i porównania parametrów |
| [ga_pso/pso/main.py](ga_pso/pso/main.py) | Skrypt uruchomieniowy: jeden przebieg domyślny + seria eksperymentów |

### A. Rdzeń algorytmu — `pso_algorithm.py`

1. **Funkcja celu (`objective_function`, linie 5–14)**:
   Bezpośredni zapis wzoru $f(x, y)$. Działa zarówno na pojedynczych liczbach (`float` w PSO), jak i na macierzach NumPy (wykorzystywane w `plotter.py` do rysowania mapy konturowej).

2. **Konstruktor `ParticleSwarmOptimizer.__init__` (linie 18–41)**:
   - Przechowuje hiperparametry domyślne: `num_particles=30`, `iterations=100`, `w=0.7`, `c1=1.5`, `c2=1.5`, `max_no_improvement=30`, `tol=1e-9`.
   - **Ograniczenie prędkości (`v_max`, linia 41)**:
     `self.v_max = (self.max_bound - self.min_bound) * 0.5` → czyli $(4.5 - (-4.5)) \cdot 0.5 = 4.5$.
     Zapobiega tzw. *eksplozji prędkości* (przelatywaniu cząstek przez całą dziedzinę w jednym kroku).

3. **Inicjalizacja roju (`init_swarm`, linie 43–64)**:
   - Każda cząstka to słownik z kluczami: `x`, `y`, `vx`, `vy`, `val`, `pbest_x`, `pbest_y`, `pbest_val`.
   - Pozycje `(x, y)` losowane są jednostajnie w całym `[-4.5, 4.5]`.
   - Prędkości początkowe `(vx, vy)` losowane są z mniejszego zakresu `[-0.1 * span, 0.1 * span]`, czyli `[-0.9, 0.9]`, aby cząstki na starcie nie uderzyły od razu w krawędzie dziedziny.
   - Na starcie `pbest` każdej cząstki jest równy jej pozycji początkowej.

4. **Główna pętla (`run`, linie 69–165)**:
   - **Start (linie 75–80)**: wyznaczany jest początkowy `gbest` jako minimum po `pbest_val` oraz inicjalizowane są tablice historii (`best_history`, `avg_history`, `gbest_path`) do wykresów.
   - **Detal #1 — niezależne losowanie po osiach (linie 91–92)**:
     Losowane są osobne `r1_x, r2_x` oraz `r1_y, r2_y`. Dzięki temu ruch nie jest ograniczony do prostych kombinacji liniowych wektorów, co daje lepszą eksplorację w 2D.
   - **Detal #2 — `clamp` prędkości i pozycji (linie 106–111)**:
     Najpierw prędkość jest przycinana do `[-v_max, v_max]`, a potem nowa pozycja do `[-4.5, 4.5]` (*clamping boundary* — cząstka zatrzymuje się na brzegu dziedziny).
   - **Detal #3 — asynchroniczna aktualizacja `gbest` (linie 117–126)**:
     `gbest` jest aktualizowany **od razu wewnątrz pętli po cząstkach**. Jeśli cząstka nr 5 poprawi `gbest`, to już cząstka nr 6 w tej samej iteracji korzysta z nowego lidera. To tzw. **asynchroniczne PSO** (zwykle zbiega szybciej niż wariant synchroniczny, gdzie `gbest` aktualizuje się dopiero po przejściu wszystkich cząstek).
   - **Early Stopping (linie 128–131 oraz 145–149)**:
     Jeśli poprawa `prev_gbest_val - gbest_val` między kolejnymi iteracjami jest mniejsza lub równa tolerancji `tol = 1e-9`, rośnie licznik `no_improvement`. Gdy osiągnie `max_no_improvement`, pętla kończy się wcześniej.

### B. Eksperymenty — `experiments.py`

- Stosuje metodę **OFAT (One-Factor-At-a-Time)** (linie 58–75): w każdej z 6 grup testowych zmieniany jest tylko **jeden** parametr, a pozostałe pozostają na wartościach bazowych (`DEFAULT_PARAMS`).
- Ze względu na stochastyczność PSO każda konfiguracja uruchamiana jest **10 razy** (`num_runs=10`), a wyniki agregowane są do: `Best`, `Avg`, `Worst`, średniej liczby iteracji, liczby wczesnych zatrzymań i średniego czasu wykonania (ms).
- **Uśrednianie krzywych różnej długości (linie 112–119)**: ponieważ early stopping kończy różne uruchomienia w różnych iteracjach, krótsze przebiegi są dopełniane ich ostatnią wartością (`r["best_history"][-1]`) przed obliczeniem średniej krzywej.
- Wyniki zapisywane są do arkusza `pso_experiment_results.xlsx` za pomocą `openpyxl`.

### C. Wizualizacja — `plotter.py`

- **Skala logarytmiczna (`LogNorm`, `set_yscale("log")`)**: na krańcach dziedziny $[-4.5, 4.5]^2$ wartości $f(x, y)$ sięgają ponad $10^5$, podczas gdy minimum to $0.0135$. Na zwykłej skali liniowej dolina wokół minimum byłaby całkowicie niewidoczna.
- Generowane wykresy:
  1. `plot_convergence` — spadek `gbest` w kolejnych iteracjach.
  2. `plot_contour_swarm` — mapa poziomicowa 2D z pozycjami początkowymi (białe), końcowymi (cyjan), trajektorią `gbest` (pomarańczowa) i najlepszym punktem (czerwona gwiazdka).
  3. `plot_parameter_comparison` — zestawienie uśrednionych krzywych zbieżności dla różnych wartości jednego parametru.

---

## 4. Najważniejsze wnioski z eksperymentów

1. **Waga bezwładności `w`** (najbardziej krytyczny parametr):
   - Dla $w \in [0.2, 0.7]$ algorytm w 10/10 prób trafia w dokładne optimum `0.013514`. Małe $w = 0.2$ zbiega najszybciej (~48 iteracji), bo ruch jest szybko tłumiony.
   - Dla $w = 0.9$ cząstki zaczynają przestrzeliwać dolinę, a dla **$w = 1.1$** (powyżej 1) pęd rośnie zamiast maleć — rój oscyluje i wynik wyraźnie się psuje (`avg = 0.0446`, `worst = 0.1296`).
2. **Współczynniki `c1` i `c2`**:
   - Zbyt niskie `c1 = 0.5` (przy `c2 = 1.5`) powoduje **przedwczesną zbieżność** (`worst = 5.495`) — cząstki ignorują własną pamięć i ślepo pędzą do pierwszego losowego `gbest`.
   - Zrównoważone wartości $c_1, c_2 \in [1.0, 2.0]$ dają 100% powtarzalności.
3. **Liczba cząstek i iteracji**:
   - Przestrzeń jest 2-wymiarowa, więc już 10–20 cząstek znajduje minimum, a 30–50 gwarantuje pełną stabilność w czasie poniżej 5 ms.
   - 20 iteracji to za mało (`avg = 0.013814`), natomiast 100 iteracji w zupełności wystarcza.
4. **Early stopping (`max_no_improvement`)**:
   - `10` jest zbyt agresywne (`avg = 0.01625`) — przerywa przed dokładnym doprecyzowaniem minimum.
   - `20–30` to złoty środek: osiąga pełną precyzję (`0.01351449`) i ucina zbędne iteracje.

---

## 5. Typowe pytania na obronie (Q&A)

1. **Dlaczego minimum nie wynosi 0, skoro funkcja wygląda jak funkcja Beale'a?**
   Bo w treści zadania w pierwszym nawiasie jest minus: $(1.5 - x - xy)^2$ zamiast $+xy$. Układ trzech równań przyrównanych do zera jest wtedy sprzeczny, więc najniższa możliwa wartość to ok. $0.013514$ w punkcie $(2.51799, -0.37463)$.

2. **Co się stanie, gdy ustawimy $w > 1$ (np. $1.1$)?**
   Człon bezwładności wzmacnia prędkość w każdej iteracji zamiast ją tłumić. Cząstki dostają zbyt dużego pędu, przestrzeliwują dolinę minimum i oscylują (ratuje je tylko `v_max` i `clamp`, ale nie potrafią dokładnie osiąść w minimum).

3. **Co się stanie, gdy $c_1 = 0$ albo jest bardzo małe, a $c_2$ duże?**
   Algorytm traci pamięć indywidualną (wariant *social-only*). Wszystkie cząstki rzucają się na początkowy `gbest`, co grozi przedwczesną zbieżnością do słabego obszaru — widać to w wynikach dla `c1 = 0.5` (`worst = 5.495`).

4. **A odwrotnie: co gdy $c_2 = 0$?**
   Wariant *cognitive-only* — cząstki nie wymieniają się informacjami, każda szuka minimum na własną rękę jak 30 niezależnych błądzeń lokalnych.

5. **Po co w kodzie jest `v_max` i `clamp`?**
   `v_max` (połowa szerokości dziedziny = 4.5) zapobiega eksplozji prędkości. `clamp` na pozycji pilnuje ograniczeń zadania $x, y \in [-4.5, 4.5]$, żeby cząstka nie wyszła poza dziedzinę.

6. **Czy w tym kodzie PSO jest synchroniczne czy asynchroniczne?**
   Asynchroniczne — `gbest` jest aktualizowany od razu wewnątrz pętli po cząstkach (`pso_algorithm.py`, linie 123–126), więc kolejne cząstki w tej samej iteracji widzą już poprawionego lidera.

7. **Dlaczego eksperymenty są puszczane po 10 razy dla każdego parametru?**
   Bo PSO jest niedeterministyczne (losowa inicjalizacja oraz losowe $r_1, r_2$). Pojedynczy przebieg może być szczęśliwy lub pechowy — dopiero statystyka (`Best`, `Avg`, `Worst`) pokazuje realny wpływ parametru.

8. **Czym różni się PSO od GA z zadania 1?**
   GA z zadania 1 operowało na dyskretnej reprezentacji (harmonogramie zadań) za pomocą selekcji, krzyżowania i mutacji. PSO jest natywnie zaprojektowane do optymalizacji ciągłej w $\mathbb{R}^n$, ma mniej operatorów i parametrów oraz zbiega znacznie szybciej dzięki kierunkowemu ruchowi z pamięcią (`pbest`, `gbest`).
