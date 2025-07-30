<div class="flushright" markdown="1">

Конспект Шорохова Сергея

Если нашли опечатку/ошибку - пишите @le9endwp

</div>

# Глава 1. Введение

## §1. Множества и их отношения

**Def.** Множество - набор каких-то элементов, т.е. либо $x \in A$, либо
$x \notin A (\forall x)$

**Def.** A, B - множества. $A \subset B$ - А подмножество В, т.е.
$\forall x \in A \Rightarrow x \in B$

**Def.**
$A = B \Leftrightarrow \begin{cases} A \subset B \\ B \subset A \end{cases}$

**Def.** $\emptyset$ - пустое множество, т.е.
$\forall x, x \notin \emptyset$

**Rem.** $\forall A \emptyset \subset A$

**Def.**
$\begin{cases} A \subset B \\ A \neq B \end{cases} \Leftrightarrow A \subsetneq B \Leftrightarrow$
А - собственное подмножество

**Операции:**

- Пересечение
  $A \bigcap B = \{ x | \begin{cases} x \in A \\ x \in B \end{cases} \}$

- Объединение $A \bigcup B = \{x | x \in A$ или $x \in B \}$

- Разность
  $A \backslash B = \{x | \begin{cases} x \in A \\ x \notin B \end{cases} \}$

- Симметрическая разность
  $A \triangle B = (A \backslash B) \bigcup (B \backslash A)$

**Способы задания множеств:**

- Перечисление

- Неполное перечисление

- Словесно

- С помощью функции

**Канонические обозначения:**

- $\mathbb{N}$ - натуральные числа

- $\mathbb{Z}$ - целые числа

- $\mathbb{Q}$ - рациональные числа

- $\mathbb{R}$ - вещественные числа

- $\mathbb{C}$ - комплексные числа

- $\mathbb{P}$ - простые числа

**Def.** $\langle a; b \rangle (a \in A, b \in B)$ - упорядоченная пара

$\langle a; b \rangle = <p, q> \Leftrightarrow \begin{cases} a = p \\ b = q \end{cases}$

**Def.** $<a_1, a_2 \ldots a_n> (a_k \in A_k \forall k)$ - кортеж
(упорядоченная n-ка)

$<a_1 \ldots a_n> = <b_1 \ldots b_n> \Leftrightarrow a_k = b_k \forall k$

**Def.** Декартово произведение
$A \times B = \{ \langle a; b \rangle | a \in A, b \in B\}$

**Правила Д’Моргана:**

1.  $A \backslash (\bigcap\limits_{\alpha \in I} B_\alpha) = \bigcup\limits_{\alpha \in I} (A \backslash B_\alpha)$

2.  $A \backslash (\bigcup\limits_{\alpha \in I} B_\alpha) = \bigcap\limits_{\alpha \in I} (A \backslash B_\alpha)$

**Доказательство 2**

$$x \in A \backslash (\bigcup\limits_{\alpha \in I} B_\alpha) \Leftrightarrow \begin{cases} x \in A \\ x \notin \bigcup\limits_{\alpha \in I} B_\alpha \end{cases} \Leftrightarrow \begin{cases} x \in A \\ x \notin B_\alpha, \forall \alpha \in I \end{cases} \Leftrightarrow x \in A \backslash B_\alpha, \forall \alpha \in I \Leftrightarrow x \in \bigcap\limits_{\alpha \in I} (A \backslash B_\alpha)$$

**Теорема**

- $A \bigcup (\bigcap\limits_{\alpha \in I} B_\alpha) = \bigcap\limits_{\alpha \in I} (A \bigcup B_\alpha)$

- $A \bigcap (\bigcup\limits_{\alpha \in I} B_\alpha) = \bigcup\limits_{\alpha \in I} (A \bigcap B_\alpha)$

**Доказательство**

$$x \in A \bigcap (\bigcup\limits_{\alpha \in I} B_\alpha) \Leftrightarrow \begin{cases} x \in A \\ x \in \bigcup\limits_{\alpha \in I} B_\alpha \end{cases} \Leftrightarrow \begin{cases} x \in A \\ \exists \alpha \in I : x \in B_\alpha \end{cases} \Leftrightarrow \exists \alpha \in I : x \in A \bigcap B_\alpha \Leftrightarrow x \in \bigcup\limits_{\alpha \in I} (A \bigcap B_\alpha)$$

**Def.** Бинарным отношением R на $A \times B$ называется
$R \subset A \times B$

$R = \{\langle a; b \rangle | a \in A, b \in B\}$

$\langle a; b \rangle \in R \Leftrightarrow aRb$

**Def.**
$\sigma_R = \{a \in A | \exists b \in B : \langle a; b \rangle \in R\}$ -
область определения бинарных отношений

**Def.**
$\rho_R = \{b \in B | \exists a \in A : \langle a; b \rangle \in R\}$ -
множество значений бинарных отношений

**Def.** $R^{-1} = \{<b, a> | \langle a; b \rangle \in R\}$ - обратное
отношение

**Def.**
$R_1 \circ R_2 \subset A \times C; \begin{cases} R_1 \subset A \times B \\ R_2 \subset B \times C \end{cases}$

$R_1 \circ R_2 = \{<a, c> | \exists b \in B \begin{cases} \langle a; b \rangle \in R_1 \\ <b, c> \in R_2 \end{cases} \}$

**Свойства бинарных отношений:**

1.  R - рефлексивное, если $\forall a \in A <a, a> \in R$

2.  R - иррефлексивное, если $\forall a \in A <a, a> \notin R$

3.  R - симметричное, если
    $\langle a; b \rangle \in R \Rightarrow <b, a> \in R$

4.  R - антисимметричное, если
    $\begin{cases} \langle a; b \rangle \in R \\ <b, a> \in R \end{cases} \Rightarrow a = b$

5.  R - транзитивное, если
    $\begin{cases} \langle a; b \rangle \in R \\ <b, c> \in R \end{cases} \Rightarrow <a, c> \in R$

**Def.** R - отношение эквивалентности, если R рефлексивно, симметрично,
транзитивно

**Def.** R - нестрогий частичный порядок, если R - рефлексивно,
антисимметрично, транзитивно

**Def.** R - строгий частичный порядок, если R - иррефлексивно,
транзитивно

**Def.**
$\begin{cases} \langle a; b \rangle \in R \\ <a, c> \in R \end{cases} \Rightarrow b = c$,
тогда R - функция f

**Def.** f - инъективная, если
$\begin{cases} f(x_1) = a \\ f(x_2) = a \end{cases} \Rightarrow x_1 = x_2$

**Def.** f - сюрьективная, если
$\forall y \in Y \exists x \in X : f(x) = y$

**Def.** f - биективная, если f - инъективная и сюрьективная

## §2. Вещественные числа

**Две операции в $\mathbb{R}$**

- Сложение

  - $a + b = b + a$ - коммутативность

  - $(a + b) + c = a + (b + c)$ - ассоциативность

  - $\exists 0 \in \mathbb{R} : a + 0 = a; \forall a \in \mathbb{R}$ -
    существование нейтрального

  - $\forall a \in \mathbb{R} \exists -a : a + (-a) = 0$ - существование
    обратного

- Умножение

  - $a \cdot b = b \cdot a$- коммутативность

  - $(a \cdot b) \cdot c = a \cdot (b \cdot c)$- ассоциативность

  - $\exists 1 \in \mathbb{R} : a \cdot 1 = a; \forall a \in \mathbb{R}$ -
    существование нейтрального

  - $\forall a \neq 0 \in \mathbb{R} \exists a^{-1} \in \mathbb{R} : a \cdot a^{-1} = 1$ -
    существование обратного

- $\forall a, b, c \in \mathbb{R} (a + b) \cdot c = a \cdot c + b \cdot c$ -
  дистрибутивность

**Rem.** Если соблюдаются все эти аксиомы, то поле

**Аксиомы порядка:**

- $\forall x, y \in \mathbb{R} x \leq y$ или $y \leq x$

- $OA$ $a \leq b \Rightarrow a + c \leq b + c$

- $OM$
  $\begin{cases} a \geq 0 \\ b \geq 0 \end{cases} \Rightarrow 0 \leq a \cdot b$

**Аксиома полноты:**

$A \neq \emptyset$, $B \neq \emptyset$, $A, B \subset R$

$\begin{gathered} \forall a \in A \\ \forall b \in B \end{gathered}$
$a \leq b \Rightarrow \exists c \in R: a \leq c \leq b (\forall a \in A, \forall b \in B)$

$\mathbb{Q}$ не удовлетворяет аксиоме полноты:

$A = \{ x \in \mathbb{Q} | x^2 < 2 \}$

$B = \{ x \in \mathbb{Q} | x^2 > 2 \}$

Между ними только $\sqrt{2} \notin \mathbb{Q}$

**Следствие (принцип Архимеда):**

$\begin{gathered}
    \forall x \in \mathbb{R} \\
    \forall y \in \mathbb{R}, y > 0
\end{gathered}$ $\exists n \in \mathbb{N} : x < ny$

**Доказательство**

$fix$ $y > 0$

$A = \{ x \in \mathbb{R} | \exists n : x < ny\}$

Пусть
$A \neq \mathbb{R} \Rightarrow \mathbb{R} \backslash A = B \neq \emptyset$

$A \neq \emptyset$, т.к. $0 \in A$

Левее ли $A$, чем $B$

Пусть
$\begin{gathered} b \in B \\ a \in A \end{gathered}: b < a < ny \Rightarrow b < ny \Rightarrow b \in A$,
но из
$\mathbb{R} \backslash A = B \Rightarrow A \bigcap B = \emptyset \Rightarrow$

$\begin{cases}
\Rightarrow \forall a \in A, b \in B, a \leq b \\
A, B \subset \mathbb{R} \\ 
A \neq \emptyset \\
B \neq \emptyset 
\end{cases}$
$\Rightarrow \exists c \in \mathbb{R}: a \leq b \leq c (\forall a \in A, b \in B)$

$\begin{cases}
c - y < c \Rightarrow c - y \in A \Rightarrow \exists n \in \mathbb{N} : c - y < ny \Rightarrow c < (n+1)y \\
c < c + y \Rightarrow c \in B
\end{cases}$ $\Rightarrow c + y < (n+2)y \Rightarrow c + y \in A$ -
противоречие $A \bigcap B = \emptyset \Rightarrow A = \mathbb{R}$

**Следствие:**

$\forall \varepsilon > 0$
$\exists n \in \mathbb{N}: \frac{1}{n} < \varepsilon$

$\frac{1}{n} < \varepsilon \Leftrightarrow 1 < n \varepsilon$ - принцип
Архимеда $x = 1, y = \varepsilon$

**Аксиома индукции** (метод математической индукции; принцип
математической индукции)

$P_1, P_2, \ldots P_n \ldots$ - последователььностьь утверждений

$\begin{cases}
P_1 \text{-- истина (база)} \\
P_n \text{-- истина} \Rightarrow P_{n+1} \text{-- истина (переход)}
\end{cases}$ $\Rightarrow \forall n \in \mathbb{N}$ $P_n$ - истина

**Th.** Во всяком конечном множестве вещественных чисел есть наибольший
и наименьший элементы

$a = maxA \Leftrightarrow \begin{cases} a \in A \\ \forall x \in A \end{cases} x \leq a$

$b = minA \Leftrightarrow \begin{cases} b \in A \\ \forall x \in A \end{cases} x \geq b$

**Доказательство**

$P_n$ - в множестве из $n$ элементов есть наибольший и наименьший
элементы

1.  $P_1$ - истина, т.к. в множестве из 1 элемента он и наибольший, и
    наименьший

2.  $P_n \Rightarrow P_{n+1}$

    $A = \{ a_1, a_2 \ldots a_{n+1} \}$

    $B = \{ b_1, b_2 \ldots b_n \}$ - $n$ элементов
    $\Rightarrow \exists maxB = \tilde{a}$

    $\tilde{a} \in B \Rightarrow \tilde{a} \in A$

    $\forall k, 1 \leq k \leq n$ $a_k \leq \tilde{a}$

    Случаи:

    - $a_k \leq \tilde{a} \leq a_{n+1} \Rightarrow a_{n+1} = maxA$

    - $a_{n+1} \leq \tilde{a} \Rightarrow \tilde{a} = maxA$

**Def.** Множество $A$ называется ограниченным сверху, если
$\exists c \in \mathbb{R}: a \leq c, \forall a \in A$

**Def.** Множество $A$ называется ограниченным снизу, если
$\exists c \in \mathbb{R}: a \geq c, \forall a \in A$

**Def.** Множество $A$ называется ограниченным, если оно ограничено и
сверху, и снизу

$\exists c_1, c_2: c_1 \leq a \leq c_2, \forall a \in A$

**Th.**

1.  В любом непустом ограниченном сверху множестве целых чисел есть
    наибольший элемент

2.  В любом непустом ограниченном снизу множестве целых чисел есть
    наименьший элемент

3.  В любом непустом ограниченном сверху множестве натуральных чисел
    есть наибольший и наименьший элементы

**Доказательство**

$A; a \in \mathbb{Z}, \forall a \in A$

$b$ - верхняя граница

$\forall a \in A$ $a \leq b$. Возьмем $\tilde{a} \in A$

$\begin{cases}
    B = \{ a \in A | a \geq \tilde{a} \} \\
    B \text{- конечное множество} \end{cases}
\Rightarrow \exists maxB = \tilde{\tilde{a}}$

$\tilde{\tilde{a}} = maxA$, т.к.
$\tilde{a} \leq \beta \in B \leq \tilde{\tilde{a}}$

**Def.** $x \in \mathbb{R}; [x] = \lfloor x \rfloor$ - целая часть числа

$[x]$ - наибольшее целое число, не превосходящее $x$

**Свойства:**

1.  $[x] \leq x \leq [x] + 1$

2.  $x - 1 \leq [x] \leq x$

**Доказательство**

1.  $[x] \leq x$ - определение

2.  Пусть $x \geq [x] + 1 \in \mathbb{Z}$, тогда $[x]$ не наиболььшее,
    что противоречит определению

**Th.** $x, y \in \mathbb{R}: y > x \Rightarrow \begin{gathered}
    1) \exists r \in \mathbb{Q}: x < r < y \\
    2) \exists s \notin \mathbb{Q}: x < s < y
\end{gathered}$

**Доказательство**

1.  $x < y \Rightarrow y - x > 0 \Rightarrow$ (по следствию из принципа
    Архимеда)
    $\exists n \in N: \frac{1}{n} < y - x \Leftrightarrow \frac{1}{n} + x < y$

    $r = \frac{[nx]+1}{n} > \frac{nx}{n} = x$

    $r = \frac{[nx]+1}{n} = \frac{[nx]}{n} + \frac{1}{n} \leq \frac{nx}{n} + \frac{1}{n} = x + \frac{1}{n} < y$

    $x < r < y$

2.  $\sqrt{2} \notin \mathbb{Q}$

    $x < y \Rightarrow x - \sqrt{2} < y - \sqrt{2} \Rightarrow$ (по п.1)
    $\exists r \in \mathbb{Q}: x - \sqrt{2} < r < y - \sqrt{2} \Rightarrow x < r + \sqrt{2} < y$

    $s = r + \sqrt{2} \notin \mathbb{Q}$

## §3. Супремум и инфимум

**Def.** $A \subset \mathbb{R}, A \neq \emptyset, A$ - ограничено сверху

$supA$ - наименьшая (точная) верхняя граница

**Def.** $A \subset \mathbb{R}, A \neq \emptyset, A$ - ограничено снизу

$infA$ - наибольшая (точная) нижняя граница

**Th.**

1.  У любого непустого ограниченного сверху множества вещественных чисел
    существует единственный супремум

2.  У любого непустого ограниченного снизу множества вещественных чисел
    существует единственный инфимум

**Доказательство**

1.  Единственность - очевидно

2.  Существование:

    $A \neq \emptyset, A \subset \mathbb{R}$

    $B$ - множество всех верхних границ

    $B \neq \emptyset, B \subset \mathbb{R}$

    $\begin{gathered}
            \forall a \in A \\
            \forall b \in B
        \end{gathered}$ $a \leq b$

    Тогджа по аксиоме полноты
    $\exists c \in \mathbb{R}: a \leq c \leq b (\forall a \in A, \forall b \in B)$

    $\forall a \in A$ $a \leq c \Rightarrow c$ - верхняя граница
    $A \Rightarrow c \in B$

    $\forall b \in B$
    $c \leq b \Rightarrow c = minB \Rightarrow c = supA$

**Следствия:**

1.  $\begin{cases}
    A \neq \emptyset \\
    A \subset B \subset \mathbb{R} \\
    B \text{-ограничено сверху}
    \end{cases}
    \Rightarrow supA \leq supB$

2.  $\begin{cases}
            A \neq \emptyset \\
            A \subset B \subset \mathbb{R} \\
            B \text{ограничено снизу} \end{cases}
        \Rightarrow infA \geq infB$

**Доказательство**

$\begin{cases}
    B \neq \emptyset \\
    B \subset \mathbb{R} \\ 
    B \text{ограничено сверху} \end{cases}
\Rightarrow \exists supB \Rightarrow \forall b \in B$
$b \leq supB \Rightarrow \forall a \in A$
$a \leq supB \Rightarrow \exists supA \Rightarrow supA \leq supB$

**Обозначения:**

1.  $A$ не является ограниченным сверху $\Rightarrow supA = + \infty$

2.  $A$ не ограничено снизу $\Rightarrow infA = - \infty$

**Th.** (характеристика супремума и инфимума)

1.  $a = supA \Leftrightarrow \begin{cases}
            \forall x \in A, x \leq a \\
            \forall \varepsilon > 0, \exists x \in A: x > a - \varepsilon
        \end{cases}$

2.  $b = infA \Leftrightarrow \begin{cases}
            \forall x \in A, x \geq b \\
            \forall \varepsilon > 0, \exists x \in A: x < b + \varepsilon
        \end{cases}$

**Доказательство**

1.  $\forall x \in A, x \geq b \Rightarrow b$ - нижняя граница $A$

2.  $\forall \varepsilon > 0, \exists x \in A: x < b + \varepsilon \Rightarrow$
    все числа $> b$ не являются нижними гранциами множества
    $A \Rightarrow b$ - наибольшая нижняя граница $\Rightarrow b = infA$

**Th.** о вложенных отрезках

$[a_1; b_1] \supset [a_2;b_2] \supset \ldots \supset [a_n;b_n] \supset \ldots$,
тогда
$\exists c \in \mathbb{R}: c \in [a_n;b_n] \forall n \in \mathbb{N}$

Другими словами
$\bigcap\limits_{n=1}^{+ \infty} [a_n;b_n] \neq \emptyset$

**Доказательство**

$a_1 \leq a_2 \leq a_3 \ldots$, $A = \{ a_1, a_2 \ldots \}$

$b_1 \geq b_2 \geq b_3 \ldots$, $B = \{ b_1, b_2 \ldots \}$

$A \neq \emptyset, B \neq \emptyset; A, B \subset R$

$\forall a_n \leq b_n$

$?a_k \leq b_m$

1.  $k < m$, $a_k \leq a_m \leq b_m$

2.  $k > m$. $a_k \leq b_k \leq b_m$

3.  $k = m$, $a_k \leq b_m$

По аксиоме полноты
$\exists c \in \mathbb{R}: a \leq c \leq b (\forall a \in A, \forall b \in B) \Rightarrow \forall n$
$a_n \leq c \leq b_n \Rightarrow c \in [a_n;b_n] \forall n \in \mathbb{N}$

**Замечания:**

1.  Таких точек может быть много

2.  Интервалов недостаточно

3.  Лучей недостаточно

# Глава 2. Последовательности вещественных чисел

## §1. Пределы последовательности

**Def.** Последовательность - функция натурального аргумента

$f: \mathbb{N} \rightarrow \mathbb{R} \Leftrightarrow \{f_n\}_{n=1}^{+\infty}$

$f(1) \leftrightarrow f_1$

Как задавать последовательность?

- Формулой (форму общего члена последовательности)

- Описательно

- Рекуррентно

- График последовательности (двумерный или одномерный, но второй
  неудобен, если какие-то точки дублируются)

**Def.** $x_n$ называется ограниченной сверху, если
$\exists M \in \mathbb{R}: \forall n \in \mathbb{N}$ $x_n \leq M$

**Def.** $y_n$ называется ограниченной снизу, если
$\exists m \in \mathbb{R} : \forall n \in \mathbb{N}$ $y_n \geq m$

**Def.** $z_n$ называется ограниченной, если она ограничена и сверху, и
снизу $\Leftrightarrow \exists c > 0 : \forall n \in \mathbb{N}$
$|z_n| < c$

**Def.** $x_n$ называется монотонно возрастающей, если
$\forall n \in \mathbb{N}$ $x_{n+1} \geq x_n$

**Def.** $y_n$ строго монотонно возрастает, если
$\forall n \in \mathbb{N}$ $y_{n+1} > y_n$

**Def.** $x_n$ монотонно убывает, если $\forall n \in \mathbb{N}$
$x_{n+1} \leq x_n$

**Def.** $y_n$ строго монотонно убывает, если $\forall n \in \mathbb{N}$
$y_{n+1} < y_n$

**Def.** $z_n$ монотонная, если она мотонно возрастает или монотонно
убывает

**Def.** $z_n$ строго монотонная, если она строго монотонно возрастает
или строго монотонно убывает

**Def.(1)** (неклассическое)

$a \in \mathbb{R}$

$a = \lim\limits_{n \rightarrow \infty}{x_n} \Leftrightarrow$ вне любого
интервала, содержащего точку $a$ находится лишь коненчое число членов
ппоследовательности

**Rem.** Можно рассматривать тольько симметричные интервалы

**Def.(2)** (классическое)

$a \in \mathbb{R}$

$a = \lim\limits_{n \rightarrow \infty}{x_n} \Leftrightarrow \forall \varepsilon > 0 \exists N \in \mathbb{N}: \forall n \geq N$
$|x_n - a| < \varepsilon$

Последнее неравенство равносильно выбору симметричного интервала, отсюда
равносильность определений

$\exists N \in \mathbb{N} \Leftrightarrow N = N(\varepsilon)$

**Свойства:**

1.  Если предел существует, то он единственный

    **Доказательство**

    От противного: $\begin{cases}
            \lim\limits_{n \rightarrow \infty}{x_n} = a \\
            \lim\limits_{n \rightarrow \infty}{x_n} = b \\
            a \neq b
        \end{cases}$

    Пусть $\varepsilon = \frac{|b-a|}{3}$, тогда окрестности будут
    непересекающимися $\Rightarrow$ либо вне
    $(a - \varepsilon, a + \varepsilon)$ бесконечно много членов и вне
    $(b - \varepsilon, b + \varepsilon)$ бесконечно много членов, либо
    число $n$ - конечно, оба варианта неверны

2.  Если из последовательности удалить конечное число членов, то предел
    не изменится

3.  Если переставить члены последовательности, то предел не изменится

4.  Если записать некоторые члены последовательности с конечной
    кратностью, то предел не изменится

5.  Если добавить конечное число членов последовательности, то предел не
    изменится

6.  Если изменить конечное число членов последовательности, то предел не
    изменится

7.  Если последовательность имеет предел, то она ограничена

    **Доказательство**

    Окрестность $(a - 1, a + 1)$

    Снаружи лишь конечное число членов, в их множестве существует
    наибольший и наименьший элемент

    Пусть $x_{\tilde{N}}$ - наибольший, а $x_{\tilde{\tilde{N}}}$ -
    наименьший, тогда

    $M = max\{a+1, x_{\tilde{N}}\}$ и
    $m = min\{a-1, x_{\tilde{\tilde{N}}}\}$

    **Lem.**

    $\begin{cases}
            \lim\limits_{n \rightarrow \infty}{x_n} = a \\
            \lim\limits_{n \rightarrow \infty}{y_n} = b
        \end{cases}$
    $\Rightarrow \forall \varepsilon > 0 \exists N : \forall n \geq N \begin{cases}
            |x_n - a| < \varepsilon \\
            |y_n - b| < \varepsilon
        \end{cases}$

    **Доказательство**

    Для $x_n$ $\forall \varepsilon_1 > 0$
    $\exists N_1 : \forall n \geq N_1$ $|x_n - a| < \varepsilon_1$

    Для $y_n$ $\forall \varepsilon_2 = \varepsilon_1$
    $\exists N_2 : \forall n \geq N_2$ $|y_n - b| < \varepsilon_2$

    $\varepsilon = \varepsilon_2 = \varepsilon_1$; $N = max\{N_1, N_2\}$

8.  Предельный переход в неравенстве

    $\begin{cases}
            \lim\limits_{n \rightarrow \infty}{x_n} = a \\
            \lim\limits_{n \rightarrow \infty}{y_n} = b \\
            \forall n \in \mathbb{N}; x_n \leq y_n
        \end{cases}$ $\Rightarrow a \leq b$

    **Доказательство**

    Пусть $b < a$

    Возьмем $\varepsilon = \frac{|a-b|}{3}$, окрестности не пересекаются

    По лемме для нашего $\varepsilon$
    $\exists N:\forall n \geq N \begin{cases}
            |x_n - a| < \varepsilon \\
            |y_n - b| < \varepsilon
        \end{cases}$

    Рассмотрим $\begin{cases}
            x_N \in (a - \varepsilon, a + \varepsilon) \\
            y_N \in (b - \varepsilon, b + \varepsilon)
        \end{cases}$ $\Rightarrow x_N > y_N$ ??

    Значит $a \leq b$

    **Rem.** $\forall n$ $x_n < y_n \not\Rightarrow a < b$

    **Rem.** Необязательно $\forall n$ $x_n \leq y_n$, можно
    использовать $x_n \leq y_n$ $\forall n \geq N_0$

9.  Стабилизация знака

    $\lim\limits_{n \rightarrow \infty}{x_n} = a \neq 0 \Rightarrow \exists N : \forall n \geq N$
    $x_n \cdot a > 0$

    **Доказательство**

    Пусть $\varepsilon = \frac{|a|}{3}$

    $\exists N : \forall n \geq N$ $|x_n - a| < \varepsilon$

10. Принцип двух миллиционеров (теорема о сжатой переменной)

    $\begin{cases}
            \lim\limits_{n \rightarrow \infty}{x_n} = \lim\limits_{n \rightarrow \infty}{z_n} = a \\
            \forall n; x_n \leq y_n \leq z_n
        \end{cases}$
    $\Rightarrow \exists \lim\limits_{n \rightarrow \infty}{y_n} = a$

    **Доказательство**

    Хотим $\varepsilon > 0$ $\exists N : n \geq N$
    $|y_n - a| < \varepsilon$

    $fix \varepsilon > 0$

    По лемме $\exists N : \forall n \geq N \begin{cases}
            |x_n - a| < \varepsilon \\
            |z_n - a| < \varepsilon
        \end{cases}$ $\Leftrightarrow \begin{cases}
            a - \varepsilon < x_n < a + \varepsilon \\
            a - \varepsilon < z_n < a + \varepsilon
        \end{cases}$

    Возьмем $\begin{cases}
            a - \varepsilon < x_n \\
            z_n < a + \varepsilon \\
            x_n \leq y_n \leq z_n
        \end{cases}$
    $\Rightarrow a - \varepsilon < x_n \leq y_n \leq z_n < a + \varepsilon \Rightarrow a - \varepsilon < y_n < a + \varepsilon \Leftrightarrow |y_n - a| < \varepsilon \Rightarrow \exists \lim\limits_{n \rightarrow \infty}{y_n} = a$

    **Rem.** Можно вместо $\forall n \in \mathbb{N}$ использовать
    $\exists N_0 : \forall n \geq N_0$

    **Следствие:** $\forall n \in \mathbb{N} \begin{cases}
            |y_n| \leq z_n \\
            \lim\limits_{n \rightarrow \infty}{z_n} = 0
        \end{cases}$
    $\Rightarrow \lim\limits_{n \rightarrow \infty}{y_n} = 0$

    Доказательство

    $|y_n| \leq z_n \Leftrightarrow -z_n \leq y_n \leq z_n$, дальше очев

    **Rem.** Вместо $\forall n \in \mathbb{N}$ можно
    $\exists N_0 : \forall n \geq N_0$

**Теорема о пределе монотонной последовательности**

1.  Если $x_n$ монотонно возрастает и ограничена сверху, то у нее
    существует пределе

2.  Если $y_n$ монотонно убывает и ограничена снизу, то у нее есть
    предел

3.  Если $z_n$ монотонна, то существование предела равносильно
    ограниченности $z_n$

**Доказательство**

1.  $\begin{cases}
            \{x_1, x_2, x_3 \ldots x_n \ldots \} = X \\
            \exists M : \forall n; x_n \leq M
        \end{cases}$ $\Rightarrow X$ - Ограничена сверху
    $\Rightarrow \exists supX = a$

    Докажем, что $\lim\limits_{n \rightarrow \infty}{x_n} = supX = a$

    $\forall \varepsilon > 0$ $\exists N : \forall n \geq N$
    $|x_n - a| < \varepsilon \Leftrightarrow a - \varepsilon < x_n < a + \varepsilon$

    При этом правая часть верна всегда, докажем левую

    $fix \varepsilon > 0$

    $a = supX \Rightarrow a \cdot \varepsilon \neq supX \Rightarrow \exists x_{\tilde{N}} : x_{\tilde{N}} > a - \varepsilon \Rightarrow \forall n \geq \tilde{N}$
    $x_n > a - \varepsilon$, так как $x_n$ монотонно возрастает

2.  - уже доказано (свойство 7)

    - $\begin{cases}
                  \exists m, M; m \leq z_n \leq M \\
                  z_n \text{- монотонная} \end{cases}$
      $\Rightarrow \left[ \begin{gathered}
                  z_n \uparrow \Rightarrow z_n \leq M \\
                  z_n \downarrow \Rightarrow m \leq z_n
              \end{gathered} \right .$

**Def.** Последовательность $x_n$ называется бесконечно малой, если
$\lim\limits_{n \rightarrow \infty}{x_n} = 0$

**Свойства:**

1.  $\begin{cases}
            x_n -\text{б/м}\\
            y_n \text{- ограничена}\end{cases} \Rightarrow x_n \cdot y_n$ -
    б/м

2.  $\begin{cases}
            \lim\limits_{n \rightarrow \infty}{x_n} = 0 \\
            \lim\limits_{n \rightarrow \infty}{y_n} = 0
        \end{cases}$
    $\Rightarrow \lim\limits_{n \rightarrow \infty}{x_n + y_n} = 0$

3.  $\lim\limits_{n \rightarrow \infty}{x_n} = a \Leftrightarrow x_n = a + \alpha_n$,
    где $\alpha_n$ - б/м

**Доказательство**

1.  $y_n$ - ограничена $\Rightarrow \exists M > 0 : |y_n| \leq M$
    $\forall n \in \mathbb{N}$

    $\lim\limits_{n \rightarrow \infty}{x_n} = 0 \Leftrightarrow \forall \varepsilon > 0$
    $\exists N : \forall n \leq N$ $|x_n| < \frac{\varepsilon}{M}$

    Хотим $\forall \varepsilon > 0$ $\exists N : \forall n \geq N$
    $|x_n \cdot y_n - 0| < \varepsilon$

    $fix \varepsilon > 0$

    Знаем, что $\exists N : \forall n \geq N$ $\begin{cases}
            |x_n| < \frac{\varepsilon}{M} \\
            |y_n| \leq M
        \end{cases}$ $\Rightarrow |x_n \cdot y_n| < \varepsilon$

2.  $fix \varepsilon > 0$

    $\begin{cases}
            \lim\limits_{n \rightarrow \infty}{x_n} = 0 \\
            \lim\limits_{n \rightarrow \infty}{y_n} = 0
        \end{cases}$ $\Rightarrow$ по лемме $\varepsilon > 0$
    $\exists N : \forall n \geq N \begin{cases}
            |x_n| < \frac{\varepsilon}{2} \\
            |y_n| < \frac{\varepsilon}{2}
        \end{cases}$

    $|x_n + y_n| \leq |x_n| + |y_n| < \frac{\varepsilon}{2} + \frac{\varepsilon}{2} = \varepsilon \Rightarrow \forall \varepsilon > 0$
    $\exists N : \forall n \geq N$
    $|(x_n + y_n) - 0| < \varepsilon \Rightarrow \lim\limits_{n \rightarrow \infty}{x_n + y_n} = 0 \Rightarrow (x_n + y_n)$ -
    б/м

3.  $\forall \varepsilon > 0$ $\exists N : \forall n \geq N$
    $|x_n - a| < \varepsilon \Leftrightarrow |(x_n - a) - 0| < \varepsilon$

    Обозначение $x_n - a = \alpha_n$, тогда

    $|\alpha_n - 0| < \varepsilon$

    $|\alpha_n| < \varepsilon$, т.е.
    $\lim\limits_{n \rightarrow \infty}{\alpha_n} = 0 \Rightarrow \alpha_n$ -
    б/м, а $x_n = a + \alpha_n$, где $\alpha_n$ - б/м

**Th.** об арифметических действиях с пределами

1.  $\begin{cases}
            \lim\limits_{n \rightarrow \infty}{x_n} = a \\
            \lim\limits_{n \rightarrow \infty}{y_n} = b
        \end{cases}$
    $\Rightarrow \exists \lim\limits_{n \rightarrow \infty}{x_n + y_n} = a + b$

2.  $\begin{cases}
            \lim\limits_{n \rightarrow \infty}{x_n} = a \\
            \lim\limits_{n \rightarrow \infty}{y_n} = b
        \end{cases}$
    $\Rightarrow \exists \lim\limits_{n \rightarrow \infty}{x_n \cdot y_n} = a \cdot b$

3.  $\begin{cases}
            \lim\limits_{n \rightarrow \infty}{x_n} = a \\
            \lim\limits_{n \rightarrow \infty}{y_n} = b \neq 0
        \end{cases}$
    $\Rightarrow \exists \lim\limits_{n \rightarrow \infty}{\frac{x_n}{y_n}} = \frac{a}{b}$

4.  $\lim\limits_{n \rightarrow \infty}{x_n} = a \Rightarrow \lim\limits_{n \rightarrow \infty}{|x_n|} = |a|$

**Доказательство**

1.  $\lim\limits_{n \rightarrow \infty}{x_n} = a \Leftrightarrow x_n = a + \alpha_n, \alpha_n$
    – б/м

    $\lim\limits_{n \rightarrow \infty}{y_n} = b \Leftrightarrow y_n = b + \beta_n, \beta_n$
    – б/м

    $x_n + y_n = a + \alpha_n + b + \beta_n = (a + b) + (\alpha_n + \beta_n) = a + b + \gamma_n \rightarrow a + b$

2.  $\lim\limits_{n \rightarrow \infty}{x_n} = a \Leftrightarrow x_n = a + \alpha_n$

    $\lim\limits_{n \rightarrow \infty}{y_n} = b \Leftrightarrow y_n = b + \beta_n$

    $x_n \cdot y_n = (a + \alpha_n)(b + \beta_n) = ab + a\beta_n + b\alpha_n + \alpha_n\beta_n = ab + \gamma_n \rightarrow ab$

3.  $\lim\limits_{n \rightarrow \infty}{y_n} = b \neq 0 \Rightarrow \exists N : \forall n \geq N$
    $y_n \neq 0$

    $\frac{x_n}{y_n}$ – определено $\forall n \geq N$

    $\lim\limits_{n \rightarrow \infty}{\frac{x_n}{y_n}} = \lim\limits_{n \rightarrow \infty}{x_n \cdot \frac{1}{y_n}}$

    Хотим
    $\lim\limits_{n \rightarrow \infty}{\frac{1}{y_n}} = \frac{1}{b}$

    $\frac{1}{y_n} - \frac{1}{b} = \frac{1}{b = \beta_n} - \frac{1}{b} = \frac{b - b - \beta_n}{b(b+\beta_n)} = (-\beta_n) \cdot \frac{1}{b(b + \beta_n)}$

    Можем выбрать окрестность
    $(b - \varepsilon, b + \varepsilon); \varepsilon = \frac{|b|}{2}$

    $|b(b + \beta_n)| = |b| \cdot |b + \beta_n|$
    $\exists N : \forall n \geq N$ $|\beta_n| < \frac{|b|}{2}$

    $|b| \cdot |b + \beta_n| \leq |b| \cdot (|b| + \frac{|b|}{2}) = k$

    $|b| \cdot |b + \beta_n| \geq |b| \cdot (|b| - |\beta_n|) \geq |b| \cdot \frac{|b|}{2} = M > 0$

    $0 < M \leq |b(b + \beta_n)| \leq k$

    $\frac{1}{k} \leq \frac{1}{|b(b + \beta_n)|} \leq \frac{1}{M} \Rightarrow \frac{1}{|b(b + \beta_n)|}$
    – ограничена $\Rightarrow (-\beta_n) \cdot \frac{1}{b(b + \beta_n)}$
    – б/м
    $\Rightarrow \lim\limits_{n \rightarrow \infty}{\frac{1}{y_n}} = \frac{1}{b} \Rightarrow \lim\limits_{n \rightarrow \infty}{x_n \cdot \frac{1}{y_n}} = a \cdot \frac{1}{b} = \frac{a}{b}$

4.  $\lim\limits_{n \rightarrow \infty}{x_n} = a$

    $x_n = a + \alpha_n$

    $|a| - |\alpha_n| \leq |x_n| = |a + \alpha_n| \leq |a| + |\alpha_n|$

    По принципу двух милиционеров

    $\begin{cases}
            |a| - |\alpha_n| \rightarrow a \\
            |a| + |\alpha_n| \rightarrow a
        \end{cases} \Rightarrow |x_n| \rightarrow a \Leftrightarrow \lim\limits_{n \rightarrow \infty}{|x_n|} = |a|$

### Бесконечные пределы

**Def.**
$\lim\limits_{n \rightarrow \infty}{x_n} = +\infty \Leftrightarrow \forall E \in \mathbb{R}$
$\exists N : \forall n \geq N$ $x_n > E$

или $\forall E \in \mathbb{R}$ вне луча $(E;+ \infty)$ лежит лишь
конечное число членов

**Rem.** Можно рассматривать только $E > 0$

**Def.**
$\lim\limits_{n \rightarrow \infty}{x_n} = - \infty \Leftrightarrow \forall E \in \mathbb{R}$
$\exists N : \forall n \geq N$ $x_n < E$

или вне любого луча вида $(-\infty; E)$ лежит лишь конечное число членов

**Rem.** Можно рассматривать только $E < 0$

**Def.**
$\lim\limits_{n \rightarrow \infty}{x_n} = \infty \Leftrightarrow \forall E > 0$
$\exists N : \forall n \geq N$ $|x_n| > E$

или вне любого множества вида $(-\infty; -E) \bigcup (E; +\infty)$ лежит
лишь конечное число членов

**Наблюдение.**
$\lim\limits_{n \rightarrow \infty}{x_n} = + \infty \Rightarrow \lim\limits_{n \rightarrow \infty}{x_n} = \infty$

$\lim\limits_{n \rightarrow \infty}{x_n} = - \infty \Rightarrow \lim\limits_{n \rightarrow \infty}{x_n} = \infty$

**Def.** $x_n$ – б/б
$\Leftrightarrow \lim\limits_{n \rightarrow \infty}{x_n} = \infty$

**Наблюдение.** $x_n$ – б/б $\Rightarrow x_n$ не является ограниченной

**Утверждение.** $x_n \neq 0$ $\forall n \in \mathbb{N}$

$x_n$ – б/м $\Leftrightarrow \frac{1}{x_n}$ – б/б

**Доказательство**

$\lim\limits_{n \rightarrow \infty}{x_n} = 0 \Leftrightarrow \forall \varepsilon > 0$
$\exists N : \forall n \geq N$
$|x_n| < \varepsilon \Leftrightarrow \frac{1}{|x_n|} > \frac{1}{\varepsilon}$

т.е. $\forall E > 0$ $\exists N : \forall n \geq N$
$|\frac{1}{x_n}| > E \Leftrightarrow \frac{1}{x_n}$ – б/б

**Def.**
$\overline{\mathbb{R}} = \mathbb{R} \bigcup \{-\infty\} \bigcup \{+\infty\}$

**Свойства пределов в $\overline{\mathbb{R}}$**

1.  Предел в $\overline{\mathbb{R}}$ – единственный

    $\begin{cases}
            \lim\limits_{n \rightarrow \infty}{x_n} = a \in \overline{\mathbb{R}} \\
            \lim\limits_{n \rightarrow \infty}{x_n} = b \in \overline{\mathbb{R}}
        \end{cases} \Rightarrow a = b$

2.  Все свойства про добавить/убрать/переставить сохраняются

3.  - $\begin{cases}
                  \forall n; x_n \leq y_n \\
                  \lim\limits_{n \rightarrow \infty}{x_n} = + \infty
              \end{cases} \Rightarrow \exists \lim\limits_{n \rightarrow \infty}{y_n} = + \infty$

    - $\begin{cases}
                  \forall n; x_n \leq y_n \\
                  \lim\limits_{n \rightarrow \infty}{y_n} = -\infty
              \end{cases} \Rightarrow \exists \lim\limits_{n \rightarrow \infty}{x_n} = - \infty$

    **Доказательство**

    $\lim\limits_{n \rightarrow \infty}{y_n} = -\infty \Leftrightarrow \forall E \in \mathbb{R}$
    $\exists N : \forall n \geq N$ $|y_n| < E$

    $x_n \leq y_n < E \Rightarrow \forall E \in \mathbb{R}$
    $\exists N: \forall n \geq N$
    $|x_n| < E \Leftrightarrow \lim\limits_{n \rightarrow \infty}{x_n} = -\infty$

4.  Арифметические действия с пределами в $\overline{\mathbb{R}}$

    Смотрите нудный, но нужный видос Александра Игоревича

## §2. Экспонента

**Неравенство Бернулли**

$\forall n \in \mathbb{N}$ $\forall x \in \mathbb{R}$ $(x > -1)$

$(1 + x)^n \geq 1 + nx$, причем равенство достигается при $x = 0$ или
$n = 1$

**Доказательство по ММИ**

База: $n = 1$

$1 + x \geq 1 + 1 \cdot x$ – верно

Переход: $n \rightarrow n + 1$

$(1 + x)^n \geq 1 + nx$

$(1 + x)^{n+1} \geq (1+x)(1 + nx) = 1 + nx + x + x^2n = 1 + (n+1)x + x^2n \geq 1 + (n+1)x$

**Наблюдение**

1.  $|a| < 1 \Rightarrow \lim\limits_{n \rightarrow \infty}{a^n} = 0 \Leftrightarrow a^n$
    – б/м

2.  $|a| > 1 \Rightarrow \lim\limits_{n \rightarrow \infty}{a^n} = \infty \Leftrightarrow a^n$
    – б/б

**Rem:**
$a > 1 \Rightarrow \lim\limits_{n \rightarrow \infty}{a^n} = + \infty$

**Rem:** Из пункта 2 $\Rightarrow$ пункт 1

**Доказательство**

1.  $|a| > 1 \Rightarrow |a| = 1 + x,\ x > 0$

    $|a|^n = (1 + x)^n \geq 1 + nx$ – б/б
    $(\lim\limits_{n \rightarrow \infty}{(1 + nx)} = + \infty) \Rightarrow \lim\limits_{n \rightarrow \infty}{|a|^n} = + \infty \Leftrightarrow a^n$
    – б/б

**Th.**

$a \in \mathbb{R}$

$x_n = (1 + \frac{a}{n})^n$

- $\{x_n\}$возрастает при $n > -a \Leftrightarrow n + a > 0$ (строго при
  $a \neq 0$)

- $\{x_n\}$ – ограничено сверху

**Доказательство**

- $\frac{x_n}{x_{n-1}} = \frac{(1 + \frac{a}{n})^n}{(1 + \frac{a}{n-1})^{n-1}} = \frac{(n + a)^n \cdot (n - 1)^n}{n^n(n - 1 + a)^{n-1}} = (\frac{(n+a)(n-1)}{n(n-1+a)})^n \cdot \frac{n-1+a}{n-1} = \frac{n-1+a}{n-1} \cdot (1 + \frac{-a}{n(n-1+a)})^n \geq \frac{n-1+a}{n-1} \cdot (1 + \frac{-a}{n(n+1-a)}) = \frac{n-1+a}{n-1} \cdot \frac{n-1+a-a}{n-1+a} = 1$

  $\frac{x_n}{x_{n-1}} \geq 1$, но нужно доказать:
  $\frac{-a}{n(n-1+a)} > -1$

  $\frac{a}{n(n-1+a)} < 1$

  $a < n(n-1+a)$

  $n^2 - n + an - a > 0$

  $n(n-1) + a(n-1) > 0$

  $(n-1)(n+a) > 0$

  Из того, что у нас есть нужно

  $(n+a) > 0 \Leftrightarrow n > -a$, что дано, значит Бернулли разрешен

- $y_n = (1 + \frac{-a}{n})^n$ монотонно возрастает при $n > a$

  $x_n \cdot y_n = (1 + \frac{a}{n})^n \cdot (1 + \frac{-a}{n})^n = (1 - \frac{a^2}{n})^n \leq 1$

  $x_n \leq \frac{1}{y_n} \leq \frac{1}{y_{min}} = const$

**Следствие**
$\exists \lim\limits_{n \rightarrow \infty}{x_n} \in \mathbb{R}$
(монотонность + ограниченность)

**Def.**
$a \in \mathbb{R}\ \exp(a) = \lim\limits_{n \rightarrow \infty}{x_n} = \lim\limits_{n \rightarrow \infty}{(1 + \frac{a}{n})^n}$

**Def.**
$e = \lim\limits_{n \rightarrow \infty}{(1 + \frac{1}{n})^n} = \exp(1)$

**Rem.** $z_n = (1 + \frac{1}{n})^{n+1}$

1.  $z_n$ строго убывает

2.  $\lim\limits_{n \rightarrow \infty}{z_n} = e$

**Доказательство**

1.  $\lim\limits_{n \rightarrow \infty}{z_n} = \lim\limits_{n \rightarrow \infty}{((1 + \frac{1}{n})^n(1 + \frac{1}{n}))} = e \cdot 1 = e$

2.  $z_n = (1 + \frac{1}{n})^{n+1} = (\frac{n+1}{n})^{n+1} = \frac{1}{(\frac{n}{n+1})^{n+1}} = \frac{1}{(1 - \frac{1}{n+1})^{n+1}} = \frac{1}{(1 + \frac{-1}{n+1})^{n+1}}$

    Знаменатель строго возрастает $\Rightarrow$ дробь строго убывает

**Свойства экспоненты:**

1.  $\exp(1) = e;\ \exp(a) = 1$

2.  Монотонность:

    $a \leq b \Rightarrow \exp(a) \leq \exp(b)$

    **Доказательство**

    $1 + \frac{a}{n} \leq 1 + \frac{b}{n}$ – верно $\forall n:$ обе
    дроби $> 0$

    $\Rightarrow (1 + \frac{a}{n})^n \leq (1 + \frac{b}{n})^n \Rightarrow \exp(a) \leq \exp(b)$

3.  $\exp(a) > 0\ \forall a \in \mathbb{R}$

    $(1 + \frac{a}{n})^n > 0$ НСНМ строго возрастает

    $\exists \delta > 0: (1 + \frac{a}{n})^n > \delta > 0 \Rightarrow \exp(a) > \delta > 0$

4.  $\exp(a) \cdot \exp(-a) \leq 1$

    $(1 + \frac{a}{n})^n \cdot (1 + \frac{-a}{n})^n = (1 + \frac{-a^2}{n})^n \leq 1 \Rightarrow \exp(a) \cdot \exp(-a) \leq 1$

5.  $\exp(a) \geq 1 + a\ \forall a \in \mathbb{R}$

    $(1 + \frac{a}{n})^n \geq 1 + n \frac{a}{n} = 1 + a; n > -a \Rightarrow \exp(a) \geq 1 + a$

6.  $a < 1$

    $\exp(a) \leq \frac{1}{1 - a}$

    $\begin{cases}
            \exp(a) \cdot \exp(-a) < 1 \Leftrightarrow \exp(a) \leq \frac{1}{\exp(-a)} \\
            \exp(-a) \geq 1 - a > 0
        \end{cases} \Rightarrow \frac{1}{\exp(-a)} \leq \frac{1}{1-a}$

7.  $\forall n \in \mathbb{N}$

    $(1 + \frac{1}{n})^n < e < (1 + \frac{1}{n})^{n+1}$

    **Доказательство**

    - Правое:

      $z_n = (1 + \frac{1}{n})^{n+1}$ строго убывает

      $\lim\limits_{n \rightarrow \infty}{z_n} = e$

      $fix\ n$

      $(1 + \frac{1}{n+1})^{n+2} < (1 + \frac{1}{n})^{n+1}$

    - Левое:

      Строго убывает и
      $\rightarrow e \Rightarrow e = inf(1 + \frac{1}{n+1})^{n+2} \Rightarrow e \leq (1 + \frac{1}{n+1})^{n+2} < (1 + \frac{1}{n})^{n+1}$

    $(1 + \frac{1}{n})^n = 2\ (n = 1) \Rightarrow 2 < e$

    $(1 + \frac{1}{n})^{n+1} = (1 + \frac{1}{5})^6\ (n = 5) < 3$

    $2 < e < 3$

    $e = 2,718281828459045 \ldots$

**Lem.**
$\lim\limits_{n \rightarrow \infty}{a_n} = a \Rightarrow \lim\limits_{n \rightarrow \infty}{(1 + \frac{a_n}{n})^n} = \exp(a)$

**Доказательство**

$A = 1 + \frac{a}{n};\ B = 1 + \frac{a_n}{n}$

$a_n$ – ограниченная $\Rightarrow \exists M : \begin{cases}
    |A| \leq 1 + \frac{M}{n} \\
    |B| \leq 1 + \frac{M}{n}
\end{cases}$

Докажем, что $\begin{cases}
    A^n - B^n \rightarrow 0 \\
    \lim\limits_{n \rightarrow \infty}{A^n} = \exp(a)
\end{cases} \Leftrightarrow \lim\limits_{n \rightarrow \infty}{B^n} = \exp(a)$

$0 \leq |A^n - B^n| = |(A-B)(A^{n-1} + A^{n-2}B + \ldots + B^{n-1}| = |A-B| \cdot |A^{n-1} + A^{n-2}B \ldots B^{n-1}| \leq |A-B| \cdot (|A^{n-1}| + |A^{n-2}B| + \ldots + |B^{n-1}|) \leq |A - B| \cdot n(1 + \frac{M}{n})^{n-1} = |1 + \frac{a}{n} - 1 - \frac{a_n}{n}| \cdot n \cdot (1 + \frac{M}{n})^{n-1} = \frac{|a - a_n|}{n} \cdot n \cdot (1 + \frac{M}{n})^{n-1} = |a - a_n| \cdot (1 + \frac{M}{n})^{n-1}$

Модуль – б/м, скобка ограничена $\Rightarrow$ выражение
$\rightarrow 0 \Rightarrow A^n - B^n \rightarrow 0$

**Следствие**

$\exp(a) \cdot \exp(b) = \exp(a + b)$

**Доказательство**

$(1 + \frac{a}{n})^n \cdot ( 1 + \frac{b}{n})^n = ( 1 + \frac{a}{n} + \frac{b}{n} + \frac{ab}{n^2})^n = ( 1 + \frac{a + b + \frac{ab}{n}}{n})^n \Leftrightarrow \exp(a) \cdot \exp(b) = \exp(a + b)$,
т.к. $(a + b + \frac{ab}{n}) \rightarrow a + b$

**Следствие:**

1.  $\exp(n) = e^n, n \in \mathbb{N}$

2.  $f(x) = \exp(x)$ – строго возрастает

**Доказательство**

1.  $\exp(n) = \exp(1\ldots1) = \exp(1) \cdot \exp(1) \ldots = e^n$

2.  $t > 0\ \exp(x+t) = \exp(x) \cdot \exp(t) \geq (1 + t)\exp(x)$

**Теорема** $\begin{cases}
    x_n > 0\ \forall n \in \mathbb{N} \\
    \lim\limits_{n \rightarrow \infty}{\frac{x_{n+1}}{x_n}} = a < 1
\end{cases} \Rightarrow x_n$ – б/м

**Доказательство**

$a < 1$, возьмем окрестность радиусом $\frac{a+1}{2}$

$\exists N : \forall n \geq N \frac{x_{n+1}}{x_n} < \frac{a+1}{2}$

$fix\ n > N$

$x_n = \frac{x_n}{x_{n-1}} \cdot \frac{x_{n-1}}{x_{n-2}} \ldots \frac{x_{N+1}}{x_N} \cdot x_N < (\frac{a+1}{2})^{n-N}\cdot x_N$

$0 < x_n < (\frac{a+1}{2})^n \cdot \frac{x_N}{(\frac{a+1}{2})^N} \Rightarrow x_n \rightarrow 0$

**Следствие**

1.  $\lim\limits_{n \rightarrow \infty}{\frac{n^k}{a^n}} = 0$

2.  $\lim\limits_{n \rightarrow \infty}{\frac{a^n}{n!}} = 0$

3.  $\lim\limits_{n \rightarrow \infty}{\frac{n!}{n^n}} = 0$

**Доказательство**

1.  $x_n = \frac{n^k}{a^n} > 0$

    $\frac{x_{n+1}}{x_n} = \frac{(n+1)^k \cdot a^n}{a^{n+1} \cdot n^k} = \frac{1}{a} \cdot (\frac{n+1}{n})^k = \frac{1}{a} \cdot (1 + \frac{1}{n})^k$

    $\lim\limits_{n \rightarrow \infty}{\frac{1}{a} \cdot(1 + \frac{1}{n})^k} = \frac{1}{a} < 1 \Rightarrow \lim\limits_{n \rightarrow \infty}{x_n} = 0$

2.  $x_n = \frac{a^n}{n!}$

    $\frac{x_{n+1}}{x_n} = \frac{a^{n+1} \cdot n!}{(n+1)! \cdot a^n} = \frac{a}{n+1}$

    $\lim\limits_{n \rightarrow \infty}{\frac{a}{n+1}} = 0 < 1 \Rightarrow \lim\limits_{n \rightarrow \infty}{x_n} = 0$

3.  $x_n = \frac{n!}{n^n}$

    $\frac{x_{n+1}}{x_n} = \frac{(n+1)! \cdot n^n}{(n+1)^{n+1} \cdot n!} = \frac{(n+1)n^n}{(n+1)^{n+1}} = \frac{n^n}{(n+1)^n} = \frac{1}{(1 + \frac{1}{n})^n}$

    $\lim\limits_{n \rightarrow \infty}{\frac{1}{(1 + \frac{1}{n})^n}} = \frac{1}{e} < \frac{1}{2} \Rightarrow \lim\limits_{n \rightarrow \infty}{x_n} = 0$

**Теорема Штольца**

$y_n$ строго возрастает и
$\lim\limits_{n \rightarrow \infty}{y_n} = + \infty$

Если
$\exists \lim\limits_{n \rightarrow \infty}{\frac{x_n - x_{n-1}}{y_n - y_{n-1}}} = l \in \overline{\mathbb{R}}$,
то $\exists \lim\limits_{n \rightarrow \infty}{\frac{x_n}{y_n}} = l$

**Доказательство**

1.  $l = 0$

    $\frac{x_n-x_{n-1}}{y_n-y_{n-1}} = z_n$ – б/м
    $\Leftrightarrow \forall \varepsilon > 0\ \exists N : \forall n \geq N\ |z_n| < \varepsilon$

    $fix\ \varepsilon > 0 \rightarrow N$

    $N \leq m < n$

    $x_n - x_{n-1} = z_n(y_n - y_{n-1})$

    $x_n - x_m = (x_n - x_{n-1}) + (x_{n-1} - x_{n-2}) + \ldots + (x_{m+1} - x_m) = z_n(y_n - y_{n-1}) + z_{n-1}(y_{n-1} - y_{n-2}) + \ldots + z_{m+1}(y_{m+1} - y_m)$

    $|x_n - x_m| = |\sum\limits_{k=m+1}^n z_k(y_k - y_{k-1})| \leq \sum\limits_{k=m+1}^n |z_k(y_k-y_{k-1})| < \varepsilon \sum\limits_{k=m+1}^n |y_k - y_{k-1}| = \varepsilon \sum\limits_{k=m+1}^n (y_k - y_{k-1}) = \varepsilon(y_n - y_m)$

    $|x_n - x_m| < \varepsilon(y_n - y_m)$

    $|x_n| - |x_m| \leq |x_n - x_m| < \varepsilon(y_n - y_m) < \varepsilon y_n$

    $|x_n| < |x_m| + \varepsilon y_n$

    $|\frac{x_n}{y_n}| < \varepsilon + \frac{|x_m|}{y_n}$

    $fix\ m; n \rightarrow + \infty \Rightarrow |x_m| = const \Rightarrow \frac{|x_m|}{y_n}$
    – б/м
    $\Rightarrow \frac{|x_n|}{y_n} < \varepsilon \Rightarrow |\frac{x_n}{y_n}| < 2 \varepsilon$

2.  $l \in \mathbb{R}; l \neq 0$

    $\tilde{x_n} = x_n - l y_n$

    $\frac{\tilde{x_n} - \tilde{x_{n-1}}}{y_n - y_{n-1}} = \frac{x_n - l y_n - (x_{n-1} - l y_{n-1})}{y_n - y_{n-1}} = \frac{x_n - x_{n-1}}{y_n - y_{n-1}} - l \rightarrow 0 \Rightarrow \lim\limits_{n \rightarrow \infty}{\frac{\tilde{x_n}}{y_n}} = 0$

    $\frac{\tilde{x_n}}{y_n} = \frac{x_n - l y_n}{y_n} = \frac{x_n}{y_n} - l \Rightarrow \lim\limits_{n \rightarrow \infty}{\frac{x_n}{y_n}} = l$

3.  $l = + \infty$

    $\lim\limits_{n \rightarrow \infty}{\frac{x_n - x_{n-1}}{y_n - y_{n-1}}} = + \infty \Rightarrow \lim\limits_{n \rightarrow \infty}{\frac{y_n - y_{n-1}}{x_n - x_{n-1}}} = 0_+ \Rightarrow \lim\limits_{n \rightarrow \infty}{\frac{y_n}{x_n}} = 0_+ \Rightarrow \lim\limits_{n \rightarrow \infty}{\frac{x_n}{y_n}} = + \infty$

    Надо доказать:

    - $x_n$ строго возрастает

    - $\lim\limits_{n \rightarrow \infty}{x_n} = + \infty$

    $\frac{x_n - x_{n-1}}{y_n - y_{n-1}} \rightarrow + \infty \Rightarrow$
    НСНМ
    $\frac{x_n - x_{n-1}}{y_n-y_{n-1}} > 1 \Rightarrow x_n - x_{n-1} > 0 \Rightarrow x_n > x_{n-1}$

    НСНМ $(N)$ $N \leq m < n$

    $\frac{x_n - x_{n-1}}{y_n - y_{n-1}} > 1 \Rightarrow x_n - x_{n-1} > y_n - y_{n-1}$

    $x_n - x_m = (x_n - x_{n-1}) + (x_{n-1} - x_{n-2}) + \ldots + (x_{m+1} - x_m) > (y_n - y_{n-1}) + (y_{n-1} - y_{n-2}) + \ldots + (y_{m+1} - y_m) = y_n - y_m$

    $x_n - x_m > y_n - y_m > y_n$

    $x_n > x_m + y_n$

    $fix\ m; n \rightarrow + \infty$

    $x_n > x_m + y_n \Rightarrow \lim\limits_{n \rightarrow \infty}{x_n} = + \infty$

4.  $l = - \infty$

    $\tilde{x_n} = -x_n \rightarrow$ случай 3

**Теорема Штольца (ver. 2)**

$y_n : 0 < y_n < y_{n-1}\ \forall n \in \mathbb{N}$

$\lim\limits_{n \rightarrow \infty}{x_n} = \lim\limits_{n \rightarrow \infty}{y_n} = 0$

Если
$\exists \lim\limits_{n \rightarrow \infty}{\frac{x_n-x_{n-1}}{y_n - y_{n-1}}} = l \in \overline{\mathbb{R}}$,
то $\exists \lim\limits_{n \rightarrow \infty}{\frac{x_n}{y_n}} = l$

**Доказательство**

1.  $l = 0$

    $\frac{x_n-x_{n-1}}{y_n-y_{n-1}} = z_n$ – б/м
    $\Rightarrow \forall \varepsilon > 0\ \exists N : \forall n \geq N\ |z_n| < \varepsilon$

    $N \leq m < n$

    $x_n - x_m = (x_n-x_{n-1}) + (x_{n-1} - x_{n-2}) + \ldots + (x_{m+1} - x_m) = z_n(y_n - y_{n-1}) + z_{n-1}(y_{n-1}-y_{n-2}) + \ldots + z_{m+1}(y_{m+1} - y_m)$

    $|x_n - x_m| \leq \sum\limits_{k=m+1}^n |z_k| \cdot |y_k - y_{k-1}| \leq \varepsilon \sum\limits_{k=m+1}^n |y_k - y_{k-1}| = \varepsilon \sum\limits_{k=m+1}^n(y_{k-1} - y_k) = \varepsilon(y_m - y_n)$

    $fix\ m; n \rightarrow + \infty$

    $|x_n - x_m| \leq \varepsilon (y_m - y_n) \Rightarrow |x_m| \leq \varepsilon y_m$

    $|\frac{x_m}{y_m}| \leq \varepsilon$

    $\forall \varepsilon > 0\ \exists N : \forall m \geq N\ |\frac{x_m}{y_m}| \leq \varepsilon \Rightarrow \lim\limits_{n \rightarrow \infty}{\frac{x_n}{y_n}} = 0$

2.  Упражнение

## §3. Подпоследовательности

**Def.** $n_k$ строго возрастающая последовательность натуральных чисел

$x_1, x_2, x_3 \ldots x_n \ldots$ – последовательность

$x_{n_1}, x_{n_2}, x_{n_3} \ldots x_{n_k} \ldots$ – ее
подпоследовательность

**Rem.**

1.  $\exists \lim{x_n} = a \Rightarrow \forall x_{n_k}\ \lim{x_{n_k}} = a$

2.  $n_k \bigcup m_l = \mathbb{N}$

    $\lim{x_{n_k}} = \lim{x_{m_l}} = a \Rightarrow \exists \lim{x_n} = a$

**Rem.** $n_k$ возрастающая последовательность индексов (т.е.
$\mathbb{N}$) $\Rightarrow n_k \geq k$

**Доказательство**

ММИ:

$n_1 \geq 1$

$n_k \geq k \Rightarrow n_{k+1} > n_k \geq k \Rightarrow n_{k+1} > k \Rightarrow n_{k+1} \geq k + 1$

**Теорема** о стягивающихся отрезках

$[a_1;b_1] \supset [a_2;b_2] \supset \ldots \supset [a_n;b_n]$

$\lim(b_n-a_n) = 0 \Rightarrow \exists! c \in [a_n;b_n]\ \forall n \in \mathbb{N}$
и $\lim{a_n} = \lim{b_n} = c$

**Доказательство**

- $\exists c : c \in [a_n;b_n]\ \forall n \in \mathbb{N}$ – знаем из
  теоремы о вложенных отрезках

- Пусть $\exists d : d \in [a_n;b_n]\ \forall n \in \mathbb{N}$

  $|c-d| \leq |a_n-b_n|$

  $|c-d| \leq 0 \Rightarrow c = d$

- $0 \leq |a_n - c| \leq |a_n - b_n|$

  $|a_n - c| \rightarrow 0 \Rightarrow \lim{a_n} = c$

**Теорема** Больцано-Вейерштрасса

Из любой ограниченной последовательности можно выделить сходящуюся
подпоследовательность

**Доказательство**

$x_n$ – ограничена
$\Rightarrow \exists a_0, b_0 : a_0 < x_n < b_n\ \forall n \in \mathbb{N}$

Возьмемь $\frac{a_0 + b_0}{2}$, выберем половину с бесконечным числом
членов. Пусть левая $\Rightarrow a_1 = a_0; b_1 = \frac{a_0 + b_0}{2}$

Возьмем $\frac{a_1 + b_1}{2}$, аналогично. Пусть правая
$\Rightarrow a_2 = \frac{a_1 + b_1}{2}; b_2 = b_1$ итд

Тогда
$[a_0;b_0] \supset [a_1;b_1] \supset \ldots \supset [a_n;b_n] \supset \ldots$

$|a_n - b_n| = |\frac{a_0 - b_0}{2^n}| \Rightarrow |a_n - b_n| \rightarrow 0$

Значит это система стягивающихся отрезков

На первом шаге выберем $x_{n_1} \in [a_0;b_0]$, на втором
$x_{n_2} \in [a_1;b_1]\ (n_2 > n_1)$ и так далее

Получили последовательность $x_{n_k}$

$x_{n_k} \in [a_{k-1};b_{k-1}]$

$a_{k-1} \leq x_{n_k} \leq b_{k-1} \Rightarrow x_{n_k} \rightarrow c$,
где $c = \bigcap [a_n;b_n]$

$\lim{x_{n_k}} = c$

**Def.** $x_n$ – фундаментальная, если
$\forall \varepsilon > 0\ \exists N : \forall n, m \geq N\ |x_m - x_m| < \varepsilon$

**Свойства:**

1.  $x_n$ – сходится $\Rightarrow x_n$ – фундаментальна

2.  $x_n$ – фундаментальна $\Rightarrow x_n$ – ограничена

3.  $x_n$ – фундаментальна и
    $\exists n_k : \lim{x_{n_k}} = a \Rightarrow \lim{x_n} = a$

**Доказательство**

1.  $\lim{x_n} = a \Leftrightarrow \forall \varepsilon > 0\ \exists N : \forall n \geq N\ |x_n - a| < \varepsilon$

    $m, n \geq N \begin{cases}
            |x_n - a| < \varepsilon \\
            |x_m - a| < \varepsilon
        \end{cases}$

    $|x_n - x_m| \ |(x_n - a) + (a - x_m)| \leq |x_n - a| + |a - x_m| < 2 \varepsilon \Rightarrow x_n$
    – фундаментальна

2.  $x_n$ – фундаментальна

    $\forall \varepsilon > 0\ \exists N : \forall n, m \geq N\ |x_n - x_m| < \varepsilon$

    $\varepsilon = 1\ \exists N : \forall n, m \geq N\ |x_n - x_m| < 1$

    $\forall n\ |x_n - x_N| < 1$

    $|x_n| - |x_N| \leq |x_n - x_N| < 1$

    $\forall n \geq N\ |x_n| \leq 1 + |x_N|$

    Значит НСНМ ограничена $\in [-(1 + |x_N|);1 + |x_n|]$

    До $N$ конечное число, их можем просто сравнить с текущей границей,
    т.е.

    $x_n \leq max\{x_1, x_2 \ldots x_{N-1}, 1 + |x_n|\}$

    $x_n \geq min\{x_1, x_2 \ldots x_{N-1}, -(1 + |x_n)\}$

3.  $fix\ \varepsilon > 0$

    $\exists K : \forall k \geq K\ |x_{n_k} - a| < \varepsilon$

    $\exists N : \forall m, n \geq N\ |x_n - x_m| < \varepsilon$

    $k \geq max \{N;K\}$

    $|x_n - a| = |x_n - x_{n_k} + x_{n_k} - a| \leq |x_n - x_{n_k}| + |x_{n_k} - a|$

    $k \geq N \Rightarrow n_k \geq k \geq N \Rightarrow |x_n - x_{n_k}| + |x_{n_k} - a| < 2 \varepsilon \Rightarrow \lim{x_n} = a$

Критерий Коши: $x_n$ – сходится $\Leftrightarrow x_n$ – фундаментальна

**Доказательство**

- уже доказано

- $x_n$ – фундаментальна $\Rightarrow x_n$ – ограничена $\Rightarrow$
  существует сходящаяся подпоследовательность $\Rightarrow x_n$ –
  сходится

**Th.**

1.  $x_n$ – монотонная и не ограниченная сверху
    $\Rightarrow \lim{x_n} = + \infty$

    $x_n$ – монотонная и не ограниченная снизу
    $\Rightarrow \lim{x_n} = - \infty$

2.  $x_n$ – неограниченная сверху
    $\Rightarrow \exists x_{n_k} : \lim{x_{n_k}} = + \infty$

3.  $x_n$ – неограниченная снизу
    $\Rightarrow \exists x_{n_k} : \lim{x_{n_k}} = - \infty$

**Доказательство**

1.  $\begin{cases}
            x_n\ \text{возрастает монотонно} \\
            x_n\ \text{неограничена сверху} \Leftrightarrow \forall M\ \exists N : x_N > M
        \end{cases} \Rightarrow \forall n \geq N\ x_n > M$

    $\forall M\ \exists N : \forall n \geq N\ x_n > M \Leftrightarrow \lim{x_n} = + \infty$

2.  $x_n$ неограничена сверху

    $\exists n_1 : x_{n_1} > 1$

    $\exists n_2 : x_{n_2} > 2 + x_{n_1};\ n_2 > n_1$

    $\exists n_3 : x_{n_3} > 2 + x_{n_2};\ n_3 > n_2$

    $\ldots$

    $\forall k\ \exists x_{n_{k+1}} > 2 + x_{n_k}$

    $x_{n_1} > 1 \Rightarrow \forall k\ x_{n_k} > k$

    $\lim{x_{n_k}} = + \infty$

3.  Аналогично второму пункту

**Def.** $a \in \overline{R};\ a$ – частичный предел последовательности
$x_n$, если $\exists x_{n_k} : \lim{x_{n_k}} = a$

**Th.** $a$ – частичный предел $x_n \Leftrightarrow$ в любой окрестности
точки $a$ содержится бесконечное число членов последовательности

**Доказательство**

- $a$ – частичный предел
  $\Leftrightarrow \exists x_{n_k} \rightarrow a \Leftrightarrow \forall \varepsilon > 0$
  в $(a - \varepsilon; a + \varepsilon)$ содержится бесконечное
  количество членов $x_{n_k}$

- Возьмем $(a-1; a+1)$, возьмем $x_{n_1} : a - 1 < x_{n_1} < a + 1$

  Возьмем $(a - \frac{1}{2}; a + \frac{1}{2})$, возьмем
  $x_{n_2} : a - \frac{1}{2} < x_{n_2} < a + \frac{1}{2}$ и $n_2 > n_1$

  …

  $\forall k\ \exists x_{n_k} : n_k > n_{k-1}$ и
  $a - \frac{1}{k} < x_{n_k} < a + \frac{1}{k} \Rightarrow \lim{x_{n_k}} = a$

**Def.** $x_n$ – последовательность

$\underline{\lim}x_n$ – нижний предел последовательности $x_n$

$\underline{\lim}x_n = \lim(infx_k) = \lim{inf\{x_k, x_{k+1} \ldots\}}$

**Def.** $x_n$ – последовательность

$\overline{\lim}x_n$ – верхний предел последовательности $x_n$

$\overline{\lim}x_n = \lim(supx_k) = \lim{sup\{x_k, x_{k+1} \ldots\}}$

**Договор:** $\lim{\pm \infty} = \pm \infty$

$\begin{cases}
    y_n = inf x_k \text{-- монотонно возрастает} \\
    z_n = sup x_k \text{-- монотонно убывает}
\end{cases} \Rightarrow \begin{cases}
    \exists \lim{y_n} = \underline{\lim} x_n \\
    \exists \lim{z_n} = \overline{\lim} x_n
\end{cases}$

**Th.**

1.  $\forall x_n\ \exists \overline{lim} x_n$ и $\underline{lim} x_n$ в
    $\overline{R}$

2.  $\underline{lim} x_n \leq \overline{lim} x_n$

**Доказательство**

1.  $y_n \uparrow;\ z_n \downarrow$

2.  $\forall n\ y_n \leq z_n \Rightarrow \underline{lim} x_n \leq \overline{lim} x_n$

**Th.**

1.  $\underline{lim} x_n$ – наименьший из частичных пределов

2.  $\overline{lim} x_n$ – наибольший из частичных пределов

    **Rem.** $\forall x_n$ множество частичных пределов непустое

3.  $\underline{lim} x_n = \overline{lim} x_n \Leftrightarrow \exists \lim{x_n} = \underline{lim} x_n = \overline{lim} x_n$

**Доказательство**

1.  $x_n \rightarrow z = sup x_k;\ z_n \downarrow$

    $\overline{lim} x_n = \lim{z_n} = a$

    $z_n$ бежит к $a$ справа

    Хотим $x_{n_k} : \lim{x_{n_k}} = a$

    $(a - 1)$ не является верхней границей для
    $x_n \Rightarrow \exists x_{n_1} > a - 1$

    $(a - \frac{1}{2})$ не является верхней границей для
    $\{x_{n_1} + 1, x_{n_1} + 2 \ldots\} \Rightarrow \exists x_{n_2} > a - \frac{1}{2}$

    …

    $\forall k\ \exists x_{n_k} > a - \frac{1}{k}$, т.к.
    $a - \frac{1}{k}$ не может быть верхней граничей для
    $\{x_{n_{k-1}} + 1, x_{n_{k-1}} + 2 \ldots \}$

    $a - \frac{1}{k} < x_{n_k} \leq z_{n_k} \Rightarrow \lim{x_{n_k}} = a \Rightarrow \overline{lim}x_n$
    – частичный предел

    Пусть $x_{n_m} \rightarrow b$

    $x_{n_m} \leq z_{n_m} \Rightarrow \lim{x_{n_m}} \leq \lim{z_{n_m}} \Rightarrow b \leq a$

    Если $a = + \infty \Rightarrow x_n$ – не ограничена свреху
    $\Rightarrow \exists x_{n_k} \rightarrow + \infty$

    $x_{n_m} \leq z_{n_m}$

    $b \leq + \infty$

2.  $\underline{lim}x_n = \overline{lim}x_n \Leftrightarrow \exists \lim{x_n}$

    - $\exists \lim{x_n} = a \Rightarrow \forall x_{n_k} \rightarrow a$

    - $\forall n\ y_n \leq x_n \leq z_n \Rightarrow \exists \lim{x_n} = \lim{y_n} = \lim{z_n}$

**Th.** Характеристика верхнего и нижнего пределов на языке
$\varepsilon, N$

$a = \underline{lim}x_n \Leftrightarrow \begin{cases}
    \forall \varepsilon > 0\ \exists N : \forall n \geq N\ x_n > a - \varepsilon \\
    \forall \varepsilon > 0\ \exists N : \forall n \geq N\ x_n < a + \varepsilon
\end{cases}$

$b = \overline{lim}x_n \Leftrightarrow \begin{cases}
    \forall \varepsilon > 0\ \exists N : \forall n \geq N\ x_n < b + \varepsilon \\
    \forall \varepsilon > 0\ \exists N : \forall n \geq N\ x_n > b - \varepsilon
\end{cases}$

**Доказательство**

$b = \lim{z_n}$

$z_n = sup\{x_n, x_{n+1} \ldots\}$

- 1.  $\forall \varepsilon > 0\ \exists N : \forall n \geq N\ z_n \leq b + \varepsilon$

  2.  В любом хвосте есть элемент больший, чем
      $b - \varepsilon \Rightarrow \forall n\ z_n > b + \varepsilon$

  Тогда НСНМ
  $b - \varepsilon < z_n \leq b + \varepsilon \Rightarrow \lim{z_n} = b$

- $\forall \varepsilon > 0\ \exists N : \forall n \geq N\ b - \varepsilon < z_n < b + \varepsilon$

  $x_n \leq z_n \Rightarrow \exists x_N : x_N > b - \varepsilon$

## §4. Ряды

**Def.** $\sum a_n$ – ряд (числовой ряд); $a_n \in R$

**Def.** $S_n = \sum a_k$ – частичная сумма ряда

$S_n = a_1 + a_2 + \ldots + a_n$

$\{S_n\}$ – последовательность частичных сумм

Если $\exists \lim{S_n} = S \in \overline{R}$, то $S$ – суммы ряда

**Def.** $\sum a_n$ – ряд – сходящийся, если $S \in R$. Т.е. если
$S = \pm \infty$ или $\not\exists \lim{S_n}$, то $\sum a_n$ –
расходящийся ряд

**Th.** Необходимый признак сходимости числового ряда

$\sum a_n$ – сходится $\Rightarrow a_n \rightarrow 0$

**Доказательство**

$\sum a_n$ сходится
$\Leftrightarrow \exists \lim{S_n} = S \in R\ a_n = S_n - S_{n-1} \Rightarrow 0 = S - S$

Действия с числовыми рядами:

1.  $\begin{cases}
            \sum a_n \text{-- сходится к } S \\
            \sum b_n \text{-- сходится к } \tilde{S}
        \end{cases} \Rightarrow \sum (a_n + b_n)$ – сходится к
    $S + \tilde{S}$

2.  $\begin{cases}
            \sum a_n \text{-- сходится к } S \\
            c \in R
        \end{cases} \Rightarrow \sum c \cdot a_n$ – cходится к
    $c \cdot S$

3.  Сумма ряда, если существует, то удинственная

4.  $\sum a_n$ сходится к $S$

    $\begin{cases}
            (a_1 + a_2) + (a_3) + (a_4 + a_5 + \ldots) \\
            b_1 + b_2 + b_3
        \end{cases} \Rightarrow \sum b_n$ – сходится к $S$

5.  Изменение (добавление, отбрасывание) конечного числа членов ряда не
    меняет сходимость, но может изменить сумму

# Глава 3. Непрерывные функции

## §1. Предел функции

**Def.**

- $a \in R;\ U_a$ – окрестность точки $a$

  $U_a = (a - \varepsilon; a + \varepsilon)$ для некоторого
  $\varepsilon > 0$

  $\mathring{U_a}$ – проколотая окрестность точки $a$

  $\mathring{U_a} = (a - \varepsilon; a) \bigcup (a; a + \varepsilon)$
  для некоторого $\varepsilon > 0$

- $a = + \infty \Rightarrow$ окрестность – луч $(\varepsilon; + \infty)$

- $a = - \infty \Rightarrow$ окрестность – луч $(- \infty; \varepsilon)$

**Def.** $E \subset R;\ a \in R$

$a$ – предельная точка множества $E$, если
$\forall \mathbb{U_a} \bigcap E \neq \emptyset$, т.е. в любой проколотой
окрестности $a$ есть элемент из $E$

**Th.** Следующие условия равносильны:

1.  $a$ – предельная точка $E$

2.  В любой окрестности точки $a$ содержится бесконечное количество
    элементов множества $E$

3.  $\exists x_n : \begin{gathered}
            x_n \neq a \\
            x_n \in E
        \end{gathered} \lim{x_n} = a$

    Более того, можно сделать так, что $|x_n - a|$ строго монотонно
    убывает

**Доказательство**

- $2 \Rightarrow 1$ очев

- $3 \Rightarrow 2$

  $\exists x_n : \lim{x_n} = a$

  $\forall \begin{gathered}
          x_n \neq a \\
          x_n \in E
      \end{gathered}$

  $\forall \varepsilon > 0\ \exists N : \forall n \geq N\ |x_n - a| < \varepsilon \Rightarrow \forall n \geq N\ \begin{gathered}
          x_n \in U_a \\
          x_n \in E
      \end{gathered}$

  Возьмем $b_1 = (a-1; a+1)\backslash\{a\}$ и $x_1 \in b_1$

  Потом $\varepsilon_2 = min(\frac{1}{2}; |x_1 - a|)$,
  $b_2 = (a - \varepsilon_2; a + \varepsilon_2)\backslash\{a\}$ и
  $x_2 \in b_2$ итд

  Знаем:

  1.  $x_n \neq a$

  2.  $|x_{n-1} - a| > |x_n - a|$

  3.  $|x_n - a| < \frac{1}{n}$

      $\lim{x_n} = a$

**Def.** $f : E \rightarrow R; a$ – предельная точка $E$

$A = \lim\limits_{x \rightarrow a}{f(x)} \Leftrightarrow$

1.  $\forall \varepsilon > 0\ \exists \delta > 0 : \forall x \in E : 0 < |x - a| < \delta \Rightarrow |f(x) - A| < \varepsilon$
    – определение предела по Коши

2.  $\forall$ окрестности
    $U_A\ \exists U_a : f(\mathring{U_a} \bigcap E) \subset U_A$ – на
    языке окрестностей

3.  $\forall \{x_n\} : \begin{cases}
            x_n \in E \\
            x_n \neq a \\
            \lim{x_n} = a
        \end{cases} \Rightarrow \lim{f(x_n)} = A$ – по Гейне

$1 \Leftrightarrow 2$

$x \in (a - \delta; a) \bigcup (a; a + \delta) = \mathring{U_a}$

$U_A = (A - \varepsilon; A + \varepsilon)$

Дальше по определению

**Rem.**

1.  Значение функции $f(x)$ в точке $a$ в окрестности не участвует

2.  Предел в точке – локальное свойство

3.  В определении по Гейне: если все последовательности $f(x_n)$ имеют
    предел $\forall x_n : \begin{cases}
            x_n \neq a \\
            x_n \in E \\
            x_n \rightarrow a
        \end{cases}$, то все последовательности $\{f(x_n)\}$ имеют
    равные пределы

    **Доказательство**

    $\begin{cases}
            x_n \rightarrow a; y_n \rightarrow a \\
            f(x_n) \rightarrow A; f(y_n) \rightarrow B
        \end{cases}$

    $z_n = x_1, y_1, x_2, y_2 \ldots$

    $z_n \rightarrow a \Rightarrow f(z_n) \rightarrow C \Rightarrow \begin{cases}
            A = C \\
            B = C
        \end{cases} \Rightarrow A = B$

**Th.** Определение предела по Коши и по Гейне равносильны

**Доказательство**

- $x_n : \begin{cases}
          x_n \neq a \\
          x_n \in E \\
          x_n \rightarrow a
      \end{cases}$

  Хотим $f(x_n) \rightarrow A$

  Знаем:
  $\forall \varepsilon > 0\ \exists \delta > 0 : \forall x \in E\ 0 < |x - a| < \delta \Rightarrow |f(x) - A| < \varepsilon$

  $fix\ \varepsilon > 0$, подбираем для нее $\delta$

  $\delta \rightarrow \exists N : \forall n \geq N\ 0 < |x_n - a| < \delta$
  и
  $x_n \in E \Rightarrow |f(x_n) - A| < \varepsilon \Leftrightarrow \lim{f(x_n)} = A$

- Надо:
  $\forall \varepsilon > 0\ \exists \delta > 0: \forall x \in E\ 0 < |x - a| < \delta \Rightarrow |f(x) - A| < \varepsilon$

  От противного

  Пусть есть $\varepsilon > 0$ для которого любая $\delta$ не подходит

  $\varepsilon \leftarrow \delta = 1\ \exists x_1 : \begin{cases}
          0 < |x_1 - a| < 1 \\
          x_1 \in E \\
          |f(x_1) - A| \geq \varepsilon
      \end{cases}$

  $\varepsilon \leftarrow \delta = \frac{1}{2}\ \exists x_2 : \begin{cases}
          0 < |x_1 - a| < \frac{1}{2} \\
          x_2 \in E \\
          |f(x_2) - A| \geq \varepsilon
      \end{cases}$

  На $n$-м шаге $\delta = \frac{1}{n}\ \exists x_n : \begin{cases}
          0 < |x_n - a| < \frac{1}{n} \\
          x_n \in E \\
          |f(x_n) - A| \geq \varepsilon
      \end{cases}$

  Получили последовательность $x_n : \forall n \begin{cases}
          x_n \in E \\
          x_n \neq a \\
          |x_n - a| < \frac{1}{n}
      \end{cases} \Rightarrow \begin{cases}
          x_n \in E \\
          x_n \neq a \\
          \lim{x_n} = a
      \end{cases} \Rightarrow \lim{f(x_n)} = A$ ?!

**Th.** Свойства пределов:

1.  Единственность пределов

    Пусть $\lim\limits_{x \rightarrow a}{f(x)} = A$ и
    $\lim\limits_{x \rightarrow a}{f(x)} = B$

    Гейне: $\begin{cases}
            x_n \rightarrow a \\
            x_n \neq a \\
            x_n \in E
        \end{cases} \Rightarrow \begin{cases}
            \lim\limits_{n \rightarrow +\infty}{f(x_n)} = A \\
            \lim\limits_{n \rightarrow +\infty}{f(x_n)} = B
        \end{cases}$

    У последовательности предел единственный $\Rightarrow A = B$

2.  Локальная ограниченность

    $\lim\limits_{x \rightarrow a}{f(x)} = A \in R$, то
    $\exists U_a : f(x)$ ограничена при $x \in U_a$

    Определение через окрестность:

    $U_A = (A - 1; A + 1) \rightarrow \exists U_a : f(E \bigcap \mathring{U_a}) \subset U_A$

    $A - 1 < f(x) < A + 1\ \forall x \in E \bigcap \mathring{U_a}$

    **Rem.** Глобальной ограниченности нет

    $f(x) = \frac{1}{x}$

3.  Стабилизация знака

    $\lim\limits_{x \rightarrow a}{f(x)} = A \neq 0 \Rightarrow \exists U_a : \forall x \in E \bigcap \mathring{U_a}\ f(x) \cdot A > 0$

    $\forall \varepsilon > 0\ \exists \delta : \forall x \in E\ 0 < |x - a| < \delta \Rightarrow |f(x) - A| < \varepsilon$

    Берем $A > 0; \varepsilon = \frac{A}{2}$ – победа

**Def.**
$\lim\limits_{x \rightarrow a}{f(x)} = +\infty \Leftrightarrow \forall M > 0\ \exists \delta > 0 : \forall x \in E\ 0 < |x - a| < \delta \Rightarrow f(x) > M$

**Def.**
$\lim\limits_{x \rightarrow +\infty}{f(x)} = A \in R \Leftrightarrow \forall \varepsilon > 0\ \exists \delta > 0 : \begin{cases}
    x \in E \\
    x > \delta
\end{cases} \Rightarrow |f(x) - A| < \varepsilon$

**Th.** Арифметические действия с пределами

$f, g : E \rightarrow R; a$ – предельная точка $E$

$\lim\limits_{x \rightarrow a}{f(x)} = A; \lim\limits_{x \rightarrow a}{g(x)} = B; A, B \in R \Rightarrow$

1.  $\lim\limits_{x \rightarrow a}{f(x)\pm g(x)} = A \pm B$

2.  $\lim\limits_{x \rightarrow a}{f(x) \cdot g(x)} = A \cdot B$

3.  $\lim\limits_{x \rightarrow a}{|f(x)|} = |A|$

4.  $B \neq 0 \Rightarrow \lim\limits_{x \rightarrow a}{\frac{f(x)}{g(x)}} = \frac{A}{B}$

**Доказательство**

Пункт 1 по Гейне:

$\begin{cases}
    \{x_n\} \begin{cases}
        x_n \in E \\
        x_n \neq a \\
        x_n \rightarrow a
    \end{cases} \\
    \lim\limits_{x \rightarrow a}{f(x)} = A
\end{cases} \Rightarrow \lim\limits_{n \rightarrow +\infty}{f(x_n)} = A$

Аналогично $\lim\limits_{n \rightarrow +\infty}{g(x_n)} = B$

$\Rightarrow \lim\limits_{n \rightarrow +\infty}{f(x_n) + g(x_n)} = A + B$

Аналогично доказываются все пункты

**Th.** Предельный переход в неравенстве

$f, g : E \rightarrow R; a$ – предельная точка $E$

В некоторой $\mathring{U_a}\ f(x) \leq g(x); \begin{cases}
    \lim\limits_{x \rightarrow a}{f(x)} = A \\
    \lim\limits_{x \rightarrow a}{g(x)} = B
\end{cases} \Rightarrow A \leq B$

**Доказательство**

По Гейне: $\{x_n\} \begin{cases}
    x_n \neq a \\
    x_n \in E \\
    x_n \rightarrow a
\end{cases} \Rightarrow \begin{cases}
    \lim{f(x_n)} = A \\
    \lim{g(x_n)} = B
\end{cases}$

$x_n \rightarrow a \Rightarrow$ в какой-то момент
$\forall n \geq N : x_n \in \mathring{U_a} \Rightarrow f(x_n) \leq g(x_n) \Rightarrow A \leq B$

**Th.** Теорема о двух миллиционерах

$f, g, h : E \rightarrow R; a$ – предельная точка $E$

В некоторой
$\mathring{U_a}\ f(x) \leq g(x) \leq h(x)\ (\forall x \in \mathring{U_a})$

$\lim\limits_{x \rightarrow a}{f(x)} = \lim\limits_{x \rightarrow a}{h(x)} = A \in R \Rightarrow \exists \lim\limits_{x \rightarrow a}{g(x)} = A$

**Доказательство**

$\{x_n\} \begin{cases}
    x_n \neq a \\
    x_n \in E \\
    x_n \rightarrow a
\end{cases} \Rightarrow \begin{cases}
    f(x_n) \rightarrow A \\
    h(x_n) \rightarrow A
\end{cases}$

$\exists N : \forall n \geq N\ x_n \in \mathring{U_a} \Rightarrow f(x_n) \leq g(x_n) \leq h(x_n) \Rightarrow \lim\limits_{x \rightarrow a}{g(x)} = A$

Критерий Коши (для функции):

$f : E \rightarrow R; a$ – предельная точка E

$\exists \lim\limits_{x \rightarrow a}{f(x)} \in R \Leftrightarrow \forall \varepsilon > 0\ \exists \delta > 0 : \forall x, y \in \mathring{U}_{_\delta (a)} \bigcap E \Rightarrow |f(x) - f(y)| < \varepsilon$

**Доказательство**

- $\lim\limits_{x \rightarrow a}{f(x)} = A \in R \Leftrightarrow \forall \varepsilon > 0\ \exists \delta > 0 : \begin{cases}
          \forall x \in E\ 0 < |x - a| < \delta \Rightarrow |f(x) - A| < \varepsilon \\
          \forall y \in E\ 0 < |y - a| < \delta \Rightarrow |f(y) - A| < \varepsilon
      \end{cases} \Rightarrow |f(x) - f(y)| = |(f(x) - A) + (A - f(y))| \leq |f(x) - A| + |f(y) - A| < \varepsilon + \varepsilon = 2 \varepsilon$

- $\forall \varepsilon > 0\ \exists \delta > 0 : \forall x, y \in \mathring{U_a} \bigcap E \Rightarrow |f(x) - f(y)| < \varepsilon$

  Гейне:

  $\begin{cases}
          x_n \neq a \\
          x_n \in E \\
          x_n \rightarrow a
      \end{cases}$

  $fix\ \varepsilon > 0$, подбираем $\delta$

  $\exists N : \forall n \geq N\ |x_n - a| < \delta \Rightarrow x_n \in \mathring{U_a} \bigcap E$

  Возьмем
  $x_n, x_m : n, m \geq N \Rightarrow |f(x_n) - f(x_m)| < \varepsilon$

  Получили
  $\forall \varepsilon > 0\ \exists N : \forall m, n \geq N\ |f(x_n) - f(x_m)| < \varepsilon \Rightarrow \{f(x_n)\}$
  – фундаментальная
  $\Leftrightarrow \exists \lim{f(x_n)} \in R \Rightarrow \exists \lim\limits_{x \rightarrow a}{f(x)}$

**Def.** $f : E \rightarrow R; E_1 = E \bigcap (-\infty; a)$

$a$ – предельная точка $E_1$

$f_1 = f|_{E_1}$. Тогда если существует
$\lim\limits_{x \rightarrow a}{f_1(x)}$, то он называется пределом слева
для $f(x)$ в точке $a$

$\lim\limits_{x \rightarrow a}{f_1(x)} = \lim\limits_{x \rightarrow a_-}{f(x)} = \lim\limits_{x \rightarrow a_{-0}}{f(x)}$

**Def.** $f : E \rightarrow R; E_2 = E \bigcap (a; + \infty)$

$f_2 = f|_{E_2}$. Тогда если существует
$\lim\limits_{x \rightarrow a}{f_2(x)}$, то он называется пределом
справа для $f(x)$ в точке $a$

$\lim\limits_{x \rightarrow a}{f_2(x)} = \lim\limits_{x \rightarrow a_+}{f(x)} = \lim\limits_{x \rightarrow a_{+0}}{f(x)}$

Это односторонние пределы

**Rem.**
$\exists \lim\limits_{x \rightarrow a}{f(x)} \Leftrightarrow \lim\limits_{x \rightarrow a_-}{f(x)} = \lim\limits_{x \rightarrow a_+}{f(x)}$

$\lim\limits_{x \rightarrow a_-}{f(x)} = A \in R \Leftrightarrow \forall \varepsilon > 0\ \exists \delta > 0 : \forall x \in E\ a - \delta < x < a \Rightarrow |f(x)-A| < \varepsilon$

**Def.** $f : E \rightarrow R$

$f$ – монотонно возрастает
$\Leftrightarrow \forall x, y \in E : x < y \Rightarrow f(x) \leq f(y)$

$f$ – cтрого монотонно возрастает
$\Leftrightarrow \forall x, y \in E : x < y \Rightarrow f(x) < f(y)$

$f$ – монотонно убывает
$\Leftrightarrow \forall x, y \in E : x < y \Rightarrow f(x) \geq f(y)$

$f$ – строго монотонно убывает
$\Leftrightarrow \forall x, y \in E : x < y \Rightarrow f(x) > f(y)$

**Th.**$f : E \rightarrow R; E_1 = (-\infty; a) \bigcap E; a$ –
предельная точка $E_1 \Rightarrow$

1.  Если $f$ монотонно возрастает и ограничена сверху, то
    $\exists \lim\limits_{x \rightarrow a_-}{f(x)} \in R$

**Th.** $f : E \rightarrow R; E_2 = (a; + \infty) \bigcap E; a$ –
предельная точка $E_2 \Rightarrow$

1.  Если $f$ монотонно убывает и ограничена снизу, то
    $\exists \lim\limits_{x \rightarrow a_+}{f(x)} \in R$

**Доказательство**

1.  $f$ – ограничена сверху $\Rightarrow \exists sup(f(x)) = A$

    Хотим доказать $\lim\limits_{x \rightarrow a_-}{f(x)} = A$

    $fix\ \varepsilon > 0$

    $A - \varepsilon$ – не верхняя граница
    $\Rightarrow \exists y \in E_1 : f(y) > A - \varepsilon \Rightarrow \forall x > y\ f(x) > f(y) > A - \varepsilon$

    $\begin{cases}
            x < a \\
            y < a
        \end{cases} \Rightarrow \forall x : a > x > y\ A + \varepsilon > A \geq f(x) > A - \varepsilon \Rightarrow |f(x) - A| < \varepsilon \Rightarrow \lim\limits_{x \rightarrow a_-}{f(x)} = A$

## §2. Непрерывность

**Def.** $f : E \rightarrow R, a \in E$

$f$ называется непрерывной в точке $a$, если

1.  $a$ – не является предельной точкой $E$

2.  $a$ – предельная точка
    $E \Rightarrow \lim\limits_{x \rightarrow a}{f(x)} = f(a)$

<!-- -->

1.  $\forall \varepsilon > 0\ \exists \delta > 0 : \forall \in E\ |x-a| < \delta \Rightarrow |f(x) - f(a)| < \varepsilon$

2.  $\forall U_{f(a)}\ \exists U_a : f(U_a \bigcap E) \subset U_{f(a)}$

3.  $\forall x_n : \begin{cases}
            x_n \in E \\
            x_n \rightarrow a
        \end{cases} \Rightarrow \lim{f(x_n)} = f(a)$

**Ex:**

- $f(x) = C \Rightarrow \lim\limits_{x \rightarrow a}{f(x)} = C$

- $f(x) = x \Rightarrow \lim\limits_{x \rightarrow a}{f(x)} = a = f(a)$

- $f(x) = sign(x)$

  Для $f(0)$ неверно, значит не непрерывна

**Th.** $f(x) = \exp(x)$ непрерывна на $R$

**Доказательство**

1.  $\exp(x)$ непрерывна в $0$

    $\lim\limits_{x \rightarrow 0}{\exp(x)} = \exp(0) = 1$

    $\frac{1}{1-x} \geq \exp(x) \geq 1 + x$

    По двум милиционерам
    $1 \geq \lim\limits_{x \rightarrow 0}{\exp(x)} \geq 1 \Rightarrow \lim\limits_{x \rightarrow 0}{\exp(x)} = 1$

2.  $x = a \neq 0$

    Хотим $\lim\limits_{x \rightarrow a}{\exp(x)} = \exp(a)$

    $\exp(x) = \exp((x - a) + a) = \exp(x - a) \cdot \exp(a)$. Первое
    стремится к 1 по первому пункту, второе – константа
    $\Rightarrow \exp(x) \rightarrow 1 \cdot \exp(a)$

**Th.** Арифметика непрерывных функций

$f, g : E \rightarrow R; a \in E$

$f, g$ – непрерывные в $a \Rightarrow$

1.  $f \pm g$ – непрерывно в $a$

2.  $f \cdot g$ – непрерывно в $a$

3.  $|f|$ – непрерывно в $a$

4.  $g(a) \neq 0 \Rightarrow \frac{f}{g}$ – непрерывно в $a$

**Доказательство**

1.  $a$ не является предельной точкой $E \Rightarrow$ очев, т.к. в ней
    все непрерывно

2.  $a$ – предельная точка $E \Rightarrow \begin{cases}
            \exists \lim\limits_{x \rightarrow a}{f(x)} = f(a) \\
            \exists \lim\limits_{x \rightarrow a}{g(x)} = g(a)
        \end{cases} \Rightarrow$ зовем теорему про арифметику пределов

**Th.** О стабилизации знака

$f : E \rightarrow R$, непрерывна в $a; a \in E$ и
$f(a) \neq 0 \Rightarrow \exists U_a : \forall x \in U_a\ f(x) \cdot f(a) > 0$

**Доказательство**

1.  $a$ – не является предельной $\Rightarrow$ можем выбрать
    окрестность, в которой будет только $a$

2.  $a$ – предельная точка
    $\Rightarrow \lim\limits_{x \rightarrow a}{f(x)} = f(a) \Rightarrow$
    смотри теорему о стабилизации знака для предела функции

**Th.** О пределе композиции

$f : D \rightarrow R; g : E \rightarrow R; f(D) \subset E$

$a$ – предельная точка
$D; \lim\limits_{x \rightarrow a}{f(x)} = b; b \in E$

Если $g(x)$ непрерывна в $b$, то
$\lim\limits_{x \rightarrow a}{g(f(x))} = g(b)$

**Доказательство**

$g$ непрерывна в
$b \Rightarrow \forall \varepsilon > 0\ \exists \delta > 0 : \forall y \in E : |y - b| < \delta \Rightarrow |g(y) - g(b)| < \varepsilon$

Для этой
$\delta > 0\ \exists \gamma > 0 : \forall x \in D : 0 < |x-a| < \gamma \Rightarrow |f(x) - b| < \delta$

$\forall \varepsilon > 0\ \exists \gamma > 0 : \forall x \in D\ 0 < |x-a| < \gamma \Rightarrow |g(f(x)) - g(b)| < \varepsilon \Leftrightarrow \lim\limits_{x \rightarrow a}{g(f(x))} = b$

**Следствие:**
$f : D \rightarrow R; g : E \rightarrow R; f(D) \subset E; a \in D; f(a) = b \in E$

Если $f$ непрерывна в $a$, а $g$ непрерывна в $b$, то композиция
$g(f(x))$ непрерывна в $a$

**Th.** $0 < x < \frac{\pi}{2} \Rightarrow \sin{x} < x < tgx$

$S_{\triangle AOB} < S_\text{сектор AOB} < S_{\triangle COB}$

$S_{\triangle AOB} = \frac{1}{2} \cdot 1 \cdot 1 \cdot \sin{x}$

$S_\text{сектор AOB} = \frac{1}{2} \cdot 1^2 \cdot x$

$S_{\triangle COB} = \frac{1}{2} \cdot 1 \cdot tg x$

$\sin{x} < x < tgx$

**Следствие:**

1.  $x \in R;\ |\sin{x}| \leq |x|$, причем равенство только при $x = 0$

    $x \in (0; \frac{\pi}{2})$ доказано

    $x \in (- \frac{\pi}{2}; 0)\ x \rightarrow -x$

    $|x| > \frac{\pi}{2} > \frac{3}{2} > 1 \Rightarrow |\sin{x}| \leq 1 < |x|$

2.  $|\sin{x} - siny| \leq |x - y|;\ |\cos{x} - cosy| \leq |x - y|$

    $|\sin{x} - siny| = |2sin\frac{x-y}{2} \cdot cos\frac{x+y}{2}| = 2 \cdot |sin\frac{x-y}{2}| \cdot |cos\frac{x+y}{2}| \leq 2 \cdot |\frac{x-y}{2}| \cdot 1 = |x - y|$

    $\cos{x} - cosy = -2sin\frac{x-y}{2} \cdot sin\frac{x+y}{2}$ –
    аналогично

**Th.**

1.  $f(x) = \sin{x};\ g(x) = \cos{x}$ – непрерывны на $R$

2.  $tgx,\ ctgx$ – непрерывны на своей области определения

**Доказательство**

1.  $\lim\limits_{x \rightarrow a}{\sin{x}} = \sin{a}$

    $0 \leq |\sin{x} - \sin{a}| \leq |x-a| \rightarrow 0 \Rightarrow \lim\limits_{x \rightarrow a}{\sin{x} - \sin{a}} = 0 \Rightarrow \lim\limits_{x \rightarrow a}{\sin{x}} = \sin{a} \Leftrightarrow \sin{x}$
    непрерывна в $a$

    $\cos{x} = sin(\frac{\pi}{2} - x)$ – внутренняя и внешняя непрерывны
    $\Rightarrow$ непрерывен $\cos{x}$

2.  $tgx = \frac{\sin{x}}{\cos{x}}$ – отношение двух непрерывных функций
    $\Rightarrow tgx$ непрерывен во всех точках, где $\cos{x} \neq 0$,
    т.е. на своей области определения ($x \neq \frac{\pi}{2} + \pi k$)

    $ctgx$ – аналогично

**Th.** Первый замечательный предел

$\lim\limits_{x \rightarrow 0}{\frac{\sin{x}}{x}} = 1$

**Доказательство**

$x \in (0; \frac{\pi}{2}) \Rightarrow sinx < x < tgx \Rightarrow \frac{sinx}{x} < 1$

$x < \frac{sinx}{cosx} \Leftrightarrow cosx < \frac{sinx}{x}$

$cosx < \frac{sinx}{x} < 1$ – все функции четные

$\Rightarrow 0 < |x| < \frac{\pi}{2}:\ 1 \leftarrow cosx < \frac{sinx}{x} < 1 \rightarrow 1 \Rightarrow \lim\limits_{x \rightarrow 0}{\frac{sinx}{x}} = 1$

**Th.** Теорема Вейерштрасса

$f : [a;b] \rightarrow R;\ f$ – непрерывна на $[a; b]$, тогда

1.  $f$ – ограничена на $[a; b]$

2.  $f$ достигает своего наибольшего и наименьшего значения на $[a; b]$

**Доказательство**

1.  От противного. Пусть $f$ не является ограниченной
    $\Rightarrow \forall n \in N\ \exists x_n \in [a;b] : |f(x_n)| > n$

    $\{x_n\};\ \forall n\ a < x_n < b \Rightarrow \exists x_{n_k}$ –
    подпоследовательность

    $\lim{x_{n_k}} = c \in R;\ a < x_{n_k} < b \Rightarrow c \in [a;b]$

    $\begin{cases}
            f \text{-- непрерывна} \\
            x_{n_k} \rightarrow c
        \end{cases} \Rightarrow \lim{f(x_{n_k})} = f(c) \in R$

    Знаем: $|f(x_{n_k})| > n_k \geq k \rightarrow + \infty$

2.  $f$ – ограничена на
    $[a;b] \Rightarrow \exists M = supf(x);\ m = inff(x);\ m, M \in R$.
    Докажем, что $\exists c : f(c) = M$

    От противного. Пусть
    $\forall x \in [a;b]\ f(x) \neq M \Rightarrow \forall x \in [a;b]\ f(x) < M$

    $g(x) = \frac{1}{M - f(x) (\neq 0)}$ – непрерывна на $[a;b]$ как
    отношение двух непрерывных; $g(x) > 0 \Rightarrow g(x)$ – ограничена
    $\exists \tilde{M} : 0 < g(x) < \tilde{M}$

    $\frac{1}{M - f(x)} < \tilde{M} \Leftrightarrow M-f(x) > \frac{1}{\tilde{M}} \Leftrightarrow f(x) < M - \frac{1}{M} \Rightarrow M \neq supf(x)\ ??$

    Для $inf$ используем $h(x) = \frac{1}{f(x) - m}$

**Rem.**

1.  Непрерывность нужна везде

    $f(x) = \begin{cases}
            \frac{1}{x}, x \in (0; 1] \\
            0, x = 0
        \end{cases} \Rightarrow f(x)$ непрерывна везде, кроме $x = 0$,
    но $f$ уже не ограничена

2.  Отрезок важен

    $f(x) = \frac{1}{x}, x \in (0; 1]$

**Th.** Теорема Больцано-Коши (о промежуточном значении)

$f$ – непрерывна на $[a;b]$, тогда:

1.  Если
    $f(a) \cdot f(b) < 0 \Rightarrow \exists c \in (a, b) : f(c) = 0$

2.  $f(x)$ принимает все значения между $f(a)$ и $f(b)$

**Доказательство**

1.  НУО $f(a) < 0; f(b) > 0$

    $a_0 = a;\ b_0 = b;\ c = \frac{a_0 + b_0}{2}:$

    - $f(c) = 0$ – победа

    - $f(c) < 0 \rightarrow a_1 = c;\ b_1 = b_0;\ c = \frac{a_1 + b_1}{2}$

    - $f(c) > 0 \rightarrow a_1 = a_0;\ b_1 = c;\ c = \frac{a_1 + b_1}{2}$

    Если продолжается бесконечно:

    $[a_0;b_0] \supset [a_1;b_1] \supset \ldots$

    $|b_n - a_n| = \frac{1}{2^n} \cdot |b_0 - a_0|$. Стягивающиеся
    отрезки $\Rightarrow \exists!c : \begin{gathered}
            a_n \leq c \leq b_n \\
            \lim{a_n} = \lim{b_n} = c
        \end{gathered}$

    $\begin{cases}
            \lim{a_n} = c \\
            f \text{-- непрерывна}
        \end{cases} \Rightarrow \begin{cases}
            \lim{f(a_n)} = f(c) \\
            f(a_n) < 0
        \end{cases} \Rightarrow f(c) \leq 0$

    $\begin{cases}
            \lim{b_n} = c \\
            f \text{-- непрерывна}
        \end{cases} \Rightarrow \begin{cases}
            \lim{f(b_n)} = f(c) \\
            f(b_n) > 0
        \end{cases} \Rightarrow f(c) \geq 0$

    Значит $\begin{cases}
            f(c) \geq 0 \\
            f(c) \leq 0
        \end{cases} \Rightarrow f(c) = 0$

2.  $\forall y$ между $f(a)$ и $f(b)\ \exists c \in (a; b) : f(c) = y$

    НУО $f(a) < y < f(b)$

    $g(x) = f(x) - y$ – непрерывна

    $g(a) = f(a) - y < 0;\ g(b) = f(b) - y > 0 \Rightarrow \exists c \in (a; b) : g(c) = 0 \Rightarrow f(c) - y = 0 \Rightarrow f(c) = y$

**Rem.**

1.  Непрерывность нужна везде

    $f(x) = \begin{cases}
            -1, x \in [-1; 0) \\
            1, x \in [0; 1]
        \end{cases}$

    $f(1) \cdot f(-1) < 0$, но $\not\exists c : f(c) = 0$

2.  Бывают не непрерывные функции, удовлетворяющие теореме Больцано-Коши

    $f(x) = \begin{cases}
            0, x = 0 \\
            \sin{\frac{1}{x}}, (0; 1]
        \end{cases}$

    Если $0 < a < b \leq 1$, то очевидно выполняются (условия соблюдены)

    Интересно $0 = a < b \leq 1$. Возьмем k такую, что в $[a;b]$ влезет
    $[\frac{1}{2\pi(k+1)};\frac{1}{2\pi k}] \Rightarrow \frac{1}{x} \in [2\pi k; 2\pi (k + 1)]$

**Th.** Непрерывный образ отрезка – отрезок

**Доказательство**

$f : [a;b] \rightarrow R;\ f$ – непрерывна

$f([a;b])$ – отрезок

По теореме Вейерштрасса
$M = max f(x);\ m = min f(x);\ \exists p \in [a;b] : f(p) = M$ и
$\exists q \in [a;b] : f(q) = m \Rightarrow f([a;b]) \subset [m;M]$

$? \forall y\ m < y < M\ \exists c : f(c) = y \Rightarrow f([a;b]) = [m; M]$

Рассмотрим $[p;q] \begin{gathered}
    f(p) = M \\
    f(q) = m \\
    f \text{ -- непрерывна на} [p;q]
\end{gathered} \Rightarrow \exists c \in (p; q) : f(c) = y$

**Def.** $\langle a; b \rangle$ – промежуток. $a, b \in \overline{R}$

$\langle a; b \rangle$ – множество одно из 4 видов:

- $(a; b)$

- $(a; b]$

- $[a; b)$

- $[a; b]$

**Th.** Непрерывный образ промежутка – промежуток (может быть другого
типа)

**Доказательство**

$f : \langle a; b \rangle \rightarrow R;\ f$ – непрерывна на
$\langle a; b \rangle$

$m = inff(x);\ M = supf(x);\ m, M \in \overline{R}$

Знаем $f(\langle a; b \rangle) \subset [m; M]$

Хотим: $(m; M) \subset f(\langle a; b \rangle)$

$y \in (m; M) \Rightarrow m < y < M$

$\begin{cases}
    m = inff(x) \\
    m < y
\end{cases} \Rightarrow \exists p \in \langle a; b \rangle : f(p) < y$
(иначе $\forall p \in \langle a; b \rangle\ f(p) \geq y$)

$\begin{cases}
    M = supf(x) \\
    y < M
\end{cases} \Rightarrow \exists q \in \langle a; b \rangle : f(q) > y$
(иначе $\forall q \in \langle a; b \rangle\ f(q) \leq y$)

$\begin{cases}
    [p; q] \subset \langle a; b \rangle \\
    f \text{ -- непрерывна на } \langle a; b \rangle \Rightarrow f \text{ -- непрерывна на } [p; q] \\
    f(p) < y < f(q)
\end{cases} \Rightarrow \exists c \in [p; q] \subset \langle a; b \rangle : f(c) = y$

**Def.** Обратная функция:

$E \subset R;\ f : E \rightarrow R$ – инъективна

$f : E \rightarrow f(E)$ – биекция (взаимно однозначное соответствие)

$\begin{cases}
    g : f(E) \rightarrow E \\
    g(f(x)) = x\ \forall x \in E \\
    f(g(y)) = y\ \forall y \in E
\end{cases} \Rightarrow g$ – обратная к $f$ функция ($g(x) = f^{-1}(x)$)

**Th.** $f : \langle a; b \rangle \rightarrow R;\ f$ – непрерывна и
строго монотонна

$m = inff(x);\ M = supf(x);\ m, M \in \overline{R}$. Тогда

1.  $f$ – обратима и $f^{-1} : <m;M> \rightarrow \langle a; b \rangle$

2.  $f^{-1}$ – строго монотонна (характер монотонности сохраняется)

3.  $f^{-1}$ – непрерывна на $<m;M>$

**Доказательство**

1.  Строго монотонная $\Rightarrow$ инъективная $\Rightarrow$ обратима

2.  НУО $f(x) \nearrow$ строго $: x > y \Leftrightarrow f(x) > f(y)$

    $f^{-1} : <m;M> \rightarrow \langle a; b \rangle$

    $\forall u, v \in <m;M>$

    $u > v \Leftrightarrow f^{-1}(u) > f^{-1}(v)$, т.к. если

    $f(x) = v;\ x = f^{-1}(v);\ f(y) = u;\ y = f^{-1}(u)$

    $f^{-1}(v) > f^{-1}(u) \Leftrightarrow v > u$

3.  $y_0 \in <m;M>$. Хотим доказать, что $f^{-1}$ непрерывна в $y$

    $\lim\limits_{y \rightarrow y_0}{f^{-1}(y)} = f^{-1}(y_0)$

    На
    $<m;y_0)\ f^{-1} \nearrow \Rightarrow f^{-1}(y_0) \geq f^{-1}(y)\ \forall y \in <m; y_0] \Rightarrow \exists \lim\limits_{y \rightarrow y_0^-}{f^{-1}(y)} = A = \sup\limits_{<m;y_0)}f^{-1}(y) \leq f^{-1}(y_0)$

    На
    $(y_0;M>\ f^{-1} \nearrow \Rightarrow f^{-1}(y_0) \leq f^{-1}(y) \forall y \in [y_0;M> \Rightarrow \exists \lim\limits_{y \rightarrow y_0^+}{f^{-1}(y)} = B = \inf\limits_{(y_0;M>}{f^{-1}(y)} \geq f^{-1}(y_0)$

    $\lim\limits_{y \rightarrow y_0^-}{f^{-1}(y)} = A \leq f^{-1}(y_0) \leq B = \lim\limits_{y \rightarrow y_0^+}{f^{-1}(y)}$

    Если $A = B$ – победа

    Что знаем: $A \leq B$, хотим отбросить часть $A < B$

    Пусть $A < B$

    $f^{-1} : <m;M> \rightarrow \langle a; b \rangle$

    $\begin{cases}
            f^{-1}(<m;M>) = \langle a; b \rangle \\
            f^{-1}(<m;M>) \subset (- \infty; A] \bigcup \{f^{-1}(y_0)\} \bigcup [B; + \infty]
        \end{cases} \Rightarrow$ емае
    …$\Rightarrow A = B \Rightarrow \lim\limits_{y \rightarrow y_0^-}{f^{-1}(y)} = f^{-1}(y_0) = \\ = \lim\limits_{y \rightarrow y_0^+}{f^{-1}(y)} \Rightarrow f^{-1}$
    непрерывна в $y_0$

## §3. Элементарные функции

$\sin : [-\frac{\pi}{2}; \frac{\pi}{2}] \rightarrow [-1; 1]$ –
непрерывен и строго возрастает

$arcsin = \sin^{-1} : [-1; 1] \rightarrow [-\frac{\pi}{2}; \frac{\pi}{2}]$
– непрерывен и строго возрастает

$\cos : [0; \pi] \rightarrow [-1; 1]$ – непрерывен и строго убывает

$arccos = \cos^{-1} : [-1; 1] \rightarrow [0; \pi]$ – непрерывен и
строго убывает

$\tg : (-\frac{\pi}{2}; \frac{\pi}{2}) \rightarrow R$ – непрерывен и
строго возрастает

$arctg = \tg^{-1} : R \rightarrow (-\frac{\pi}{2}; \frac{\pi}{2})$ –
непрерывен и строго возрастает

$\ctg : (0; \pi) \rightarrow R$ – непрерывен и строго убывает

$arcctg = \ctg^{-1} : R \rightarrow (0; \pi)$ – непрерывен и строго
убывает

**Def.** $\exp : R \rightarrow (0; + \infty)$ – непрерывна и строго
возрастает

$\exp^{-1} = \ln : (0; + \infty) \rightarrow R$ – непрерывен и строго
возрастает

**Свойства:**

1.  $\lim\limits_{x \rightarrow 0_+}{\ln{x}} = - \infty$

    $\lim\limits_{x \rightarrow + \infty}{\ln{x}} = + \infty$

2.  $\forall x > -1\ \ \ln{(1+x)} \leq x$

    $y = \ln{(1+x)} \Leftrightarrow 1 + x = \exp(y) \geq 1 + y \Rightarrow x \geq y \Rightarrow x \geq \ln{(1+x)}$

3.  $\forall x \in (-1; 1)\ \ \ln{(1+x)} \geq 1 - \frac{1}{1 + x}$

    $y = \ln{(1+x)} \Leftrightarrow 1 + x = \exp(y) \leq \frac{1}{1-y}\ (y < 1)$

    $1 + x \leq \frac{1}{1 - y} \Leftrightarrow 1 - y \leq \frac{1}{1+x} \Leftrightarrow y \geq 1 - \frac{1}{1+x}$

    $\ln{(1+x)} \geq 1 - \frac{1}{1+x}$

    Условие из
    $\ln{(1+x)} < 1 = \ln{e} \Leftrightarrow 1 + x < e \Leftrightarrow x < e - 1$

4.  $\lim\limits_{x \rightarrow 0}{\frac{\ln{(1+x)}}{x}} = 1$

    $\frac{x}{x+1} = 1 - \frac{1}{1+x} \leq \ln{(1+x)} \leq x;\ -1 < x < 1$

    - $x \in (0; 1)$

      $\frac{1}{x+1} \leq \frac{\ln{(1+x)}}{x} \leq 1$

      По двум милиционерам
      $\lim\limits_{x \rightarrow 0_+}{\frac{\ln{(1+x)}}{x}} = 1$

    - $x \in (-1; 0)$

      $\frac{1}{x+1} \geq \frac{\ln{(1+x)}}{x} \geq 1$

      По двум милиционерам
      $\lim\limits_{x \rightarrow 0_-}{\frac{\ln{(1+x)}}{x}} = 1$

    Односторонние пределы равны
    $\Rightarrow \lim\limits_{x \rightarrow 0}{\frac{\ln{(1+x)}}{x}} = 1$

5.  $\ln{(ab)} = \ln{a} + \ln{b}$

    $\begin{cases}
            \ln{a} = x \Rightarrow a = \exp(x) \\
            \ln{b} = y \Rightarrow b = \exp(y)
        \end{cases}$

    $ab = \exp(x) \cdot \exp(y) = \exp(x + y) \Leftrightarrow \ln{(ab)} = x + y = \ln{a} + \ln{b}$

**Def.** $a > 0;\ b \in R$

$a^b = \exp(b \cdot \ln{a})$

**Свойства:**

1.  $b \in N;\ b = n, n \in N$

    $a^n = \exp(n \cdot \ln{a}) = \exp(\ln{a} + \ln{a} + \ldots + \ln{a}) = \exp(\ln{a}) \cdot \ldots \cdot \exp(\ln{a}) = a \cdot \ldots \cdot a$

2.  $b \in Z;\ b = -n, n \in N$

    $a^{-n} = \exp(-n \cdot \ln{a}) = \frac{1}{\exp(n \cdot \ln{a})} = \frac{1}{a^n}$

3.  $a^0 = 1$, т.к. $\exp(0) = 1$

4.  $b \in Q;\ b = \frac{m}{n}, \begin{gathered}
            n \in N \\
            m \in Z
        \end{gathered}$

    $a^\frac{m}{n} = \exp(\frac{m}{n} \cdot \ln{a})$

    $(a^\frac{m}{n})^n = (\exp(\frac{m}{n} \cdot \ln{a}))^n = \exp(n \cdot \frac{m}{n} \cdot \ln{a}) = \exp(m \cdot \ln{a}) = a^m$

**Th.** $\lim\limits_{x \rightarrow 0}{(1 + x)^\frac{1}{x}} = e$

$\lim\limits_{x \rightarrow + \infty}{(1 + \frac{1}{x})^x} = \lim\limits_{x \rightarrow - \infty}{(1 + \frac{1}{x})^x} = e$

**Доказательство**

1.  $(1 + x)^\frac{1}{x} = exp(\frac{1}{x} \cdot \ln{(1 + x)})$

    $\frac{\ln{(1+x)}}{x} \rightarrow 1$

    $\lim\limits_{x \rightarrow 0}{exp(\frac{1}{x} \cdot \ln{(1 + x)})} = exp(\lim\limits_{x \rightarrow 0}{\frac{\ln{(1 + x)}}{x}} = exp(1) = e$

2.  $y = \frac{1}{x};\ x \rightarrow + \infty \Rightarrow y \rightarrow 0_+$

    А если $x \rightarrow - \infty \Rightarrow y \rightarrow 0_-$

    $(1 + \frac{1}{x})^x = (1 + y)^\frac{1}{y} = e$

**Def.** Показательная функция:

$a > 0;\ a \neq 1;\ x \in R$

$a^x = exp(x \cdot \ln{a})$

**Свойства:**

1.  $a^x : R \rightarrow (0; + \infty)$

2.  $a > 1;\ a^x \nearrow$ строго и непрерывна

    $0 < a < 1;\ a^x \searrow$ строго и непрерывна

3.  $a^x \geq 1 + x \cdot \ln{a},\ \forall x$

    $a^x = exp(x \cdot \ln{a}) \geq 1 + x \cdot \ln{a}$

**Th.**
$\lim\limits_{x \rightarrow 0}{\frac{a^x - 1}{x}} = \ln{a};\ \forall a > 0, a \neq 1$

**Доказательство**

$a^x \geq 1 + x \ln{a} \Rightarrow a^x - 1 \geq x \cdot \ln{a}$

$a^{-x} \geq 1 - x \cdot \ln{a}$

В окрестности нуля
$a^x \leq \frac{1}{1 - x \ln{a}} \Rightarrow a^x - 1 \leq \frac{1}{1 - x\ln{a}} - 1 = \frac{x\ln{a}}{1 - x\ln{a}}$

$x\ln{a} \leq a^x - 1 \leq \frac{x\ln{a}}{1 - x\ln{a}}$

- $x > 0$

  $\ln{a} \leq \frac{a^x-1}{x} \leq \frac{\ln{a}}{1 - x\ln{a}}$

  По двум милиционерам
  $\lim\limits_{x \rightarrow 0_+}{\frac{a^x-1}{x}} = \ln{a}$

- $x < 0$

  $\ln{a} \geq \frac{a^x-1}{x} \geq \frac{\ln{a}}{1 - x\ln{a}}$

  По двум милиционерам
  $\lim\limits_{x \rightarrow 0_-}{\frac{a^x-1}{x}} = \ln{a}$

Односторонние пределы равны
$\Rightarrow \lim\limits_{x \rightarrow 0}{\frac{a^x-1}{x}} = \ln{a}$

**Def.** Степенная функция

$x \in (0; + \infty);\ p \in R$

$x^p = exp(p \cdot \ln{x})$

$x^p : (0 + \infty) \rightarrow (0; + \infty)$

1.  Непрерывная

2.  - $p > 0 \Rightarrow x^p \nearrow$ строго

    - $p < 0 \Rightarrow x^p \searrow$ строго

**Th.** $\lim\limits_{x \rightarrow 0}{\frac{(1+x)^p-1}{x}} = p$

**Доказательство**

$(1 + x)^p = exp(p \cdot \ln{(1+x)})$

$\frac{(1+x)^p-1}{x} = \frac{exp(p \cdot \ln{(1+x)})-1}{x} = \frac{(exp(p \cdot \ln{(1+x)})-1) \cdot p \cdot \ln{(1+x)}}{p \cdot \ln{(1+x)} \cdot x}$

$x \rightarrow 0 \Rightarrow 1 + x \rightarrow 1 \Rightarrow \ln{(1+x)} \rightarrow 0$

$\frac{e^t-1}{t} \rightarrow 1$ при $t \rightarrow 0$ знаем

$\frac{exp(p \cdot \ln{(1+x)})-1}{p \cdot \ln{(1 + x)}} \rightarrow 1$

$\frac{\ln{(1+x)}}{x} \rightarrow 1$

Значит исходное стремится к $1 \cdot p \cdot 1$

## §4. Сравнение функций

**Def.** $f, g : E \Rightarrow R;\ a$ – предельная точка $E$

Если $\exists \varphi : E \Rightarrow R$ такая что
$f(x) = \varphi(x) \cdot g(x)$ при $x \in \mathring{U_a} \bigcap E$ и

1.  $\varphi(x)$ – ограниченная
    $\Rightarrow f(x) = O(g(x)),\ x \rightarrow a$

2.  $\lim\limits_{x \rightarrow a}{\varphi(x)} = 0$, то
    $f(x) = o(g(x)),\ x \rightarrow a$

3.  $\lim\limits_{x \rightarrow a}{\varphi(x)} = 1$, то
    $f(x) \sim g(x),\ x \rightarrow a$

$O, o$ – символы Ландау

**Rem.**

1.  $f(x) = O(g(x)),\ x \rightarrow a \Leftrightarrow |f(x)| \leq c \cdot |g(x)|$
    в некоторой $\mathring{U_a}$

2.  $f(x) = o(g(x)),\ x \rightarrow a \Leftrightarrow \lim\limits_{x \rightarrow a}{\frac{f(x)}{g(x)}} = 0$,
    но соглашение $\frac{0}{0} = 0$

3.  $f(x) \sim g(x),\ x \rightarrow a \Leftrightarrow \lim\limits_{x \rightarrow a}{\frac{f(x)}{g(x)}} = 1$,
    но соглашение $\frac{0}{0} = 1$

**Def.** $f = O(g)$ на
$E \Leftrightarrow \exists c > 0 : |f(x)| \leq c \cdot |g(x)|\ \forall x \in E$

**Свойства:**

1.  $\sim$ – отношение эквивалентности

    - Рефлексивность: $f \sim f$, т.к. $f(x) = 1 \cdot f(x)$

    - Симметричность: $f \sim g \xRightarrow[]{?} g \sim f$

      $f \sim g \Rightarrow \begin{cases}
                  f(x) = \varphi(x) \cdot g(x) \\
                  \varphi(x) \rightarrow 1,\ x \rightarrow a
              \end{cases} \Rightarrow g(x) = \frac{1}{\varphi(x)} \cdot f(x)$

    - $\begin{cases}
                  f \sim g \\
                  g \sim h \\
              \end{cases} \xRightarrow[]{?} f \sim h$

      $f(x) = \varphi(x) \cdot g(x)$

      $g(x) = \psi(x) \cdot h(x)$

      $f(x) = \varphi(x) \cdot \psi(x) \cdot h(x)$

2.  $\begin{cases}
            f_1 \sim g_1 \\
            f_2 \sim g_2 
        \end{cases} \Rightarrow f_1 \cdot f_2 \sim g_1 \cdot g_2,\ x \rightarrow a$

    $f_1 = \varphi_1 \cdot g_1;\ f_2 = \varphi_2 \cdot g_2;\ \varphi_1, \varphi_2 \rightarrow 1$

    $\Rightarrow f_1 \cdot g_2 = (\varphi_1 \cdot \varphi_2) \cdot g_1 \cdot g_2 \Rightarrow f_1 \cdot f_2 \sim g_1 \cdot g_2$

3.  $\begin{cases}
            f_1 \sim g_1 \\
            f_2 \sim g_2 \\
            x \rightarrow a \\
            f_2, g_2 \neq 0 \text{ в } \mathring{U_a}
        \end{cases} \Rightarrow \frac{f_1}{f_2} \sim \frac{g_1}{g_2}$

    $\begin{cases}
            f_1 = \varphi_1 \cdot g_1 \\
            f_2 = \varphi_2 \cdot g_2
            \varphi_1. \varphi_2 \rightarrow 1
        \end{cases} \Rightarrow \frac{f_1}{f_2} = \frac{\varphi_1 \cdot g_1}{\varphi_2 \cdot g_2} = \frac{\varphi_1}{\varphi_2} \cdot \frac{g_1}{g_2}$

4.  $f \sim g,\ x \rightarrow a \Rightarrow \begin{cases}
            f = g + o(g) \\
            g = f + o(f)
        \end{cases}$

    $f \sim g \Rightarrow \exists \varphi \rightarrow 1 : f(x) = \varphi(x) \cdot g(x) \Rightarrow f(x) = g(x) + (\varphi(x) - 1) \cdot g(x)$

    $(\varphi(x) - 1) \cdot g(x) = \psi(x) \cdot g(x),\ \psi(x) \rightarrow 0,\ x \rightarrow a \Rightarrow \psi(x) \cdot g(x) = o(g(x))$

    $f(x) = g(x) + o(g(x))$

5.  $f \sim g \Rightarrow f = O(g),\ x \rightarrow a$

    $f(x) = \varphi(x) \cdot g(x),\ \varphi(x) \rightarrow 1$ –
    ограничена в $\mathring{U_a}$

    $f = o(g) \Rightarrow f = O(g)$

    $f(x) = \varphi(x) \cdot g(x), \varphi(x) \rightarrow 0$ –
    ограничена в $\mathring{U_a}$

6.  $o(f) + o(f) = o(f),\ x \rightarrow a$

    $\begin{cases}
            \varphi(x) \cdot f(x) = o(f) \\
            \psi(x) \cdot f(x) = o(f)
        \end{cases} \Rightarrow$ одностороннее свойство

    $o(f) + o(f) = o(f)$

    $\begin{cases}
            h(x) = \varphi(x) \cdot f(x), \varphi(x) \rightarrow 0 \\
            g(x) = \psi(x) \cdot f(x), \psi(x) \rightarrow 0
        \end{cases} \Rightarrow h(x) + g(x) = (\varphi(x) + \psi(x)) \cdot f(x)$

    Но
    $\varphi(x) + \psi(x) \rightarrow 0 \Rightarrow h(x) + g(x) \in o(f)$

    6.5: $O(f) + O(f) = O(f)$

    $\begin{cases}
            h = \varphi \cdot f \\
            g = \psi \cdot f \\
            \varphi, \psi \text{ -- ограничены}
        \end{cases} \Rightarrow h + g = (\varphi + \psi) \cdot \Rightarrow h + g = O(f)$

7.  $f \cdot o(g) = o(fg),\ x \rightarrow a$

    $h \in o(g) \Rightarrow h = \varphi \cdot g,\ \varphi \rightarrow 0$

    $f \cdot h = f \cdot \varphi \cdot g = \varphi \cdot (fg) \Rightarrow fh = o(fg)$

    $k \in o(fg) \Rightarrow k(x) = \varphi(x) \cdot f(x) \cdot g(x),\ \varphi \rightarrow 0$

    $k(x) = f(x) \cdot (\varphi(x) \cdot g(x)) = f(x) \cdot o(g(x))$

8.  $\lim\limits_{x \rightarrow a}{f(x)} = b \Leftrightarrow f(x) = b + o(1),\ x \rightarrow a$

    $? o(1)$

    $h \in o(1) \Rightarrow h(x) = \varphi(x) \cdot 1,\ \varphi(x) \rightarrow 0$

    $h(x) = o(1) \Leftrightarrow h(x)$ – б/м

    $\lim\limits_{x \rightarrow a}{f(x)} = b \Leftrightarrow \lim\limits_{x \rightarrow a}{f(x)-b} = 0 \Leftrightarrow f(x) - b = \varphi(x),\ \varphi(x) \rightarrow 0$

    $f(x) = b + \varphi(x) = b + o(1)$

**E.g.**

1.  $\frac{\sin{x}}{x} \rightarrow 1 \Leftrightarrow \sin{x} \sim x,\ x \rightarrow 0$

2.  $\frac{\ln{(1+x)}}{x} \rightarrow 1 \Leftrightarrow \ln{(1+x)} \sim x,\ x \rightarrow 0$

3.  $\frac{\tg{x}}{x} \rightarrow 1 \Leftrightarrow \tg{x} \sim x,\ x \rightarrow 0$

Или

1.  $\frac{\sin{x}}{x} = 1 + o(1)$

    $\sin{x} = x + x \cdot o(1)$

    $\sin{x} = x + o(x),\ x \rightarrow 0$

2.  $\ln{(1+x)} = x + o(x)$

3.  $\tg{x} = x + o(x)$

4.  $\frac{e^x -1}{x} \rightarrow 1$

    $\frac{e^x-1}{x} = 1 + o(1)$

    $e^x - 1 = x + o(x)$

    $e^x = 1 + x + o(x),\ x \rightarrow 1$

5.  $\frac{(1+x)^p-1}{x} \rightarrow p$

    $(1+x)^p = 1 + px + o(x)$

# Глава 4. Дифференциальное исчисление

**Def.**

$f : \langle a; b \rangle \rightarrow R;\ x_0 \in \langle a; b \rangle$

$f$ – дифференцируема в
$x_0 \Leftrightarrow \exists k \in R : f(x) = f(x_0) + k(x - x_0) + o(x - x_0),\ x \rightarrow x_0$

**Def.**
$f : \langle a; b \rangle \rightarrow R;\ x_0 \in \langle a; b \rangle$

Производная функции $f(x)$ в точке $x_0$:
$f'(x_0) = \lim\limits_{x \rightarrow x_0}{\frac{f(x) - f(x_0)}{x - x_0}} = \lim\limits_{h \rightarrow 0}{\frac{f(x_0 + h) - f(x_0)}{h}}$
при условии существования этого предела

**Th.** Критерий дифференцируемости

$f : \langle a; b \rangle \rightarrow R;\ x_0 \in \langle a; b \rangle$

Следующие условия равносильны

1.  $f$ – дифференцируема в точке $x_0$

2.  $\exists$ конечная производная в точке $x_0$ ($f'(x_0) \in R$)

3.  $\exists \varphi : \langle a; b \rangle \rightarrow R$

    $f(x) - f(x_0) = \varphi(x) \cdot (x - x_0),\ \forall x$;
    $\varphi(x)$ – непрерывна в $x_0$

**Rem.** Если все утверждения верны, то $k = f'(x_0) = \varphi(x_0)$

**Доказательство**

- $f(x) = f(x_0) + k(x - x_0) + o(x - x_0)$

  $\frac{f(x) - f(x_0)}{x-x_0} = k + \frac{o(x-x_0)}{x - x_0}$

  $f'(x_0) = \lim\limits_{x \rightarrow x_0}{\frac{f(x)-f(x_0)}{x-x_0}} = \lim\limits_{x \rightarrow x_0}{(k + \frac{o(x-x_0)}{x-x_0}}) = k \in R$

- $\varphi(x) = \begin{cases}
          \frac{f(x)-f(x_0)}{x-x_0}, x \neq x_0 \\
          f'(x_0), x = x_0
      \end{cases}$

  $f(x)-f(x_0) = \varphi(x)(x-x_0)$

  $\varphi$ непрерывна в $x_0$ и $\varphi(x_0) = f'(x_0)$

- $f(x) - f(x_0) = \varphi(x)(x-x_0)$

  $f(x) = f(x_0) + \varphi(x_0)(x-x_0) - \varphi(x_0)(x-x_0) + \varphi(x)(x-x_0) = f(x_0) + \varphi(x_0)(x-x_0) + (\varphi(x) - \varphi(x_0))(x-x_0)$

  Знаем, что
  $\varphi(x) - \varphi(x_0) \rightarrow 0 \Rightarrow (\varphi(x) - \varphi(x_0))(x-x_0) = o(x-x_0)$

  $f(x) = f(x_0) + k(x-x_0) + o(x-x_0)$

**Def.** Бесконечная производная

$f(x) = \sqrt[3]{x};\ x_0 = 0$

$f'(x_0) = f'(0) = \lim\limits_{h \rightarrow 0} \frac{\sqrt[0]{0 + h} - \sqrt[3]{0}}{h} = \lim\limits_{h \rightarrow 0} \frac{\sqrt[3]{h}}{h} = \lim\limits_{h \rightarrow 0} \frac{1}{\sqrt[3]{h^2}} = + \infty$

$f'(x_0) = \lim\limits_{h \rightarrow 0} \frac{f(x_0 + h) - f(x_0)}{h}$

**Def.** Односторонние производные

$f'_+(x_0) = \lim\limits_{h \rightarrow 0_+} \frac{f(x_0 + h) - f(x_0)}{h} = \lim\limits_{x \rightarrow x_{0_+}} \frac{f(x) - f(x_0)}{x - x_0}$

$f'_-(x_0) = \lim\limits_{h \rightarrow 0_-} \frac{f(x_0 + h) - f(x_0)}{h} = \lim\limits_{x \rightarrow x_{0_-}} \frac{f(x) - f(x_0)}{x - x_0}$

**Ex.** $f(x) = |x|, x_0 = 0$

$f'_+(0) = \lim\limits_{x \rightarrow 0_+} \frac{|x| - 0}{x} = 1$

$f'_-(0) = \lim\limits_{x \rightarrow 0_-} \frac{|x| - 0}{x} = -1$

**Rem.** $\exists f'(x_0) \Leftrightarrow \exists f'_+(x_0) = f'_-(x_0)$

**Def.** Касательная – предельное положение секущей

**Утверждение** $f$ – дифференцируема в $x_0 \Rightarrow$ прямая
$y = f(x_0) + f'(x_0)(x - x_0)$ – касательная к графику функции $f(x)$ в
точке $x_0$

**Доказательство**

$f$ – дифференцируема в $u$

$\frac{f(v) - f(u)}{v - u}(x - u) + f(u) = y$
($x = u \rightarrow f(u);\ x = v \rightarrow f(v)$)

$x_0 \leftrightarrow u$

$\lim\limits_{v \rightarrow u} \frac{f(v) - f(u)}{v - u} = f'(u)$

$y = f(u) + f'(u)(x - u) = f(x_0) + f'(x_0)(x - x_0)$

**Def.** Дифференциал функции – линейная часть приращения функции (для
дифференцируемых функций)

$f$ – дифференцируема в $x_0 \Leftrightarrow \exists k \in R$

$f(x) = f(x_0) + k(x - x_0) + o(x - x_0),\ x \rightarrow x_0$

$f(x) - f(x_0) = k(x - x_0) + o(x - x_0)$. Слева от равно приращение
функции, справа – линейная часть + о малое

$df_{x_0} : R \rightarrow R$. $df_{x_0} = kx$

$f(x) = f(x_0) + df_{x_0}(x - x_0) + o(x - x_0)$

**Утверждение** $f(x)$ дифференцируема в $x_0$, то $f$ непрерывна в
$x_0$

**Доказательство**

$f$ – дифференцируема в
$x_0 \Rightarrow \exists \varphi(x) : f(x) = f(x_0) + \varphi(x)(x - x_0)$,
причем $\varphi(x)$ непрерывна в $x_0$

$\lim\limits_{x \rightarrow x_0} f(x) = \lim\limits_{x \rightarrow x_0} (f(x_0) + \varphi(x)(x - x_0)) = f(x_0) + \varphi(x_0) \cdot 0 = f(x_0) \Rightarrow f$
непрерывна в $x_0$

**Th.** Про арифметические действия с производной

$f, g : \langle a; b \rangle \rightarrow R;\ x_0 \in \langle a; b \rangle;\ f, g$
– дифференцируемы в $x_0$, тогда

1.  $f \pm g$ – дифференцируема в $x_0$ и
    $(f \pm g)'(x_0) = f'(x_0) \pm g'(x_0)$

2.  $f \cdot g$ – дифференцируема в $x_0$ и
    $(f \cdot g)'(x_0) = f'(x_0) \cdot g(x_0) + f(x_0) \cdot g'(x_0)$

3.  Если $g(x_0) \neq 0$, то $\frac{f}{g}$ – дифференцируема в $x_0$ и
    $(\frac{f}{g})'(x_0) = \frac{f'(x_0) \cdot g(x_0) - f(x_0) \cdot g'(x_0)}{g^2(x_0)}$

**Доказательство**

1.  $f$ – дифференцируема в
    $x_0 \Leftrightarrow \exists \varphi(x) : \langle a; b \rangle \rightarrow R$,
    $\varphi(x)$ непрерывна в $x_0$

    $f(x) = f(x_0) + \varphi(x)(x - x_0)$
    ($\forall x \in \langle a; b \rangle$))

    $g$ – дифференцируема в
    $x_0 \Leftrightarrow \exists \psi(x) : \langle a; b \rangle \rightarrow R$,
    $\psi(x)$ непрерывна в $x_0$

    $g(x) = g(x_0) + \psi(x)(x - x_0)$
    ($\forall x \in \langle a; b \rangle$))

    $f(x) \pm g(x) = (f(x_0) \pm g(x_0)) + (\varphi(x) \pm \psi(x))(x - x_0)$

    $\xi(x) = \varphi(x) \pm \psi(x)$ – непрерывна в $x_0$

    $\Rightarrow f(x) \pm g(x)$ – дифференцируема в $x_0$ и
    $(f(x) \pm g(x))'(x_0) = \xi(x_0) = \varphi(x_0) \pm \psi(x_0) = f'(x_0) \pm g'(x_0)$

2.  $f(x) \cdot g(x) = (f(x_0) + \varphi(x) \cdot (x - x_0)) \cdot (g(x_0) + \psi(x) \cdot (x - x_0)) =$

    $= f(x_0) \cdot g(x_0) + f(x_0) \cdot \psi(x) \cdot (x - x_0) + g(x_0) \cdot \varphi(x) \cdot (x - x_0) + \varphi(x) \cdot \psi(x) \cdot (x - x_0)^2 =$

    $= f(x_0) \cdot g(x_0) + (f(x_0) \cdot \psi(x) + g(x_0) \cdot \varphi(x) + \varphi(x) \cdot \psi(x) \cdot (x - x_0)) \cdot (x - x_0)$

    Большая скобка
    $= \xi(x) : \langle a; b \rangle \rightarrow R,\ \xi(x)$ непрерывна
    в $x_0$

    $f(x) \cdot g(x) = f(x_0) \cdot g(x_0) + \xi(x) \cdot (x - x_0) \Rightarrow f \cdot g$
    – дифференцируема в $x_0$ и $(f \cdot g)'(x_0) = \xi(x_0) =$

    $= f'(x_0) \cdot \psi(x_0) + g(x_0) \cdot \varphi(x_0) + 0 = f'(x_0) \cdot g(x_0) + f(x_0) \cdot g'(x_0)$

3.  $(\frac{f}{g})' = (f \cdot \frac{1}{g})'$

    $(\frac{1}{g})'(x_0) = \lim\limits_{x \rightarrow x_0} \frac{\frac{1}{g(x)} - \frac{1}{g(x_0)}}{x - x_0} = \lim\limits_{x \rightarrow x_0} \frac{g(x_0) - g(x)}{g(x) \cdot g(x_0) \cdot (x - x_0)} = \frac{-g'(x_0)}{g^2(x_0)}$

    $(f \cdot \frac{1}{g})'(x_0) = f'(x_0) \cdot \frac{1}{g(x_0)} - f(x_0) \cdot \frac{g'(x_0)}{g^2(x_0)} = \frac{f'(x_0) \cdot g(x_0) - f(x_0) \cdot g'(x_0)}{g^2(x_0)}$

**Th.** Дифференцируемость композиции

$f : \langle a; b \rangle \rightarrow R;\ g : <c;d> \rightarrow \langle a; b \rangle$

$x_0 \in <c;d>;\ y_0 = g(x_0) \in \langle a; b \rangle$

$g$ – дифференцируема в $x_0$ и $f$ – дифференцируема в $y_0 = g(x_0)$.
Тогда $f \circ g$ – дифференцируема в $x_0$ и
$(f \circ g)'(x_0) = f'(g(x_0)) \cdot g'(x_0) = f'(y_0) \cdot g'(x_0)$

**Доказательство**

$g$ – дифференцируема в
$x_0 \Leftrightarrow \exists \psi(x) : g(x) = g(x_0) + \psi(x)(x - x_0);\ \psi$
– непрерывна в $x_0$

$f$ – дифференцируема в
$y_0 \Leftrightarrow \exists \varphi(y) : f(y) = f(y_0) + \varphi(y)(y - y_0);\ \varphi$
– непрерывна в $y_0$

$f(g(x)) = f(g(x_0)) + \varphi(g(x))(g(x) - g(x_0)) = f(g(x_0)) + \varphi(g(x))\psi(x)(x - x_0);\ \xi(x) = \varphi(g(x))\psi(x)$

$f(g(x)) = f(g(x_0)) + \xi(x)(x - x_0)$

$\xi(x)$ – непрерывна в $x_0$? $\psi$ – непрерывна в $x_0$, $g$ –
непрерывна в $x_0$, $\varphi$ – непрерывна в
$y_0 = g(x_0) \Rightarrow \varphi(g(x))$ – непрерывна в $x_0$

Значит $f(g(x))$ – дифференцируема в $x_0$

$(f \circ g)'(x_0) = \xi(x_0) = \varphi(g(x_0))\psi(x_0) = f'(g(x_0)) \cdot g'(x_0) = f'(y_0) \cdot g'(x_0)$

**Th.** Дифференцируемость обратной функции

$f : \langle a; b \rangle \rightarrow <m, M>$ – строго монотонная и
непрерывная

$x_0 \in \langle a; b \rangle : f'(x_0) \neq 0$ ($f$ – дифференцируема в
$x_0$)

Тогда $f^{-1}$ – дифференцируема в $y_0 = f(x_0)$ и
$(f^{-1})'(y_0) = \frac{1}{f'(x_0)}$

**Доказательство**

$\exists f^{-1}$, более того $f^{-1}$ – непрерывная

$f$ – дифференцируема в
$x_0 \Rightarrow \exists \varphi(x) : f(x) = f(x_0) + \varphi(x)(x - x_0);\ \varphi(x)$
– непрерывна в $x_0$

$y = f(x) \Rightarrow x = f^{-1}(y)$

$y_0 = f(x_0) \Rightarrow x_0 = f^{-1}(y_0)$

$y = y_0 + \varphi(f^{-1}(y))(f^{-1}(y) - f^{-1}(y_0)) \Rightarrow y - y_0 = \varphi(f^{-1}(y))(f^{-1}(y) - f^{-1}(y_0))$

$f^{-1}(y) - f^{-1}(y_0) = \frac{1}{\varphi(f^{-1}(y))}(y - y_0)$

$\varphi(f^{-1}(y)) = \varphi(x);\ \varphi(x_0) = f'(x_0) \neq 0$ и
$\varphi$ – непрерывна в $x_0$

В окрестности $x_0\ \varphi(x) \neq 0$

$\varphi(f^{-1}(y))$ непрерывна по непрерывности композиции

$f^{-1}$ – дифференцируема в $y_0$ и
$(f^{-1})'(y_0) = \frac{1}{\varphi(f^{-1}(y_0))} = \frac{1}{\varphi(x_0)} = \frac{1}{f'(x_0)}$

**Rem.** $(f^{-1})(y_0) = \frac{1}{f'(f^{-1}(y_0))}$

**Def.** Производные элементарных функций

1.  $c \in R;\ (c)' = 0$

2.  $(x^p)' = p \cdot x^{p-1}$

3.  $(a^x)' = a^x \ln{a}$

    $(e^x)' = e^x$

4.  $(\ln{x})' = \frac{1}{x}$

5.  $(\sin{x})' = \cos{x}$

6.  $(\cos{x})' = -\sin{x}$

7.  $(\tg{x})' = \frac{1}{\cos^2{x}}$

8.  $(\ctg{x})' = - \frac{1}{\sin^2{x}}$

9.  $(\arcsin{x})' = \frac{1}{\sqrt{1 - x^2}}$

10. $(\arccos{x})' = - \frac{1}{\sqrt{1 - x^2}}$

11. $(\arctg{x})' = \frac{1}{1 + x^2}$

12. $(\arcctg{x})' = - \frac{1}{1 + x^2}$

**Доказательство**

1.  $\lim\limits_{h \rightarrow 0} \frac{(x + h)^p - x^p}{h} = \lim\limits_{h \rightarrow 0} \frac{x^p((1 + \frac{h}{p})^p - 1)}{h}$

    $\frac{(x + 1)^p - 1}{x} \rightarrow p,\ x \rightarrow 0 \Rightarrow (1 + x)^p - 1 \sim px,\ x \rightarrow 0$

    $= \lim\limits_{h \rightarrow 0} \frac{x^p \cdot p \cdot \frac{h}{x}}{h} = p \cdot x^{p-1}$

2.  $\lim\limits_{h \rightarrow 0} \frac{a^{x + h} - a^x}{h} = a^x \lim\limits_{h \rightarrow 0} \frac{a^h - 1}{h} = a^x \ln{a}$

3.  $(\ln{x})' = \lim\limits_{h \rightarrow 0} \frac{\ln{(x + h)} - \ln{x}}{h} = \lim\limits_{h \rightarrow 0} \frac{\ln{(\frac{x + h}{x})}}{h} = \lim\limits_{h \rightarrow 0} \frac{\ln{(1 + \frac{h}{x})}}{h} = \frac{1}{x}$

4.  $(\sin{x})' = \lim\limits_{h \rightarrow 0} \frac{\sin{(x + h)} - \sin{x}}{h} = \lim\limits_{h \rightarrow 0} \frac{2 \sin{\frac{h}{2}} \cos{(x + \frac{h}{2})}}{h} = \lim\limits_{h \rightarrow 0} \frac{\sin{\frac{h}{2}}}{\frac{h}{2}} \cos{(x + \frac{h}{2})} = 1 \cdot \cos{x} = \cos{x}$

5.  $(\tg{x})' = (\frac{\sin{x}}{\cos{x}})' = \frac{(\sin{x})' \cdot \cos{x} - \sin{x} \cdot (\cos{x})'}{\cos^2{x}} = \frac{\cos^2{x} + \sin^2{x}}{\cos^2{x}} = \frac{1}{\cos^2{x}}$

6.  $(\arcsin(x))' = (\sin^{-1})'(x) = \frac{1}{\cos{(\arcsin{x})}} = \frac{1}{\sqrt{1 - \sin^2{(\arcsin{x})}}} = \frac{1}{\sqrt{1 - x^2}}$

7.  $(\arctg{x})' = (\tg^{-1})'(x) = \frac{1}{\frac{1}{\cos^2{(\arctg{x})}}} = \cos^2{(\arctg{x})} = \frac{1}{1 + \tg^2{(\arctg{x})}} = \frac{1}{1 + x^2}$

## §1. Теорема о среднем

**Th.** Теорема Ферма.
$f : \langle a; b \rangle \rightarrow R;\ x_0 \in (a, b)$

$f$ – дифференцируема в $x_0$. $f(x_0)$ – наибольшее/наименьшее значение
функции $f(x)$ на $\langle a; b \rangle$. Тогда $f'(x_0) = 0$

**Доказательство**

НУО $f(x_0) \geq f(x),\ \forall x \in \langle a; b \rangle$

$f'_+(x_0) = \lim\limits_{x \rightarrow x_{0_+}} \frac{f(x) - f(x_0)}{x - x_0} \Rightarrow f'(x_0) \leq 0$

$f'_-(x_0) = \lim\limits_{x \rightarrow x_{0_-}} \frac{f(x) - f(x_0)}{x - x_0} \Rightarrow f'(x_0) \geq 0$

Но $f'_+(x_0) = f'_-(x_0) = f'(x_0) \Rightarrow f'(x_0) = 0$

**Rem.**

1.  Важна дифференцируемость. $f(x) = -|x|;\ x_0 = 0$

2.  Теорема не работает на концах. $f(x) = x$, определена на $[-1; 1]$

**Rem.** $f(x_0)$ – наибольшее/наименьшее значение
$\Rightarrow f'(x_0) = 0 \Rightarrow$ касательная горизонтальна

**Th.** Теорема Ролля. $f : [a, b] \rightarrow R;\ f$ – непрерывна на
$[a, b]$ и дифференцируема на $(a, b)$. $f(a) = f(b)$

Тогда $\exists c \in (a, b) : f'(c) = 0$

$f(x)$ непрерывна на $[a, b] \Rightarrow f$ – достигает наибольшего и
наименьшего значения (по Вейерштрассу)

$\exists p, q \in [a; b] : f(p) \leq f(x) \leq f(q),\ \forall x \in [a; b]$

- Если $p \in (a; b)$ или $q \in (a;b)$, то по теореме Ферма все хорошо

- Если $p$ и $q$ – концы отрезка
  $\Rightarrow f(p) = f(q) \Rightarrow f(x) = const \Rightarrow f'(x) = 0;\ \forall x \in [a;b]$

**Rem.**

1.  Дифференцируемость важна везде. $f(x) = |x|$ на $[-1; 1]$

2.  Геометрический смысл теоремы Ролля: если график функции $f(x)$
    проходит через две точки на одной горизонтальной прямой, то
    существует точка, в которой касательная горизонтальна

**Th.** Теорема Лагранжа (теорема о конечном приращении)

$f : [a, b] \rightarrow R$, непрерывна на $[a, b]$, дифференцируема на
$(a, b)$

Тогда $\exists c \in (a, b) : f(b) - f(a) = f'(c) \cdot (b - a)$ (или
$\frac{f(b) - f(a)}{b - a} = f'(c)$)

**Доказательство**

$g(x) = f(x) - kx$, $k$ – подбираем так, чтобы $g(a) = g(b)$

$g(a) - f(a) - ka = g(b) = f(b) - kb$

$k = \frac{f(b) - f(a)}{b - a}$ ($b \neq a$)

$g : [a; b] \rightarrow R$, непрерывна на $[a,b]$, дифференцируема на
$(a, b)$, $g(b) = g(a) \Rightarrow$ по теореме Ролля
$\exists c \in (a, b) : g'(c) = 0$

$g'(x) = (f(x) - kx)' = f'(x) - k \Rightarrow 0 = g'(c) = f'(c) - k \Rightarrow f'(c) = k$,
а $k$ мы задали ранее

**Th.** Теорема Коши (о среднем)

$f, g : [a, b] \rightarrow R$, непрерывны на $[a, b]$ и дифференцируемы
на $(a, b)$

$g'(x) \neq 0,\ \forall x \in (a, b)$

Тогда
$\exists c \in (a, b) : \frac{f(b) - f(a)}{g(b) - g(a)} = \frac{f'(c)}{g'(c)}$

**Доказательство**

$h(x) = f(x) - k \cdot g(x)$. $k$ подбираем так, чтобы $h(a) = h(b)$

$f(a) - k \cdot g(a) = f(b) - k \cdot g(b) \Leftrightarrow k = \frac{f(b) - f(a)}{g(b) - g(a)}$
(знаменатель не ноль, т.к. иначе производная не везде ноль по теореме
Ролля)

$h(x)$ непрерывна на $[a,b]$, дифференцируема на $(a, b)$,
$h(a) = h(b) \Rightarrow \exists c \in (a, b) : h'(c) = 0$

$h'(x) = f'(x) - k \cdot g'(x);\ h'(c) = 0 \Rightarrow f'(c) - k \cdot g'(c) = 0 \Rightarrow \frac{f'(c)}{g'(c)} = k$,
а $k$ мы задали ранее

**Rem.** Геометрический смысл: $k$ – угловой коэффициент наклона хорды;
$\exists c : f'(c) = k$ – есть точка, в которой касательная параллельна
хорде

**Rem2.** Физический смысл: тело движества по плоскости $(g(t), f(t))$ –
координаты тела в момент времени $t$. Опять нарисуем хорду, тогда
$\tg(\alpha) = \frac{f(b) - f(a)}{g(b) - g(a)}$, а $\frac{f'(c)}{g'(c)}$
– крутая штука. Вектор мгновенной скорости в точке $c$ параллелен хорде

**Th.** Следствия из теоремы Лагранжа

$f : \langle a; b \rangle \rightarrow R$

1.  $f$ – непрерывна на $\langle a; b \rangle$ и дифференцируема на
    $(a, b)$ и $\forall x \in (a, b)\ |f'(x)| \leq M\ (\exists M > 0)$

    Тогда
    $|f(x) - f(y)| \leq M \cdot |x - y|\ \forall x, y \in \langle a; b \rangle$

    $[x; y]\ |f(y) - f(x)| = |f'(c)| \cdot |y - x| \leq M \cdot |x - y|$

    **Def.** $f : E \Rightarrow R;\ f$ – липшицева с константой $M$,
    если $\forall x, y \in E\ |f(x) - f(y)| \leq M \cdot |x - y|$

2.  $f$ непрерывна на $\langle a; b \rangle$ и дифференцируема на
    $(a, b)$

    Тогда $f'(x) \geq 0 \Leftrightarrow f(x)$ монотонно возрастает на
    $\langle a; b \rangle$

    **Доказательство**

    - $x < y;\ x, y \in (a, b)$

      $[x; y]$ – Лагранж

      $f(y) - f(x) = f'(c) \cdot (y - x) \geq 0 \Rightarrow f(x) \leq f(y)$

    - $x_0 \in (a, b);\ f'(x_0) = f_+'(x_0) = \lim\limits_{h \rightarrow 0_+} \frac{f(x_0 + h) - f(x_0)}{h} \geq 0$

3.  $f$ непрерывна на $\langle a; b \rangle$ и дифференцируема на
    $(a, b)$

    $f'(x) > 0\ \forall x \in (a, b) \Rightarrow f(x)$ строго возрастает
    на $\langle a; b \rangle$

    **Доказательство**

    $x, y \in (a, b);\ x < y$

    На $[x, y]$ теорема Лагранжа

    $f(y) - f(x) = f'(c) \cdot (y - x) > 0 \Rightarrow f(x) < f(y)$

4.  $f'(x) \leq 0$ на $(a, b) \Leftrightarrow f(x)$ монотонно убывает на
    $\langle a; b \rangle$

5.  $f'(x) < 0$ на $(a, b) \Rightarrow f(x)$ строго убывает на
    $\langle a; b \rangle$

6.  $f$ – непрерывна на $\langle a; b \rangle$ и дифференцируема на
    $(a, b)$

    $\forall x \in (a, b)\ f'(x) = 0 \Rightarrow f(x)$ – постоянная на
    $\langle a; b \rangle$

    **Доказательство**

    $x, y \in q{a, b};\ x < y$

    $[x; y]$ – Лагранж

    $f(x) - f(y) = f'(c) \cdot (x - y) = 0 \Rightarrow f(x) = f(y)$

**Th.** Теорема Дарбу

$f : [a; b] \rightarrow R;\ f$ – дифференцируема на $[a; b]$. $M$ лежит
между $f'(a)$ и $f'(b)$

Тогда $\exists c \in (a; b) : f'(c) = M$

**Доказательство**

1.  $M = 0$

    $f'(c) = M$. НУО $f'(a) < 0 < f'(b)$

    Хочу: $\exists c \in (a; b) : f'(c) = 0$

    $f : [a; b] \rightarrow R;\ f$ – дифференцируема на
    $[a; b] \Rightarrow f$ – непрерывна на $[a; b] \Rightarrow$ по
    теореме Вейерштрасса
    $\exists p, q \in [a; b] : f(p) \leq f(x) \leq f(q),\ \forall x \in [a; b]$

    Если $p$ или $q$ внутри $(a; b)$, то по теореме Ферма $f'(p) = 0$
    или $f'(q) = 0$

    1.  $p = a$

        $f'(a) = f_+'(a) = \lim\limits_{h \rightarrow 0_+} \frac{f(a + h) - f(a)}{h} \geq 0$,
        но у нас $f'(a) < 0 \Rightarrow$ противоречие

    2.  $p = b$

        $f'(b) = f_-'(b) = \lim\limits_{h \rightarrow 0_-} \frac{f(b + h) - f(b)}{h} \leq 0$,
        но у нас $f'(b) > 0 \Rightarrow$ противоречие

    Значит $p \in (a; b)$; Ферма $f'(p) = 0$

2.  $M \neq 0$

    $g(x) = f(x) - Mx$

    $g(x)$ дифференцируема на $[a; b]$

    $g'(x) = f'(x) - M \Rightarrow \begin{cases}
            g'(a) = f'(a) - M < 0\\
            g'(b) = f'(b) - M > 0
        \end{cases}$

    По пункту 1
    $\Rightarrow \exists c : g'(c) = 0 \Rightarrow f'(c) - M = 0 \Rightarrow f'(c) = M$

**Th.** Следствие из теоремы Дарбу

$f : \langle a; b \rangle \rightarrow R$, $f$ – дифференцируема на
$\langle a; b \rangle$ и
$f'(x) \neq 0 \forall x \in \langle a; b \rangle$

Тогда $f(x)$ строго монотонна на $\langle a; b \rangle$

**Доказательство**

$f'(x) > 0$ на $\langle a; b \rangle$ или $f'(x) < 0$ на
$\langle a; b \rangle$

Если не так, то $\exists x \in \langle a; b \rangle : f'(x) < 0$ и
$\exists y \in \langle a; b \rangle : f'(y) > 0$

На $[x; y]$ по теореме Дарбу $\exists c : f'(c) = 0$ – противоречие

**Th.** Правило Лопиталя

$- \infty \leq a < b \leq + \infty;\ f, g : (a; b) \rightarrow R;\ f, g$
– дифференцируемы на $(a; b)$

$g'(x) \neq 0$ на
$(a; b);\ \lim\limits_{x \rightarrow a_+} f(x) = \lim\limits_{x \rightarrow a_+} g(x) = 0$

Тогда, если
$\exists \lim\limits_{x \rightarrow a_+} \frac{f'(x)}{g'(x)} = l \in \overline{R}$,
то $\exists \lim\limits_{x \rightarrow a_+} \frac{f(x)}{g(x)} = l$

**Доказательство**

Зовем Гейне: $\{x_n\} : \begin{cases}
    x_n \neq a \\
    x_n \rightarrow a \\
    x_n \searrow
\end{cases}$

Хочу: $\lim\limits_{n \rightarrow +\infty} \frac{f(x_n)}{g(x_n)} = l$

Зовем Штольца (почуяли кровь):
$\lim\limits_{n \rightarrow +\infty} f(x_n) = \lim\limits_{n \rightarrow +\infty} g(x_n) = 0$

$g(x)$ строго монотонная (т.к. производная не зануляется и следствие из
Дарбу), $x_n$ монотонная по заданию $\Rightarrow g(x_n)$ монотонная

$\lim\limits_{n \rightarrow +\infty} \frac{f(x_{n + 1}) - f(x_n)}{g(x_{n + 1}) - g(x_n)} = l?$
проверяем

По теореме Коши
$\frac{f(x_{n + 1}) - f(x_n)}{g(x_{n + 1}) - g(x_n)} = \frac{f'(c_n)}{g'(c_n)}$
($\exists c_n \in (x_{n + 1}; x_n)$)

Родили последовательность $c_n$, которую по двум милиционерам устремили
к $a$

$\lim\limits_{n \rightarrow +\infty} \frac{f(x_{n + 1}) - f(x_n)}{g(x_{n + 1}) - g(x_n)} = \lim\limits_{n \rightarrow +\infty} \frac{f'(c_n)}{g'(c_n)} = l$

**Th.** Правило Лопиталя

$- \infty \leq a < b \leq + \infty;\ f, g : (a; b) \rightarrow R;\ f, g$
– дифференцируемы на $(a; b)$

$g'(x) \neq 0$ на
$(a; b);\ \lim\limits_{x \rightarrow a_+} g(x) = + \infty$

Тогда если
$\exists \lim\limits_{x \rightarrow a_+} \frac{f'(x)}{g'(x)} = l \in \overline{R}$,
то $\exists \lim\limits_{x \rightarrow a_+} \frac{f(x)}{g(x)} = l$

**Доказательство**

Штольц для $\frac{\infty}{\infty}$

**Ex.**

1.  $\lim\limits_{x \rightarrow 0_+} x^x = \lim\limits_{x \rightarrow 0_+} e^{x\ln{x}}$

    $\lim\limits_{x \rightarrow 0_+} x\ln{x} = \lim\limits_{x \rightarrow 0_+} \frac{\ln{x}}{\frac{1}{x}} = \lim\limits_{x \rightarrow 0_+} \frac{\frac{1}{x}}{-\frac{1}{x^2}} = \lim\limits_{x \rightarrow 0_+} -x = 0$

    $\Rightarrow \lim\limits_{x \rightarrow 0_+} x^x = e^0 = 1$

2.  $\lim\limits_{x \rightarrow + \infty} \frac{\ln{x}}{x^p} = \lim\limits_{x \rightarrow + \infty} \frac{\frac{1}{x}}{px^{p - 1}} = \lim\limits_{x \rightarrow + \infty} \frac{1}{px^p} = 0$

## §2. Производные высших порядков

**Def.**
$f : \langle a; b \rangle \rightarrow R;\ x_0 \in \langle a; b \rangle$,
$f$ – дифференцируема в окрестности $x_0$

Тогда если $f'(x)$ дифференцируема в $x_0$, то $f(x)$ – дважды
дифференцируема в $x_0$ и $f''(x_0) = (f')'(x_0)$

Аналогично $f$ – трижды дифференцируема в $x_0$, если $f$ дважды
дифференцируема в окрестности $x_0$ и $f''(x)$ дифференцируема в $x_0$ и
$f'''(x) = (f'')'(x_0)$

**Def.** $f : \langle a; b \rangle \rightarrow R;\ f$ – дифференцируема
на $\langle a; b \rangle$ и $f'$ – непрерывна на $\langle a; b \rangle$

Тогда говорят, что $f$ – непрерывно дифференцируема на
$\langle a; b \rangle$

**Обозначения**

1.  $f : E \rightarrow R$

    $f \in C(E) \Leftrightarrow f$ – непрерывна на $E$

2.  $f : \langle a; b \rangle \rightarrow R$

    $f \in C^n(\langle a; b \rangle) \Leftrightarrow \begin{cases}
            f \text{ -- } n \text{ раз дифференцируема на } \langle a; b \rangle \\
            \text{все производные непрерывны}
        \end{cases}$

3.  $f : \langle a; b \rangle \rightarrow R$

    $f \in C^{\infty}(\langle a; b \rangle) \Leftrightarrow \forall n \in N\ \ f \in C^n(\langle a; b \rangle)$

**Rem.**
$C(\langle a; b \rangle) \supset C^1(\langle a; b \rangle) \supset C^2(\langle a; b \rangle) \supset ... \supset C^{\infty}(\langle a; b \rangle)$

Все вложения строгие, т.к. $f_n(x) = x^{n + \frac{1}{3}}$;
$f_n(x) = x^n \cdot \sqrt[3]{x}$

$f_n(x) \in C^n(R);\ f_n(x) \not\in C^{n + 1}(R)$

$(f_n(x))' = (n + \frac{1}{3})x^{(n - 1) + \frac{1}{3}};\ (f_n(x))'' = (n + \frac{1}{3})((n - 1) + \frac{1}{3})x^{(n - 2) + \frac{1}{3}}$

$(f_n(x))^{(n)} = (n + \frac{1}{3})((n - 1) + \frac{1}{3})...(\frac{1}{3})x^{\frac{1}{3}} = k \cdot x^{\frac{1}{3}}$

$g(x) = x^{\frac{1}{3}}$ не является дифференцируемой в $x = 0$

**Th.** Теорема об арифметических действиях

$f, g : \langle a; b \rangle \rightarrow R;\ x_0 \in \langle a; b \rangle;\ f, g$
$n$ раз дифференцируемы в $x_0$

1.  $\alpha f + \beta g$ – $n$ раз дифференцируема в $x_0$ и
    $(\alpha f + \beta g)^{(n)}(x_0) = \alpha f^{(n)}(x_0) + \beta g^{(n)}(x_0)$

2.  $f \cdot g$ – $n$ раз дифференцируема в $x_0$ и
    $(f \cdot g)^{(n)}(x_0) = \sum\limits_{k = 0}^{n} C_n^k f^{(k)}(x_0) \cdot g^{(n - k)}(x_0)$
    – формула Лейбница

3.  $f(\alpha x + \beta)$ – $n$ раз дифференцируема в $x_0$ и
    $(f(\alpha x + \beta))^{(n)} = \alpha^n \cdot f^{(n)}(\alpha x_0 + \beta)$

**Доказательство**

1.  По индукции. База $n = 1$ – теорема о производной суммы

    Переход $n \rightarrow n + 1$

    $(\alpha f + \beta g)^{(n + 1)}(x_0) = ((\alpha f + \beta g)^{(n)})'(x_0) = (\alpha f^{(n)}(x_0) + \beta g^{(n)})'(x_0) = \alpha f^{(n + 1)}(x_0) + \beta g^{(n + 1)}(x_0)$

2.  ММИ. База $n = 1$ – теорема о производной произведения

    $(f \cdot g)'(x_0) = f'(x_0)g(x_0) + f(x_0)g'(x_0) = \sum\limits_{k = 0}^{1} C_1^k f^{(k)}(x_0) \cdot g^{(1 - k)}(x_0)$

    Переход $n \rightarrow n + 1$

    $(f \cdot g)^{(n + 1)}(x_0) = ((f \cdot g)^{(n)})'(x_0) = (\sum\limits_{k = 0}^{n} C_n^k f^{(k)}(x_0) \cdot g^{(n - k)}(x_0))' = \sum\limits_{k = 0}^{n} C_n^k (f^{(k)}(x_0) \cdot g^{(n - k)})'(x_0) =$

    $\sum\limits_{k = 0}^{n} C_n^k (f^{(k + 1)}(x_0) \cdot g^{(n - k)}(x_0) + f^{(k)}(x_0) \cdot g^{(n - k + 1)}(x_0)) = \sum\limits_{k = 0}^{n} C_n^k f^{(k + 1)}(x_0) \cdot g^{(n - k)}(x_0) + \sum\limits_{k = 0}^{n} C_n^k f^{(k)}(x_0) \cdot g^{(n - k + 1)}(x_0) =$

    $= \sum\limits_{m = 1}^{n + 1} C_n^{m - 1} f^{(m)}(x_0) \cdot g^{(n + 1 - m)}(x_0) + \sum\limits_{m = 0}^{n} C_n^m f^{(m)}(x_0) \cdot g^{(n + 1 - m)}(x_0) = f(x_0) \cdot g^{(n + 1)}(x_0) + \sum\limits_{m = 1}^{n} C_{n + 1}^m \cdot f^{(m)}(x_0) \cdot g^{(n + 1 - m)}(x_0) + f^{(n + 1)}(x_0) \cdot g(x_0) =$

    $= \sum\limits_{m = 0}^{n + 1} C_{n + 1}^m \cdot f^{(m)}(x_0) \cdot g^{(n + 1 - m)}(x_0)$

3.  на экзамене писать, что это упражнение

**Ex.**

1.  $(x^p)^{(n)} = p(p - 1)...(p - n + 1)x^{p - n}$

2.  $(\frac{1}{x})^{(n)} = (-1)(-2)(-3)...(-n)\frac{1}{x^{n + 1}} = \frac{n!}{x^{n + 1}}$

3.  $(\ln{x})^{(n)} = ((\ln{x})')^{(n - 1)} = (\frac{1}{x})^{(n - 1)} = \frac{(n - 1)!(-1)^{n - 1}}{x^n}$

4.  $(a^x)^{(n)} = (\ln{a} \cdot a^x)^{(n - 1)} = (\ln{a})^n \cdot a^x$

    $(e^x)^{(n)} = e^x$

5.  $(\sin{x})^{(n)} = \sin{(x + \frac{\pi}{2}n)}$

    $(\cos{x})^{(n)} = \cos{(x + \frac{\pi}{2}n)}$

**Th.** Формула Тейлора для многочлена

$T(x)$ – многочлен степени $n$, тогда
$T(x) = \sum\limits_{k = 0}^n \frac{T^{(k)}(x_0)}{k!}(x - x_0)^k$

**Доказательство**

**Lm. 1** $T(x) = \sum\limits_{k = 0}^n a_kx^k$, то его можно
представить в виде $T(x) = \sum\limits_{k = 0}^n c_k(x - x_0)^k$

$T(x) = \sum\limits_{k = 0}^n a_kx^k = \sum\limits_{k = 0}^n a_k (x - x_0 + x_0)^k = \sum\limits_{k = 0}^n a_k (t + x_0)^k$
– раскроем скобки по биному

$= \sum\limits_{k = 0}^n c_k \cdot t^k = \sum\limits_{k = 0}^n c_k(x - x_0)^k$

**Lm. 2** $f(x) = (x - x_0)^k$, то $f^{(m)}(x_0) = \begin{cases}
    m!,\ m = k \\
    0,\ m \neq k
\end{cases}$

$f^{(m)}(x) = k(k - 1)\ldots(k - m + 1) \cdot (x - x_0)^{k - m}$

Если $k > m$, то степень у $x - x_0$ будет больше нуля
$\Rightarrow f^{(m)}(x_0) = 0$

Если $k < m$, то при дифференцировании вылезет 0 в множителе
$\Rightarrow f^{(m)}(x) = 0\ \forall x$

Доказываем теорему:

$T(x) \stackrel{Lm 1}{=} \sum\limits_{k = 0}^n c_k (x - x_0)^k$

$T^{(m)}(x_0) = (\sum\limits_{k = 0}^n c_k(x - x_0)^k)^{(m)}|_{x = x_0} \stackrel{Lm 2}{=} c_m \cdot m!$

$c_m = \frac{T^{(m)}(x_0)}{m!}$

$T(x) = \sum\limits_{k = 0}^n$

**Def.** $f(x)\ n$ раз дифференцируема в точке $x_0$, тогда

$T_{n, x_0}f(x) = \sum\limits_{k = 0}^n \frac{f^{(k)}(x_0)}{k!}(x - x_0)^k$
– многочлен Тейлора степени $n$ для функции $f(x)$ в точке $x_0$

$f(x) - T_{n, x_0}f(x) = R_{n, x_0}f(x)$ – остаток в формуле Тейлора
(будем записывать в разной форме)

$f(x) = T_{n, x_0}f(x) + R_{n, x_0} f(x)$ – формула Тейлора для $f(x)$ в
точке $x_0$

Иногда
$f(x) = \sum\limits_{k = 0}^n \frac{f^{(k)}(x_0)}{k!}(x - x_0)^k + R_{n, x_0}f(x)$

**Lm.** $g(x) - n$ раз дифференцируема в точке $x_0$ и
$g(x_0) = g'(x_0) = g''(x_0) = \ldots = g^{(n)}(x_0) = 0$

Тогда $g(x) = o((x - x_0)^n),\ x \rightarrow x_0$

**Доказательство**

$g(x) = o((x - x_0)^n) \Leftrightarrow \lim\limits_{x \rightarrow x_0} \frac{g(x)}{(x - x_0)^n} = 0$

$\lim\limits_{x \rightarrow x_0} \frac{g(x)}{(x - x_0)^n} \stackrel{\{ \frac{0}{0} \}}{=} \lim\limits_{x \rightarrow x_0} \frac{g'(x)}{n(x - x_0)^{n-1}} = \lim\limits_{x \rightarrow x_0} \frac{g''(x)}{n(n-1)(x - x_0)^{n-2}} = \ldots = \lim\limits_{x \rightarrow x_0} \frac{g^{(n-1)}(x)}{n(n-1)\ldots 2(x - x_0)} = \lim\limits_{x \rightarrow x_0} \frac{o(x - x_0)}{n!(x - x_0)} = 0$

$g^{(n - 1)}(x)$ – дифференцируема в точке
$x_0 \Leftrightarrow g^{(n - 1)}(x) = g^{(n - 1)}(x_0) + g^{(n)}(x_0)(x - x_0) + o(x - x_0)$

**Th.** Формула Тейлора с остатком в форме Пеано

$f$ – $n$ раз дифференцируема в точке $x_0$, тогда

$f(x) = T_{n, x_0}f(x) + o((x - x_0)^n) = \sum\limits_{k = 0}^n \frac{f^{(k)}(x_0)}{k!}(x - x_0)^k + o((x - x_0)^n),\ x \rightarrow x_0$

**Доказательство**

$f(x) - T_{n, x_0}f(x) = g(x)$ – дифференцируема в точке $x_0\ n$ раз

$g^{(m)}(x_0) = f^{(m)}(x_0) - (T_{n, x_0}f(x))^{(m)}|_{x = x_0} = f^{(m)}(x_0) - \frac{f^{(m)}(x_0)}{m!} \cdot m! = 0$

$g(x_0) = T_{n, x_0}f(x_0) - f(x_0) = 0$

$\forall 0 \leq m \leq n\ g^{(m)}(x_0) = 0 \xRightarrow{Lm} g(x) = o((x - x_0)^n),\ x \rightarrow x_0$

**Следствие** Единственность многочлена Тейлора

$f$ – $n$ раз дифференцируема в точке $x_0;\ P(x)$ – многочлен степени
$\leq n;\ f(x) = P(x) + o((x - x_0)^n),\ x \rightarrow x_0$

Тогда $P(x) = T_{n, x_0}f(x)$

**Доказательство**

$\begin{cases}
    P(x) - T_{n, x_0}f(x) = o(x - x_0)^n \\
    P(x) - T_{n, x_0}f(x) = \sum\limits_{k = 0}^n a_k(x - x_0)^k
\end{cases}$

Пусть $a_m \neq 0,\ m$ – наименьший номер

$\sum\limits_{k = 0}^n a_k(x - x_0)^k = \sum\limits_{k = m}^n a_k(x - x_0)^k = o(x - x_0)^n,\ x \rightarrow x_0 \Leftrightarrow \begin{cases}
    \lim\limits_{x \rightarrow x_0} \frac{o(x - x_0)^n}{x - x_0}^m = 0 \\
    \lim\limits_{x \rightarrow x_0} \frac{\sum\limits_{k = m}^n a_k(x - x_0)^k}{x - x_0}^m = a_m
\end{cases}$ ?!

**Th.** Формула Тейлора с остатком в форме Лагранжа

$f : \langle a; b \rangle \rightarrow R\ f$ – $n$ раз дифференцируема на
$\langle a; b \rangle;\ x, x_0 \in \langle a; b \rangle$

Тогда $\exists c$ между $x$ и
$x_0 : f(x) = T_{n, x_0}f(x) + \frac{f^{(n + 1)}(c)}{(n + 1)!} (x - x_0)^{n + 1} = \sum\limits_{k = 0}^n \frac{f^{(k)}(x_0)}{k!}(x - x_0)^k + \frac{f^{(n + 1)}(c)}{(n + 1)!}(x - x_0)^{n + 1}$

**Доказательство**

$fix\ x \in \langle a; b \rangle$

$f(x) = T_{n, x_0}f(x) + M(x - x_0)^{n + 1}$

Найдем такое $M$, что выполняется равенство. Хотим
$M = \frac{f^{(n + 1)}(c)}{(n + 1)!}$

$g(t) = f(t) - T_{n, x_0}f(t) - M(t - x_0)^{n + 1}$

$g(t)$ – $(n + 1)$ раз дифференцируема на $\langle a; b \rangle$

$g(x) = 0;\ g(x_0) = g'(x_0) = g''(x_0) = \ldots = g^{(n)}(x_0) = 0$

$f^{(m)}(x_0) = (T_{n, x_0}f(t))^{(m)}_{t = x_0}$

На $[x; x_0]$ зовем теорему Ролля для $g(t)$

$g(x) = g(x_0) = 0 \Rightarrow \exists c_1 \in [x; x_0] : g'(c) = 0$

$\begin{cases}
    [c_1; x_0]\ g'(t) \\
    g'(c_1) = 0 \\
    g'(x_0) = 0
\end{cases} \xRightarrow{\text{т. Ролля}} \exists c_2 \in [c_1; x_0] : g''(c_2) = 0$
итд

$\begin{cases}
    g^{(n)}(c_n) = 0 \\
    g^{(n)}(x_0) = 0
\end{cases} g^{(n)}(t) \xRightarrow{\text{т. Ролля}} \exists c \in [c_n; x_0] : g^{(n + 1)}(c) = 0$

$g^{(n + 1)}(t) = f^{(n + 1)}(t) + 0 - M(n + 1)!$

$0 = g^{(n + 1)}(c) = f^{(n + 1)}(c) - M(n + 1)!$

**Следствие**

1.  $\forall t \in \langle a; b \rangle\ |f^{(n+1)}(t) \leq k$, тогда
    $R_{n, x_0}f(x) = O((x - x_0)^{n + 1})$

    $|R_{n, x_0}f(x)| = |\frac{f^{n + 1}(c)}{(n + 1)!}(x - x_0)^{n + 1}| \leq \frac{k}{(n + 1)!} \cdot |x - x_0|^{n + 1} \Rightarrow R_{n, x_0} f(x) = O((x - x_0)^{n + 1})$

2.  $\forall n \in N\ |f^{(n)}(t)| \leq k\ \forall t \in \langle a; b \rangle$,
    то
    $\lim\limits_{n \rightarrow + \infty} T_{n, x_0}f(x) = f(x)\ \forall x \in \langle a; b \rangle$

    Это
    $\Leftrightarrow \lim\limits_{n \rightarrow + \infty} (f(x) - T_{n, x_0}f(x)) = 0$

    $|\frac{f^{(n + 1)}(c)}{(n + 1)!}(x - x_0)^n| \leq \frac{k(x - x_0)^n}{(n + 1)!} \xrightarrow[n \rightarrow + \infty]{} 0$

**Формулы Тейлора для элементарных функций** ($x_0 = 0$)

1.  $e^x = \sum\limits_{k = 0}^n \frac{x^k}{k!} + o(x^n)$

    $\forall k\ f^{(k)}(0) = 1 \Rightarrow (e^x)^{(k)} = e^x\ \ e^0 = 1$

2.  $\sin{x} = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \ldots + \frac{(-1)^n x^{2n + 1}}{(2n + 1)!} + o(x^{2n + 1})$

    $(\sin{x})^{(k)} = \sin{(x + \frac{\pi}{2}k)}|_{x = 0}\ f^{(k)}(0) = \sin{(\frac{\pi}{2}k)}$

    $\cos{x} = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \ldots + \frac{(-1)^n x^{2n}}{(2n)!} + o(x^{2n})$

    $f^{(k)}(0) = \cos{(\frac{\pi}{2}k)}$

3.  $\ln{(1 + x)} = x - \frac{x^2}{2} + \frac{x^3}{3} - \frac{x^4}{4} + \ldots + \frac{(-1)^nx^n}{n} + o(x^n)$

    $f^{(k)}(x) = \frac{(-1)^{k - 1}(k - 1)!}{(1 + x)^k}\ f^{(k)}(0) = (-1)^{k - 1} \cdot (k - 1)!$

4.  $(1 + x)^p = 1 + px + \frac{p(p-1)x^2}{2!} + \frac{p(p-1)(p-2)x^3}{3!} + \ldots + \frac{p(p-1)\ldots(p-n+1)x^n}{n!} + o(x^n)$

**Ряды Тейлора для $\sin{x}/\cos{x}/exp(x)$**

1.  $\sin{x}/\cos{x}$

    $\forall n\ |\sin^{(n)}(x)| = |\sin{(x + \frac{\pi}{2}n)}| \leq 1 \xRightarrow{\text{След.}} \lim\limits_{n \rightarrow + \infty} (T_{n, x_0}f(x)) = \sin{x}\ \forall x$

    $\lim\limits_{n \rightarrow + \infty} (\sum\limits_{k = 0}^{+ \infty} \frac{(-1)^k x^{2k+1}}{(2k+1)!}) = \sin{x}$

    Частичная сумма ряда
    $\sum\limits_{n = 0}^n \frac{(-1)^k x^{2k+1}}{(2k+1)!} \Rightarrow \sin{x} = \sum\limits_{k = 0}^{+ \infty} \frac{(-1)^k x^{2k+1}}{(2k+1)!}\ \forall x \in R$

    То есть можем $o(x^{2n+1}) \rightarrow o(x^{2n + 2})$

    С $\cos{x}$ аналогично

2.  $f(x) = e^x$

    Рассмотрим
    $x \leq b\ e^x = \sum\limits_{k = 0}^{+ \infty} \frac{x^n}{n!}$

    $(e^x)^{(n)} = e^x$

    $|e^x| \leq e^b = k \xRightarrow{\text{След.}} T_{n, x_0}f(x) \xrightarrow[n \rightarrow + \infty]{} e^x$

    $\lim\limits_{n \rightarrow + \infty} \sum\limits_{k = 0}^{n} \frac{x^k}{k!} = e^x\ \forall x \leq b$

    Частичная сумма ряда
    $\Rightarrow e^x = \sum\limits_{k = 0}^{+ \infty} \frac{x^k}{k!}$

**Th.** Число $e$ – иррациональное

**Доказательство**

Пусть $e = \frac{m}{n};\ m, n \in N;\ n \geq 2$, т.к. $2 < e < 3$

$e^x = \sum\limits_{k = 0}^n \frac{x^k}{k!} + \frac{e^c}{(n + 1)!} x^{n + 1}$

$\frac{m}{n} = e = 1 + \frac{1}{1!} + \frac{1}{2!} + \ldots + \frac{1}{n!} + \frac{e^c}{(n + 1)!}$,
где $0 < c < 1$

$m (n - 1)! = n! + n! + \frac{n!}{2!} + \frac{n!}{3!} + \ldots + \frac{n!}{n!} + \frac{e^c n!}{(n + 1)!}$

Слева натуральное, справа сумма факториалов точно натуральна
$\Rightarrow \frac{e^c}{n + 1}$ – натуральное
$\Rightarrow \frac{e^c}{n + 1} \geq 1$

$\frac{e^c}{n + 1} < \frac{e}{n + 1} < \frac{3}{n + 1} \Rightarrow n + 1 \geq 3 \Rightarrow \frac{e^c}{n + 1} \leq 1$
?!

## §3. Экстремум функций

**Точки экстремума:**

$f : E \rightarrow R;\ a \in E$

1.  **Def.** Точка $a$ – точка локального минимума для $f(x)$, если
    существует окрестность $U$ точки $a$ такая, что

    $\forall x \in U \bigcap E\ f(x) \geq f(a)$

2.  **Def.** Точка $a$ – точка локального максимума, если $\exists U$ –
    окрестность точки $a$ такая, что

    $\forall x \in U \bigcap E\ f(x) \leq f(a)$

3.  **Def.** Точка $a$ – точка строгого локального минимума (максимума),
    если $\exists U$ – окрестность точки $a$ такая

    $\forall x \in \mathring{U}(a) \bigcap E\ f(x) > f(a)\ \ (f(x) < f(a))$

**Th.** Необходимые условия экстремума

$f : \langle a; b \rangle \rightarrow R;\ f$ – дифференцируема в точке
$x_0;\ x_0 \in (a; b)$

Если $x_0$ – точка экстремума, то $f'(x_0) = 0$

**Доказательство**

НУО $x_0$ – локальный минимум

$\exists \delta\ \forall x \in (x_0 - \delta; x_0 + \delta)\ f(x) \geq f(x_0)$

Рассмотрим $f : U \rightarrow R$

Точка $x_0$ – глобальный минимум на $U$. $x_0$ – внутренняя точка
$U \xRightarrow{\text{т. Ферма}} f'(x_0) = 0$

**Rem.**

1.  $\not\Leftarrow f(x) = x^3\ f'(0) = 0$, но $x = 0$ не экстремум

2.  Экстремумы бывают в точках, в которых нет дифференцируемости
    $f(x) = |x|$

3.  Экстремумы бывают на концах $x : [0; 1] \rightarrow R$

**Th.** Достаточные условия экстремума в терминах первой производной

$x_0 \in (a, b);\ f : \langle a; b \rangle \rightarrow R;\ f$ –
непрерывна в $x_0$ и $f$ – дифференцируема на
$(x_0 - \delta; x_0) \bigcup (x_0; x_0 + \delta)$

Тогда если

1.  $f'(x) < 0$ на $(x_0 - \delta; x_0)$ и $f'(x) > 0$ на
    $(x_0; x_0 + \delta)$, то $x_0$ – строгий локальный минимум

2.  $f'(x) > 0$ на $(x_0 - \delta; x_0)$ и $f'(x) < 0$ на
    $(x_0; x_0 + \delta)$, то $x_0$ – строгий локальный максимум

3.  $f'(x)$ не меняет знак в точке $x_0$, то точка $x_0$ не экстремум

**Доказательство**

1.  На $[x_0 - \frac{\delta}{2}; x_0]$ непрерывность +
    $x_0 - \frac{\delta}{2}; x_0$ дифференцируемость + $f'(x) < 0$ на
    $(x_0 - \frac{\delta}{2}; x_0)$

    $\xRightarrow{\text{Сл. т. Лагранжа}} f(x)$ строго убывает на
    $[x_0 - \frac{\delta}{2}; x_0] \Rightarrow \forall x \in [x_0 - \frac{\delta}{2}; x_0)\ f(x) > f(x_0)$

    На $[x_0; x_0 + \frac{\delta}{2}]$ непрерывна + дифференцируема
    внутри + $f'(x) > 0$

    $\xRightarrow{\text{Сл. т. Лагранжа}} f(x)$ строго возрастает на
    $[x_0; x_0 + \frac{\delta}{2}] \Rightarrow \forall x \in (x_0; x_0 + \frac{\delta}{2}]\ f(x) > f(x_0)$

**Th.** Достаточное условие экстремума в терминах второй производной

$f : \langle a; b \rangle \rightarrow R;\ x_0 \in (a; b);\ f$ – дважды
дифференцируема в точке $x_0$ и $f'(x_0) = 0$. Тогда

1.  Если $f''(x_0) > 0$, то $x_0$ – строгий локальный минимум

2.  Если $f''(x_0) < 0$, то $x_0$ – строгий локальный максимум

**Th.** Достаточное условие экстремума в терминах $n$-ой производной

$f : \langle a; b \rangle \rightarrow R;\ x \in (a; b);\ f$ – $n$ раз
дифференцируема в точке $x_0$.
$f'(x_0) = f''(x_0) = \ldots = f^{(n - 1)}(x_0) = 0$. Тогда

1.  Если $n \vdots 2;\ f^{(n)}(x_0) > 0$, то $x_0$ – строгий локальный
    минимум

2.  Если $n \vdots 2;\ f^{(n)}(x_0) < 0$, то $x_0$ – строгий локальный
    максимум

3.  Если $n \not \vdots 2$ и $f^{(n)}(x_0) \neq 0$, то $x_0$ – не
    экстремум

**Доказательство**

Тейлор + Пеано

$f(x) = f(x_0) + \sum\limits_{k = 1}^n \frac{f^{(k)}(x_0)}{k!}(x - x_0)^k + o((x - x_0)^n) = f(x_0) + \frac{f^{(n)}(x_0)}{n!}(x - x_0)^n + o((x - x_0)^n)$

$f(x) - f(x_0) = (x - x_0)^n (\frac{f^{(n)}(x_0)}{n!} + o(1))$

## §4. Выпуклые функции

**Def.** $f : \langle a; b \rangle \rightarrow R;\ f$ – выпуклая на
$\langle a; b \rangle$, если
$\forall x, y \in \langle a; b \rangle,\ \forall \lambda \in [0; 1]$

$f(\lambda x + (1 - \lambda)y) \leq \lambda f(x) + (1 - \lambda)f(y)$

Если знак $<$ – строго выпуклая. Если знак $\geq$ – вогнутая. Если знак
$>$ – строго вогнутая

**Rem.**

1.  Выпуклая $\Leftrightarrow$ выпуклая вниз

2.  Вогнутая $\Leftrightarrow$ выпуклая вверх

**Ex.** $y = x^2$ – выпуклая (проверим по определению)

$\forall \lambda \in [0; 1]\ \forall x, y\ (\lambda x + (1 - \lambda)y)^2 \leq \lambda x^2 + (1 - \lambda)y^2$

$2\lambda(1 - \lambda)xy \leq x^2(\lambda - \lambda^2) + y^2((1 - \lambda) - (1 - \lambda)^2)$

$2\lambda(1 - \lambda)xy \leq \lambda(1 - \lambda)x^2 + (1 - \lambda)\lambda y^2$

$2xy \leq x^2 + y^2$

**Геометрический смысл**

$z = \lambda x + (1 - \lambda) y$. НУО $x < y$

$x < \lambda x + (1 - \lambda)x < z < \lambda y + (1 - \lambda)y < y \Rightarrow z \in (x; y)$

$\lambda (y - x) = y - z \Leftrightarrow z = y - \lambda (y - x) = \lambda x + (1 - \lambda)y$

Проведем хорду по двум точка $u$ и $w$:

$y = \frac{f(w) - f(u)}{w - u} \cdot (x - u) + f(u)$

Возьмем произвольную $v \in (u, w)$

$\frac{f(w) - f(u)}{w - u} \cdot (v - u) + f(u) = f(w) \cdot \frac{v - u}{w - u} + f(u) \cdot (1 - \frac{v - u}{w - u}) = f(u) \cdot \frac{w - v}{w - u} + f(w) \cdot \frac{v - u}{w - u}$

Если возьмем
$\lambda = \frac{w - v}{w - u} \Leftrightarrow 1 - \lambda = \frac{v - u}{w - u}$,
то получим

$f$ – выпуклая $\Leftrightarrow$ график $f(x)$ лежит под хордой

**Переформулировка определения.**
$x \rightarrow u;\ \lambda x + (1 - \lambda)y \rightarrow v;\ y \rightarrow w$

$\lambda = \frac{w - v}{w - u};\ 1 - \lambda = \frac{v - u}{w - u}$

$f(v) \leq \frac{w - v}{w - u} \cdot f(u) + \frac{v - u}{w - u} \cdot f(w)\ |\ \cdot (w - u)$

$(w - u)f(v) \leq (w - v)f(u) + (v - u)f(w)$

Если это выполняется
$\forall u, v, w \in \langle a; b \rangle : u < v < w$, то $f$ –
выпуклая

**Свойства выпуклой функции:**
($f, g : \langle a; b \rangle \rightarrow R$)

1.  $f, g$ – выпуклые $\Rightarrow f + g$ – выпуклая

2.  $\alpha > 0;\ f$ – выпуклая $\Rightarrow \alpha f$ – выпуклая

3.  $f$ – выпуклая $\Rightarrow (-f)$ – вогнутая

**Lm.** Лемма о трех хордах

$f: \langle a; b \rangle \rightarrow R$ – выпуклая, тогда
$\forall u, v, w \in \langle a; b \rangle : u < v < w$

$\frac{f(v) - f(u)}{v - u} \leq \frac{f(w) - f(u)}{w - u} \leq \frac{f(w) - f(v)}{w - v}$,
причем каждое из трех неравенств равносильно выпуклости

**Rem.** Если неравенства строгие, то $f$ – строго выпуклая

**Доказательство**

1.  $\frac{f(v) - f(u)}{v - u} \leq \frac{f(w) - f(u)}{w - u} \Leftrightarrow (w - u)(f(v) - f(u)) \leq (v - u)(f(w) - f(u)) \Leftrightarrow$

    $\Leftrightarrow f(v) (w - u) \leq f(u) (v - u - w + u) + f(w) (v - u) \Leftrightarrow f(v) (w - u) \leq f(u) (v - w) + f(w) (v - u) \Leftrightarrow$

    $\Leftrightarrow f$ – выпуклая

2.  $(w - v)(f(w) - f(u)) \leq (w - u)(f(w) - f(v)) \Leftrightarrow (w - u)f(v) \leq (w - v)f(u) + (v - u)f(w) \Leftrightarrow f$
    – выпуклая

3.  Упражнение

**Th.** $f : \langle a; b \rangle \rightarrow R$ – выпуклая, тогда

$\forall x_0 \in \langle a; b \rangle$ существуют конечные $f'_+(x_0)$ и
$f'_-(x_0)$, причем $f'_+(x_0) \geq f'_-(x_0)$

**Доказательство**

$fix\ x;\ u, v, w$ из определения ($u < v < w;\ u < x < v$)

$\frac{f(v) - f(x)}{v - x}$ – возрастает по $v$

$\frac{f(v) - f(w)}{v - u} \leq \frac{f(v) - f(x)}{v - x}$ по лемме о
трех хордах

$\lim\limits_{v \rightarrow x_+} \frac{f(v) - f(x)}{v - x} = f'_+(x)$
если существует

$\frac{f(v) - f(x)}{v - x}$ убывает при $v \rightarrow x_+$ + есть
ограниченность снизу, т.к.
$\frac{f(x) - f(u)}{x - u} \leq \frac{f(v) - f(x)}{v - x}$

Значит
$\exists \lim\limits_{v \rightarrow x_+} \frac{f(v) - f(x)}{v - x} = f'_+(x) \geq \frac{f(x) - f(u)}{x - u}$

$\frac{f(x) - f(u)}{x - u}$ – возрастает по $u$ + ограничена сверху
$f'_+(x)$

Значит
$\exists \lim\limits_{u \rightarrow x_-} \frac{f(x) - f(u)}{x - u} = \lim\limits_{u \rightarrow x_-} \frac{f(u) - f(x)}{u - x} = f'_-(x)$

Итого: $f'_-(x) \leq f'_+(x)$

**Следствие.** $f : \langle a; b \rangle \rightarrow R$ – выпуклая, то
$f$ – непрерывная на $(a; b)$

**Доказательство**

Выпуклая $\Rightarrow \forall x_0 \in (a; b)$

$\begin{cases}
    \exists f'_+(x_0) \in R \Rightarrow f(x) \text{ -- непрерывная в точке } x_0 \text{ справа} \\
    \exists f'_-(x_0) \in R \Rightarrow f(x) \text{ -- непрерывная в точке } x_0 \text{ слева}
\end{cases} \Rightarrow f(x)$ – непрерывная в точке $x_0$

**Rem.** Про концы ничего сказать нельзя

**Th.** $f : \langle a; b \rangle \rightarrow R$ – дифференцируема,
тогда

$f$ – выпуклая
$\Leftrightarrow f(x) \geq f(x_0) + f'(x_0)(x - x_0)\ \forall x, x_0 \in \langle a; b \rangle$

(т.е. график функции лежит над касательной)

**Доказательство**

- $u < v < w$ ($x_0 \leftrightarrow v$)

  $f(u) \geq f(v) + f'(v)(u - v)\ |\ (w - v) > 0$

  $f(w) \geq f(v) + f'(v)(w - v)\ |\ (v - u) > 0$

  $f(u)(w - v) + f(w)(v - u) \geq f(v)(w - u)$

- Хотим: $\forall x\ \ f(x) \geq f(x_0) + f'(x_0)(x - x_0)$. НУО
  $x > x_0$

  $\Leftrightarrow \frac{f(x) - f(x_0)}{x - x_0} \geq f'(x_0) = f'_+(x_0) = \lim\limits_{y \rightarrow x_0^+} \frac{f(y) - f(x_0)}{y - x_0}$

  По лемме о трех хордах:
  $\frac{f(x) - f(x_0)}{x - x_0} \geq \frac{f(y) - f(x_0)}{y - x_0} \xrightarrow[y \rightarrow x_0^+]{} f'_+(x_0)$

**Критерий выпуклости**

1.  $f : \langle a; b \rangle \rightarrow R$, непрерывна на
    $\langle a; b \rangle$ и дифференцируема на $(a; b)$, тогда

    $f$ – (строго) выпуклая $\Leftrightarrow f'(x)$ (строго) монотонно
    возрастает на $(a; b)$

2.  $f : \langle a; b \rangle \rightarrow R$, непрерывна на
    $\langle a; b \rangle$ и дважды дифференцируема на $(a; b)$, тогда

    $f$ – выпуклая $\Leftrightarrow f''(x) \geq 0$

**Доказательство**

**Rem.** $1 \Rightarrow 2$

- $u < v$

  $f'(u) \leq \frac{f(v) - f(u)}{v - u} \leq f'(v) \Rightarrow f'(x)$ –
  возрастает

- $\frac{f(v) - f(u)}{v - u}$ и $\frac{f(w) - f(v)}{w - v}$. Хотим
  $(1) \leq (2)$

  $\frac{f(v) - f(u)}{v - u} \stackrel{\text{Лагранж}}{=} f'(\xi_1)$

  $\frac{f(w) - f(v)}{w - v} \stackrel{\text{Лагранж}}{=} f'(\xi_2)$

  Получаем $u < \xi_1 < v < \xi_2 < w$

  $f'(\xi_1) \leq f'(\xi_2)$, т.к. $f'$ – возрастает, тогда
  $\frac{f(v) - f(u)}{v - u} \leq \frac{f(w) - f(v)}{w - v} \xRightarrow{\text{Лемма о трех хордах}} f$
  – выпуклая

**Ex.**

1.  $a^x$ – строго выпуклая ($a \neq 1$)

    $(a^x)'' = a^x \ln^2{a} > 0$

2.  $\ln{x}$ – строго вогнутый

    $(\ln{x})'' = -\frac{1}{x^2} < 0$

3.  $x^p,\ x > 0$

    $(x^p)'' = p(p - 1)x^{p - 2}$

    При $p \in (0; 1)$ – строго вогнутая, при $p > 1$ и $p < 0$ – строго
    выпуклая

## §6. Классические неравенства

**Неравенство Йенсена**

$f$ – выпуклая на
$\langle a; b \rangle;\ x_1, x_2, \ldots, x_n \in \langle a; b \rangle;\ \lambda_1, \lambda_2, \ldots, \lambda_n > 0$
и $\sum\limits_{i = 1}^n \lambda_i = 1$. Тогда

$f(\sum\limits_{i = 1}^n \lambda_i x_i) \leq \sum\limits_{i = 1}^n \lambda_i f(x_i)$

**Доказательство**

ММИ. База $n = 2$ – определение выпуклости

$n = 2\ f(\lambda_1 x_1 + \lambda_2 x_2) \leq \lambda_1 f(x_1) + \lambda_2 f(x_2)$
и $\lambda_1 + \lambda_2 = 1$

Переход: $n \rightarrow n + 1$

Пусть $\lambda_1 + \ldots + \lambda_n = \lambda$, тогда
$\lambda + \lambda_{n + 1} = 1$

$\lambda_1x_1 + \ldots + \lambda_n x_n = \lambda x$ ($\exists x$)

$f(\lambda_1x_1 + \ldots + \lambda_n x_n + \lambda_{n + 1} x_{n + 1}) = f(\lambda x + \lambda_{n + 1}x_{n + 1}) \leq \lambda f(x) + \lambda_{n + 1} f(x_{n + 1})$

Это
$\lambda f(\frac{\lambda_1}{\lambda}x_1 + \ldots + \frac{\lambda_n}{\lambda} x_n) + \lambda_{n + 1} f(x_{n + 1}) \leq \lambda (\frac{\lambda_1}{\lambda} f(x_1) + \ldots + \frac{\lambda_n}{\lambda}f(x_n)) + \lambda_{n + 1} f(x_{n + 1}) = \sum\limits_{i = 0}^{n + 1} \lambda_i f(x_i)$

**Неравенство о средних (неравенство Коши)**

$x_1, x_2 \ldots x_n \geq 0$, тогда

$\sqrt[n]{x_1 x_2 \ldots x_n} \leq \frac{x_1 + x_2 + \ldots + x_n}{n}$

**Доказательство**

НУО $x_1, x_2 \ldots x_n > 0$

$f(x) = -\ln{x}$ – выпуклая;
$\lambda_1 = \lambda_2 = \ldots = \lambda_n = \frac{1}{n}$

Йенсен:
$-\ln{\frac{x_1}{n} + \ldots + \frac{x_n}{n}} \leq \frac{1}{n}(-\ln{x_1}) + \ldots + \frac{1}{n}(-\ln{x_n})$

$\ln{\frac{x_1 + \ldots + x_n}{n}} \geq \frac{1}{n}(\ln{x_1} + \ldots + \ln{x_n}) = \ln{(x_1x_2\ldots x_n)^\frac{1}{n}}$

Т.к. $\ln{x} \nearrow$, то
$\frac{x_1 + \ldots + x_n}{n} \geq \sqrt[n]{x_1x_2\ldots x_n}$

**Неравенство между средними степенными**

$x_1, x_2 \ldots x_n > 0;\ p \in R$

$(\frac{x_1^p + \ldots + x_n^p}{n})^\frac{1}{p}$ – среднее степенное

Например, при $p = 2$ – среднее квадратическое

А при $p = -1$ – среднее гармоническое

$x_1, x_2 \ldots x_n > 0;\ p < q$

$(\frac{x_1^p + \ldots + x_n^p}{n})^\frac{1}{p} \leq (\frac{x_1^q + \ldots + x_n^q}{n})^\frac{1}{q}$

**Доказательство**

1.  $p = 1 < q;\ f(x) = x^q$ – выпуклая

    $\lambda_1 = \lambda_2 = \ldots = \lambda_n = \frac{1}{n}$

    $\xRightarrow{\text{Йенсен}} f(\lambda_1x_1 + \ldots + \lambda_nx_n) = (\frac{x_1 + x_2 + \ldots + x_n}{n})^q \leq \frac{x_1^q + x_2^q + \ldots + x_n^q}{n} = \lambda_1 f(x_1) + \ldots + \lambda_n f(x_n)$.
    Возведем в степень $\frac{1}{q}$

    $\frac{x_1 + x_2 + \ldots + x_n}{n} \leq (\frac{x_1^q + x_2^q + \ldots + x_n^q}{n})^\frac{1}{q}$

2.  $0 < p < q;\ y_k = x_k^p;\ r = \frac{q}{p} > 1$

    $\xRightarrow{1)} \frac{x_1^p + \ldots + x_n^p}{n} = \frac{y_1 + \ldots + y_n}{n} \leq (\frac{y_1^r + \ldots + y_n^r}{n})^\frac{1}{r} = (\frac{x_1^q + \ldots + x_n^q}{n})^\frac{p}{q}$
    Возводим в степень $\frac{1}{p}$

3.  $p < q < 0;\ y_k = x_k^q;\ r = \frac{p}{q} > 1$

    $\xRightarrow{1)} \frac{x_1^q + \ldots + x_n^q}{n} = \frac{y_1 + \ldots + y_n}{n} \leq (\frac{y_1^r + \ldots + y_n^r}{n})^\frac{1}{r} = (\frac{x_1^p + \ldots + x_n^p}{n})^\frac{q}{p}$
    Возводим в степень $\frac{1}{q} < 0$ (поменяли знак)

    $(\frac{x_1^q + \ldots + x_n^q}{n})^\frac{1}{q} \geq (\frac{x_1^p + \ldots + x_n^p}{n})^\frac{1}{p}$

4.  $p < 0 < q$

    $\frac{x_1^q + \ldots + x_n^q}{n} \geq \sqrt[n]{x_1^q\ldots x_n^q} \xRightarrow[\text{в степень} \frac{1}{q}]{q > 0} (\frac{x_1^q + \ldots + x_n^q}{n})^\frac{1}{q} \geq \sqrt[n]{x_1\ldots x_n}$

    $\frac{x_1^p + \ldots + x_n^p}{n} \geq \sqrt[n]{x_1^p\ldots x_n^p} \xRightarrow[\text{в степень} \frac{1}{p}]{p < 0} (\frac{x_1^p + \ldots + x_n^p}{n})^\frac{1}{p} \leq \sqrt[n]{x_1\ldots x_n}$

**Неравенство Гёльдера**

$a_k, b_k \geq 0;\ \frac{1}{p} + \frac{1}{q} = 1;\ p, q > 1$. Тогда

$(*)\ \sum\limits_{k = 1}^n a_k b_k \leq (\sum\limits_{k = 1}^n a_k^p)^\frac{1}{p} \cdot (\sum\limits_{k = 1}^n b_k^q)^\frac{1}{q}$

**Доказательство**

Пусть $B = (\sum\limits_{k = 1}^n b_k^q)^\frac{1}{q} > 0$

$(*) \Leftrightarrow \sum\limits_{k = 1}^n a_k \frac{b_k}{B} \leq (\sum\limits_{k = 1}^n a_k^p)^\frac{1}{p} \Leftrightarrow (\sum\limits_{k = 1}^n a_k \frac{b_k}{B})^p \leq \sum\limits_{k = 1}^n a_k^p$

$f(x) = x^p$ – выпуклая
$\Rightarrow \sum\limits_{k = 1}^n (\lambda_k x_k)^p \leq \sum\limits_{k = 1}^n \lambda_k x_k^p$

Хотим: $\begin{cases}
    \lambda_k x_k = a_k \frac{b_k}{B} \\
    \lambda_k x_k^p = a_k^p
\end{cases}$
$x_k^{p - 1} = \frac{a_k^{p - 1} B}{b_k};\ x_k = \frac{a_k B^\frac{1}{p - 1}}{b_k^\frac{1}{p - 1}}$

$\frac{1}{p} + \frac{1}{q} = 1 \Rightarrow \frac{1}{q} = 1 - \frac{1}{p} = \frac{p - 1}{p} \Rightarrow q = \frac{p}{p - 1}$.
Тогда
$x_k^p = \frac{a_k^p B^\frac{p}{p - 1}}{b_k^\frac{p}{p - 1}} = \frac{a_k^p B^q}{b_k^q} \Rightarrow \lambda_k = \frac{b_k^q}{B^q}$

$\sum\limits_{k = 1}^n \lambda_k = \sum\limits_{k = 1}^n \frac{b_k^q}{B^q} = 1$
по заданию

**Неравенство Коши-Буняковского**

$(x_1^2 + \ldots + x_n^2)(y_1^2 + \ldots + y_n^2) \geq (x_1y_1 + \ldots + x_ny_n)^2$

**Доказательство**

$\begin{cases}
    p = q = 2\ (\frac{1}{2} + \frac{1}{2} = 1) \\
    a_k = |x_k|;\ b_k = |y_k|
\end{cases}$ Гёльдер

$(\sum\limits_{k = 1}^n |x_k|^2)^\frac{1}{2} \cdot (\sum\limits_{k = 1}^n |y_k|^2)^\frac{1}{2} \geq \sum\limits_{k = 1}^n |x_k||y_k| \geq \sum\limits_{k = 1}^n x_ky_k$ +
возведем в квадрат

**Неравенство Минковского**

$p \geq 1;\ a_k, b_k \geq 0$. Тогда

$(\sum\limits_{k = 1}^n (a_k + b_k)^p)^\frac{1}{p} \leq (\sum\limits_{k = 1}^n a_k^p)^\frac{1}{p} + (\sum\limits_{k = 1}^n b_k^p)^\frac{1}{p}$

**Доказательство**

$p > 1 \Rightarrow \exists q > 1 : \frac{1}{p} + \frac{1}{q} = 1 \Leftrightarrow p + q = pq$

$\sum\limits_{k = 1}^n (a_k + b_k)^p = \sum\limits_{k = 1}^n a_k \cdot (a_k + b_k)^{p - 1} + \sum\limits_{k = 1}^n b_k \cdot (a_k + b_k)^{p - 1}$

$\sum\limits_{k = 1}^n a_k \cdot (a_k + b_k)^{p - 1} \stackrel{\text{Гёльдер}}{\leq} (\sum\limits_{k = 1}^n a_k^p)^\frac{1}{p} \cdot (\sum\limits_{k = 1}^n ((a_k + b_k)^{p - 1})^q)^\frac{1}{q} = (*)$

$(p - 1)q = pq - q = p$

$(*) = (\sum\limits_{k = 1}^n a_k^p)^\frac{1}{p} \cdot (\sum\limits_{k = 1}^n (a_k + b_k)^p)^\frac{1}{q}$

$\sum\limits_{k = 1}^n (a_k + b_k)^p \leq (\sum\limits_{k = 1}^n a_k^p)^{\frac{1}{p}} \cdot (\sum\limits_{k = 1}^n (a_k + b_k)^p)^{\frac{1}{q}} + (\sum\limits_{k = 1}^n b_k^p)^{\frac{1}{p}} \cdot (\sum\limits_{k = 1}^n (a_k + b_k)^p)^{\frac{1}{q}} = (\sum\limits_{k = 1}^n (a_k + b_k)^p)^{\frac{1}{q}} \cdot ((\sum\limits_{k = 1}^n a_k^p)^{\frac{1}{p}} + (\sum\limits_{k = 1}^n b_k^p)^{\frac{1}{p}})$

$(\sum\limits_{k = 1}^n (a_k + b_k)^p)^\frac{1}{p} = (\sum\limits_{k = 1}^n (a_k + b_k)^p)^{1 - \frac{1}{q}} \leq (\sum\limits_{k = 1}^n a_k^p)^{\frac{1}{p}} + (\sum\limits_{k = 1}^n b_k^p)^{\frac{1}{p}}$
