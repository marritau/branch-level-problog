# BranchNet + ProbLog: Архитектура и изменения

## Краткая суть

BranchNet из оригинальной статьи (Rodríguez-Salas & Riess, 2025) переведён в нейро-символическую ProbLog-архитектуру.
Каждая ветвь дерева становится **явным логическим правилом** и **латентной переменной** одновременно.

---

## 1. Архитектура BranchNet (оригинал из статьи)

### Branching: Parent-of-Leaf

Ансамбль `ExtraTreesClassifier` обучается на данных. Из каждого дерева извлекаются **parent-of-leaf** узлы — внутренние узлы, у которых хотя бы один ребёнок является листом. Каждый такой узел становится одним скрытым нейроном.

Это значит: не каждый лист создаёт нейрон, а каждый **родитель листа**. Для дерева с 37 листьями может быть, например, 27 parent-of-leaf узлов — и значит 27 нейронов (а не 37).

### Формулы из статьи

```
n_estimators = n_classes + ⌊log₂(n_features)⌋
max_leaf_nodes = 2^(⌊log₂(n_features)⌋ + 4)
```

### Инициализация весов

| Матрица | Размерность | Источник | Обучается? |
|---------|-------------|----------|------------|
| **m1** | `[hidden, features]` | Бинарная маска: `(w1 ≠ 0)` | ❌ buffer |
| **w1** | `[hidden, features]` | Feature importance × mask | ✅ trainable |
| **w2** | `[classes, hidden]` | Распределение классов из **parent** node | ❌ frozen |

Для **w1**: по каждой ветви берётся путь от корня до parent-of-leaf. Признаки, встречающиеся на пути (включая split самого parent'а), формируют ненулевые элементы строки w1. Значения — feature importance дерева.

Для **w2**: распределение классов берётся из **parent node** (не из leaf), взвешенное долей сэмплов: `factor × tree.value[parent][0]`.

### Forward Pass

```
x → BN₀(x) → Linear(w1 · m1) → BN₁ → Sigmoid → BN₂ → Linear(w2) → logits
```

- **m1** применяется при обучении (маска фиксирует, какие признаки видит каждый нейрон)
- **Sigmoid** — активация (как в статье)
- **w2** заморожен — символический выход из RF сохраняется

---

## 2. Символическое представление: Branch / Condition

Каждая ветвь хранится как объект `Branch` (`branch_schema.py`):

```python
@dataclass
class Branch:
    branch_id: str           # "b0", "b1", ...
    tree_id: int             # номер дерева в ансамбле
    parent_node_id: int      # ID parent-of-leaf узла
    conditions: List[Condition]  # путь от корня ДО parent'а (le/gt)
    class_proportions: List[float]  # распределение классов (для w2)
    split_feature_idx: int   # split-признак самого parent'а (для w1 mask)
    split_threshold: float   # порог split'а parent'а
```

`Condition` описывает одно условие на пути:

```python
@dataclass
class Condition:
    feature_idx: int   # индекс признака
    threshold: float   # порог
    direction: str     # "le" (≤) или "gt" (>)
    node_id: int       # ID узла дерева
```

**Важно**: `conditions` содержат путь **до** parent'а (с направлениями le/gt), а `split_feature_idx` — это split самого parent'а (без направления, т.к. parent не ведёт к конкретному листу). Оба участвуют в формировании маски m1 через метод `feature_indices_for_w1()`.

---

## 3. ProbLog-экспорт: Структурные правила

`export_branches_to_problog()` записывает каждую ветвь как логическое правило:

```prolog
% Пороги
threshold(t0_0, 101.2066098).
threshold(t0_1, 1.796446539).

% Структурные правила ветвей
branch_struct(b0, X) :- le(f4,t0_0,X), le(f5,t0_1,X), le(f1,t0_5,X).
branch_struct(b1, X) :- le(f4,t0_0,X), le(f5,t0_1,X), gt(f1,t0_5,X), gt(f1,t0_11,X).
```

Каждое правило читается: «Ветвь bN активна для объекта X, если все условия на пути от корня до parent-of-leaf выполняются».

---

## 4. ProbLog-экспорт: Латентная модель

`export_branches_to_problog_latent()` реализует полную нейро-символическую модель:

### 4.1 Латентные переменные z(b, X)

Нейронная часть (метод `branch_probs()`) вычисляет для каждого объекта X и каждой ветви b вероятность латентной активации:

```
P(z(b,X) = true | x) = Sigmoid(BN₁(Linear(BN₀(x), w1 · m1)))
```

Это записывается в ProbLog:

```prolog
0.95756269::z(b33, 0).    % P(z(b33, id_0) = true) = 0.957
0.61640000::z(b42, 0).    % P(z(b42, id_0) = true) = 0.616
```

### 4.2 Manifestation Rules: условия как «симптомы» z

Условия le/gt — не независимые случайные факты, а **проявления** латентного события z(b,X). Одна скрытая причина управляет всей группой условий ветви:

```prolog
% Если z(b0) активно — условие выполняется с высокой вероятностью
0.95000000::le(b0,f4,t0_0,X) :- z(b0,X).

% Если z(b0) не активно — условие выполняется с низкой вероятностью
0.05000000::le(b0,f4,t0_0,X) :- not_z(b0,X).
```

Это автоматически создаёт **зависимость** между условиями внутри ветви: они коррелированы через общего «родителя» z(b,X).

### 4.3 Evidence: наблюдения из данных

Для каждого объекта условия вычисляются детерминированно (сравнение feature value с threshold) и передаются ProbLog как evidence:

```prolog
evidence(le(b0,f4,t0_0,0)).            % x[f4] ≤ 101.2 → true
evidence(gt(b206,f8,t6_12,4), false).  % x[f8] > threshold → false
```

### 4.4 Вывод ProbLog

ProbLog получает:
- **Априорные вероятности** z(b,X) из нейросети
- **Manifestation model**: условия зависят от z
- **Evidence**: детерминированные наблюдения из данных

И выполняет **апостериорный вывод**: если наблюдения совпадают со структурой ветви, P(z(b,X) | evidence) растёт; если противоречат — падает. Корреляции между условиями учитываются автоматически через общую скрытую причину z.

---

## 5. Полная схема пайплайна

```
                    TRAINING
                    ========
ExtraTreesClassifier(data)
         │
         ▼
    ┌─────────────────┐
    │ bf_search:       │
    │ parent-of-leaf   │──→ Branch[] (conditions, class_proportions)
    │ extraction       │
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐
    │ BranchNet:       │
    │ w1 (trainable)   │
    │ m1 (frozen mask) │
    │ w2 (frozen RF)   │
    └─────────────────┘
         │
         ▼
    fit(x_train, y_train)
         │
         ▼
    branch_probs(x) → P(z(b,X) | x)

                    EXPORT
                    ======
    Branch[] ──→ branch_struct(b, X) :- le(...), gt(...)    [structural]
    P(z)     ──→ pZ::z(b, x_id).                            [latent]
    p_high   ──→ 0.95::le(...) :- z(b,X).                   [manifestation]
    p_low    ──→ 0.05::le(...) :- not_z(b,X).               [manifestation]
    data     ──→ evidence(le(...,x_id)). / evidence(...,false). [observed]

                    INFERENCE
                    =========
    ProbLog: P(z(b,X) | evidence) — апостериорный вывод
```

---

## 6. Последние изменения

### Исправление BranchNet.py: leaf → parent-of-leaf

**Было (до исправления):**
- `bf_search` создавал одну ветвь на каждый **лист** дерева
- w2 брал class distribution из **leaf node** (`tree.value[leaf_node_id]`)
- Sample weight: `n_node_samples[leaf]`
- Количество hidden neurons = количество листьев

**Стало (оригинал из статьи):**
- `bf_search` создаёт одну ветвь на каждый **parent-of-leaf** узел
- w2 берёт class distribution из **parent node** (`tree.value[index]`)
- Sample weight: `n_node_samples[parent]`
- Количество hidden neurons ≤ количество листьев

**Пример (Wine dataset, первое дерево):**
```
Листьев в дереве:           37
Parent-of-leaf ветвей:      27  (← меньше нейронов)
Верификация: 0 missing, 0 extra ✅
```

### Удалённые файлы

- `BranchNetWithDropout.py` — Dropout-вариант не из статьи, не использовался в train.py
- `BranchNetFramworkWithDropout.py` — обёртка над ним

### Обновлённые файлы

| Файл | Изменение |
|------|-----------|
| `BranchNet.py` | `bf_search` переписан на parent-of-leaf; `Branch`/`Condition` сохранены для ProbLog |
| `run_one_export_check.py` | Добавлена `extract_expected_parent_of_leaf_paths()`; верификация теперь по parent-of-leaf |
| `branch_schema.py` | Без изменений — `split_feature_idx` и `feature_indices_for_w1()` уже были |
| `BranchNetFramwork.py` | Без изменений — наследует исправленный BranchNet |
| `problog_export.py` | Без изменений — работает с любыми Branch[] |
| `train.py` | Без изменений — использует только BranchNetModel |

---

## 7. Файлы проекта

```
branch-level-problog/
├── BranchNet.py              # Оригинальная архитектура (parent-of-leaf, Sigmoid, w2 frozen)
├── BranchNetFramwork.py      # fit/predict обёртка + predict_branch_proba()
├── branch_schema.py          # Branch, Condition dataclasses
├── problog_export.py         # Экспорт: structural, latent, evidence
├── train.py                  # Trainer: ExtraTrees → BranchNet → ProbLog export
├── run_one_export_check.py   # Верификация parent-of-leaf extraction
├── verify_branch_export.py   # Spot-check JSON ↔ ProbLog consistency
├── test_branchnet_branch_probs.py       # Тест: P(z) range, shape
├── test_problog_export_latent.py        # Тест: содержимое latent KB
├── test_problog_consistency.py          # Тест: predict до/после = идентично
├── test_problog_end_to_end_consistency.py
├── test_problog_evidence_inference.py
└── test_problog_latent_branch_scoping.py
```

---

## 8. Почему это нейро-символическая архитектура

| Компонент | Нейронный | Символический |
|-----------|-----------|---------------|
| **w1 (input weights)** | ✅ обучается | Инициализирован из RF feature importance |
| **m1 (mask)** | — | ✅ frozen boolean mask из RF |
| **w2 (output weights)** | — | ✅ frozen class distributions из RF parent nodes |
| **Branch conditions** | — | ✅ IF-THEN правила из деревьев (le/gt) |
| **z(b,X)** | ✅ P(z) из Sigmoid | ✅ латентная переменная в ProbLog |
| **Manifestation** | — | ✅ условия как проявления z (p_high/p_low) |
| **Evidence** | — | ✅ детерминированные наблюдения из данных |
| **Inference** | — | ✅ ProbLog апостериорный вывод P(z│evidence) |

Нейронная часть вычисляет вероятности латентных состояний.
Символическая часть содержит структуру правил, модель manifestation и выполняет формальный вывод.
Они связаны через единое пространство ветвей — каждая ветвь одновременно нейрон в сети и правило в ProbLog.
