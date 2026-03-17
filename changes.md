# Изменения для BranchNet + ProbLog (пункты 1-7)

## Архитектурный смысл (пункты задачи)

1. BranchNet даёт структуру: извлекаются ветви из дерева.
2. Ветви переводятся в ProbLog branch_struct правило.
3. Нейронная часть дает P(z(b,X)=true).
4. Условия le/gt зависят от z (p_high/p_low).
5. Надо не сломать логику сети при добавлении ProbLog.
6. Проверки и тесты (проверка образцов, метрик, стабильности).
7. Проверка, что ProbLog совпадает с логикой сети (без нарушения).

---

## Измененные файлы (по пунктам)

### 1. BranchNet → structure
- `branchnet/BranchNet.py`
  - `build_model_from_ensemble` уже был, содержит ветви и условия.
  - добавлен метод `branch_probs`:
    - делает BN0 -> W1 -> BN1 -> sigmoid
    - возвращает `P(z(b,X))` для каждой ветки.

### 2. branches → ProbLog branch_struct
- `branchnet/problog_export.py`
  - функция `export_branches_to_problog` была и оставлена без изменения.

### 3. Нейронная `P(z)`
- `branchnet/BranchNetFramwork.py`
  - добавлен метод `predict_branch_proba`.
- `branchnet/train.py`
  - в `Trainer.train`: после `fit`, сохраняются `branch_probs_data` на `x_train`.

### 4. z -> condition (p_high/p_low)
- `branchnet/problog_export.py`
  - добавлен метод `export_branches_to_problog_latent`:
    - записывает threshold, branch_struct, z-факты для каждого x_id,
    - добавляет правила для `le` и `gt`:
      - `p_high :: le(...) :- z(...)`
      - `p_low  :: le(...) :- \+ z(...)`

### 5. Функциональность не сломана (core)
- `branchnet/train.py`
  - дописано условие вызова `export_branches_to_problog_latent` при наличии branch_probs.

### 6. Тесты
- `branchnet/test_branchnet_branch_probs.py`
  - training + predict_branch_proba + assert диапазон, размер.
- `branchnet/test_problog_export_latent.py`
  - проверка с фиксированными ветвями и pZ/p_high/p_low содержимого файла.

### 7. Соответствие (ProbLog не сломал)
- `branchnet/test_problog_consistency.py`
  - проверка:
    - `predict` до и после экспорта совпадает,
    - `branch_struct` файл создан,
    - `latent` файл содержит z-факты и pZ совпадает с моделью.

---

## Почему это ок
1. **Пункты 1-4** теперь покрыты кодом, добавлены методы и выводы в ProbLog.
2. **Пункт 5** — мы сделали side effect-прописку (экспорт), без изменения предсказаний.
3. **Пункты 6-7** (тесты) гарантируют, что изменения не ломают логику.

---

## Примеры вывода / визуальная проверка

- ``BranchNet`` предсказывает ``y`` как раньше (`model.predict`).
- ``predict_branch_proba`` показывает ``branch_scores`` (0..1) в ``test_branchnet_branch_probs.py``.
- ``export_branches_to_problog_latent`` формирует файл `tmp_problog_latent.pl`, который содержит:
  - branch_struct правила
  - threshold-факты
  - строки вида `0.75000000::z(b0,0).`
  - правила manifestation из z -> le/gt
- В тесте ``test_problog_consistency.py``:
  - сохраняется поведение ``y``, равное до/после экспортов,
  - строится проверка совпадения первого `z` с `P(z)` из сети.

---

## Вывод (что было / что стало)

### До ProbLog
- `Trainer` обучал BranchNet и выдавал `y`.
- структура ветвей сохранялась в Python `Branch`.
- `branch_struct` уже существовал в `problog_export.py`.

### После ProbLog (новое)
- добавлен `branch_probs` и latent `z(b,X)` таблица.
- добавлен ProbLog latent exporter.
- добавлены тесты 6 и 7:
  - `test_branchnet_branch_probs.py` (latent probabilities)
  - `test_problog_export_latent.py` (контент Tupl of rules)
  - `test_problog_consistency.py` (сравнение с до-экспортом)

### Что проверено в test_flow
- будучи без ProbLog сеть выдавала `y` и branch_probs.
- с ProbLog(implicit) (просто экспорт в файл) сеть не изменила свои предсказания.
- содержимое нового file-представления имеет соответствие ветвям и вероятностям.

---

## Результаты (грубо)
- branch_struct, threshold были и остаются.
- данные z(b,X) привязаны к скорости branch_probs из сети.
- тест показывает: `P(z)` в файле = `predict_branch_proba`.
- модели y остаются идентичными (экспорт не поменял инвокацию).
