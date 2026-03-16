# Основы машинного обучения

## Что такое машинное обучение

Машинное обучение (ML) — это область искусственного интеллекта, где системы обучаются на данных без явного программирования. Вместо написания правил, мы предоставляем данные, и алгоритм находит закономерности.

## Типы машинного обучения

### Обучение с учителем (Supervised Learning)

Алгоритм обучается на размеченных данных (парах вход-выход):

- **Классификация**: предсказание категории (спам/не спам, болен/здоров)
- **Регрессия**: предсказание числового значения (цена, температура)

Примеры алгоритмов:
- Линейная регрессия
- Логистическая регрессия
- Random Forest
- Gradient Boosting
- Нейронные сети

### Обучение без учителя (Unsupervised Learning)

Алгоритм находит структуру в немаркированных данных:

- **Кластеризация**: группировка похожих объектов
- **Понижение размерности**: сжатие данных с сохранением структуры
- **Ассоциации**: поиск правил взаимосвязи

Примеры алгоритмов:
- K-means
- DBSCAN
- Иерархическая кластеризация
- PCA (Principal Component Analysis)
- t-SNE

### Обучение с подкреплением (Reinforcement Learning)

Агент обучается через взаимодействие со средой, получая награды или штрафы:

- Q-learning
- Policy Gradient
- Actor-Critic
- PPO (Proximal Policy Optimization)

## Жизненный цикл ML-проекта

### 1. Постановка задачи

- Что предсказываем?
- Какая метрика важна?
- Какие данные доступны?

### 2. Сбор данных

Источники данных:
- Внутренние базы данных
- Публичные датасеты (Kaggle, UCI, Google Dataset Search)
- Веб-скрапинг
- API сторонних сервисов
- Генерация синтетических данных

### 3. Подготовка данных (Data Preparation)

#### Очистка данных

- Обработка пропущенных значений
- Удаление дубликатов
- Коррекция ошибок
- Обработка выбросов

#### Feature Engineering

Создание новых признаков из существующих:

```python
# Временные признаки
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Агрегации
df['avg_purchase_30d'] = df.groupby('customer_id')['amount'].rolling(30).mean()

# Взаимодействия признаков
df['price_per_unit'] = df['total_price'] / df['quantity']
```

#### Кодирование категориальных признаков

- One-hot encoding
- Label encoding
- Target encoding
- Embeddings для высокой кардинальности

#### Масштабирование

- Min-Max Scaling: приводит к диапазону [0, 1]
- Standard Scaling: среднее 0, стандартное отклонение 1
- Robust Scaling: устойчив к выбросам

### 4. Разделение данных

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Для валидации во время обучения
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
```

### 5. Выбор модели

Факторы при выборе:
- Размер данных (малые vs большие)
- Интерпретируемость vs производительность
- Требования к latency
- Ресурсы (CPU vs GPU)
- Регулярность обновления

### 6. Обучение и валидация

#### Кросс-валидация

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
print(f"CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

#### Регуляризация

Предотвращение переобучения:
- L1 (Lasso): создает разреженные модели
- L2 (Ridge): сглаживает веса
- Elastic Net: комбинация L1 и L2
- Dropout (для нейронных сетей)

### 7. Оценка модели

#### Метрики классификации

- **Accuracy**: доля правильных предсказаний
- **Precision**: точность (доля истинных положительных)
- **Recall**: полнота (доля найденных положительных)
- **F1-score**: гармоническое средство precision и recall
- **ROC-AUC**: площадь под ROC-кривой
- **Log Loss**: логарифмическая функция потерь

#### Метрики регрессии

- **MSE** (Mean Squared Error): средний квадрат ошибки
- **RMSE**: квадратный корень из MSE
- **MAE** (Mean Absolute Error): средняя абсолютная ошибка
- **R²**: коэффициент детерминации
- **MAPE**: средняя абсолютная процентная ошибка

### 8. Гиперпараметрическая оптимизация

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

## Глубокое обучение

### Нейронные сети

Архитектура нейрона:
```
output = activation_function(sum(weights * inputs) + bias)
```

### Типы слоев

- **Dense (Fully Connected)**: каждый нейрон связан со всеми
- **Convolutional**: фильтры для извлечения признаков
- **Recurrent**: имеют память (LSTM, GRU)
- **Attention**: механизм внимания
- **Normalization**: нормализация активаций

### Фреймворки

- **PyTorch**: динамический граф, research-friendly
- **TensorFlow/Keras**: production-ready, экосистема
- **JAX**: функциональный, высокая производительность
- **ONNX**: универсальный формат моделей

## MLOps

### CI/CD для ML

- Автоматическое тестирование данных
- Версионирование моделей
- A/B тестирование
- Мониторинг деградации

### Инструменты

- **MLflow**: отслеживание экспериментов
- **DVC**: версионирование данных
- **Kubeflow**: оркестрация ML-воркфлоу
- **Weights & Biases**: визуализация экспериментов
- **Evidently**: мониторинг качества моделей

## Этика и ответственность

### Проблемы

- Смещение данных (bias)
- Приватность
- Прозрачность решений
- Безопасность (adversarial attacks)

### Решения

- Fairness-aware ML
- Explainable AI (XAI)
- Дифференциальная приватность
- Federated Learning

## Заключение

Машинное обучение — это мощный инструмент, требующий системного подхода. Успех проекта зависит от качества данных, правильного выбора модели и тщательной оценки. Начинайте с простых решений и усложняйте по мере необходимости.
