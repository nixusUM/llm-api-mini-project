# Оптимизация баз данных

## Введение

Оптимизация баз данных — это комплекс мер по повышению производительности, масштабируемости и надежности хранилищ данных.

## Индексы

### Типы индексов

#### B-Tree Index

Стандартный индекс для большинства СУБД:
- Сбалансированное дерево
- O(log n) поиск
- Хорош для равенства и диапазонов

```sql
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_orders_date ON orders(created_at);
```

#### Hash Index

Хеш-таблица для точного поиска:
- O(1) поиск
- Только равенство (=)
- Не поддерживает диапазоны

```sql
CREATE INDEX idx_products_code ON products USING HASH (product_code);
```

#### GiST и GIN (PostgreSQL)

Для сложных типов данных:
- GiST: обобщенное дерево поиска
- GIN: обобщенный инвертированный индекс
- Full-text search, JSONB, arrays

```sql
-- GIN для полнотекстового поиска
CREATE INDEX idx_articles_content ON articles USING GIN (to_tsvector('english', content));

-- GIN для JSONB
CREATE INDEX idx_events_data ON events USING GIN (data);
```

#### Partial Index

Индекс по подмножеству строк:

```sql
CREATE INDEX idx_active_users ON users(email) WHERE is_active = true;
```

Преимущества:
- Меньший размер
- Быстрее обновление
- Специфичность

#### Covering Index

Индекс, содержащий все нужные колонки:

```sql
CREATE INDEX idx_orders_cover ON orders(user_id, status, created_at) 
INCLUDE (total_amount, shipping_address);
```

Позволяет Index-Only Scan.

### Стратегии индексирования

#### Index Selectivity

```sql
-- Высокая селективность (хорошо)
SELECT * FROM users WHERE email = 'test@example.com';

-- Низкая селективность (плохо для индекса)
SELECT * FROM users WHERE country = 'US';  -- 50% строк
```

#### Composite Index Column Order

```sql
-- Правильный порядок: равенства, затем диапазоны
CREATE INDEX idx_optimal ON table(a, b, c) WHERE a = 'val' AND b > 100 AND c < 50;

-- Плохой порядок: диапазон первый ограничивает использование остальных
CREATE INDEX idx_suboptimal ON table(c, b, a);
```

## Оптимизация запросов

### EXPLAIN ANALYZE

```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at > '2024-01-01'
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 5;
```

### Common Table Expressions (CTE)

```sql
-- Recursive CTE для иерархических данных
WITH RECURSIVE category_tree AS (
    SELECT id, name, parent_id, 0 as level
    FROM categories
    WHERE parent_id IS NULL
    
    UNION ALL
    
    SELECT c.id, c.name, c.parent_id, ct.level + 1
    FROM categories c
    JOIN category_tree ct ON c.parent_id = ct.id
)
SELECT * FROM category_tree;
```

### Window Functions

```sql
-- Ранжирование без self-join
SELECT 
    employee_id,
    department,
    salary,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) as dept_rank,
    salary - LAG(salary) OVER (ORDER BY salary) as diff_from_prev
FROM employees;
```

## Нормализация vs Денормализация

### Нормализация (3NF)

Преимущества:
- Минимальная избыточность
- Целостность данных
- Эффективные обновления

```
users(id, name, email)
orders(id, user_id, total, created_at)
order_items(id, order_id, product_id, quantity, price)
products(id, name, category_id, price)
categories(id, name)
```

### Денормализация

Применение:
- Read-heavy workloads
- OLAP системы
- Кеширование агрегаций

```sql
-- Materialized View для отчетов
CREATE MATERIALIZED VIEW daily_sales AS
SELECT 
    DATE(created_at) as date,
    SUM(total) as revenue,
    COUNT(*) as order_count
FROM orders
GROUP BY DATE(created_at);

-- Обновление
REFRESH MATERIALIZED VIEW CONCURRENTLY daily_sales;
```

## Партиционирование

### Типы партиционирования

#### Range Partitioning

```sql
-- PostgreSQL
CREATE TABLE events (
    id bigint,
    event_type varchar(50),
    created_at timestamp,
    data jsonb
) PARTITION BY RANGE (created_at);

CREATE TABLE events_2024_q1 PARTITION OF events
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE events_2024_q2 PARTITION OF events
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');
```

#### List Partitioning

```sql
CREATE TABLE sales (
    id bigint,
    region varchar(20),
    amount decimal
) PARTITION BY LIST (region);

CREATE TABLE sales_north PARTITION OF sales FOR VALUES IN ('NYC', 'BOS', 'CHI');
CREATE TABLE sales_south PARTITION OF sales FOR VALUES IN ('ATL', 'MIA', 'HOU');
```

#### Hash Partitioning

```sql
CREATE TABLE transactions (
    id bigint,
    user_id bigint,
    amount decimal
) PARTITION BY HASH (user_id);

CREATE TABLE transactions_p0 PARTITION OF transactions FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE transactions_p1 PARTITION OF transactions FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE transactions_p2 PARTITION OF transactions FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE transactions_p3 PARTITION OF transactions FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

## Репликация

### Master-Slave (Primary-Replica)

```
┌─────────┐      Replication      ┌─────────┐
│ Master  │──────────────────────►│  Slave  │
│ (Write) │  (async/sync/hybrid)  │ (Read)  │
└─────────┘                       └─────────┘
```

Режимы:
- **Asynchronous**: минимальная latency, риск потери данных
- **Synchronous**: гарантия консистентности, latency
- **Semi-synchronous**: компромисс

### Multi-Master

```
┌─────────┐◄────────────────────►┌─────────┐
│ Master  │      Bi-directional │ Master  │
│   A     │      Replication    │   B     │
└─────────┘◄────────────────────►└─────────┘
```

Проблемы:
- Конфликты записи
- Разрешение конфликтов (Last-Write-Wins, CRDTs)
- Split-brain

## Кеширование на уровне БД

### PostgreSQL

```sql
-- Shared buffers (в памяти)
SHOW shared_buffers;

-- Effective cache size (для планировщика)
SHOW effective_cache_size;

-- Work mem (для операций)
SHOW work_mem;
```

### Query Cache

```sql
-- MySQL Query Cache (deprecated в 8.0)
SET GLOBAL query_cache_type = ON;
SET GLOBAL query_cache_size = 268435456;
```

## Мониторинг

### Key Metrics

| Метрика | Что показывает | Целевое значение |
|---------|---------------|------------------|
| QPS | Queries per second | Зависит от workload |
| Latency p99 | Время отклика | < 100ms |
| Connection usage | Использование пула | < 80% |
| Cache hit ratio | Эффективность кеша | > 95% |
| Lock wait time | Конкуренция | < 10ms |
| Replication lag | Отставание реплик | < 1s |

### pg_stat_statements

```sql
-- Установка
CREATE EXTENSION pg_stat_statements;

-- Топ медленных запросов
SELECT 
    query,
    calls,
    mean_exec_time,
    total_exec_time,
    rows
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

## Транзакции и изоляция

### Уровни изоляции

| Уровень | Dirty Read | Non-Repeatable | Phantom |
|---------|------------|----------------|---------|
| READ UNCOMMITTED | Да | Да | Да |
| READ COMMITTED | Нет | Да | Да |
| REPEATABLE READ | Нет | Нет | Да |
| SERIALIZABLE | Нет | Нет | Нет |

```sql
-- Optimistic locking
UPDATE accounts 
SET balance = balance - 100, version = version + 1
WHERE id = 1 AND version = 5;

-- Проверка affected rows
```

## Оптимизация схемы

### Правильные типы данных

```sql
-- UUID vs BIGINT
-- UUID: distributed generation, непредсказуемость
-- BIGINT: компактность, сортировка, читаемость

-- JSONB vs нормализация
-- JSONB: гибкость, индексы GIN
-- Нормализация: целостность, эффективные joins

-- TEXT vs VARCHAR
-- В PostgreSQL нет разницы внутри
-- VARCHAR(n) для семантического ограничения
```

### Foreign Keys

```sql
-- Создание FK
ALTER TABLE orders
ADD CONSTRAINT fk_orders_user
FOREIGN KEY (user_id) REFERENCES users(id)
ON DELETE RESTRICT  -- или CASCADE, SET NULL
ON UPDATE CASCADE;
```

## Заключение

Оптимизация БД — это итеративный процесс:
1. Измерить текущую производительность
2. Найти узкие места
3. Внедрить оптимизацию
4. Проверить улучшения
5. Документировать изменения

Начинайте с индексов и оптимизации запросов, затем переходите к архитектурным изменениям (шардирование, партиционирование).
