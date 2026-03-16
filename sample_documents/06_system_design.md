# System Design: Проектирование распределённых систем

## Введение

System design — это процесс проектирования архитектуры программных систем, способных масштабироваться и выдерживать высокие нагрузки.

## Основные концепции

### Масштабирование

#### Вертикальное масштабирование (Scale Up)

Увеличение мощности одного сервера:
- Больше CPU ядер
- Больше RAM
- Быстрые диски (SSD, NVMe)

Ограничения:
- Физический предел железа
- Единая точка отказа
- Высокая стоимость

#### Горизонтальное масштабирование (Scale Out)

Добавление серверов:
- Больше инстансов
- Распределение нагрузки
- Отказоустойчивость

Преимущества:
- Практически неограниченный рост
- Эластичность (elastic scaling)
- Географическое распределение

### CAP Theorem

В распределённой системе можно гарантировать только два из трёх:

- **Consistency**: все узлы видят одинаковые данные
- **Availability**: система отвечает на каждый запрос
- **Partition Tolerance**: система работает при сетевых разделениях

Типы систем:
- **CP**: Consistency + Partition Tolerance (MongoDB, HBase)
- **AP**: Availability + Partition Tolerance (Cassandra, DynamoDB)
- **CA**: Consistency + Availability (традиционные реляционные БД)

### PACELC Theorem

Расширение CAP с учётом latency:
- Если есть Partition: выбор между Availability и Consistency
- Иначе: выбор между Latency и Consistency

## Компоненты системы

### Load Balancer

Распределение трафика:

```
                    ┌─────────────┐
                    │             │
    ┌───────────────┤  Load       ├───────────────┐
    │               │  Balancer   │               │
    │               │             │               │
    ▼               └─────────────┘               ▼
┌─────────┐                               ┌─────────┐
│ Server  │◄─────────────────────────────►│ Server  │
│   A     │         Health Checks         │   B     │
└─────────┘                               └─────────┘
```

Алгоритмы балансировки:
- **Round Robin**: по кругу
- **Least Connections**: минимум активных соединений
- **IP Hash**: хеш IP для sticky sessions
- **Weighted**: с учётом весов серверов

### CDN (Content Delivery Network)

Распределение контента географически:

```
┌─────────────────────────────────────────┐
│           Origin Server                 │
│         (Main Data Center)              │
└─────────────────┬───────────────────────┘
                  │ Push / Pull
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌────────┐   ┌────────┐   ┌────────┐
│  Edge  │   │  Edge  │   │  Edge  │
│  NYC   │   │ London │   │ Tokyo  │
└────────┘   └────────┘   └────────┘
```

Типы кеширования:
- **Static content**: CSS, JS, изображения
- **Dynamic content**: API responses
- **Streaming**: видео, audio

### Базы данных

#### Master-Slave Replication

```
┌─────────┐      Write        ┌─────────┐
│  Master │◄────────────────── │ Clients │
│  (Write)│                   └─────────┘
└────┬────┘
     │ Replication
     │
     ▼
┌─────────┐     Read
│  Slave  │◄─────────────────────┐
│ (Read)  │                      │
└─────────┘                      │
┌─────────┐                      │
│  Slave  │◄─────────────────────┤
│ (Read)  │                      │
└─────────┘                      │
                                 │
                           ┌─────────┐
                           │ Clients │
                           └─────────┘
```

#### Sharding

Горизонтальное разделение данных:

```
┌─────────────────────────────────────┐
│           Query Router              │
└────────────────┬────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌────────┐  ┌────────┐  ┌────────┐
│Shard A │  │Shard B │  │Shard C │
│User ID │  │User ID │  │User ID │
│1-1000  │  │1001-200│  │2001-   │
└────────┘  │0       │  │3000    │
            └────────┘  └────────┘
```

Стратегии шардирования:
- **Hash-based**: hash(key) % N
- **Range-based**: диапазоны значений
- **Directory-based**: lookup table

### Кеширование

#### Уровни кеширования

```
┌─────────────────────────────────────────┐
│  L1: Browser Cache                       │
│  - LocalStorage, IndexedDB               │
├─────────────────────────────────────────┤
│  L2: CDN Edge                            │
│  - Static assets, API responses          │
├─────────────────────────────────────────┤
│  L3: Application Cache                   │
│  - Redis, Memcached                      │
├─────────────────────────────────────────┤
│  L4: Database Cache                      │
│  - Query cache, Buffer pool              │
└─────────────────────────────────────────┘
```

#### Стратегии инвалидации

- **TTL (Time To Live)**: время жизни записи
- **LRU (Least Recently Used)**: вытеснение редко используемых
- **Write-through**: запись и в кеш, и в БД
- **Write-behind**: асинхронная запись в БД
- **Cache-aside**: явное управление кешем

### Message Queue

Асинхронная коммуникация:

```
┌─────────┐     Publish      ┌─────────┐
│ Service │──────────────►   │  Queue  │
│    A    │                  │         │
└─────────┘                  └────┬────┘
                                  │
                         ┌────────┴────────┐
                         │                 │
                         ▼                 ▼
                   ┌─────────┐       ┌─────────┐
                   │Consumer │       │Consumer │
                   │    1    │       │    2    │
                   └─────────┘       └─────────┘
```

Паттерны:
- **Point-to-Point**: один consumer
- **Publish-Subscribe**: множество subscribers
- **Request-Reply**: синхронный через асинхронный транспорт
- **Dead Letter Queue**: обработка ошибок

## Паттерны проектирования

### Microservices

```
┌─────────────────────────────────────────┐
│              API Gateway                │
│  - Auth, Rate Limiting, Routing         │
└─────────────────────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
┌────────┐   ┌────────┐   ┌────────┐
│Users   │   │Orders  │   │Products│
│Service │   │Service │   │Service │
└────┬───┘   └────┬───┘   └───┬────┘
     │            │           │
     └────────────┼───────────┘
                  │
         ┌────────┴────────┐
         │                 │
    ┌────┴────┐       ┌────┴────┐
    │User DB  │       │Order DB │
    └─────────┘       └─────────┘
```

Принципы:
- Single Responsibility
- Database per Service
- Event-Driven Communication
- Circuit Breaker
- Bulkhead Pattern

### Event Sourcing

Хранение событий как источника истины:

```
┌─────────────────────────────────────────┐
│           Event Store                   │
│                                         │
│  UserCreated {id: 1, name: "John"}     │
│  UserNameChanged {id: 1, to: "Jane"}   │
│  OrderPlaced {id: 101, user: 1}        │
│  PaymentReceived {order: 101, amt: 100} │
│                                         │
└─────────────────────────────────────────┘
                   │
                   │ Projections
                   ▼
    ┌─────────────────────────────┐
    │   Read Models               │
    │   - Current User State      │
    │   - Order Summary           │
    │   - Analytics Views         │
    └─────────────────────────────┘
```

### CQRS (Command Query Responsibility Segregation)

Разделение чтения и записи:

```
         ┌─────────────────┐
         │                 │
    ┌────┴────┐       ┌────┴────┐
    │Commands │       │ Queries │
    │         │       │         │
    └────┬────┘       └────┬────┘
         │                 │
         ▼                 ▼
    ┌─────────┐       ┌─────────┐
    │  Write  │       │  Read   │
    │  Model  │       │  Model  │
    │(Complex)│       │(Denorm.)│
    └────┬────┘       └────┬────┘
         │                 │
         ▼                 ▼
    ┌─────────┐       ┌─────────┐
    │ Write   │       │ Read    │
    │   DB    │       │   DB    │
    │         │       │         │
    │SQL,     │       │NoSQL,   │
    │Graph    │       │Cache    │
    └─────────┘       └─────────┘
```

## Проектирование под нагрузку

### Rate Limiting

```python
# Token Bucket Algorithm
import time

class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
    
    def allow_request(self, tokens=1):
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        now = time.time()
        delta = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + delta * self.refill_rate
        )
        self.last_refill = now
```

### Circuit Breaker

```
┌─────────┐     Request      ┌─────────┐
│ Client  │─────────────────►│ Service │
│         │                  │         │
│         │◄─────────────────│         │
└────┬────┘    Success       └─────────┘
     │
     │  ┌─────────────────┐
     │  │  Circuit        │
     └──┤  Breaker        │
        │                 │
        │  CLOSED (ok)    │
        │  OPEN (failing) │
        │  HALF-OPEN      │
        └─────────────────┘
```

Состояния:
- **Closed**: нормальная работа
- **Open**: превышен лимит ошибок, запросы блокируются
- **Half-Open**: проверка восстановления

## Примеры систем

### URL Shortener

```
┌─────────┐   Short URL    ┌─────────┐   Hash/ID    ┌─────────┐
│ Client  │───────────────►│  API    │─────────────►│   DB    │
│         │                │ Server  │              │         │
│         │◄───────────────│         │◄────────────│         │
└─────────┘   Long URL     └─────────┘  Redirect    └─────────┘
```

Требования:
- 100M URLs/день
- 10B чтений/день
- 100 bytes/URL
- 99.99% availability

### News Feed (Twitter)

```
Pull Model:
┌─────────┐     Request     ┌─────────┐
│  User   │────────────────►│  API    │
│         │                 │ Server  │
│         │◄────────────────│         │
└─────────┘   Aggregated    └────┬────┘
              Feed                │
                                  │ Query
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
              ┌─────────┐   ┌─────────┐   ┌─────────┐
              │Follower │   │Following│   │  Tweets  │
              │  List   │   │  List   │   │  Store   │
              └─────────┘   └─────────┘   └─────────┘

Push Model (Fan-out):
┌─────────┐  New Tweet    ┌─────────┐
│  User   │──────────────►│  Queue  │
│ Writes  │               │         │
└─────────┘               └────┬────┘
                               │
                    ┌──────────┴──────────┐
                    │   Fan-out Service   │
                    │                     │
                    │ Write to followers' │
                    │      timelines      │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │       Redis           │
                    │  (Timeline Cache)     │
                    └───────────────────────┘
```

## Заключение

System design требует:
- Понимания требований (functional + non-functional)
- Знания паттернов и компромиссов
- Опыта с масштабируемыми системами
- Умение делать обоснованные trade-offs

Практикуйтесь на mock interviews, изучайте реальные архитектуры крупных компаний, экспериментируйте с облачными сервисами.
