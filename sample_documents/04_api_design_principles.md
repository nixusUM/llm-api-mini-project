# Принципы проектирования API

## Введение

API (Application Programming Interface) — это контракт между системами. Хорошо спроектированный API прост в использовании, надежен и масштабируем.

## REST API

### Принципы REST

1. **Client-Server**: разделение ответственности
2. **Stateless**: каждый запрос содержит всю информацию
3. **Cacheable**: ответы должны определять кешируемость
4. **Uniform Interface**: единый интерфейс взаимодействия
5. **Layered System**: иерархия слоёв
6. **Code on Demand** (опционально): передача кода

### HTTP методы

- **GET**: чтение ресурса (идемпотентный)
- **POST**: создание ресурса
- **PUT**: полное обновление (идемпотентный)
- **PATCH**: частичное обновление
- **DELETE**: удаление (идемпотентный)
- **HEAD**: метаданные ресурса
- **OPTIONS**: доступные методы

### Коды статуса

#### 2xx Success
- `200 OK` — успешный запрос
- `201 Created` — ресурс создан
- `204 No Content` — успех без тела ответа

#### 3xx Redirection
- `301 Moved Permanently` — постоянное перенаправление
- `304 Not Modified` — ресурс не изменился (кеш)

#### 4xx Client Error
- `400 Bad Request` — неверный запрос
- `401 Unauthorized` — требуется аутентификация
- `403 Forbidden` — доступ запрещён
- `404 Not Found` — ресурс не найден
- `409 Conflict` — конфликт состояния
- `422 Unprocessable Entity` — семантическая ошибка
- `429 Too Many Requests` — превышен rate limit

#### 5xx Server Error
- `500 Internal Server Error`
- `502 Bad Gateway`
- `503 Service Unavailable`
- `504 Gateway Timeout`

### Структура URL

```
https://api.example.com/v1/users/{user_id}/orders/{order_id}
```

Принципы:
- Существительные, не глаголы
- Множественное число
- Иерархия через `/`
- Версионирование через префикс

## GraphQL

### Особенности

- Клиент определяет структуру ответа
- Единый endpoint
- Типизированная схема
- Introspection API

### Пример запроса

```graphql
query GetUserWithOrders($userId: ID!, $limit: Int = 10) {
  user(id: $userId) {
    id
    name
    email
    orders(first: $limit) {
      edges {
        node {
          id
          total
          status
        }
      }
    }
  }
}
```

### Мутации

```graphql
mutation CreateOrder($input: CreateOrderInput!) {
  createOrder(input: $input) {
    order {
      id
      status
    }
    errors {
      field
      message
    }
  }
}
```

### Подписки

```graphql
subscription OnOrderStatusChanged($orderId: ID!) {
  orderStatusChanged(orderId: $orderId) {
    status
    updatedAt
  }
}
```

## gRPC

### Преимущества

- Бинарный формат (Protocol Buffers)
- HTTP/2 с multiplexing
- Строгая типизация
- Поддержка стриминга
- Кросс-платформенность

### Определение сервиса

```protobuf
syntax = "proto3";

package orders;

service OrderService {
  rpc GetOrder(GetOrderRequest) returns (Order);
  rpc ListOrders(ListOrdersRequest) returns (ListOrdersResponse);
  rpc CreateOrder(CreateOrderRequest) returns (Order);
  rpc StreamOrderUpdates(StreamRequest) returns (stream OrderUpdate);
}

message Order {
  string id = 1;
  string customer_id = 2;
  repeated OrderItem items = 3;
  double total = 4;
  OrderStatus status = 5;
}

enum OrderStatus {
  PENDING = 0;
  CONFIRMED = 1;
  SHIPPED = 2;
  DELIVERED = 3;
  CANCELLED = 4;
}
```

## Аутентификация и авторизация

### JWT (JSON Web Tokens)

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

Структура:
- **Header**: алгоритм и тип
- **Payload**: данные (claims)
- **Signature**: подпись

### OAuth 2.0

Потоки:
1. **Authorization Code** — для серверных приложений
2. **PKCE** — для мобильных/SPA
3. **Client Credentials** — machine-to-machine
4. **Device Code** — для устройств с ограниченным вводом

### API Keys

```
X-API-Key: your_api_key_here
```

Использование:
- Регистрация приложений
- Простая аутентификация
- Rate limiting per key

## Rate Limiting

### Стратегии

#### Fixed Window

```
Limit: 100 requests per hour
Window: [10:00, 11:00)
```

Проблема: пик в конце и начале окна.

#### Sliding Window

Учитывает окно, скользящее во времени:

```
Current: 14:23
Window: [13:23, 14:23)
```

#### Token Bucket

- Bucket с фиксированной ёмкостью
- Пополнение токенов со скоростью r
- Запрос потребляет 1 токен

### Заголовки

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
X-RateLimit-Retry-After: 3600
```

## Версионирование

### Стратегии

#### URL Path

```
/api/v1/users
/api/v2/users
```

Плюсы: явность, простота кеширования
Минусы: дублирование endpoints

#### Header

```http
Accept: application/vnd.api+json;version=2
API-Version: 2
```

Плюсы: чистые URL
Минусы: сложнее отладка

#### Content Negotiation

```http
Accept: application/vnd.company.v2+json
```

## Пагинация

### Offset-based

```
GET /api/users?offset=20&limit=10
```

Проблемы:
- Непроизводительно при больших offset
- Дублирование при изменениях
- Пропуск данных

### Cursor-based

```
GET /api/users?cursor=eyJpZCI6MTAwfQ==&limit=10
```

Преимущества:
- Консистентность
- Производительность
- Подходит для реального времени

### Пример ответа

```json
{
  "data": [...],
  "pagination": {
    "total": 1000,
    "per_page": 10,
    "current_page": 2,
    "total_pages": 100,
    "next_cursor": "...",
    "prev_cursor": "...",
    "has_more": true
  }
}
```

## Документирование

### OpenAPI (Swagger)

```yaml
openapi: 3.0.0
info:
  title: Orders API
  version: 1.0.0
paths:
  /orders:
    get:
      summary: List orders
      parameters:
        - name: status
          in: query
          schema:
            type: string
            enum: [pending, confirmed, shipped]
      responses:
        '200':
          description: List of orders
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Order'
```

### Инструменты генерации

- Swagger UI: интерактивная документация
- ReDoc: альтернативный рендерер
- OpenAPI Generator: генерация клиентов
- Postman: коллекции для тестирования

## Обработка ошибок

### Структура ошибки

```json
{
  "error": {
    "code": "INVALID_PAYMENT_METHOD",
    "message": "The provided payment method is not valid",
    "details": [
      {
        "field": "card_number",
        "issue": "Invalid checksum"
      }
    ],
    "request_id": "req_123456",
    "documentation_url": "https://api.example.com/docs/errors/INVALID_PAYMENT_METHOD"
  }
}
```

### Логирование

- Корреляционный ID для трассировки
- Структурированные логи
- PII scrubbing
- Различные уровни детализации

## Тестирование API

### Unit тесты

```python
def test_create_order_success(client):
    response = client.post('/api/v1/orders', json={
        'items': [{'product_id': 1, 'quantity': 2}],
        'shipping_address': {...}
    })
    assert response.status_code == 201
    assert 'id' in response.json()
```

### Contract тесты

- Pact: consumer-driven contracts
- Spring Cloud Contract
- Schemathesis

### Нагрузочное тестирование

Инструменты:
- k6: JavaScript-скрипты
- Locust: Python, распределенное
- Artillery: YAML-конфигурации
- JMeter: GUI для сложных сценариев

## Заключение

Хороший API — это результат внимания к деталям на всех этапах: от проектирования и документирования до мониторинга и эволюции. Выбирайте подходящий стиль в зависимости от требований: REST для универсальности, GraphQL для гибкости, gRPC для производительности.
