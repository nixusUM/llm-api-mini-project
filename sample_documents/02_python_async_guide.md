# Руководство по асинхронному Python

## Введение в asyncio

Модуль asyncio — это библиотека для написания конкурентного кода с использованием синтаксиса async/await. Она предоставляет основу для многозадачности через кооперативную многозадачность.

## Основные концепции

### Корутины

Корутина — это функция, объявленная с async def, которая может приостанавливать свое выполнение:

```python
import asyncio

async def greet(name):
    await asyncio.sleep(1)
    return f"Hello, {name}!"

async def main():
    result = await greet("World")
    print(result)

asyncio.run(main())
```

### Event Loop

Event Loop — сердце asyncio. Он управляет выполнением корутин:

```python
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```

### Задачи (Tasks)

Task — обертка для корутины, позволяющая управлять её выполнением:

```python
async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)

async def main():
    task1 = asyncio.create_task(say_after(1, 'hello'))
    task2 = asyncio.create_task(say_after(2, 'world'))
    
    await task1
    await task2
```

## Параллельное выполнение

### gather

asyncio.gather позволяет выполнять несколько корутин параллельно:

```python
async def fetch_data(url):
    await asyncio.sleep(1)
    return f"Data from {url}"

async def main():
    urls = ['url1', 'url2', 'url3']
    results = await asyncio.gather(
        *[fetch_data(url) for url in urls]
    )
    print(results)
```

### wait и wait_for

```python
# wait_for с таймаутом
try:
    result = await asyncio.wait_for(
        long_operation(), 
        timeout=5.0
    )
except asyncio.TimeoutError:
    print("Timeout!")

# wait с возвратом по первому завершению
done, pending = await asyncio.wait(
    tasks, 
    return_when=asyncio.FIRST_COMPLETED
)
```

## Синхронизация

### Lock

```python
lock = asyncio.Lock()

async def critical_section():
    async with lock:
        # только один корутина может выполнять этот код
        await modify_shared_resource()
```

### Semaphore

```python
# ограничение количества одновременных операций
semaphore = asyncio.Semaphore(10)

async def limited_operation():
    async with semaphore:
        await make_request()
```

### Event

```python
event = asyncio.Event()

async def waiter():
    await event.wait()
    print("Event received!")

async def setter():
    await asyncio.sleep(1)
    event.set()
```

## Работа с сетью

### aiohttp

```python
import aiohttp

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, 'http://example.com')
        print(html[:100])
```

### aiofiles

```python
import aiofiles

async def read_file(path):
    async with aiofiles.open(path, 'r') as f:
        return await f.read()
```

## Продвинутые паттерны

### Connection Pool

```python
class ConnectionPool:
    def __init__(self, max_size=10):
        self._pool = asyncio.Queue(maxsize=max_size)
        self._semaphore = asyncio.Semaphore(max_size)
    
    async def acquire(self):
        async with self._semaphore:
            return await self._pool.get()
    
    async def release(self, conn):
        await self._pool.put(conn)
```

### Rate Limiting

```python
class RateLimiter:
    def __init__(self, rate_per_second):
        self.rate = rate_per_second
        self.tokens = rate_per_second
        self.last_update = asyncio.get_event_loop().time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.rate,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            if self.tokens < 1:
                sleep_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(sleep_time)
                self.tokens = 0
            else:
                self.tokens -= 1
```

### Backpressure

```python
async def process_with_backpressure(queue, worker_count):
    semaphore = asyncio.Semaphore(worker_count)
    
    async def worker(item):
        async with semaphore:
            await process(item)
    
    while True:
        item = await queue.get()
        asyncio.create_task(worker(item))
```

## Интеграция с синхронным кодом

### run_in_executor

```python
import concurrent.futures

def blocking_io():
    # синхронная операция
    time.sleep(1)
    return "result"

async def main():
    loop = asyncio.get_running_loop()
    
    # выполнение в thread pool
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, blocking_io)
```

### to_thread

```python
async def main():
    result = await asyncio.to_thread(blocking_io)
```

## Отладка и профилирование

### Отладка

```python
# Включение отладки
asyncio.get_event_loop().set_debug(True)

# Таймауты для обнаружения зависших корутин
asyncio.wait_for(coro, timeout=30)
```

### Профилирование

```python
import cProfile
import pstats

async def profiled_main():
    profiler = cProfile.Profile()
    profiler.enable()
    
    await actual_work()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
```

## Антипаттерны

### Блокирование event loop

```python
# ПЛОХО: блокирует весь event loop
time.sleep(10)

# ХОРОШО: дает возможность другим корутинам работать
await asyncio.sleep(10)
```

### Забытые await

```python
# ПЛОХО: корутина не выполнится
async def main():
    coro = some_async_function()
    # coro никогда не запустится!

# ХОРОШО:
async def main():
    result = await some_async_function()
```

### Игнорирование исключений в задачах

```python
# ПЛОХО: исключение будет потеряно
task = asyncio.create_task(coro())

# ХОРОШО: обработка исключений
task = asyncio.create_task(coro())
try:
    await task
except Exception as e:
    logger.error(f"Task failed: {e}")
```

## Заключение

Asyncio — мощный инструмент для IO-bound задач. Правильное использование требует понимания event loop, корутин и механизмов синхронизации. Начинайте с простых случаев и постепенно усложняйте архитектуру.
