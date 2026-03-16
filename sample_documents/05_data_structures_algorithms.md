# Структуры данных и алгоритмы

## Введение

Алгоритмы и структуры данных — фундамент computer science. Понимание их эффективности критично для написания производительного кода.

## Анализ сложности

### Big O Notation

Обозначает асимптотическую сложность алгоритма:

- **O(1)** — константное время
- **O(log n)** — логарифмическое
- **O(n)** — линейное
- **O(n log n)** — линейно-логарифмическое
- **O(n²)** — квадратичное
- **O(2^n)** — экспоненциальное
- **O(n!)** — факториальное

### Пространственная сложность

Оценка используемой памяти в зависимости от входных данных.

## Основные структуры данных

### Массив (Array)

Фиксированный блок памяти с индексным доступом.

```python
# Python list (динамический массив)
arr = [1, 2, 3, 4, 5]
arr.append(6)      # O(1) amortized
arr.insert(0, 0)   # O(n)
arr[2]             # O(1) доступ
```

Сложности:
- Доступ: O(1)
- Поиск: O(n)
- Вставка/удаление: O(n)
- Добавление в конец: O(1)

### Связный список (Linked List)

Элементы связаны указателями.

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def prepend(self, val):  # O(1)
        new_node = Node(val)
        new_node.next = self.head
        self.head = new_node
    
    def delete(self, val):   # O(n)
        if not self.head:
            return
        if self.head.val == val:
            self.head = self.head.next
            return
        current = self.head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
                return
            current = current.next
```

Сложности:
- Доступ: O(n)
- Поиск: O(n)
- Вставка в начало: O(1)
- Вставка в конец: O(1) с tail
- Удаление: O(n)

### Стек (Stack)

LIFO (Last In, First Out).

```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):      # O(1)
        self.items.append(item)
    
    def pop(self):             # O(1)
        if not self.is_empty():
            return self.items.pop()
    
    def peek(self):            # O(1)
        if not self.is_empty():
            return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
```

Применение:
- Undo/Redo операции
- Обработка вызовов функций
- Проверка скобочной структуры
- DFS алгоритм

### Очередь (Queue)

FIFO (First In, First Out).

```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):   # O(1)
        self.items.append(item)
    
    def dequeue(self):         # O(1)
        if not self.is_empty():
            return self.items.popleft()
    
    def peek(self):
        if not self.is_empty():
            return self.items[0]
```

Применение:
- BFS алгоритм
- Обработка задач
- Кеширование (LRU)
- Потоковая обработка

### Хеш-таблица (Hash Table)

Ключ-значение с O(1) доступом.

```python
# Python dict — реализация хеш-таблицы
hash_table = {}
hash_table['key'] = 'value'  # O(1) average
value = hash_table['key']    # O(1) average

# Ручная реализация (упрощенная)
class HashTable:
    def __init__(self, size=100):
        self.size = size
        self.buckets = [[] for _ in range(size)]
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def put(self, key, value):     # O(1) average
        index = self._hash(key)
        bucket = self.buckets[index]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
    
    def get(self, key):            # O(1) average
        index = self._hash(key)
        bucket = self.buckets[index]
        for k, v in bucket:
            if k == key:
                return v
        return None
```

Коллизии:
- Separate chaining (цепочки)
- Open addressing (открытая адресация)
- Robin Hood hashing

### Деревья

#### Бинарное дерево поиска (BST)

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

class BST:
    def insert(self, root, val):      # O(log n) balanced
        if not root:
            return TreeNode(val)
        if val < root.val:
            root.left = self.insert(root.left, val)
        else:
            root.right = self.insert(root.right, val)
        return root
    
    def search(self, root, val):      # O(log n) balanced
        if not root or root.val == val:
            return root
        if val < root.val:
            return self.search(root.left, val)
        return self.search(root.right, val)
```

#### Самобалансирующиеся деревья

- **AVL Tree**: строгий баланс (разница высот ≤ 1)
- **Red-Black Tree**: приблизительный баланс
- **B-Tree**: для баз данных и файловых систем
- **B+ Tree**: все данные в листьях

### Куча (Heap)

Полное бинарное дерево с heap property.

```python
import heapq

# Минимальная куча
heap = []
heapq.heappush(heap, 3)    # O(log n)
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)

min_val = heapq.heappop(heap)  # O(log n)

# Преобразование списка в кучу
heapq.heapify(data)        # O(n)

# n-largest / n-smallest
largest = heapq.nlargest(3, data)   # O(n log k)
smallest = heapq.nsmallest(3, data)
```

Применение:
- Priority Queue
- Dijkstra алгоритм
- Heap Sort
- Median Maintenance

### Граф

```python
# Adjacency List
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}

# Adjacency Matrix (плотные графы)
n = 4  # vertices
matrix = [[0] * n for _ in range(n)]
matrix[0][1] = 1  # edge A->B
```

## Алгоритмы

### Сортировка

| Алгоритм | Средняя | Худшая | Пространство | Стабильность |
|----------|---------|--------|--------------|--------------|
| QuickSort | O(n log n) | O(n²) | O(log n) | Нет |
| MergeSort | O(n log n) | O(n log n) | O(n) | Да |
| HeapSort | O(n log n) | O(n log n) | O(1) | Нет |
| Insertion | O(n²) | O(n²) | O(1) | Да |
| Counting | O(n + k) | O(n + k) | O(k) | Да |
| Radix | O(nk) | O(nk) | O(n + k) | Да |

```python
# QuickSort
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# MergeSort
def mergesort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

### Поиск

```python
# Binary Search (отсортированный массив)
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Поиск в ширину (BFS)
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        vertex = queue.popleft()
        print(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Поиск в глубину (DFS)
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
```

### Динамическое программирование

```python
# Fibonacci с мемоизацией
def fib(n, memo=None):
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]

# Longest Common Subsequence
def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# Knapsack 0/1
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i-1][w],
                    dp[i-1][w-weights[i-1]] + values[i-1]
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]
```

## Заключение

Понимание структур данных и алгоритмов позволяет:
- Оценивать эффективность решений
- Выбирать правильные инструменты для задачи
- Оптимизировать узкие места
- Проходить технические интервью

Практикуйтесь на LeetCode, HackerRank, Codeforces для закрепления материала.
