# Kotlin для мобильной разработки

## Почему Kotlin используют для Android

Kotlin стал основным языком Android-разработки благодаря безопасной работе с null, лаконичному синтаксису и хорошей интеграции с Java. В типичном Android-проекте Kotlin используется для UI-слоя, бизнес-логики и data-слоя.

Ключевые причины выбора Kotlin:
- Null safety и уменьшение числа runtime ошибок.
- Coroutines для асинхронных операций без callback hell.
- Extension functions для выразительного API.
- Data classes и sealed classes для удобного моделирования состояний.

## Null safety

Kotlin различает nullable и non-nullable типы:

```kotlin
val nonNullName: String = "Alice"
val nullableName: String? = null
```

Операторы:
- `?.` safe call
- `?:` Elvis
- `!!` принудительное разыменование (использовать осторожно)

## Coroutines на Android

Coroutines нужны для сетевых запросов, работы с БД и long-running операций.

Основные элементы:
- `suspend` функции
- `CoroutineScope`
- `Dispatchers.IO` для I/O
- `Dispatchers.Main` для UI
- `viewModelScope` для запуска задач во ViewModel

Пример:

```kotlin
class UserViewModel(
    private val repo: UserRepository
) : ViewModel() {

    private val _state = MutableStateFlow<UserUiState>(UserUiState.Loading)
    val state: StateFlow<UserUiState> = _state

    fun loadUser(id: String) {
        viewModelScope.launch {
            _state.value = UserUiState.Loading
            _state.value = try {
                val user = withContext(Dispatchers.IO) { repo.getUser(id) }
                UserUiState.Success(user)
            } catch (e: Exception) {
                UserUiState.Error(e.message ?: "Unknown error")
            }
        }
    }
}
```

## Работа с состоянием UI

На Android (особенно в Compose) распространен uni-directional data flow:
1. UI отправляет event.
2. ViewModel обрабатывает event.
3. ViewModel обновляет state.
4. UI рендерит state.

Частые модели состояния:
- `StateFlow`
- `MutableStateFlow`
- `SharedFlow` для одноразовых событий

## Sealed classes для UI-state

```kotlin
sealed interface UserUiState {
    data object Loading : UserUiState
    data class Success(val user: User) : UserUiState
    data class Error(val message: String) : UserUiState
}
```

Преимущество: компилятор заставляет обработать все варианты в `when`.

## Рекомендации по структуре кода

Для среднего мобильного приложения удобно использовать слои:
- `ui/` — экраны, composables, ui-state
- `domain/` — use cases, бизнес-правила
- `data/` — repository, network, database

Принцип: UI не должен напрямую обращаться к network/database.

## Dependency Injection

На Android часто применяют Hilt или Koin:
- `Hilt` — стандартный вариант в экосистеме Google.
- `Koin` — более легковесный DSL-подход.

DI позволяет:
- изолировать зависимости;
- проще писать тесты;
- заменять реализации (например, fake repository).

## Ошибки и best practices

Частые ошибки:
- запуск тяжелых операций на Main dispatcher;
- хранение mutable состояния прямо в UI;
- отсутствие обработки исключений в coroutine.

Best practices:
- для I/O всегда использовать `Dispatchers.IO`;
- хранить source of truth во ViewModel;
- возвращать из репозитория domain-модели, а не raw DTO;
- использовать immutable UI state.

## Краткий вывод

Kotlin дает мобильной команде безопасный и быстрый в разработке стек: null safety, coroutines и выразительный язык. В связке с ViewModel + StateFlow это создает стабильную основу для масштабируемого Android-приложения.
