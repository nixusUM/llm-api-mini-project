# Compose Multiplatform: практический гид

## Что такое Compose Multiplatform

Compose Multiplatform (CMP) — это UI-фреймворк от JetBrains, который позволяет писать интерфейс на Kotlin и переиспользовать его между Android, iOS, Desktop и Web (в зависимости от зрелости платформы).

CMP строится на идеях Jetpack Compose:
- декларативный UI;
- composable-функции;
- управление состоянием через `remember` и `State`.

## Архитектура в MPP-проекте

Обычно код делят на модули:
- `shared` (Kotlin Multiplatform): domain + data + часть UI (общие composables).
- `androidApp`: Android entry point.
- `iosApp`: iOS entry point (Swift/ObjC оболочка + Compose host).

В `shared` можно держать:
- UI-компоненты, не зависящие от платформы;
- бизнес-логику;
- use cases;
- сетевой слой через Ktor;
- сериализацию через kotlinx.serialization.

## expect/actual механизм

Для платформенно-специфичных API используют `expect/actual`:

```kotlin
// commonMain
expect class PlatformLogger() {
    fun log(message: String)
}
```

```kotlin
// androidMain
actual class PlatformLogger {
    actual fun log(message: String) {
        android.util.Log.d("App", message)
    }
}
```

```kotlin
// iosMain
actual class PlatformLogger {
    actual fun log(message: String) {
        println("iOS: $message")
    }
}
```

## Управление состоянием в Compose

Базовые инструменты:
- `remember { mutableStateOf(...) }`
- `rememberSaveable` для восстановления состояния
- `derivedStateOf` для вычисляемых значений
- `LaunchedEffect` для side effects

Пример:

```kotlin
@Composable
fun CounterCard() {
    var count by rememberSaveable { mutableStateOf(0) }
    Column {
        Text("Count: $count")
        Button(onClick = { count++ }) {
            Text("Increment")
        }
    }
}
```

## Оптимизация recomposition

Чтобы уменьшить лишние перерисовки:
- использовать стабильные модели данных;
- выносить тяжелые вычисления в `remember`;
- задавать `key` в списках;
- не создавать новые объекты в каждой composable-функции без необходимости.

## Навигация в мультиплатформенном UI

Подходы:
- Decompose (Ark Ivanov);
- Voyager;
- собственный state-based роутер.

Важно, чтобы навигация была:
- предсказуемой;
- тестируемой;
- отделенной от UI-деталей платформы.

## Работа с ресурсами

В MPP-проектах часто используют:
- `org.jetbrains.compose.resources`;
- обертки для строк/изображений;
- платформенные адаптеры для шрифтов и системных иконок.

Рекомендация: держать ключи ресурсов и форматирование строк в `shared`, а platform-файлы подключать через build-конфигурацию.

## Тестирование Compose Multiplatform

Минимальный набор:
- unit-тесты для domain/use cases;
- snapshot/preview проверки UI (где возможно);
- integration-тесты на уровне platform apps.

Для Android можно использовать стандартные Compose UI tests. Для iOS часть тестов обычно выполняется в Xcode-инфраструктуре.

## Ограничения и риски

- Не все Android API доступны в `commonMain`.
- Некоторые UI-компоненты могут вести себя по-разному на платформах.
- Нужно внимательно следить за версиями Kotlin, Compose и Gradle plugins.

## Когда стоит использовать CMP

CMP полезен, когда:
- команда хочет общий UI-код между Android и iOS;
- есть сильная экспертиза в Kotlin;
- критична скорость выпуска одинаковых фич на обе платформы.

Если проекту нужен только Android — Jetpack Compose без MPP проще.
