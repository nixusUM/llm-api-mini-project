# Архитектура мобильного приложения и релизный процесс

## Цели архитектуры мобильного проекта

Хорошая архитектура в мобильной разработке нужна для:
- ускорения поставки фич;
- уменьшения регрессий;
- удобства тестирования;
- предсказуемого масштабирования команды.

## Рекомендуемая архитектура: Clean + MVVM/MVI

На практике часто используют комбинацию:
- UI слой: Compose screen + ViewModel;
- Domain слой: use cases;
- Data слой: repositories + network/database.

Поток данных:
1. UI отправляет `Intent/Event`.
2. ViewModel вызывает use case.
3. Use case обращается к repository.
4. Repository читает/пишет API и локальную БД.
5. Result возвращается в ViewModel, обновляется UI state.

## Offline-first и кэширование

Для стабильности мобильного UX:
- локальный кэш (Room/SQLDelight);
- стратегия stale-while-revalidate;
- синхронизация в фоне.

Паттерн:
- UI всегда читает локальный source of truth;
- сеть обновляет локальное хранилище;
- UI автоматически получает обновления через flow/observable.

## Работа с сетью

Рекомендации:
- единый API client (Ktor/Retrofit);
- централизованный перехват ошибок;
- retry с backoff для временных ошибок;
- таймауты и отмена запросов.

Коды ошибок полезно маппить в доменные ошибки:
- `NetworkUnavailable`
- `Unauthorized`
- `ServerError`
- `ValidationError`

## Feature modules

Для больших приложений удобен модульный подход:
- `feature:profile`
- `feature:catalog`
- `feature:checkout`
- `core:ui`, `core:network`, `core:analytics`

Плюсы:
- параллельная разработка;
- более быстрые инкрементальные сборки;
- четкие границы ответственности.

## Аналитика и observability

Минимальный набор в проде:
- crash reporting (Crashlytics/Sentry);
- performance traces (startup time, frame drops);
- product analytics (ключевые события).

Для каждой фичи желательно иметь:
- success rate;
- latency;
- drop-off по шагам пользовательского сценария.

## CI/CD для мобильных приложений

Стандартный pipeline:
1. Lint + unit tests.
2. Static analysis.
3. Сборка debug/release.
4. UI/integration тесты.
5. Подпись артефактов.
6. Деплой в internal track/TestFlight.

Инструменты:
- GitHub Actions / Bitrise / Jenkins;
- Fastlane для автоматизации сборок и публикаций.

## Версионирование и релизы

Рекомендуется semantic-like схема для app version:
- `major.minor.patch+build`.

Типичный процесс:
- weekly release train;
- feature flags для рисковых фич;
- staged rollout (например 5% -> 20% -> 100%);
- мониторинг crash-free sessions после выката.

## Безопасность в мобильной разработке

Базовые правила:
- не хранить секреты в репозитории;
- использовать encrypted storage (Android Keystore/Keychain);
- certificate pinning для чувствительных API;
- обфускация и минимизация release build.

## Тестовая стратегия

Сбалансированный набор:
- unit tests для бизнес-логики;
- integration tests для data слоя;
- UI tests для ключевых сценариев;
- smoke tests перед релизом.

Важные метрики:
- test pass rate;
- flaky test ratio;
- среднее время pipeline.

## Что обычно ломает мобильные релизы

- отсутствие feature flags;
- ручная публикация без checklist;
- слишком большой релиз без staged rollout;
- нет rollback-плана.

Решение: автоматизировать pipeline и поддерживать release playbook.

## Вывод

Для mobile-проектов критично сочетать:
- понятную архитектуру (Clean + MVVM/MVI),
- стабильный data flow,
- автоматизированный CI/CD,
- контролируемый релизный процесс.

Это уменьшает риски и ускоряет поставку функций на Android и iOS.
