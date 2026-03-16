# Лучшие практики безопасности

## Введение

Безопасность приложений — это процесс, а не продукт. Требуется постоянная бдительность и обновление подходов.

## Аутентификация

### Парольные политики

```python
import re
import bcrypt

def validate_password(password: str) -> tuple[bool, str]:
    """Валидация пароля согласно NIST guidelines."""
    if len(password) < 12:
        return False, "Пароль должен быть минимум 12 символов"
    
    if len(password) > 128:
        return False, "Пароль слишком длинный"
    
    # Проверка на common passwords
    common_passwords = load_common_passwords_list()
    if password.lower() in common_passwords:
        return False, "Пароль слишком распространенный"
    
    # Проверка сложности (опционально, NIST рекомендует длину > сложность)
    has_upper = bool(re.search(r'[A-Z]', password))
    has_lower = bool(re.search(r'[a-z]', password))
    has_digit = bool(re.search(r'\d', password))
    has_special = bool(re.search(r'[!@#$%^&*]', password))
    
    score = sum([has_upper, has_lower, has_digit, has_special])
    if score < 3:
        return False, "Пароль должен содержать символы из 3 категорий"
    
    return True, "OK"


# Хеширование паролей
def hash_password(password: str) -> str:
    """Безопасное хеширование с bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed.decode()


def verify_password(password: str, hashed: str) -> bool:
    """Проверка пароля."""
    return bcrypt.checkpw(password.encode(), hashed.encode())
```

### Multi-Factor Authentication (MFA)

```python
import pyotp
import qrcode
from io import BytesIO

class MFAHandler:
    def generate_secret(self) -> str:
        """Генерация секрета для TOTP."""
        return pyotp.random_base32()
    
    def get_provisioning_uri(self, secret: str, user_email: str, issuer: str) -> str:
        """URI для QR-кода."""
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(user_email, issuer_name=issuer)
    
    def verify_totp(self, secret: str, code: str) -> bool:
        """Проверка TOTP кода."""
        totp = pyotp.TOTP(secret)
        # Допуск 1 шага для временной рассинхронизации
        return totp.verify(code, valid_window=1)
    
    def generate_qr_code(self, provisioning_uri: str) -> bytes:
        """Генерация QR-кода."""
        qr = qrcode.make(provisioning_uri)
        buffer = BytesIO()
        qr.save(buffer, format='PNG')
        return buffer.getvalue()


# Backup codes
def generate_backup_codes(count: int = 10) -> list[str]:
    """Генерация одноразовых резервных кодов."""
    import secrets
    return [secrets.token_hex(4).upper() for _ in range(count)]
```

### JWT Security

```python
import jwt
from datetime import datetime, timedelta
from typing import Optional

class JWTHandler:
    def __init__(self, secret: str, algorithm: str = 'HS256'):
        self.secret = secret
        self.algorithm = algorithm
    
    def create_token(
        self,
        user_id: str,
        expires_in: int = 3600,
        additional_claims: Optional[dict] = None
    ) -> str:
        """Создание JWT access token."""
        now = datetime.utcnow()
        payload = {
            'sub': user_id,
            'iat': now,
            'exp': now + timedelta(seconds=expires_in),
            'type': 'access',
            'jti': generate_token_id(),  # JWT ID для отзыва
        }
        if additional_claims:
            payload.update(additional_claims)
        
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str, expires_in: int = 604800) -> str:
        """Создание refresh token."""
        now = datetime.utcnow()
        payload = {
            'sub': user_id,
            'iat': now,
            'exp': now + timedelta(seconds=expires_in),
            'type': 'refresh',
            'jti': generate_token_id(),
        }
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)
    
    def verify_token(self, token: str, expected_type: str = 'access') -> dict:
        """Верификация и декодирование токена."""
        try:
            payload = jwt.decode(
                token,
                self.secret,
                algorithms=[self.algorithm],
                options={'require': ['exp', 'iat', 'sub']}
            )
            
            if payload.get('type') != expected_type:
                raise jwt.InvalidTokenError(f"Expected {expected_type} token")
            
            # Проверка отзыва (blacklist)
            if is_token_revoked(payload.get('jti')):
                raise jwt.InvalidTokenError("Token has been revoked")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {e}")


def generate_token_id() -> str:
    """Генерация уникального ID для токена."""
    import uuid
    return str(uuid.uuid4())


def is_token_revoked(jti: str) -> bool:
    """Проверка отзыва токена (в Redis или БД)."""
    # Реализация зависит от storage
    pass
```

## Авторизация

### RBAC (Role-Based Access Control)

```python
from enum import Enum
from functools import wraps
from typing import List

class Permission(Enum):
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_DELETE = "user:delete"
    ORDER_READ = "order:read"
    ORDER_WRITE = "order:write"
    ADMIN_FULL = "admin:full"

# Роли и их разрешения
ROLE_PERMISSIONS = {
    'user': [Permission.USER_READ, Permission.ORDER_READ, Permission.ORDER_WRITE],
    'manager': [Permission.USER_READ, Permission.ORDER_READ, Permission.ORDER_WRITE],
    'admin': list(Permission),  # Все разрешения
}


class RBACHandler:
    def __init__(self):
        self.user_roles = {}  # user_id -> roles
        self.user_permissions = {}  # user_id -> permissions
    
    def assign_role(self, user_id: str, role: str):
        """Назначение роли пользователю."""
        if role not in ROLE_PERMISSIONS:
            raise ValueError(f"Unknown role: {role}")
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        self.user_roles[user_id].add(role)
        
        # Обновление кеша разрешений
        self._update_user_permissions(user_id)
    
    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Проверка разрешения."""
        perms = self.user_permissions.get(user_id, set())
        return permission in perms
    
    def _update_user_permissions(self, user_id: str):
        """Обновление кеша разрешений."""
        roles = self.user_roles.get(user_id, set())
        permissions = set()
        for role in roles:
            permissions.update(ROLE_PERMISSIONS.get(role, []))
        self.user_permissions[user_id] = permissions


def require_permission(permission: Permission):
    """Декоратор для проверки разрешений."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Получение user_id из request context
            user_id = get_current_user_id()
            rbac = get_rbac_handler()
            
            if not rbac.has_permission(user_id, permission):
                raise PermissionError(f"Permission {permission.value} required")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Использование
@app.route('/api/users', methods=['POST'])
@require_permission(Permission.USER_WRITE)
def create_user():
    pass
```

### ABAC (Attribute-Based Access Control)

```python
from dataclasses import dataclass
from typing import Any, Dict, Callable

@dataclass
class AccessContext:
    subject: Dict[str, Any]  # пользователь
    resource: Dict[str, Any]  # объект
    action: str  # действие
    environment: Dict[str, Any]  # контекст (время, IP и т.д.)


class ABACPolicy:
    def __init__(self):
        self.rules: List[Callable[[AccessContext], bool]] = []
    
    def add_rule(self, rule: Callable[[AccessContext], bool]):
        """Добавление правила доступа."""
        self.rules.append(rule)
    
    def evaluate(self, context: AccessContext) -> bool:
        """Оценка всех правил (все должны пройти)."""
        return all(rule(context) for rule in self.rules)


# Примеры правил
def owner_can_edit(context: AccessContext) -> bool:
    """Владелец может редактировать ресурс."""
    return context.resource.get('owner_id') == context.subject.get('id')

def working_hours_only(context: AccessContext) -> bool:
    """Доступ только в рабочие часы."""
    from datetime import datetime
    now = datetime.now()
    return 9 <= now.hour < 18

def from_trusted_ip(context: AccessContext) -> bool:
    """Доступ только из доверенных сетей."""
    user_ip = context.environment.get('ip')
    return user_ip.startswith('10.0.') or user_ip.startswith('192.168.')
```

## Защита от атак

### SQL Injection Prevention

```python
# ПЛОХО: строковая интерполяция
def get_user_unsafe(user_id: str):
    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    # Атака: ' OR '1'='1

# ХОРОШО: параметризованные запросы
def get_user_safe(user_id: str):
    query = "SELECT * FROM users WHERE id = %s"
    cursor.execute(query, (user_id,))  # PostgreSQL
    # или
    cursor.execute(query, [user_id])   # SQLite

# ХОРОШО: ORM
from sqlalchemy.orm import Session

def get_user_orm(session: Session, user_id: int):
    return session.query(User).filter(User.id == user_id).first()
```

### XSS Prevention

```python
import html
from markupsafe import Markup, escape

# Экранирование HTML
def render_user_content(user_input: str) -> str:
    """Безопасный вывод пользовательского контента."""
    # Escape all HTML entities
    safe_content = escape(user_input)
    return Markup(safe_content)


# Content Security Policy
CSP_HEADER = (
    "default-src 'self'; "
    "script-src 'self' 'nonce-{nonce}'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: https:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self';"
)


def generate_nonce() -> str:
    """Генерация nonce для inline scripts."""
    import secrets
    return secrets.token_urlsafe(16)
```

### CSRF Protection

```python
import secrets
import hmac
import hashlib
from functools import wraps
from flask import session, request, abort

class CSRFProtection:
    TOKEN_NAME = 'csrf_token'
    
    @staticmethod
    def generate_token() -> str:
        """Генерация CSRF токена."""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def validate_token(token: str, session_token: str) -> bool:
        """Валидация токена через constant-time comparison."""
        return hmac.compare_digest(token, session_token)


def csrf_protect(func):
    """Декоратор для CSRF защиты."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
            token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')
            session_token = session.get(CSRFProtection.TOKEN_NAME)
            
            if not token or not session_token:
                abort(403, "CSRF token missing")
            
            if not CSRFProtection.validate_token(token, session_token):
                abort(403, "Invalid CSRF token")
        
        return func(*args, **kwargs)
    return wrapper
```

## Заключение

Безопасность требует:
1. Defense in depth (многослойная защита)
2. Least privilege (минимальные привилегии)
3. Fail securely (безопасное поведение при ошибках)
4. Security by design (безопасность на этапе проектирования)

Постоянно обновляйте зависимости, проводите аудиты (SAST, DAST) и следите за security advisories.
