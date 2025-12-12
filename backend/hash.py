import hashlib
import secrets
import base64
import sys


def hash_password(password: str) -> tuple[str, str]:
    """Хэширует пароль с солью"""
    salt = secrets.token_bytes(32)
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return base64.b64encode(pwd_hash).decode('utf-8'), base64.b64encode(salt).decode('utf-8')


def verify_password(password: str, stored_hash: str, stored_salt: str) -> bool:
    """Проверяет пароль"""
    salt = base64.b64decode(stored_salt.encode('utf-8'))
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return base64.b64encode(pwd_hash).decode('utf-8') == stored_hash

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    
    if sys.argv[1] == 'verify':
        password = sys.argv[2]
        stored_hash = sys.argv[3]
        stored_salt = sys.argv[4]
        result = verify_password(password, stored_hash, stored_salt)
        print(result)
    else:
        password = sys.argv[1]
        hash_result, salt = hash_password(password)
        print(hash_result)
        print(salt)