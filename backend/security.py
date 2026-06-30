"""Securite : hachage des mots de passe et jetons JWT.

Hachage via pbkdf2_sha256 (pur Python, inclus dans passlib) : pas de dependance
native, donc aucun probleme de version comme avec bcrypt sur certaines images.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from backend.config import ACCESS_TOKEN_EXPIRE_MINUTES, JWT_ALG, JWT_SECRET

_pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    return _pwd.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return _pwd.verify(password, hashed)


def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": subject, "exp": expire}, JWT_SECRET, algorithm=JWT_ALG)


def decode_token(token: str) -> Optional[str]:
    """Renvoie le `sub` (username) si le jeton est valide, sinon None."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG]).get("sub")
    except JWTError:
        return None
