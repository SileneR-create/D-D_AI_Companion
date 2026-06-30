"""Dependances FastAPI liees a l'authentification."""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session, select

from backend.db import get_session
from backend.models import User
from backend.security import decode_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Jeton invalide ou expire",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> User:
    username = decode_token(token)
    if not username:
        raise _UNAUTHORIZED
    user = session.exec(select(User).where(User.username == username)).first()
    if not user:
        raise _UNAUTHORIZED
    return user
