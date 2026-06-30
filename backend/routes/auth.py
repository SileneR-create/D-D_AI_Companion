"""Route /api/auth -- inscription, connexion, profil courant."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from backend.db import get_session
from backend.deps import get_current_user
from backend.models import User
from backend.schemas import LoginRequest, Token, UserCreate, UserRead
from backend.security import create_access_token, hash_password, verify_password

router = APIRouter()


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def register(data: UserCreate, session: Session = Depends(get_session)) -> User:
    if session.exec(select(User).where(User.username == data.username)).first():
        raise HTTPException(status_code=400, detail="Ce nom d'utilisateur est deja pris.")
    user = User(
        username=data.username,
        email=data.email,
        hashed_password=hash_password(data.password),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@router.post("/login", response_model=Token)
def login(data: LoginRequest, session: Session = Depends(get_session)) -> Token:
    user = session.exec(select(User).where(User.username == data.username)).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Identifiants invalides.")
    return Token(access_token=create_access_token(user.username))


@router.get("/me", response_model=UserRead)
def me(user: User = Depends(get_current_user)) -> User:
    return user
