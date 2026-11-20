from sqlalchemy import select
from sqlalchemy.orm import Session
from app.core.security import get_password_hash, verify_password
from app.models.user import User
from app.schemas.auth import SignupRequest

def get_user_by_id(db: Session, user_id: str) -> User | None:
    return db.scalar(select(User).where(User.id == user_id))

def get_user_by_username(db: Session, username: str) -> User | None:
    return db.scalar(select(User).where(User.username == username))

def get_user_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == email))

def create_user(db: Session, signup_in: SignupRequest) -> User:
    user_data = signup_in.model_dump(exclude={"password"})
    user = User(
        **user_data,
        hashed_password=get_password_hash(signup_in.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def authenticate_user(db: Session, username: str, password: str) -> User | None:
    user = get_user_by_username(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user