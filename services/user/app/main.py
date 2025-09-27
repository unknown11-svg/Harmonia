from datetime import datetime, timedelta
from typing import Optional
import os

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# In-memory "database" for MVP; replace with SQLAlchemy + SQLite/Postgres later
fake_users_db = {}

# Security settings (override via env in production)
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-dev-secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

app = FastAPI(title="User Service", version="0.1.0")


class UserCreate(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None


class UserPublic(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    sub: str
    exp: int
    typ: str  # "access" or "refresh"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_token(subject: str, expires_delta: timedelta, typ: str) -> str:
    to_encode = {"sub": subject, "exp": datetime.utcnow() + expires_delta, "typ": typ}
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def create_access_token(subject: str) -> str:
    return create_token(subject, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES), "access")


def create_refresh_token(subject: str) -> str:
    return create_token(subject, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS), "refresh")


def decode_token(token: str) -> TokenPayload:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return TokenPayload(**payload)
    except JWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from e


def get_current_user(token: str = Depends(oauth2_scheme)) -> UserPublic:
    payload = decode_token(token)
    if payload.typ != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
    user = fake_users_db.get(payload.sub)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return UserPublic(id=payload.sub, email=user["email"], full_name=user.get("full_name"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/register", response_model=UserPublic, status_code=201)
def register(user: UserCreate):
    # Use email as id for MVP
    user_id = user.email.lower().strip()
    if user_id in fake_users_db:
        raise HTTPException(status_code=409, detail="User already exists")
    fake_users_db[user_id] = {
        "email": user.email,
        "full_name": user.full_name,
        "password_hash": get_password_hash(user.password),
    }
    return UserPublic(id=user_id, email=user.email, full_name=user.full_name)


@app.post("/auth/login", response_model=TokenPair)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # OAuth2PasswordRequestForm uses 'username' field for compatibility
    user_id = form_data.username.lower().strip()
    user = fake_users_db.get(user_id)
    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    return TokenPair(
        access_token=create_access_token(user_id),
        refresh_token=create_refresh_token(user_id),
    )


class RefreshRequest(BaseModel):
    refresh_token: str


@app.post("/auth/refresh", response_model=TokenPair)
def refresh_tokens(req: RefreshRequest):
    payload = decode_token(req.refresh_token)
    if payload.typ != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")
    user = fake_users_db.get(payload.sub)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return TokenPair(
        access_token=create_access_token(payload.sub),
        refresh_token=create_refresh_token(payload.sub),
    )


@app.get("/auth/me", response_model=UserPublic)
def me(current_user: UserPublic = Depends(get_current_user)):
    return current_user


# For local dev convenience
# uvicorn main:app --reload --port 8000