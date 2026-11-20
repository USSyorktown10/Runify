from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.api.deps import DbSession
from app.core.security import create_access_token
from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
    SignupRequest,
    SignupResponse,
)
from app.services.user_service import (
    authenticate_user,
    create_user,
    get_user_by_email,
    get_user_by_username,
)

router = APIRouter()

@router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
def signup(signup_in: SignupRequest, db: DbSession):
    if get_user_by_username(db, username=signup_in.username):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=SignupResponse(
                success=False, error_message="Username is already taken"
            ).model_dump(),
        )

    if get_user_by_email(db, email=signup_in.email):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=SignupResponse(
                success=False, error_message="Email is already registered"
            ).model_dump(),
        )

    create_user(db, signup_in=signup_in)
    return SignupResponse(success=True)

@router.post("/login", response_model=LoginResponse)
def login(login_in: LoginRequest, db: DbSession):
    user = authenticate_user(
        db, username=login_in.username, password=login_in.password
    )
    if not user:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=LoginResponse(
                error_message="Invalid username or password"
            ).model_dump(),
        )

    token = create_access_token(subject=str(user.id))
    return LoginResponse(session_token=token)