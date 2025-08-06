import hashlib

from fastapi import FastAPI, Depends, HTTPException, Response, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
import secrets
import time
import uvicorn

app = FastAPI()


# Pydantic models for the web requests
class SignupFlowUser(BaseModel):
    username: str
    email: str
    password: str


class LoginFlowUser(BaseModel):
    username: str
    email: str


class CreateSessionData(BaseModel):
    session_id: str
    username: str
    expiry: float


# Password hashing with SHA-256 and random salt, pretty basic but should be enough
def hash_password(password: str) -> tuple[str, bytes]:

    salt = secrets.token_bytes(32)  # Generate a 32-byte random salt
    # Combine password and salt, then hash with SHA-256
    hashed = hashlib.sha256(password.encode('utf-8') + salt).hexdigest()
    return hashed, salt

# Check if a password is valid against the salt and hash
def verify_password(password: str, stored_hash: str, salt: bytes) -> bool:
    hashed, _ = hash_password(password, salt)
    return hashed == stored_hash

def is_secure_password(password) -> tuple[bool, str]:
    """
    Check if a password meets security requirements.
    Returns a tuple of (is_secure: bool, message: str)

    Criteria (for the purposes of this):
    - At least 8 characters long
    - Contains at least one uppercase letter
    - Contains at least one lowercase letter
    - Contains at least one digit
    - Contains at least one special character
    - No spaces allowed
    """
    import re

    # Check minimum length
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"

    # Check for spaces
    if ' ' in password:
        return False, "Password cannot contain spaces"

    # Check for uppercase letter
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"

    # Check for lowercase letter
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"

    # Check for digit
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"

    # Check for special character
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"

    return True, "" # No error, no string



# Create session ID



# Signup endpoint
@app.post("/signup", response_model=User)
async def signup(user: UserCreate):
    # TODO: database call
    if user.username in users:
        raise HTTPException(status_code=400, detail="Username already registered")
    if any(u["email"] == user.email for u in users.values()):
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password, salt = hash_password(user.password)

    # TODO: database call
    users[user.username] = {
        "username": user.username,
        "email": user.email,
        "password": hashed_password,
        "salt": salt
    }

    return User(username=user.username, email=user.email)


# Login endpoint with rate limiting
@app.post("/login", response_model=SessionData)
@limiter.limit("5/minute")  # 5 requests per minute per IP
async def login(form_data: OAuth2PasswordRequestForm = Depends(), request: Request = None):
    user = users.get(form_data.username)
    if not user or verify_password(form_data.password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    # Create session
    session_id = create_session_id()
    expiry = time.time() + 3600  # 1 hour expiry
    sessions[session_id] = {"username": user["username"], "expiry": expiry}

    # Set session cookie
    response = Response(content={"session_id": session_id, "username": user["username"], "expiry": expiry})
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        samesite="strict",
        max_age=3600
    )
    return SessionData(session_id=session_id, username=user["username"], expiry=expiry)


# Protected endpoint to get current user
@app.get("/users/me", response_model=User)
async def get_current_user(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    session = sessions[session_id]
    if session["expiry"] < time.time():
        del sessions[session_id]
        raise HTTPException(status_code=401, detail="Session expired")

    user = users.get(session["username"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return User(username=user["username"], email=user["email"])


# Logout endpoint
@app.post("/logout")
async def logout(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if session_id in sessions:
        del sessions[session_id]
    response.delete_cookie(key="session_id")
    return {"message": "Logged out successfully"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)