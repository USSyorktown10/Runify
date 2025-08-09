import hashlib
import re
from typing import Literal

from fastapi import FastAPI, Depends, HTTPException, Response, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
import secrets
import time
import uvicorn

from runify.utilities import is_secure_password, is_valid_handle

app = FastAPI()




# Pydantic models for the web requests

# /
class SignupFlowUser(BaseModel):
    username: str
    firstname: str
    lastname: str
    email: str
    gender: Literal["male", "female", "other"]
    password: str

class SignupResponse(BaseModel):
    status: Literal["success", "error"]
    error: str


class LoginFlowUser(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    status: Literal["success", "error"]
    error: str


class SessionData(BaseModel):
    session_id: str
    username: str
    expiry: float

class LogoutFlow(BaseModel):
    session_id: str

class LogoutResponse(BaseModel):
    status: Literal["success", "error"]
    error: str


# POST /api/signup
@app.post("/api/signup", response_model=SignupResponse)
async def signup(user: SignupFlowUser):
    if user.gender not in ["male", "female", "other"]:
        return {"status": "error", "error": "Invalid gender."}
    if not is_secure_password(user.password)[0]:
        return {"status": "error", "error": "Password is not secure."}
    if not user.firstname:
        return {"status": "error", "error": "No first name."}
    if not user.lastname:
        return {"status": "error", "error": "No last name."}
    if not user.email:
        return {"status": "error", "error": "No email address."}
    is_real_email = re.match("""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])""", user.email)
    if not is_real_email:
        return {"status": "error", "error": "Invalid email address."}
    if not is_valid_handle(user.username):
        return {"status": "error", "error": "Invalid username."}
    if not is_valid_handle(user.firstname):
        return {"status": "error", "error": "Invalid first name."}
    if not is_valid_handle(user.lastname):
        return {"status": "error", "error": "Invalid last name."}
    return {"status": "success", "error": "Signup successful"}

# POST /api/login
# POST /api/logout






if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)