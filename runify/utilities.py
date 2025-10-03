import hashlib
import re
import secrets


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



def is_valid_handle(handle):
    # Check if handle is a string and length is between 3-12 characters
    if not isinstance(handle, str) or len(handle) < 3 or len(handle) > 12:
        return False

    # Check for valid characters (letters, numbers, and common keyboard characters)
    # Allows: a-z, A-Z, 0-9, _, -, @, #, $, %, &, *
    valid_pattern = r'^[a-zA-Z0-9_@#$%&*-]+$'

    # Check for no spaces and valid pattern
    return ' ' not in handle and bool(re.match(valid_pattern, handle))

# Password hashing with SHA-256 and random salt, pretty basic but should be enough
def hash_password(password: str, salt: bytes = None) -> tuple[str, bytes]:
    if salt is None:
        salt = secrets.token_bytes(32)  # Generate a 32-byte random salt
    # Combine password and salt, then hash with SHA-512
    hashed = hashlib.sha512(password.encode('utf-8') + salt).hexdigest()
    return hashed, salt

# Check if a password is valid against the salt and hash
def verify_password(password: str, stored_hash: str, salt: bytes) -> bool:
    hashed, _ = hash_password(password, salt)
    return hashed == stored_hash


def create_session_id() -> str:
    return secrets.token_urlsafe(32)
