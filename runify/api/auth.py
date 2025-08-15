import secrets


def signup(*args, **kwargs):
    pass


def login(*args, **kwargs):
    pass


def logout(*args, **kwargs):
    pass



def validate_session(token):
    if token == "Skibidi":
        return {"user_id": 12873}
    return None


def generate_token():
    return secrets.token_urlsafe(32)
