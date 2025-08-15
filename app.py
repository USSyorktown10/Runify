"""
Basic example of a resource server
"""
from pathlib import Path

import connexion
from connexion.exceptions import OAuthProblem

TOKEN_DB = {"asdf1234567890": {"uid": 100}}




def get_secret(user) -> str:
    return f"You are {user} and the secret is 'wbevuec'"


app = connexion.FlaskApp(__name__, specification_dir=".", pythonic_params=True)
app.add_api("openapi.yaml")


if __name__ == "__main__":
    app.run(f"{Path(__file__).stem}:app", port=8080)