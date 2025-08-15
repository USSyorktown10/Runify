import connexion
from connexion.middleware import MiddlewarePosition
from starlette.middleware.cors import CORSMiddleware

from runify.api.auth import validate_session

app = connexion.FlaskApp(__name__, specification_dir='.', pythonic_params=True)


app.add_middleware(
    CORSMiddleware,
    position=MiddlewarePosition.BEFORE_EXCEPTION,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_api(
    "./spec/openapi.yaml",
    strict_validation=True,
    validate_responses=True,
    security_map={"sessionAuth": validate_session},
    name="main_api"  # fixes blueprint name conflict
)

if __name__ == '__main__':
    app.run(port=8080)