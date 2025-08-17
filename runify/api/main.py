import connexion
from connexion.middleware import MiddlewarePosition
from starlette.middleware.cors import CORSMiddleware

from runify.api.auth import validate_session
from flask import jsonify, request

app = connexion.FlaskApp(__name__, specification_dir='.', pythonic_params=True)


app.add_middleware(
    CORSMiddleware,
    position=MiddlewarePosition.BEFORE_EXCEPTION,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging middleware (Flask level)
flask_app = app.app  # underlying Flask app

@flask_app.before_request
def log_request():
    print(f"[REQUEST] {request.method} {request.url}")
    if request.data:
        print(f"Payload: {request.data}")

@flask_app.after_request
def log_response(response):
    print(f"[RESPONSE] Status: {response.status}")
    return response

# JSON error handler
@flask_app.errorhandler(Exception)
def handle_exception(e):
    return jsonify(error=str(e)), 500


app.add_api(
    "./spec/openapi.yaml",
    validate_responses=True,
    security_map={"sessionAuth": validate_session},
    name="main_api",  # fixes blueprint name conflict
    swagger_ui=True,            # enable Swagger UI
    swagger_url="/ui",          # where Swagger UI is served
)

if __name__ == '__main__':
    app.run(port=8080)