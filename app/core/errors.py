from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class RunifyError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class NotFoundError(RunifyError):
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404)


class ForbiddenError(RunifyError):
    def __init__(self, message: str = "Forbidden"):
        super().__init__(message, status_code=403)


class UnauthorizedError(RunifyError):
    def __init__(self, message: str = "Unauthorized"):
        super().__init__(message, status_code=401)


class ConflictError(RunifyError):
    def __init__(self, message: str = "Conflict"):
        super().__init__(message, status_code=409)


async def runify_error_handler(_request: Request, exc: RunifyError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error_message": exc.message},
    )


def http_error(message: str, status_code: int = 400) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"success": False, "error_message": message})
