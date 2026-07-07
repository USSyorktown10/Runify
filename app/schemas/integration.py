from pydantic import BaseModel


class IntegrationStatus(BaseModel):
    provider: str
    is_connected: bool
    connected_at: str | None = None


class ConnectResponse(BaseModel):
    redirect_url: str
