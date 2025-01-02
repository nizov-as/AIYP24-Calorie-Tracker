import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from service.api.v1.api_route import router
from service.logger_config import setup_logger


app = FastAPI(
    title="Detection_Model_Inference",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)


class StatusResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App healthy"}]}
    )


@app.get("/", response_model=StatusResponse)
async def root():
    return StatusResponse(status="App healthy")


app.include_router(router, prefix="/api/v1/models")

logger = setup_logger()
logger.info("Сервис запущен")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
