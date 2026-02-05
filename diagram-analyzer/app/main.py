from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ui.router import router as ui_router
from api.api import router as api_router

app = FastAPI()

app.include_router(ui_router)
app.include_router(api_router)

app.mount("/static", StaticFiles(directory="ui/static"), name="static")
