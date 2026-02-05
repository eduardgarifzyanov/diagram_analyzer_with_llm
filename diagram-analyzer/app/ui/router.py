from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="ui/templates")


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/upload-diagram", response_class=HTMLResponse)
async def upload_diagram_page(request: Request):
    return templates.TemplateResponse("upload-diagram.html", {"request": request})


@router.get("/upload-csv", response_class=HTMLResponse)
async def upload_csv_page(request: Request):
    return templates.TemplateResponse("upload-csv.html", {"request": request})
