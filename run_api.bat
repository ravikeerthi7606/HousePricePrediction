
@echo off
setlocal
cd /d "%~dp0"
call "venv\Scripts\activate.bat"
set "PYTHONPATH=%CD%\src"
set "HOUSE_PRICE_MODEL_VERSION=2dc83fac14bf4e278d593d126899d7a3"
python -m uvicorn HousePrice.api:app --app-dir "%CD%\src"
