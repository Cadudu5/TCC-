@echo off
setlocal
cd /d "%~dp0"

set "LABELER_ENV=.venv-rotulador-windows"

if not exist "%LABELER_ENV%\Scripts\python.exe" (
  echo Preparando o ambiente do rotulador...
  py -3.12 -m venv "%LABELER_ENV%"
  if errorlevel 1 goto :error
  "%LABELER_ENV%\Scripts\python.exe" -m pip install --upgrade pip
  if errorlevel 1 goto :error
  "%LABELER_ENV%\Scripts\python.exe" -m pip install -r requirements-rotulador.txt
  if errorlevel 1 goto :error
)

"%LABELER_ENV%\Scripts\python.exe" rotulador_lite.py %*
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Nao foi possivel iniciar o rotulador. Verifique se o Python 3.12 esta instalado.
pause
exit /b 1
