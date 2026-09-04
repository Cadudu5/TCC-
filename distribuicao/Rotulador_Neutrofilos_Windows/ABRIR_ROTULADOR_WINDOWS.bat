@echo off
setlocal
cd /d "%~dp0"
set "LABELER_ENV=.venv-rotulador"

if not exist "%LABELER_ENV%\Scripts\python.exe" (
  echo O rotulador ainda precisa ser instalado.
  call INSTALAR_WINDOWS.bat
  if errorlevel 1 exit /b 1
)

"%LABELER_ENV%\Scripts\python.exe" rotulador_lite.py
if errorlevel 1 (
  echo.
  echo O rotulador terminou com erro. Fotografe esta janela e envie ao responsavel.
  pause
  exit /b 1
)
exit /b 0
