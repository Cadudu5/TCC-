@echo off
setlocal
cd /d "%~dp0"
set "LABELER_ENV=.venv-rotulador"

echo Instalacao do Rotulador de Neutrofilos
echo ======================================
echo.

where py >nul 2>nul
if errorlevel 1 goto :python_missing

py -3.12 --version >nul 2>nul
if errorlevel 1 goto :python_missing

if not exist "%LABELER_ENV%\Scripts\python.exe" (
  echo Criando ambiente Python isolado...
  py -3.12 -m venv "%LABELER_ENV%"
  if errorlevel 1 goto :error
)

echo Instalando as dependencias. E necessario estar conectado a internet...
"%LABELER_ENV%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
"%LABELER_ENV%\Scripts\python.exe" -m pip install -r requirements-rotulador.txt
if errorlevel 1 goto :error

echo.
echo Instalacao concluida com sucesso.
echo Agora use ABRIR_ROTULADOR_WINDOWS.bat.
pause
exit /b 0

:python_missing
echo.
echo Python 3.12 nao foi encontrado.
echo Instale-o por https://www.python.org/downloads/windows/
echo Mantenha habilitado o Python Launcher durante a instalacao.
pause
exit /b 1

:error
echo.
echo A instalacao nao foi concluida. Verifique a conexao com a internet
echo e tente executar este arquivo novamente.
pause
exit /b 1
