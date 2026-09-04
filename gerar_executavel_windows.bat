@echo off
setlocal
cd /d "%~dp0"

set "BUILD_ENV=.venv-build-windows"
set "APP_DIR=dist\Rotulador_Neutrofilos"
set "PACKAGE_ZIP=dist\Rotulador_Neutrofilos_Windows_Executavel.zip"
set "PACKAGE_SOURCE=distribuicao\Rotulador_Neutrofilos_Windows"

echo Geracao do executavel Windows
echo =============================
echo.

where py >nul 2>nul
if errorlevel 1 goto :python_missing
py -3.12 --version >nul 2>nul
if errorlevel 1 goto :python_missing

if not exist "%BUILD_ENV%\Scripts\python.exe" (
  echo Criando ambiente de compilacao...
  py -3.12 -m venv "%BUILD_ENV%"
  if errorlevel 1 goto :error
)

echo Instalando ferramentas de compilacao...
"%BUILD_ENV%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
"%BUILD_ENV%\Scripts\python.exe" -m pip install -r requirements-build-windows.txt
if errorlevel 1 goto :error

echo Gerando Rotulador_Neutrofilos.exe...
"%BUILD_ENV%\Scripts\python.exe" -m PyInstaller --noconfirm --clean rotulador_windows.spec
if errorlevel 1 goto :error

echo Copiando imagens e instrucoes para o pacote final...
xcopy "%PACKAGE_SOURCE%\IMAGENS_PARA_ROTULAR" "%APP_DIR%\IMAGENS_PARA_ROTULAR" /E /I /Y >nul
if errorlevel 1 goto :error
xcopy "%PACKAGE_SOURCE%\MARCACOES_CONCLUIDAS" "%APP_DIR%\MARCACOES_CONCLUIDAS" /E /I /Y >nul
if errorlevel 1 goto :error
copy /Y "%PACKAGE_SOURCE%\INSTRUCOES_EXECUTAVEL_WINDOWS.txt" "%APP_DIR%\INSTRUCOES.txt" >nul
if errorlevel 1 goto :error
copy /Y "%PACKAGE_SOURCE%\VERSAO.txt" "%APP_DIR%\VERSAO.txt" >nul
if errorlevel 1 goto :error

echo Criando o ZIP portatil...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "if (Test-Path '%PACKAGE_ZIP%') { Remove-Item -Force '%PACKAGE_ZIP%' }; Compress-Archive -Path '%APP_DIR%' -DestinationPath '%PACKAGE_ZIP%' -CompressionLevel Optimal"
if errorlevel 1 goto :error

echo.
echo Executavel e pacote gerados com sucesso:
echo   %APP_DIR%\Rotulador_Neutrofilos.exe
echo   %PACKAGE_ZIP%
echo.
echo Teste o executavel neste Windows antes de envia-lo.
pause
exit /b 0

:python_missing
echo Python 3.12 com o Python Launcher nao foi encontrado nesta maquina.
echo Instale-o por https://www.python.org/downloads/windows/
pause
exit /b 1

:error
echo.
echo A geracao do executavel falhou. Revise as mensagens acima.
pause
exit /b 1
