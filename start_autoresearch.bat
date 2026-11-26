@echo off
rem Porneste AutoResearch (server + agenti) si deschide dashboard-ul in browser.
rem Daca serverul ruleaza deja, doar deschide dashboard-ul.
title AutoResearch - server (inchide fereastra ca sa opresti)
cd /d "%~dp0"
set "PORT=8020"
if not "%AUTORESEARCH_PORT%"=="" set "PORT=%AUTORESEARCH_PORT%"

curl -s -m 3 http://localhost:%PORT%/api/health >nul 2>&1
if not errorlevel 1 (
    start "" http://localhost:%PORT%
    exit /b 0
)

if not exist ".venv\Scripts\python.exe" (
    echo Nu gasesc .venv\Scripts\python.exe in %CD%
    pause
    exit /b 1
)

rem deschide browserul cand serverul raspunde (maxim 90 s)
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 90;$i++){try{Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://localhost:%PORT%/api/health | Out-Null; Start-Process http://localhost:%PORT%; break}catch{Start-Sleep 1}}"

echo Pornesc AutoResearch pe http://localhost:%PORT% ...
".venv\Scripts\python.exe" server.py --port %PORT%
echo.
echo Serverul s-a oprit.
pause
