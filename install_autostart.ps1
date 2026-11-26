# Înregistrează watchdog-ul AutoResearch ca sarcină programată Windows:
# pornește la logare, repornește dacă moare, nu se oprește pe baterie.
#
#   Instalare:     powershell -ExecutionPolicy Bypass -File install_autostart.ps1
#   Dezinstalare:  powershell -ExecutionPolicy Bypass -File install_autostart.ps1 -Remove

param([switch]$Remove)

$TaskName = "AutoResearch Watchdog"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectDir ".venv\Scripts\pythonw.exe"
$Watchdog = Join-Path $ProjectDir "watchdog.py"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Sarcina '$TaskName' a fost eliminata."
    exit 0
}

if (-not (Test-Path $Python)) { Write-Error "Nu gasesc $Python"; exit 1 }

$Action = New-ScheduledTaskAction -Execute $Python -Argument "`"$Watchdog`"" -WorkingDirectory $ProjectDir
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
    -Description "Tine serverul AutoResearch pornit (watchdog.py)" -Force | Out-Null

Write-Host "Sarcina '$TaskName' instalata: porneste la logare."
Write-Host "Pornire acum:  Start-ScheduledTask -TaskName '$TaskName'"
