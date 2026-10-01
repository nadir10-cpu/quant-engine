# setup_task_scheduler.ps1
# Crea il task Windows per eseguire il paper trader ogni ora (utente corrente)

$pythonPath = (Get-Command python).Source
$scriptPath = "c:\Users\MASFI\Desktop\quantitative math\quant_engine\run_hourly_paper_trader.py"

$action  = New-ScheduledTaskAction `
    -Execute $pythonPath `
    -Argument "`"$scriptPath`"" `
    -WorkingDirectory "c:\Users\MASFI\Desktop\quantitative math"

# Trigger: ogni ora, partendo subito, fino al 1 Novembre 2026
$trigger = New-ScheduledTaskTrigger `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -Once -At (Get-Date)

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit  (New-TimeSpan -Minutes 10) `
    -RestartCount        3 `
    -RestartInterval     (New-TimeSpan -Minutes 5) `
    -StartWhenAvailable

$principal = New-ScheduledTaskPrincipal `
    -UserId    ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel  Limited

Register-ScheduledTask `
    -TaskName  "QuantEngine_HourlyPaperTrader" `
    -Action    $action `
    -Trigger   $trigger `
    -Settings  $settings `
    -Principal $principal `
    -Force

Write-Host ""
Write-Host "Task creato con successo!" -ForegroundColor Green
Write-Host "Il bot girera' ogni ora finche' il PC e' acceso e connesso."
Write-Host "Per l'esecuzione con PC SPENTO -> serve GitHub Actions (vedi istruzioni)."
