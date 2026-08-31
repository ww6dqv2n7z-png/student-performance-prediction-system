$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Error "Python environment not found. Run scripts\team_first_run.ps1 first."
}

if (-not (Test-Path (Join-Path $ProjectDir "web\node_modules"))) {
    Write-Error "Website dependencies not found. Run scripts\team_first_run.ps1 first."
}

Set-Location $ProjectDir
$Api = Start-Process -FilePath $Python -ArgumentList "-m", "student_performance.api" -PassThru -NoNewWindow

try {
    Set-Location (Join-Path $ProjectDir "web")
    npm run dev
}
finally {
    if ($Api -and -not $Api.HasExited) {
        Stop-Process -Id $Api.Id
    }
}
