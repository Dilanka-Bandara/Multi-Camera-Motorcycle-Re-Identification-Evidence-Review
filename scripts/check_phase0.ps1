# Run from a normal PowerShell terminal with Docker Desktop available.
# Uses a disposable container; does not alter the running application or Git history.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$logDirectory = Join-Path $projectRoot 'data\checks'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$logPath = Join-Path $logDirectory 'phase0.log'

function Assert-Success([string] $Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE. See $logPath"
    }
}

Start-Transcript -Path $logPath -Force
Push-Location $projectRoot
try {
    docker info --format '{{.ServerVersion}}'
    Assert-Success 'Docker engine access'
    docker compose config --quiet
    Assert-Success 'Compose validation'

    Push-Location (Join-Path $projectRoot 'frontend')
    try {
        npm.cmd ci
        Assert-Success 'Frontend dependency installation'
        npm.cmd run build
        Assert-Success 'Frontend production build'
    }
    finally { Pop-Location }

    # Run checks against current source, not the potentially older image source.
    # Root applies only to this temporary container so test tools can be installed.
    docker compose run --rm --no-deps --user root --volume "${projectRoot}:/workspace" --workdir /workspace app python scripts/check_phase0_container.py
    Assert-Success 'Python 3.11 lint and tests'

    $health = Invoke-RestMethod -Uri 'http://localhost:8080/api/health' -TimeoutSec 15
    if ($health.status -ne 'ok') { throw 'Running API health check failed.' }
    $page = Invoke-WebRequest -Uri 'http://localhost:8080/' -UseBasicParsing -TimeoutSec 15
    if ($page.StatusCode -ne 200 -or $page.Content -notmatch 'MotoReID') {
        throw 'Running frontend HTTP check failed.'
    }
    Write-Host 'Phase 0 automated checks passed. Browser visual verification remains required.'
}
finally {
    Pop-Location
    Stop-Transcript
}
