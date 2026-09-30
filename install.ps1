param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'

Push-Location $projectRoot
try {
    $python311 = $null
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        & py -3.11 --version 2>$null
        if ($LASTEXITCODE -eq 0) { $python311 = 'py' }
    }
    if (-not $python311) {
        $uvPythonRoot = Join-Path $env:APPDATA 'uv\python'
        if (Test-Path -LiteralPath $uvPythonRoot) {
            $candidates = Get-ChildItem -LiteralPath $uvPythonRoot -Directory -Filter 'cpython-3.11.*' -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending
            foreach ($candidate in $candidates) {
                $candidatePython = Join-Path $candidate.FullName 'python.exe'
                if (Test-Path -LiteralPath $candidatePython) {
                    $candidateVersion = & $candidatePython --version
                    if ($LASTEXITCODE -eq 0 -and $candidateVersion -match '^Python 3\.11\.') {
                        $python311 = $candidatePython
                        break
                    }
                }
            }
        }
    }
    if (-not $python311) {
        throw 'Python 3.11 was not found. Install Python 3.11 or the uv-managed CPython 3.11 runtime, then rerun this setup.'
    }

    if (-not (Test-Path -LiteralPath $python)) {
        if ($python311 -eq 'py') {
            & py -3.11 -m venv .venv
        }
        else {
            & $python311 -m venv .venv
        }
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the project-local Python environment.' }
    }

    $venvVersion = & $python --version
    if ($LASTEXITCODE -ne 0 -or $venvVersion -notmatch '^Python 3\.11\.') {
        throw "The existing project environment is not Python 3.11 ($venvVersion). Preserve it and create a clean project copy before retrying."
    }

    & $python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'Could not update pip in the project-local environment.' }

    & $python -m pip install 'torch>=2.11,<3' --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'Could not install CUDA-enabled PyTorch from the official PyTorch wheel index.' }

    & $python -m pip install -e '.[documents]'
    if ($LASTEXITCODE -ne 0) { throw 'Could not install ELLM Desktop Agent and its document readers.' }

    Write-Host 'Setup finished. Start the agent with ./run.ps1 -Workspace <folder>.'
}
finally {
    Pop-Location
}
