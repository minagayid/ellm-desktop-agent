param(
    [string]$Workspace = (Get-Location).Path,
    [string]$Adapter
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw 'The local environment is missing. Run ./install.ps1 first.'
}
if (-not (Test-Path -LiteralPath $Workspace -PathType Container)) {
    throw "Workspace folder does not exist: $Workspace"
}

$arguments = @('-m', 'ellm_agent.cli', '--workspace', (Resolve-Path -LiteralPath $Workspace).Path)
if ($Adapter) {
    $adapterPath = if ([System.IO.Path]::IsPathRooted($Adapter)) { $Adapter } else { Join-Path $projectRoot $Adapter }
    if (Test-Path -LiteralPath $adapterPath -PathType Container) {
        $Adapter = (Resolve-Path -LiteralPath $adapterPath).Path
    }
    $arguments += @('--adapter', $Adapter)
}

$agentExitCode = 1
Push-Location $projectRoot
try {
    & $python @arguments
    $agentExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $agentExitCode
