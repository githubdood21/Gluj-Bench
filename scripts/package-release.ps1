param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d+\.\d+\.\d+([-.][0-9A-Za-z.-]+)?$')]
    [string]$Version,

    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ManifestPath = Join-Path $RepositoryRoot 'Cargo.toml'
$Manifest = Get-Content -Raw -LiteralPath $ManifestPath
$VersionMatch = [regex]::Match($Manifest, '(?m)^version\s*=\s*"([^"]+)"\s*$')

if (-not $VersionMatch.Success) {
    throw 'Unable to read the workspace version from Cargo.toml.'
}
if ($VersionMatch.Groups[1].Value -ne $Version) {
    throw "Release version $Version does not match Cargo.toml version $($VersionMatch.Groups[1].Value)."
}

Push-Location $RepositoryRoot
try {
    if (-not $SkipBuild) {
        & (Join-Path $PSScriptRoot 'cargo.ps1') build --workspace --release --locked
        if ($LASTEXITCODE -ne 0) {
            throw "Release build failed with exit code $LASTEXITCODE."
        }
    }

    $BinaryDirectory = Join-Path $RepositoryRoot 'target\x86_64-pc-windows-gnullvm\release'
    $UiBinary = Join-Path $BinaryDirectory 'gluj-bench-ui.exe'
    $WorkerBinary = Join-Path $BinaryDirectory 'gluj-bench-worker.exe'
    foreach ($binary in @($UiBinary, $WorkerBinary)) {
        if (-not (Test-Path -LiteralPath $binary)) {
            throw "Required release binary is missing: $binary"
        }
    }

    $DistDirectory = Join-Path $RepositoryRoot 'dist'
    $PackageName = "Gluj-Bench-$Version-windows-x64"
    $StageDirectory = Join-Path $DistDirectory $PackageName
    $ArchivePath = Join-Path $DistDirectory "$PackageName.zip"
    $ChecksumPath = "$ArchivePath.sha256"

    New-Item -ItemType Directory -Force -Path $DistDirectory | Out-Null
    foreach ($path in @($StageDirectory, $ArchivePath, $ChecksumPath)) {
        $ResolvedTarget = [IO.Path]::GetFullPath($path)
        $ResolvedDist = [IO.Path]::GetFullPath($DistDirectory) + [IO.Path]::DirectorySeparatorChar
        if (-not $ResolvedTarget.StartsWith($ResolvedDist, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Package target is outside the dist directory: $ResolvedTarget"
        }
        if (Test-Path -LiteralPath $path) {
            Remove-Item -Recurse -Force -LiteralPath $path
        }
    }
    New-Item -ItemType Directory -Path $StageDirectory | Out-Null

    Copy-Item -LiteralPath $UiBinary, $WorkerBinary -Destination $StageDirectory
    Copy-Item -LiteralPath 'README.md', 'CHANGELOG.md', 'SECURITY.md' -Destination $StageDirectory
    Copy-Item -LiteralPath 'SysInfo.png', 'Benchmarks.png', 'ScalingCompute.png' -Destination $StageDirectory
    $LogoDirectory = Join-Path $StageDirectory 'design\logo-drafts'
    New-Item -ItemType Directory -Force -Path $LogoDirectory | Out-Null
    Copy-Item -LiteralPath 'design\logo-drafts\loop-chip-g-preview.png' -Destination $LogoDirectory
    Copy-Item -LiteralPath 'LICENSE', 'NOTICE', 'BRANDING.md' -Destination $StageDirectory
    New-Item -ItemType Directory -Path (Join-Path $StageDirectory 'docs') | Out-Null
    Copy-Item -LiteralPath 'docs\protocol.md' -Destination (Join-Path $StageDirectory 'docs')
    $Utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText(
        (Join-Path $StageDirectory 'VERSION.txt'),
        "$Version`n",
        $Utf8NoBom
    )

    $LegacyEntries = @(
        Get-ChildItem -Recurse -Force -LiteralPath $StageDirectory |
            Where-Object { $_.Name -match '(?i)perfcalc|perfbench' }
    )
    if ($LegacyEntries.Count -ne 0) {
        $Names = $LegacyEntries.FullName -join ', '
        throw "Release staging contains legacy project artifacts: $Names"
    }

    Compress-Archive -LiteralPath $StageDirectory -DestinationPath $ArchivePath -CompressionLevel Optimal
    $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ArchivePath).Hash.ToLowerInvariant()
    Set-Content -LiteralPath $ChecksumPath -Value "$Hash  $([IO.Path]::GetFileName($ArchivePath))" -Encoding ascii

    Write-Host "Created $ArchivePath"
    Write-Host "Created $ChecksumPath"
}
finally {
    Pop-Location
}
