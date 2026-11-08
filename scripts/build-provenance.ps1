param()
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$files = @(Get-ChildItem -LiteralPath (Join-Path $root 'crates') -File -Recurse)
foreach ($name in @('Cargo.toml','Cargo.lock','rust-toolchain.toml','.cargo/config.toml','assets/branding/loop-chip-g-preview.png','scripts/cargo.ps1','scripts/build-provenance.ps1')) {
    $path = Join-Path $root $name
    if (Test-Path -LiteralPath $path) { $files += Get-Item -LiteralPath $path }
}
$lines = foreach ($file in ($files | Sort-Object FullName)) {
    $relative = $file.FullName.Substring($root.Length + 1).Replace('\','/')
    "$relative $((Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant())"
}
$hasher = [Security.Cryptography.SHA256]::Create()
try { $digest = ([BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes(($lines -join "`n"))))).Replace('-','').ToLowerInvariant() } finally { $hasher.Dispose() }
$head = (& git -C $root rev-parse --short HEAD).Trim()
$dirty = [bool](& git -C $root status --porcelain)
$manifest = Get-Content -LiteralPath (Join-Path $root 'Cargo.toml') -Raw
$version = [regex]::Match($manifest, '(?m)^version\s*=\s*"([^"]+)"').Groups[1].Value
[pscustomobject]@{
    version = $version
    revision = "$head-source-$($digest.Substring(0,12))"
    source_sha256 = $digest
    git_head = $head
    working_tree_dirty = $dirty
    target = 'x86_64-pc-windows-gnullvm'
    rustc = (& rustc --version)
    cargo = (& cargo --version)
    prepared_utc = [DateTime]::UtcNow.ToString('o')
}
