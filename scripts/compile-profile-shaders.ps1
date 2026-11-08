param(
    [string]$Compiler = "$PSScriptRoot/../target/shader-tools/bin/glslang.exe",
    [switch]$Check
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path "$PSScriptRoot/..").Path
$shaderDir = Join-Path $repo 'crates/gluj-bench-gpu/shaders'
$scratch = Join-Path $repo 'target/profile-shader-check'
New-Item -ItemType Directory -Path $scratch -Force | Out-Null
$version = (& $Compiler --version) -join "`n"
if ($LASTEXITCODE -ne 0) { throw 'Cannot query shader compiler version.' }
$entries = @()
foreach ($precision in @('fp16', 'fp32', 'fp64')) {
    $source = Join-Path $shaderDir "compute_profile_$precision.comp"
    $binary = Join-Path $shaderDir "spv/compute_profile_$precision.spv"
    $compiled = Join-Path $scratch "compute_profile_$precision.spv"
    & $Compiler -V --target-env vulkan1.1 --spirv-val -o $compiled $source
    if ($LASTEXITCODE -ne 0) { throw "Shader compilation/validation failed: $precision" }
    $hash = (Get-FileHash -LiteralPath $compiled -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Check) {
        if ($hash -ne (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()) {
            throw "Embedded shader differs from rebuilt source: $precision"
        }
    } else {
        Copy-Item -LiteralPath $compiled -Destination $binary -Force
    }
    $entries += [ordered]@{
        precision = $precision
        source_sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
        spirv_sha256 = $hash
    }
}
$manifest = [ordered]@{ compiler = $version; target = 'vulkan1.1 / SPIR-V 1.3'; validation = 'glslang --spirv-val'; shaders = $entries }
$json = ($manifest | ConvertTo-Json -Depth 5) + "`n"
$manifestPath = Join-Path $shaderDir 'profile-build.json'
if ($Check) {
    $recorded = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($recorded.compiler -ne $version -or ($recorded.shaders | ConvertTo-Json -Depth 5 -Compress) -ne ($entries | ConvertTo-Json -Depth 5 -Compress)) {
        throw 'Shader build manifest differs; regenerate with the recorded compiler or review the compiler upgrade.'
    }
} else {
    [IO.File]::WriteAllText($manifestPath, $json, [Text.UTF8Encoding]::new($false))
}
Write-Output 'All three profile shaders compiled, validated and matched their sources.'
