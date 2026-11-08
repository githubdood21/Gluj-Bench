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
$jobs = @()
$names = @('read_cache','write_cache','copy_cache','read_stream','write_stream','copy_stream')
foreach ($index in 0..5) {
    $jobs += [ordered]@{ name = "bandwidth_$($names[$index])"; source = 'bandwidth.comp'; define = "-DKERNEL=$index" }
}
foreach ($kind in @('fp16','int8','uint8')) {
    $jobs += [ordered]@{ name = "matrix_profile_$kind"; source = "matrix_profile_$kind.comp"; define = '' }
}
$entries = @()
foreach ($job in $jobs) {
    $source = Join-Path $shaderDir $job.source
    $binary = Join-Path $shaderDir "spv/$($job.name).spv"
    $compiled = Join-Path $scratch "$($job.name).spv"
    $compileArguments = @('-V','--target-env','vulkan1.1','--spirv-val','-o',$compiled)
    if ($job.define) { $compileArguments += $job.define }
    $compileArguments += $source
    & $Compiler @compileArguments
    if ($LASTEXITCODE -ne 0) { throw "Shader compilation/validation failed: $($job.name)" }
    $hash = (Get-FileHash -LiteralPath $compiled -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Check) {
        if ($hash -ne (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()) {
            throw "Embedded shader differs from rebuilt source: $($job.name)"
        }
    } else { Copy-Item -LiteralPath $compiled -Destination $binary -Force }
    $entries += [ordered]@{
        name = $job.name
        source = $job.source
        define = $job.define
        source_sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
        spirv_sha256 = $hash
    }
}
$manifest = [ordered]@{ compiler = $version; target = 'vulkan1.1 / SPIR-V 1.3'; validation = 'glslang --spirv-val'; shaders = $entries }
$manifestPath = Join-Path $shaderDir 'memory-matrix-build.json'
if ($Check) {
    $recorded = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($recorded.compiler -ne $version -or ($recorded.shaders | ConvertTo-Json -Depth 5 -Compress) -ne ($entries | ConvertTo-Json -Depth 5 -Compress)) {
        throw 'Shader build manifest differs; regenerate and review the compiler/hash changes.'
    }
} else {
    [IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 5) + "`n", [Text.UTF8Encoding]::new($false))
}
Write-Output 'All nine memory/matrix shaders compiled, validated and matched their sources.'
