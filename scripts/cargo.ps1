$ErrorActionPreference = 'Stop'
$CargoArguments = $args

$cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE '.cargo' }
$rustcCommand = Get-Command rustc.exe -ErrorAction SilentlyContinue
$cargoCommand = Get-Command cargo.exe -ErrorAction SilentlyContinue
$rustc = if ($rustcCommand) { $rustcCommand.Source } else { Join-Path $cargoHome 'bin\rustc.exe' }
$cargo = if ($cargoCommand) { $cargoCommand.Source } else { Join-Path $cargoHome 'bin\cargo.exe' }

if (-not (Test-Path $rustc) -or -not (Test-Path $cargo)) {
    throw 'Rust is not installed or its Cargo bin directory could not be found.'
}

$sysroot = (& $rustc --print sysroot).Trim()
$hostLine = & $rustc -vV | Where-Object { $_ -like 'host: *' } | Select-Object -First 1
if (-not $hostLine) {
    throw 'Unable to determine the active Rust host target.'
}

$hostTarget = $hostLine.Substring('host: '.Length).Trim()
$selfContainedTools = Join-Path $sysroot "lib\rustlib\$hostTarget\bin\self-contained"
$selfContainedLibraries = Join-Path $sysroot "lib\rustlib\$hostTarget\lib\self-contained"
if (Test-Path $selfContainedTools) {
    $env:PATH = "$selfContainedTools;$env:PATH"
}
if (Test-Path $selfContainedLibraries) {
    $env:LIBRARY_PATH = "$selfContainedLibraries;$env:LIBRARY_PATH"
}

$clangCommand = Get-Command x86_64-w64-mingw32-clang.exe -ErrorAction SilentlyContinue
$clang = if ($clangCommand) { $clangCommand.Source } else { $null }
if (-not $clang) {
    $wingetPackages = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
    $clang = Get-ChildItem `
        (Join-Path $wingetPackages 'MartinStorsjo.LLVM-MinGW.MSVCRT_*') `
        -Recurse `
        -Filter 'x86_64-w64-mingw32-clang.exe' `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $clang -or -not (Test-Path $clang)) {
    throw 'LLVM-MinGW was not found. Install it with: winget install --id MartinStorsjo.LLVM-MinGW.MSVCRT --exact'
}

$llvmMingwBin = Split-Path -Parent $clang
$llvmDlltool = Join-Path $llvmMingwBin 'x86_64-w64-mingw32-dlltool.exe'
if (-not (Test-Path $llvmDlltool)) {
    throw 'The LLVM-MinGW dlltool was not found beside its Clang linker.'
}
$env:PATH = "$llvmMingwBin;$env:PATH"

$env:CARGO_BUILD_TARGET = 'x86_64-pc-windows-gnullvm'
$env:CARGO_TARGET_X86_64_PC_WINDOWS_GNULLVM_LINKER = $clang

$separator = [char]0x1f
$encodedFlags = @()
if ($env:CARGO_ENCODED_RUSTFLAGS) {
    $encodedFlags += $env:CARGO_ENCODED_RUSTFLAGS.Split($separator)
}
$encodedFlags += '-C'
$encodedFlags += "dlltool=$llvmDlltool"
$encodedFlags += '-C'
$encodedFlags += 'target-feature=+crt-static'
$encodedFlags += '-A'
$encodedFlags += 'linker_messages'
$env:CARGO_ENCODED_RUSTFLAGS = $encodedFlags -join $separator

& $cargo @CargoArguments
exit $LASTEXITCODE
