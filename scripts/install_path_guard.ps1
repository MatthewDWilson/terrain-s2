<#
.SYNOPSIS
    Install a PATH guard into conda environments, so other programs' GDAL-family DLLs cannot
    shadow the environment's own (e.g. F:\LAStools\bin\gdal.dll, C:\Program Files\GDAL).

.DESCRIPTION
    Why: conda's Python adds PATH folders to the DLL search, and Windows searches them in an
    unspecified order, so a folder's position on PATH does not protect the environment.

    What: writes three files into each environment:
      etc\conda\eddie_path_guard.ps1               filter: prints PATH without offending folders
      etc\conda\activate.d\zz_eddie_path_guard.ps1 / .bat     apply on `conda activate`
      etc\conda\deactivate.d\zz_eddie_path_guard.ps1 / .bat   restore on `conda deactivate`
    A folder is removed when it lies outside the environment and Windows, and contains a DLL
    whose name matches -Pattern. Removed folders are listed on activation. Nothing outside the
    environments is changed, so LAStools/GDAL keep working in other shells.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_path_guard.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_path_guard.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [string[]]$EnvName = @("terrain-s2", "terrain-s2-gpu"),
    [string]$CondaRoot = "$env:USERPROFILE\miniforge3",
    [string]$Pattern = '^(gdal|geos|proj|spatialite|geotiff|netcdf|hdf5|sqlite3|libcurl)[^\\/]*\.dll$',
    [switch]$Uninstall
)
$ErrorActionPreference = "Stop"
$EnvName = @($EnvName | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
$enc = New-Object System.Text.UTF8Encoding($false)

$filter = @'
# EDDIE PATH guard filter. Prints PATH without folders (outside this environment and Windows)
# that ship GDAL-family DLLs. Messages go to stderr so cmd's `for /f` captures only the PATH.
param([string]$Prefix = $env:CONDA_PREFIX)
$pattern = '__PATTERN__'
$sysroot = if ($env:SystemRoot) { $env:SystemRoot } else { "C:\Windows" }
$keep = New-Object System.Collections.Generic.List[string]
$removed = New-Object System.Collections.Generic.List[string]
foreach ($p in ($env:PATH -split ';')) {
    if (-not $p) { continue }
    if (($Prefix -and $p.StartsWith($Prefix, [StringComparison]::OrdinalIgnoreCase)) -or
        $p.StartsWith($sysroot, [StringComparison]::OrdinalIgnoreCase)) { $keep.Add($p); continue }
    $hit = $false
    try {
        foreach ($f in [IO.Directory]::GetFiles($p, "*.dll")) {
            if ([IO.Path]::GetFileName($f) -match $pattern) { $hit = $true; break }
        }
    } catch { }                                     # missing or unreadable folder: leave as is
    if ($hit) { $removed.Add($p) } else { $keep.Add($p) }
}
if ($removed.Count) { [Console]::Error.WriteLine("path guard: hid $($removed.Count) folder(s) with GDAL-family DLLs: " + ($removed -join '; ')) }
Write-Output ($keep -join ';')        # pipeline output: captured by PowerShell and by cmd's for /f
'@

$actPs1 = @'
$env:_EDDIE_PATH_BACKUP = $env:PATH
$_eddie_new = (& "$env:CONDA_PREFIX\etc\conda\eddie_path_guard.ps1") | Select-Object -Last 1
if ($_eddie_new) { $env:PATH = $_eddie_new }      # never replace PATH with nothing
Remove-Variable _eddie_new -ErrorAction SilentlyContinue
'@
$actBat = @'
@echo off
set "_EDDIE_PATH_BACKUP=%PATH%"
set "_EDDIE_NEW="
for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%CONDA_PREFIX%\etc\conda\eddie_path_guard.ps1"`) do set "_EDDIE_NEW=%%P"
if defined _EDDIE_NEW set "PATH=%_EDDIE_NEW%"
set "_EDDIE_NEW="
'@
$deaPs1 = @'
if ($env:_EDDIE_PATH_BACKUP) { $env:PATH = $env:_EDDIE_PATH_BACKUP; Remove-Item Env:_EDDIE_PATH_BACKUP }
'@
$deaBat = @'
@echo off
if defined _EDDIE_PATH_BACKUP set "PATH=%_EDDIE_PATH_BACKUP%"
set "_EDDIE_PATH_BACKUP="
'@

foreach ($name in $EnvName) {
    $prefix = [IO.Path]::Combine($CondaRoot, "envs", $name)
    if (-not (Test-Path ([IO.Path]::Combine($prefix, "conda-meta")))) { Write-Host "skip $name (not found at $prefix)" -ForegroundColor Yellow; continue }
    $etc = [IO.Path]::Combine($prefix, "etc", "conda")
    $files = @{
        ([IO.Path]::Combine($etc, "eddie_path_guard.ps1"))                  = $filter.Replace('__PATTERN__', $Pattern.Replace("'", "''"))
        ([IO.Path]::Combine($etc, "activate.d", "zz_eddie_path_guard.ps1"))   = $actPs1
        ([IO.Path]::Combine($etc, "activate.d", "zz_eddie_path_guard.bat"))   = $actBat
        ([IO.Path]::Combine($etc, "deactivate.d", "zz_eddie_path_guard.ps1")) = $deaPs1
        ([IO.Path]::Combine($etc, "deactivate.d", "zz_eddie_path_guard.bat")) = $deaBat
    }
    foreach ($f in $files.Keys) {
        if ($Uninstall) { if (Test-Path $f) { Remove-Item $f } ; continue }
        New-Item -ItemType Directory -Force -Path (Split-Path $f) | Out-Null
        $text = ($files[$f] -split '\r?\n') -join "`r`n"     # CRLF: cmd needs it for .bat files
        [IO.File]::WriteAllText($f, $text, $enc)
    }
    Write-Host "$(if ($Uninstall) {'removed from'} else {'installed in'}) $name" -ForegroundColor Green
}
if (-not $Uninstall) { Write-Host "Re-activate the environment (conda deactivate; conda activate terrain-s2) for it to take effect." }
