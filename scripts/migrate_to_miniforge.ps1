<#
.SYNOPSIS
    Move conda environments from r-miniconda (reticulate's Miniconda) to Miniforge.

.DESCRIPTION
    Steps (run all, or any subset with -Steps):
      Export     Write, per environment, a full export, a from-history export, an explicit
                 package list, pip freeze, and a conda-forge-only "recreate" YAML. terrain-s2
                 and terrain-s2-gpu are recreated from the repo's environment files instead.
      Uninstall  Remove r-miniconda: stop its processes, remove it from the user PATH and the
                 cmd AutoRun key, remove conda/mamba profile blocks that reference it or a
                 missing executable, set aside ~\.condarc, run the
                 uninstaller, delete the folder. Refuses unless Export has completed.
      Install    Download a pinned Miniforge release, verify its SHA-256, install for the
                 current user without touching PATH, set strict channel priority, and run
                 `conda init powershell`.
      Recreate   Create each environment in Miniforge from its recreate YAML, falling back to
                 pinned, then unpinned, then no-pip variants. Runs check_env.py and pytest for
                 terrain-s2 afterwards.

    Everything is logged to <BackupDir>\migration.log. Exports are never deleted.

.PARAMETER ReExport
    Redo the export even though BackupDir already has one (the default reuses it, so a
    half-uninstalled installation can never overwrite a good export).

.PARAMETER DryRun
    Run Export (it only writes to BackupDir) and print what the other steps would do.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\migrate_to_miniforge.ps1 -DryRun
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\migrate_to_miniforge.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\migrate_to_miniforge.ps1 -Steps Recreate
#>
[CmdletBinding()]
param(
    [string]$OldRoot = "$env:LOCALAPPDATA\r-miniconda",
    [string]$NewRoot = "$env:USERPROFILE\miniforge3",
    [string]$BackupDir = "$env:USERPROFILE\conda_migration",
    [string]$MiniforgeVersion = "26.7.2-0",
    [string]$RepoPath = "F:\OneDrive - University of Canterbury\EDDIE_Terrain\terrain-s2",
    [string]$CudaVersion = "13",
    [string[]]$Steps = @("All"),          # e.g. -Steps Uninstall,Install,Recreate
    [string[]]$Exclude = @(),
    [string]$NumbaCacheDir = "",
    [switch]$DryRun,
    [switch]$Force,
    [switch]$ReExport
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"          # Invoke-WebRequest is very slow with a progress bar on 5.1
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
# `powershell -File` passes "A,B" as one string, so split and validate here.
$Steps = @($Steps | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
foreach ($s in $Steps) {
    if (@("Export", "Uninstall", "Install", "Recreate", "All") -notcontains $s) { throw "Unknown step '$s' (use Export, Uninstall, Install, Recreate or All)" }
}
$Run = @{}
foreach ($s in "Export", "Uninstall", "Install", "Recreate") { $Run[$s] = ($Steps -contains "All") -or ($Steps -contains $s) }
$Manifest = [IO.Path]::Combine($BackupDir, "manifest.json")

# ---------------------------------------------------------------------------- helpers
function Say([string]$msg, [string]$color = "Gray") { Write-Host $msg -ForegroundColor $color }
function Head([string]$msg) { Write-Host ""; Write-Host "=== $msg ===" -ForegroundColor Cyan }
function Act([string]$what, [scriptblock]$action) {
    if ($DryRun) { Say "  [dry-run] $what" "Yellow" } else { Say "  $what"; & $action }
}
function Confirm-Or-Stop([string]$question) {
    if ($Force -or $DryRun) { return }
    $a = Read-Host "$question  Type YES to continue"
    if ($a -ne "YES") { throw "Stopped by user." }
}
function Write-Lines([string]$path, [string[]]$lines) { [IO.File]::WriteAllLines($path, $lines, $Utf8NoBom) }
function Invoke-Loose([string]$exe, [string[]]$arguments) {
    # Windows PowerShell 5.1 turns redirected native stderr into terminating errors under 'Stop'.
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $out = & $exe @arguments 2>$null } finally { $ErrorActionPreference = $old }
    return $out
}
function Invoke-Capture([string]$exe, [string[]]$arguments) {
    # stdout as lines; stderr goes to the console. Avoids conda 26's filename-based `-f` exporters.
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $out = @(& $exe @arguments) } finally { $ErrorActionPreference = $old }
    if ($LASTEXITCODE -ne 0) { throw "$exe $($arguments -join ' ') failed (exit $LASTEXITCODE)" }
    return ,[string[]]$out
}
function Get-ProfilePaths {
    $docs = [Environment]::GetFolderPath("MyDocuments")      # may be redirected (OneDrive, network share)
    $list = @()
    foreach ($dir in "WindowsPowerShell", "PowerShell") {
        foreach ($file in "profile.ps1", "Microsoft.PowerShell_profile.ps1") { $list += [IO.Path]::Combine($docs, $dir, $file) }
    }
    if ($PROFILE) { $list += $PROFILE.CurrentUserAllHosts, $PROFILE.CurrentUserCurrentHost }
    return @($list | Where-Object { $_ -and (Test-Path $_) } | Sort-Object -Unique)
}
function Repair-Profiles {
    # Remove conda/mamba init blocks that reference $OldRoot or an executable that no longer exists.
    $rx = '(?s)#region (conda|mamba) initialize.*?#endregion[^\r\n]*\r?\n?'
    $i = 0
    foreach ($pf in Get-ProfilePaths) {
        $txt = [IO.File]::ReadAllText($pf)
        $drop = @()
        foreach ($m in [regex]::Matches($txt, $rx)) {
            $blk = $m.Value
            $exes = @([regex]::Matches($blk, '[A-Za-z]:\\[^"''\r\n]*?\.(exe|bat)') | ForEach-Object { $_.Value })
            $missing = @($exes | Where-Object { -not (Test-Path $_) })
            $old = $blk.ToLower().Contains($OldRoot.ToLower())
            if ($old -or $missing.Count -gt 0) {
                $why = if ($old) { "references $OldRoot" } else { "points at missing $($missing[0])" }
                Say "  $pf : $($m.Groups[1].Value) block $why" "Yellow"
                $drop += $blk
            } else { Say "  $pf : $($m.Groups[1].Value) block OK ($(if ($exes) { $exes[0] } else { 'no exe path' }))" }
        }
        if ($drop.Count -gt 0) {
            $i++
            Act "remove $($drop.Count) block(s) from $pf (backup kept)" {
                Copy-Item $pf ([IO.Path]::Combine($BackupDir, "profile$i.$([IO.Path]::GetFileName($pf)).bak")) -Force
                $new = $txt; foreach ($b in $drop) { $new = $new.Replace($b, "") }
                [IO.File]::WriteAllText($pf, $new, $Utf8NoBom)
            }
        }
    }
}
function Show-Survey {
    Head "0. Conda installations and PATH"
    Say "  this shell: CONDA_PREFIX=$($Orig.CONDA_PREFIX)  CONDA_EXE=$($Orig.CONDA_EXE)  MAMBA_EXE=$($Orig.MAMBA_EXE)"
    $roots = @("$env:LOCALAPPDATA\r-miniconda", "$env:LOCALAPPDATA\miniconda3", "$env:LOCALAPPDATA\anaconda3",
               "$env:USERPROFILE\miniconda3", "$env:USERPROFILE\anaconda3", "$env:USERPROFILE\miniforge3",
               "$env:USERPROFILE\mambaforge", "$env:USERPROFILE\micromamba", "$env:USERPROFILE\.julia\conda\3\x86_64",
               "$env:ProgramData\miniconda3", "$env:ProgramData\anaconda3", "$env:ProgramData\miniforge3")
    foreach ($r in $roots) {
        if (Test-Path ([IO.Path]::Combine($r, "conda-meta"))) { Say "  installation: $r" }
        elseif (Test-Path $r) { Say "  leftover folder (no conda-meta): $r" "Yellow" }
    }
    foreach ($scope in "User", "Machine", "Process") {
        $pp = if ($scope -eq "Process") { $Orig.PATH } else { [Environment]::GetEnvironmentVariable("Path", $scope) }
        foreach ($e in @($pp -split ';' | Where-Object { $_ })) {
            $hit = $false
            foreach ($up in ".", "..", "..\..") {
                try { if (Test-Path ([IO.Path]::Combine($e, $up, "conda-meta"))) { $hit = $true } } catch { }
            }
            if ($hit) { Say "  $scope PATH has conda entry: $e" $(if ($scope -eq "Process") { "Gray" } else { "Yellow" }) }
        }
    }
}
function Get-Lockers([string]$root) {
    # Processes running from $root, or with a DLL/.pyd from it loaded (R/reticulate, VS Code,
    # Positron and Jupyter embed Python, so their executables live elsewhere).
    $found = @()
    foreach ($p in Get-Process) {
        $hit = $null
        try { if ($p.Path -and $p.Path.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { $hit = $p.Path } } catch { }
        if (-not $hit) {
            try {
                foreach ($m in $p.Modules) {
                    if ($m.FileName -and $m.FileName.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { $hit = $m.FileName; break }
                }
            } catch { }        # protected or elevated processes: cannot be inspected without admin
        }
        if ($hit) { $found += [pscustomobject]@{ Name = $p.ProcessName; Id = $p.Id; Exe = $p.Path; Uses = $hit } }
    }
    return $found
}
function Show-Lockers($lockers) {
    foreach ($l in $lockers) { Say ("      {0} (PID {1})  {2}`n          uses {3}" -f $l.Name, $l.Id, $l.Exe, $l.Uses) "Yellow" }
}
function Invoke-Native([string]$exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$exe $($arguments -join ' ') failed (exit $LASTEXITCODE)" }
}

function Read-Deps([string]$yamlPath) {
    # Returns @{ Conda = [string[]]; Pip = [string[]] } from a conda env YAML (export format).
    $conda = New-Object System.Collections.Generic.List[string]
    $pip = New-Object System.Collections.Generic.List[string]
    $mode = ""
    foreach ($l in [IO.File]::ReadAllLines($yamlPath)) {
        if ($l -match '^dependencies:\s*$') { $mode = "deps"; continue }
        if ($mode -eq "") { continue }
        if ($l -match '^\s*$' -or $l -match '^\s*#') { continue }
        if ($l -match '^\S') { $mode = ""; continue }                    # next top-level key (e.g. prefix:)
        if ($l -match '^\s{0,3}-\s+pip:\s*$') { $mode = "pip"; continue }
        if ($mode -eq "pip" -and $l -match '^\s{4,}-\s+(.+?)\s*$') { $pip.Add($Matches[1]); continue }
        if ($l -match '^\s{0,3}-\s+(.+?)\s*$') {
            $mode = "deps"
            $conda.Add(($Matches[1] -replace '^[^:\s=<>]+::', ''))     # drop channel prefixes (defaults::, pkgs/main::)
        }
    }
    return @{ Conda = @($conda); Pip = @($pip) }
}

function Write-EnvYaml([string]$path, [string]$name, [string[]]$conda, [string[]]$pip) {
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("name: $name"); $lines.Add("channels:"); $lines.Add("  - conda-forge"); $lines.Add("  - nodefaults")
    $lines.Add("dependencies:")
    $c = @($conda | Where-Object { $_ -and $_ -notmatch '^pip([=<>]|$)' })
    foreach ($d in $c) { $lines.Add("  - $d") }
    if ($pip.Count -gt 0) {
        $lines.Add("  - pip"); $lines.Add("  - pip:")
        foreach ($p in $pip) { $lines.Add("      - $p") }
    } elseif ($conda -match '^pip([=<>]|$)') { $lines.Add("  - pip") }
    Write-Lines $path $lines
}

function Get-RepoYaml([string]$name) {
    $map = @{ "terrain-s2" = "environment.yml"; "terrain-s2-gpu" = "environment-gpu.yml" }
    if (-not $map.ContainsKey($name)) { return $null }
    $src = [IO.Path]::Combine($RepoPath, $map[$name])
    if (-not (Test-Path $src)) { Say "  repo file not found: $src (falling back to export)" "Yellow"; return $null }
    $lines = [IO.File]::ReadAllLines($src) | Where-Object { $_ -notmatch '^\s*-\s+-e\s' }    # no editable installs
    $lines = @($lines | ForEach-Object { $_ -replace '^(\s*-\s*cuda-version\s*=\s*)\S+', "`${1}$CudaVersion.*" })
    # drop a "- pip:" key left with no entries (conda rejects an empty pip section)
    $kept = New-Object System.Collections.Generic.List[string]
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^\s{0,3}-\s+pip:\s*$') {
            $j = $i + 1
            while ($j -lt $lines.Count -and $lines[$j] -match '^\s*(#.*)?$') { $j++ }
            if ($j -ge $lines.Count -or $lines[$j] -notmatch '^\s{4,}-\s') { continue }
        }
        $kept.Add($lines[$i])
    }
    $lines = $kept
    $dst = [IO.Path]::Combine($BackupDir, "$name.recreate.yml")
    Write-Lines $dst ([string[]]$lines)
    return $dst
}

# ---------------------------------------------------------------------------- start
New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
try { Stop-Transcript | Out-Null } catch { }
Start-Transcript -Append -Path ([IO.Path]::Combine($BackupDir, "migration.log")) | Out-Null
Say "Old root : $OldRoot`nNew root : $NewRoot`nBackups  : $BackupDir`nSteps    : $($Steps -join ', ')$(if ($DryRun) {'  (dry run)'})"
# Sanitise this process: an activated conda shell (e.g. "(base)") leaks CONDA_*/MAMBA_* and PATH
# entries from another installation into every conda.exe we run.
$Orig = @{ CONDA_PREFIX = $env:CONDA_PREFIX; CONDA_EXE = $env:CONDA_EXE; MAMBA_EXE = $env:MAMBA_EXE; PATH = $env:PATH }
Get-ChildItem Env: | Where-Object { $_.Name -match '^(CONDA_|MAMBA_|_CE_)' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
$condaish = { param($e) foreach ($up in ".", "..", "..\..") { try { if (Test-Path ([IO.Path]::Combine($e, $up, "conda-meta"))) { return $true } } catch { } }; return $false }
$env:PATH = (@($env:PATH -split ';' | Where-Object { $_ -and -not (& $condaish $_) }) -join ';')
$env:PYTHONUTF8 = "1"
$OrigOutEnc = [Console]::OutputEncoding
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false) } catch { }
if ($NewRoot -match '\s') { throw "NewRoot must not contain spaces (Miniforge installer limitation): $NewRoot" }

try {
Show-Survey
# ---------------------------------------------------------------------------- 1. export
if ($Run.Export) {
    Head "1. Export environments"
    $oldConda = [IO.Path]::Combine($OldRoot, "Scripts", "conda.exe")
    $skipExport = $false
    if (-not (Test-Path $oldConda)) {
        if (Test-Path $Manifest) { Say "  $OldRoot is gone but $Manifest exists: using the earlier export" "Yellow"; $skipExport = $true }
        else { throw "conda.exe not found under $OldRoot and no earlier export in $BackupDir" }
    } elseif ((Test-Path $Manifest) -and -not $ReExport) {
        # Never overwrite a good export from a possibly half-uninstalled installation.
        Say "  using the existing export in $BackupDir (pass -ReExport to redo it)" "Yellow"; $skipExport = $true
    }
}
if ($Run.Export -and -not $skipExport) {
    # Re-exporting over an earlier (e.g. dry-run) export is safe: the source still exists.
    Invoke-Native $oldConda @("--version")
    $envs = (& $oldConda env list --json | Out-String | ConvertFrom-Json).envs
    $records = @()
    foreach ($p in $envs) {
        $full = [IO.Path]::GetFullPath($p).TrimEnd('\')
        $isBase = ($full -ieq [IO.Path]::GetFullPath($OldRoot).TrimEnd('\'))
        $name = if ($isBase) { "base" } else { Split-Path $full -Leaf }
        if (-not $isBase -and -not $full.StartsWith($OldRoot, [StringComparison]::OrdinalIgnoreCase)) {
            Say "  skip $full (outside $OldRoot; not affected by the uninstall)" "Yellow"; continue
        }
        Say "  $name  ($full)" "White"
        $stem = [IO.Path]::Combine($BackupDir, $name)
        $fullYml = "$stem.full.yml"; $histYml = "$stem.history.yml"
        foreach ($pair in @(@($fullYml, "--no-builds"), @($histYml, "--from-history"))) {
            $lines = Invoke-Capture $oldConda @("env", "export", "-p", $full, $pair[1])
            if (-not ($lines -match '^dependencies:')) { throw "unexpected export output for $name ($($pair[1]))" }
            Write-Lines $pair[0] $lines
        }
        Write-Lines "$stem.explicit.txt" (Invoke-Capture $oldConda @("list", "-p", $full, "--explicit", "--md5"))
        $py = [IO.Path]::Combine($full, "python.exe")
        if (Test-Path $py) { Invoke-Loose $py @("-m", "pip", "list", "--format=freeze") | Out-File -Encoding utf8 "$stem.pip-freeze.txt" }

        $f = Read-Deps $fullYml; $h = Read-Deps $histYml
        $cands = @()
        $repo = Get-RepoYaml $name
        if ($repo) { $cands += $repo; $src = "repo" } else { $src = "history" }
        if (-not $repo -and -not $isBase) {
            # a) requested specs + pip, b) full pinned list + pip, c) unpinned, d) as (a) without pip
            $conda = if ($h.Conda.Count -gt 0) { $h.Conda } else { $f.Conda }
            $a = "$stem.recreate.yml";         Write-EnvYaml $a $name $conda $f.Pip
            $b = "$stem.recreate-pinned.yml";  Write-EnvYaml $b $name $f.Conda $f.Pip
            $unp = $f.Conda | ForEach-Object { ($_ -split '=')[0] } | Where-Object { $_ -notmatch '^(vc|vs20\d\d_runtime|ucrt|vc14_runtime)$' }
            $c = "$stem.recreate-unpinned.yml"; Write-EnvYaml $c $name $unp $f.Pip
            $d = "$stem.recreate-nopip.yml";   Write-EnvYaml $d $name $conda @()
            $cands += $a, $b, $c, $d
        }
        Say ("    conda specs: {0} requested / {1} installed; pip: {2}; recreate from: {3}" -f $h.Conda.Count, $f.Conda.Count, $f.Pip.Count, $src)
        $records += [pscustomobject]@{ name = $name; path = $full; base = $isBase; source = $src; candidates = $cands; pip = $f.Pip }
    }
    $condarc = [IO.Path]::Combine($env:USERPROFILE, ".condarc")
    if (Test-Path $condarc) { Copy-Item $condarc ([IO.Path]::Combine($BackupDir, "user.condarc")) -Force }
    $records | ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 $Manifest
    Say "  wrote $Manifest ($($records.Count) environments incl. base; base is not recreated)" "Green"
}

# ---------------------------------------------------------------------------- 2. uninstall
if ($Run.Uninstall) {
    Head "2. Uninstall r-miniconda"
    if (-not (Test-Path $Manifest)) { throw "No export manifest at $Manifest; run -Steps Export first." }
    $recs = Get-Content $Manifest -Raw | ConvertFrom-Json
    foreach ($r in $recs) {
        if (-not $r.base -and $r.candidates.Count -eq 0) { throw "No recreate file for $($r.name); not uninstalling." }
    }
    Say "  PowerShell profiles:"
    Repair-Profiles
    if (-not (Test-Path $OldRoot)) { Say "  $OldRoot not present; nothing to uninstall" "Yellow" }
    else {
        Confirm-Or-Stop "This deletes $OldRoot and every environment in it."
        Say "  looking for processes using r-miniconda (executables and loaded DLLs)..."
        $lockers = @(Get-Lockers $OldRoot)
        if ($lockers.Count -gt 0) {
            Say "  in use by:" "Yellow"; Show-Lockers $lockers
            Confirm-Or-Stop "Stop these processes (better: close R/RStudio/VS Code/Jupyter yourself first)?"
            Act "stop $($lockers.Count) processes" {
                $lockers | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
                Start-Sleep -Seconds 3
            }
        } else { Say "  none found" }
        # user PATH (a likely cause of the DLL conflict: r-miniconda's base DLLs shadow the env's)
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        $parts = @($userPath -split ';' | Where-Object { $_ })
        $bad = @($parts | Where-Object { $_.StartsWith($OldRoot, [StringComparison]::OrdinalIgnoreCase) })
        if ($bad.Count -gt 0) {
            Say "  r-miniconda is on the USER PATH (probable cause of the rasterio DLL error):" "Yellow"
            $bad | ForEach-Object { Say "      $_" "Yellow" }
            Write-Lines ([IO.Path]::Combine($BackupDir, "user_path_before.txt")) $parts
            Act "remove from user PATH" {
                [Environment]::SetEnvironmentVariable("Path", (($parts | Where-Object { $bad -notcontains $_ }) -join ';'), "User")
            }
        } else { Say "  not on the user PATH" }
        $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
        if ($machine -and $machine.ToLower().Contains($OldRoot.ToLower())) { Say "  r-miniconda is on the MACHINE PATH: remove it as administrator" "Red" }
        # cmd AutoRun
        $key = "HKCU:\Software\Microsoft\Command Processor"
        $auto = (Get-ItemProperty -Path $key -Name AutoRun -ErrorAction SilentlyContinue).AutoRun
        if ($auto -and $auto.ToLower().Contains($OldRoot.ToLower())) {
            Write-Lines ([IO.Path]::Combine($BackupDir, "cmd_autorun_before.txt")) @($auto)
            Act "remove r-miniconda cmd AutoRun ($auto)" { Remove-ItemProperty -Path $key -Name AutoRun }
        }
        # ~\.condarc may carry 'defaults' channels into Miniforge
        $condarc = [IO.Path]::Combine($env:USERPROFILE, ".condarc")
        if (Test-Path $condarc) { Act "set aside $condarc (copy in $BackupDir)" { Move-Item $condarc "$condarc.r-miniconda.bak" -Force } }
        $envtxt = [IO.Path]::Combine($env:USERPROFILE, ".conda", "environments.txt")
        if (Test-Path $envtxt) {
            Act "remove r-miniconda entries from $envtxt" {
                Copy-Item $envtxt ([IO.Path]::Combine($BackupDir, "environments.txt.bak")) -Force
                $keep = @(Get-Content $envtxt | Where-Object { $_ -and -not $_.StartsWith($OldRoot, [StringComparison]::OrdinalIgnoreCase) })
                Write-Lines $envtxt $keep
            }
        }
        foreach ($v in "RETICULATE_PYTHON", "RETICULATE_MINICONDA_PATH", "CONDA_PREFIX", "PYTHONPATH") {
            $val = [Environment]::GetEnvironmentVariable($v, "User")
            if ($val) { Say "  note: user variable $v=$val (review manually)" "Yellow" }
        }
        $un = Get-ChildItem -Path $OldRoot -Filter "Uninstall-*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
        $partial = -not (Test-Path ([IO.Path]::Combine($OldRoot, "Scripts", "conda.exe")))
        if ($un -and $partial) { Say "  installation already partly removed; skipping the uninstaller" "Yellow"; $un = $null }
        if ($un) { Act "run $($un.Name) /S" { Start-Process -Wait -FilePath $un.FullName -ArgumentList "/S", "_?=$OldRoot" } }
        Act "delete $OldRoot" {
            for ($try = 1; $try -le 3 -and (Test-Path $OldRoot); $try++) {
                try { Remove-Item -LiteralPath $OldRoot -Recurse -Force -ErrorAction Stop }
                catch { cmd /c rmdir /s /q $OldRoot 2>$null }
                if (Test-Path $OldRoot) { Say "    attempt $try incomplete; waiting (antivirus scans can hold files briefly)" "Yellow"; Start-Sleep -Seconds 5 }
            }
            if (Test-Path $OldRoot) {
                $left = @(Get-ChildItem -LiteralPath $OldRoot -Recurse -File -Force -ErrorAction SilentlyContinue)
                Say "  $($left.Count) files could not be deleted. Still in use by:" "Red"
                $l2 = @(Get-Lockers $OldRoot)
                if ($l2.Count) { Show-Lockers $l2 } else { Say "      (no visible process: an elevated or service process, or a pending antivirus scan)" "Red" }
                throw ("Could not fully delete $OldRoot. Close the programs above (or sign out/reboot), delete the folder, " +
                       "then re-run with -Steps Uninstall,Install,Recreate. Your exports in $BackupDir are intact.")
            }
        }
    }
}

# ---------------------------------------------------------------------------- 3. install
$conda = [IO.Path]::Combine($NewRoot, "Scripts", "conda.exe")
if ($Run.Install) {
    Head "3. Install Miniforge $MiniforgeVersion"
    if (Test-Path $conda) { Say "  already installed at $NewRoot" "Yellow" }
    else {
        $exe = "Miniforge3-$MiniforgeVersion-Windows-x86_64.exe"
        $base = "https://github.com/conda-forge/miniforge/releases/download/$MiniforgeVersion"
        $dl = [IO.Path]::Combine($env:TEMP, $exe)
        Act "download $base/$exe" {
            Invoke-WebRequest -UseBasicParsing -Uri "$base/$exe" -OutFile $dl
            $content = (Invoke-WebRequest -UseBasicParsing -Uri "$base/$exe.sha256").Content
            if ($content -is [byte[]]) { $content = [Text.Encoding]::ASCII.GetString($content) }
            $expected = ($content.Trim() -split '\s+')[0]
            $actual = (Get-FileHash -Algorithm SHA256 $dl).Hash
            if ($actual -ine $expected.Trim()) { throw "SHA-256 mismatch for $exe (expected $expected, got $actual)" }
            Say "  SHA-256 verified: $actual" "Green"
        }
        Act "install to $NewRoot (current user, not on PATH, not registered as system Python)" {
            $p = Start-Process -Wait -PassThru -FilePath $dl -ArgumentList "/InstallationType=JustMe", "/RegisterPython=0", "/AddToPath=0", "/S", "/D=$NewRoot"
            if ($p.ExitCode -ne 0 -or -not (Test-Path $conda)) { throw "Miniforge installer failed (exit $($p.ExitCode))" }
        }
    }
    Act "strict channel priority in $NewRoot\.condarc" {
        Invoke-Native $conda @("config", "--file", [IO.Path]::Combine($NewRoot, ".condarc"), "--set", "channel_priority", "strict")
    }
    Act "conda init powershell" { Invoke-Native $conda @("init", "powershell") }
    $pol = Get-ExecutionPolicy -Scope CurrentUser
    if ($pol -in @("Restricted", "Undefined", "AllSigned")) {
        Say "  PowerShell execution policy is '$pol': conda activation in PowerShell needs" "Yellow"
        Say "      Set-ExecutionPolicy -Scope CurrentUser RemoteSigned" "Yellow"
    }
    if ($NumbaCacheDir) {
        Act "set user NUMBA_CACHE_DIR=$NumbaCacheDir" {
            New-Item -ItemType Directory -Force -Path $NumbaCacheDir | Out-Null
            [Environment]::SetEnvironmentVariable("NUMBA_CACHE_DIR", $NumbaCacheDir, "User")
        }
    }
}

# ---------------------------------------------------------------------------- 4. recreate
if ($Run.Recreate) {
    Head "4. Recreate environments"
    if (-not (Test-Path $Manifest)) { throw "No export manifest at $Manifest" }
    if (-not (Test-Path $conda) -and -not $DryRun) { throw "Miniforge not found at $NewRoot" }
    $recs = Get-Content $Manifest -Raw | ConvertFrom-Json
    $existing = @()
    if (Test-Path $conda) { $existing = @((& $conda env list --json | Out-String | ConvertFrom-Json).envs | ForEach-Object { Split-Path $_ -Leaf }) }
    $summary = @()
    foreach ($r in $recs) {
        if ($r.base -or ($Exclude -contains $r.name)) { continue }
        if ($existing -contains $r.name) { Say "  $($r.name): already exists, skipped" "Yellow"; $summary += "$($r.name): exists"; continue }
        $ok = $false
        foreach ($yml in @($r.candidates)) {
            if (-not (Test-Path $yml)) { continue }
            Say "  $($r.name) <- $(Split-Path $yml -Leaf)" "White"
            if ($DryRun) { Say "  [dry-run] conda env create -n $($r.name) -f <copy of $(Split-Path $yml -Leaf) as environment.yml>" "Yellow"; $ok = $true; break }
            # conda 26 picks the spec format from the file name: give it the canonical one
            $tmp = [IO.Path]::Combine([IO.Path]::GetTempPath(), "conda_recreate", $r.name)
            New-Item -ItemType Directory -Force -Path $tmp | Out-Null
            $spec = [IO.Path]::Combine($tmp, "environment.yml")
            Copy-Item $yml $spec -Force
            & $conda env create -n $r.name -f $spec
            if ($LASTEXITCODE -eq 0) { $ok = $true; $summary += "$($r.name): OK from $(Split-Path $yml -Leaf)"; break }
            Say "    failed; removing partial environment and trying the next variant" "Yellow"
            Invoke-Loose $conda @("env", "remove", "-n", $r.name, "-y") | Out-Null
        }
        if (-not $ok) { $summary += "$($r.name): FAILED (exports in $BackupDir)" }
        elseif ($r.pip.Count -gt 0 -and -not $DryRun -and $summary[-1] -like "*nopip*") {
            $summary[-1] += "; pip packages NOT installed, see $($r.name).pip-freeze.txt"
        }
    }
    Head "Summary"
    $summary | ForEach-Object { Say "  $_" $(if ($_ -like "*FAILED*" -or $_ -like "*NOT*") { "Red" } else { "Green" }) }

    if (-not $DryRun -and ($summary -like "terrain-s2: OK*")) {
        Head "terrain-s2 checks"
        & $conda run -n terrain-s2 --no-capture-output --cwd $RepoPath python scripts\check_env.py
        & $conda run -n terrain-s2 --no-capture-output --cwd $RepoPath python -m pytest -q
    }
}

Head "Done"
Say "Open a NEW PowerShell window, then:  conda activate terrain-s2"
Say "R/reticulate: set RETICULATE_MINICONDA_ENABLED=FALSE so reticulate does not reinstall r-miniconda,"
Say "and point it at Miniforge, e.g. reticulate::use_condaenv('r-reticulate', conda = '$($NewRoot -replace '\\','/')/condabin/conda.bat')."
}
finally {
    try { [Console]::OutputEncoding = $OrigOutEnc } catch { }
    try { Stop-Transcript | Out-Null } catch { }
}
