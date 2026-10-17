# terrain-s2 on the high-performance workstation (WSL, no Docker)

For the full batch runs, dataset building and (from ML-1b) training on the RTX 6000 machine (192 GB, 32 cores).
One script, `docs/wsl/terrain_hpc.sh`, does the installation and runs the work. It is the Stage 2 chain only:
the EDDIE stack and GeoFabrics (Docker, `stage1_workstation.md`) are not needed for it.

## 1. Windows side, once

1. **NVIDIA driver**: a current Windows driver. WSL uses it; never install a Linux driver inside WSL.
2. **WSL memory**: WSL takes half the RAM by default. Create `%UserProfile%\.wslconfig` (for 192 GB of RAM):

   ```ini
   [wsl2]
   memory=150GB
   processors=64
   swap=32GB

   [experimental]
   autoMemoryReclaim=dropCache
   ```

   then `wsl --shutdown`. Leave Windows at least ~32 GB beside WSL, and give Windows a page file (System >
   Advanced > Performance > Virtual memory). On 16 Oct 2026, with `memory=176GB` and no page file, Linux's file cache
   (the rasters being written) filled WSL's memory, Windows ran out ("low virtual memory", event 2004 of
   Resource-Exhaustion-Detector, naming vmmemWSL at ~162 GB) and stopped the VM every ~20 minutes. `check` reports
   the margin, and `start`, `channels`, `chips` and `unet` empty the file cache every minute while they run. Run
   `bash ~/terrain/terrain_hpc.sh guard-setup` once (it asks for your password): a sudoers rule that lets them do it
   without a password, so detached runs (`setsid nohup ...`) are guarded too.
3. **The distro, on the big data drive** (PowerShell; `E:` is an example):

   ```powershell
   wsl --install -d Ubuntu-24.04 --name terrain
   wsl --manage terrain --move E:\WSL\terrain
   ```

   The data live inside the distro's disk (ext4, fast), so put it where the space is: a national run needs about
   1 TB for the mirrored LINZ tiles plus about 0.5 GB per tile for bundles, runs and patches.
4. **GitHub access** from the distro (as on the current workstation), to clone the repository.

## 2. Install (inside the `terrain` distro)

```bash
mkdir -p ~/terrain && tr -d '\r' < /mnt/d/EDDIE_Terrain/terrain-s2/docs/wsl/terrain_hpc.sh > ~/terrain/terrain_hpc.sh
bash ~/terrain/terrain_hpc.sh setup 2>&1 | tee /mnt/d/EDDIE/terrain_hpc_setup_report.txt
```

`setup` checks the machine (cores, the memory WSL sees, disk, GPU), installs micromamba, creates the `terrain-s2`
environment from `environment.yml` (conda-forge) with CuPy, clones the mirror (`~/terrain/terrain-s2`, branch
`main`), installs the package, writes `~/terrain/env.sh` (set `LINZ_API_KEY` there; `TERRAIN_DATA_DIR` defaults
to `~/terrain_data`) and the `terrain-hpc` command, then runs `check` (versions, a CuPy kernel, the test suite).
Run it again at any time; `update` fast-forwards the mirror and updates the environment when `environment.yml`
changed. Git work (patches, commits, pushes) stays in the Windows clone, as on the current workstation.

## 3. Data to bring across

```bash
bash ~/terrain/terrain_hpc.sh labels /mnt/d/eddie_data/labels/manual      # Matt's review files
mkdir -p ~/terrain_data/partition && cp /mnt/d/eddie_data/partition/nz_tiles.gpkg ~/terrain_data/partition/
```

(or `terrain_hpc.sh partition` to build the partition here). The LINZ tiles need not be copied: the shared reader
downloads them into `~/terrain_data/cog` once. Copying `D:\eddie_data\cog` and `stac` across (e.g. with
`robocopy` to `\\wsl.localhost\terrain\home\<user>\terrain_data\`) saves those downloads for the tiles already done.

## 4. Running

```bash
bash ~/terrain/terrain_hpc.sh start ~/terrain/terrain-s2/ml/manifests/wave2.csv     # detached
bash ~/terrain/terrain_hpc.sh status                                                 # progress, memory, last lines
bash ~/terrain/terrain_hpc.sh stop
bash ~/terrain/terrain_hpc.sh dataset ~/terrain/terrain-s2/ml/manifests/wave1.csv ~/terrain/terrain-s2/ml/manifests/wave2.csv --name waves12
```

`start` sizes the workers from memory (about 12 GB per LINZ tile with features) and cores, and the GPU workers
from GPU memory; add `--workers N --gpu-workers N` to choose (a warning says when the workers would need more than
85 % of the memory WSL has). It runs detached from the terminal, restarts the batch after an interruption until
every tile is done (at most 5 attempts; finished tiles are skipped), keeps Windows from sleeping while it runs, and
records out-of-memory kills in `~/terrain/run/supervisor.log`. `batch` runs the same in the terminal (closing the
terminal stops it). Results are readable from Windows at `\\wsl.localhost\terrain\home\<user>\terrain_data`, or at
`\\wsl.localhost\terrain\mnt\wsl\terrain_data` with the data on their own disk (section 5).

If a batch stops on its own: `status` shows the last log lines and any out-of-memory kills. If WSL itself closed
(the terminal drops back to PowerShell, and the supervisor log has no "ended" line), look for low-memory events:
`Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Resource-Exhaustion-Detector'} -MaxEvents 5`
(section 1, item 2). A Windows restart (for updates) stops WSL too (`Get-WinEvent -FilterHashtable @{LogName='System'; Id=1074,6006,6008,42} -MaxEvents 10` in
PowerShell lists restarts and sleeps); then `start` again.

## 5. The data on their own disk (optional)

`~/terrain_data` lives on the distro's system disk by default. To keep that disk small (and the data separable: a
disk that can be backed up, moved, or attached on the other workstation), put the data on their own ext4 disk: a
VHDX file on a data SSD, attached to WSL with `wsl --mount`, and linked from `~/terrain_data`. It is as fast as the
system disk (both are VHDX files on NTFS). Not a Windows folder through `/mnt/<drive>`: that path is many times
slower, and GeoPackage (SQLite) locking over it is unreliable.

Do it with no batch running. In an administrator PowerShell (`E:` is an example: the drive with the most free space;
`maximum` in MB, here 3 TB, below that drive's free space; the file grows as data are written):

```powershell
$vhd = 'E:\WSL\terrain_data.vhdx'
New-Item -ItemType Directory -Force (Split-Path $vhd) | Out-Null
Set-Content $env:TEMP\mkvhd.txt "create vdisk file=`"$vhd`" maximum=3145728 type=expandable"
diskpart /s $env:TEMP\mkvhd.txt
wsl --mount --vhd $vhd --bare
```

In WSL, find the new disk (the one of that size with no mount point; check twice: formatting the wrong disk loses
it) and format it:

```bash
lsblk -dno NAME,SIZE,MOUNTPOINTS
sudo mkfs.ext4 -L terrain_data /dev/sdX          # X from lsblk
```

Back in the administrator PowerShell: attach it by name (it appears in every distro at `/mnt/wsl/terrain_data`), and
add a logon task that attaches it again (also runnable later without a UAC prompt, after `wsl --shutdown` or a
crash, with `schtasks /run /tn "WSL terrain_data disk"`):

```powershell
wsl --unmount $vhd
wsl --mount --vhd $vhd --name terrain_data
schtasks /create /tn "WSL terrain_data disk" /sc onlogon /rl highest /f /tr "wsl.exe --mount --vhd $vhd --name terrain_data"
```

In WSL: copy, check, swap in a link, and only then delete the old copy (about 1 TB: hours):

```bash
sudo chown "$USER": /mnt/wsl/terrain_data
rsync -a --info=progress2 ~/terrain_data/ /mnt/wsl/terrain_data/
rsync -an --delete --itemize-changes --exclude=lost+found ~/terrain_data/ /mnt/wsl/terrain_data/ | head   # nothing: identical
mv ~/terrain_data ~/terrain_data.old && ln -s /mnt/wsl/terrain_data ~/terrain_data
bash ~/terrain/terrain_hpc.sh status                                                   # reads through the link
rm -rf ~/terrain_data.old && sudo fstrim -av
```

If the link is there but the disk is not attached (WSL restarted), `terrain_hpc.sh` runs the logon task itself
(through Windows interop: elevated, no prompt) and waits for the disk; if that fails, it stops with the `schtasks` command rather than
writing into an empty folder.

**Optional, and at your own risk: shrinking the system disk's file.** On 16 Oct 2026 the compaction below reported
success, but WSL then refused the compacted `ext4.vhdx` (`Wsl/Service/CreateInstance/MountDisk/HCS/ERROR_FILE_CORRUPT`;
the file copied cleanly, the drive checked clean, diskpart could still open it), and the distro had to be rebuilt
(below). The data, on their own disk, were untouched. If you do it, copy `ext4.vhdx` elsewhere first. In an
administrator PowerShell:

```powershell
$base = (Get-ChildItem HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss | Get-ItemProperty |
         Where-Object DistributionName -eq terrain).BasePath -replace '^\\\\\?\\', ''
wsl --shutdown
Set-Content $env:TEMP\compact.txt "select vdisk file=`"$base\ext4.vhdx`"`r`nattach vdisk readonly`r`ncompact vdisk`r`ndetach vdisk"
diskpart /s $env:TEMP\compact.txt
schtasks /run /tn "WSL terrain_data disk"
```

The system disk keeps its 2 TB limit (only the file shrinks), which costs nothing.

**Rebuilding the distro** (the system disk lost; the data disk intact). Administrator PowerShell:
`wsl --unregister terrain`, `wsl --install -d Ubuntu-24.04 --name terrain` (the same user name),
`wsl --manage terrain --move <drive>:\WSL\terrain`, `schtasks /run /tn "WSL terrain_data disk"`. In the distro, link
the data *before* setup, and fetch the script from GitHub (a copy from a Windows clone has CRLF line endings, which
bash rejects):

```bash
ln -s /mnt/wsl/terrain_data ~/terrain_data
curl -fsSL https://raw.githubusercontent.com/MatthewDWilson/terrain-s2/main/docs/wsl/terrain_hpc.sh -o ~/terrain_hpc.sh
bash ~/terrain_hpc.sh setup && bash ~/terrain_hpc.sh update && rm ~/terrain_hpc.sh
```

`setup` does not install the script into `~/terrain`; `update` does. PyTorch again with `torch` when needed. From
Windows the data disk is at `\\wsl.localhost\terrain\mnt\wsl\terrain_data` (the home-folder link does not open from
Explorer); `net use T: <that path> /persistent:yes` gives it a letter while the distro runs. Never attach the data
VHDX in Windows while WSL uses it.

## Notes

- Untested on the target machine: the script passes `bash -n` and ShellCheck, but not every step could run in the
  sandbox (no conda-forge access there). The report from `setup` is the first check.
- Per-worker threads are capped (`OMP_NUM_THREADS`, `NUMBA_NUM_THREADS` = 2 in `env.sh`), because the parallelism is
  across tiles.
- The patch CNN (ML-1b) needs PyTorch: `bash ~/terrain/terrain_hpc.sh torch` installs the CUDA 12.8 build (the
  RTX PRO 6000, a Blackwell GPU, needs 12.8 or later). On Windows: `python -m pip install torch --index-url
  https://download.pytorch.org/whl/cu128` in the conda environment.
