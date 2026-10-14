# terrain-s2 on the high-performance workstation (WSL, no Docker)

For the full batch runs, dataset building and (from ML-1b) training on the RTX 6000 machine (192 GB, 32 cores).
One script, `docs/wsl/terrain_hpc.sh`, does the installation and runs the work. It is the Stage 2 chain only:
the EDDIE stack and GeoFabrics (Docker, `stage1_workstation.md`) are not needed for it.

## 1. Windows side, once

1. **NVIDIA driver**: a current Windows driver. WSL uses it; never install a Linux driver inside WSL.
2. **WSL memory**: WSL takes half the RAM by default. Create `%UserProfile%\.wslconfig`:

   ```ini
   [wsl2]
   memory=176GB
   processors=32
   swap=32GB
   ```

   then `wsl --shutdown`.
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
terminal stops it). Results are readable from Windows at `\\wsl.localhost\terrain\home\<user>\terrain_data`.

If a batch stops on its own: `status` shows the last log lines and any out-of-memory kills; a Windows restart (for
updates) stops WSL too (`Get-WinEvent -FilterHashtable @{LogName='System'; Id=1074,6006,6008,42} -MaxEvents 10` in
PowerShell lists restarts and sleeps); then `start` again.

## Notes

- Untested on the target machine: the script passes `bash -n` and ShellCheck, but not every step could run in the
  sandbox (no conda-forge access there). The report from `setup` is the first check.
- Per-worker threads are capped (`OMP_NUM_THREADS`, `NUMBA_NUM_THREADS` = 2 in `env.sh`), because the parallelism is
  across tiles.
- Training (ML-1b) adds PyTorch with CUDA; it will be added to `setup` with that step.
