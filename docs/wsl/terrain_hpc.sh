#!/usr/bin/env bash
# terrain_hpc.sh - terrain-s2 on a high-performance workstation, in WSL, without Docker: the Stage 2 chain, the
# ML dataset builder and (later) training, on the GPU. One script; run it again at any time (each mode is
# idempotent). Same conventions as the eddie_step scripts: a WSL mirror of GitHub that is never edited (git work
# stays in the Windows clone), data on the distro's own filesystem, a report you can tee to D:.
#
#   setup  (default)  check the machine, install micromamba, create the environment from environment.yml
#                     (conda-forge, with CuPy for the GPU), clone the mirror, install terrain-s2 into it, write
#                     ~/terrain/env.sh (TERRAIN_DATA_DIR, keys) and the 'terrain-hpc' command
#   update            fast-forward the mirror from GitHub; update the environment if environment.yml changed
#   check             versions, CPU, memory (and the WSL limit), disk, GPU (CuPy kernel), the test suite
#   labels <dir>      copy Matt's review files from a Windows folder (e.g. /mnt/d/eddie_data/labels/manual)
#   partition         build the national partition and the source coverage index on this machine (once, ~1 h)
#   batch <manifest> [run_tiles options]
#                     the batch runner in this terminal, workers sized from memory and cores (override with
#                     --workers N --gpu-workers N); logs to $TERRAIN_DATA_DIR/logs. Stops if the terminal closes.
#   start <manifest> [run_tiles options]
#                     the same, detached from the terminal, restarted after an interruption until every tile is
#                     done (at most 5 attempts), and keeping Windows awake while it runs; see 'status', 'stop'
#   status            progress (tiles done, failed, running), the last log lines, peak memory per tile, and
#                     out-of-memory kills if the kernel log shows any
#   stop              stop a detached batch
#   torch             install PyTorch (CUDA 12.8 build: Blackwell GPUs need it) into the environment, for ML-1b
#   channels <manifest...> [--workers N]  council channel lines snapped to the bed, then label rasters (ML-3a)
#   chips <name> <manifest...> [--workers N]  training chips for the channel model (ML-3a step 5)
#   unet <dataset> <name> [--folds ...]       train and evaluate the channel U-Net (ML-3b; needs 'torch')
#   predict <dataset> <name> <fold> [...]  the U-Net over whole tiles; --review for a review set
#   guard-setup                             allow the file-cache guard without a password (once)
#   check-runs <manifest...> [--stage S]    completeness and data issues of bundles and runs
#   dataset <manifest...> --name N
#                     build_labels + build_dataset for the manifest(s)
#   pipeline <plan.yml> [--from STEP | --only STEP]
#                     a plan of long steps (scripts/pipeline.py; e.g. ml/pipelines/weekend_2026_10_31.yml), detached,
#                     resumable, keeping Windows awake and the file cache emptied; 'pipeline-status', 'pipeline-stop'
#   pipeline-status   where the plan stands: each step's status, the running step's last log lines
#   pipeline-stop     stop the plan (a finished step stays done; start it again to carry on)
#
# Windows side, once (PowerShell, as yourself; section 1 of docs/wsl/terrain_hpc.md):
#   wsl --install -d Ubuntu-24.04 --name terrain        # or: wsl --install Ubuntu-24.04, then rename
#   wsl --manage terrain --move E:\WSL\terrain          # the distro's disk on the big data drive
#   and %UserProfile%\.wslconfig with memory= and processors= (WSL takes half the RAM by default)
# Then, inside the 'terrain' distro:
#   mkdir -p ~/terrain && tr -d '\r' < /mnt/d/EDDIE_Terrain/terrain-s2/docs/wsl/terrain_hpc.sh > ~/terrain/terrain_hpc.sh
#   bash ~/terrain/terrain_hpc.sh setup 2>&1 | tee /mnt/d/EDDIE/terrain_hpc_setup_report.txt
set -uo pipefail

HOME_T="$HOME/terrain"
TS2="$HOME_T/terrain-s2"                   # mirror of GitHub (main); never edited here
TS2_URL="${TS2_URL:-https://github.com/MatthewDWilson/terrain-s2.git}"
TS2_BRANCH="${TS2_BRANCH:-main}"
MAMBA="$HOME/.local/bin/micromamba"
ENVNAME=terrain-s2
ENVROOT="$HOME/micromamba"
DATA_DEFAULT="$HOME/terrain_data"          # on the distro's ext4 disk: much faster than /mnt/<drive>
ENVSH="$HOME_T/env.sh"
GB_PER_TILE=12                             # peak per LINZ tile with features (DESIGN_NOTES 7j, 11) plus margin

# cores: GNU nproc honours OMP_NUM_THREADS (2 per worker in env.sh), so it reported 2 on a 64-thread machine
cores() { getconf _NPROCESSORS_ONLN; }
gpu_mem_gb() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{printf "%d", $1/1024}'; }
GB_GPU_PER_WORKER=8                        # CuPy pool for a LINZ tile's feature stage, freed after it (pipeline)

say()  { printf '\n==== %s\n' "$1"; }
note() { printf '  %s\n' "$1"; }
die()  { printf '\nERROR: %s\n' "$1" >&2; exit 1; }

in_env() { "$MAMBA" run -r "$ENVROOT" -n "$ENVNAME" "$@"; }

load_env() {
    [ -f "$ENVSH" ] || die "run 'setup' first ($ENVSH missing)"
    # shellcheck disable=SC1090
    . "$ENVSH"
    data_dir_check
}

data_dir_check() {  # the data may live on their own ext4 disk (a VHDX attached with 'wsl --mount'; terrain_hpc.md
                    # section 5), linked from $TERRAIN_DATA_DIR. A WSL restart detaches it: attach it again through
                    # the logon task (Windows interop; the task runs elevated without a prompt), else stop rather
                    # than write into an empty folder on the system disk.
    if [ -L "$TERRAIN_DATA_DIR" ] && [ ! -e "$TERRAIN_DATA_DIR" ]; then
        if command -v schtasks.exe >/dev/null 2>&1; then
            note "data disk not attached: running the task 'WSL terrain_data disk'"
            schtasks.exe /run /tn "WSL terrain_data disk" >/dev/null 2>&1 || true
            local k
            for k in $(seq 1 30); do [ -e "$TERRAIN_DATA_DIR" ] && break; sleep 1; done
        fi
        [ -e "$TERRAIN_DATA_DIR" ] && { note "data disk attached ($(readlink "$TERRAIN_DATA_DIR"))"; return 0; }
        die "$TERRAIN_DATA_DIR links to $(readlink "$TERRAIN_DATA_DIR"), which is not attached (WSL restarted?).
       In PowerShell: schtasks /run /tn \"WSL terrain_data disk\"   (or, as administrator:
       wsl --mount --vhd <the data .vhdx> --name terrain_data), then run this again. terrain_hpc.md section 5."
    fi
}

preflight() {
    say "Machine"
    grep -qi microsoft /proc/version || note "WARNING: not WSL (fine on plain Linux, but the GPU notes assume WSL)"
    . /etc/os-release && note "OS: $PRETTY_NAME"
    note "CPU: $(cores) logical cores"
    local mem; mem=$(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo)
    note "memory visible to WSL: ${mem} GB"
    if [ "$mem" -lt 100 ]; then
        note "WARNING: WSL sees ${mem} GB. WSL takes half the host's RAM unless %UserProfile%\\.wslconfig sets"
        note "         [wsl2] memory=150GB (for 192 GB) and processors=64; then 'wsl --shutdown' and reopen."
    fi
    host_memory_check
    note "free disk in \$HOME: $(df -h --output=avail "$HOME" | tail -1 | tr -d ' ')"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  GPU: /'
    else
        note "WARNING: nvidia-smi not found. In WSL the NVIDIA driver is the Windows one (do not install a Linux"
        note "         driver); update the Windows driver, then 'wsl --shutdown'. CPU-only runs still work."
    fi
}

host_memory_check() {  # Windows' memory beside WSL's (16 Oct 2026: Windows ran out and stopped the VM)
    command -v powershell.exe >/dev/null 2>&1 || return 0
    local v ram commit wsl head
    # shellcheck disable=SC2016
    v=$(powershell.exe -NoProfile -NonInteractive -Command '$o = Get-CimInstance Win32_OperatingSystem; "$([math]::Round($o.TotalVisibleMemorySize/1MB)) $([math]::Round($o.TotalVirtualMemorySize/1MB))"' 2>/dev/null | tr -d '\r')
    read -r ram commit <<< "$v"
    [[ "$ram" =~ ^[0-9]+$ && "$commit" =~ ^[0-9]+$ ]] || return 0
    wsl=$(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo)
    head=$(( commit - wsl ))
    note "Windows: ${ram} GB RAM, page file $(( commit - ram )) GB; WSL may use ${wsl} GB, leaving Windows ${head} GB"
    if [ "$head" -lt 32 ]; then
        note "WARNING: Windows keeps only ${head} GB beside WSL. When Linux fills its memory (the file cache does, while"
        note "         rasters are written), Windows runs out and stops the WSL VM. Lower memory= in .wslconfig (e.g."
        note "         150GB for 192 GB of RAM), add a page file (System > Advanced > Performance > Virtual memory), or both."
    fi
}

DROP_RULE=/etc/sudoers.d/terrain-dropcache

cache_guard() {  # empty Linux's file cache every minute while process $1 runs; WSL then returns that memory to Windows
                 # (16 Oct: the batch's written rasters; 17 Oct: the U-Net's memory-mapped chips). With the sudoers
                 # rule of 'guard-setup' this works in detached runs too (no password needed).
    local sup="$1"
    if sudo -n -l /usr/bin/tee /proc/sys/vm/drop_caches >/dev/null 2>&1; then
        setsid nohup sh -c "while kill -0 $sup 2>/dev/null; do sync; echo 1 | sudo -n /usr/bin/tee /proc/sys/vm/drop_caches >/dev/null; sleep 60; done" \
            >/dev/null 2>&1 < /dev/null &
        note "file cache emptied every minute while this runs"
    elif [ -t 0 ] && { sudo -n true 2>/dev/null || sudo -v; }; then
        sudo -b sh -c "while kill -0 $sup 2>/dev/null; do sync; echo 1 > /proc/sys/vm/drop_caches; sleep 60; done" \
            && note "file cache emptied every minute while this runs"
    else
        note "WARNING: the file cache is not emptied (no sudo here): Windows may run short of memory and stop WSL."
        note "         Run 'bash ~/terrain/terrain_hpc.sh guard-setup' once (asks for your password) to allow it."
    fi
}

guard_setup() {  # one sudoers rule: this user may write /proc/sys/vm/drop_caches without a password
    local rule
    rule="$(id -un) ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches"
    echo "$rule" | sudo tee "$DROP_RULE.tmp" >/dev/null && sudo chmod 440 "$DROP_RULE.tmp" \
        && sudo visudo -cf "$DROP_RULE.tmp" >/dev/null && sudo mv "$DROP_RULE.tmp" "$DROP_RULE" \
        || die "could not install $DROP_RULE"
    note "installed $DROP_RULE: $rule"
}

setup() {
    preflight
    say "System packages"
    sudo apt-get update -qq && sudo apt-get install -y -qq git curl ca-certificates bzip2 >/dev/null \
        && note "git, curl, bzip2: ok" || die "apt-get failed"
    say "micromamba"
    if [ ! -x "$MAMBA" ]; then
        mkdir -p "$HOME/.local/bin"
        curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$HOME/.local" bin/micromamba \
            || die "micromamba download failed"
    fi
    note "$("$MAMBA" --version)"
    say "Mirror of terrain-s2 ($TS2_BRANCH)"
    mkdir -p "$HOME_T"
    if [ ! -d "$TS2/.git" ]; then
        git clone -q --branch "$TS2_BRANCH" "$TS2_URL" "$TS2" || die "clone failed (GitHub access?)"
    fi
    sync_mirror
    say "Environment $ENVNAME (conda-forge; a few minutes the first time)"
    if "$MAMBA" env list -r "$ENVROOT" | grep -qE "^\s*$ENVNAME\s"; then
        "$MAMBA" install -r "$ENVROOT" -n "$ENVNAME" -y -f "$TS2/environment.yml" >/dev/null || die "update failed"
    else
        "$MAMBA" create -r "$ENVROOT" -n "$ENVNAME" -y -f "$TS2/environment.yml" >/dev/null || die "create failed"
    fi
    # CuPy from conda-forge brings its own CUDA runtime; libcuda comes from the Windows driver (/usr/lib/wsl/lib)
    if command -v nvidia-smi >/dev/null 2>&1; then
        "$MAMBA" install -r "$ENVROOT" -n "$ENVNAME" -y -c conda-forge cupy "cuda-version>=12,<13" >/dev/null \
            && note "CuPy installed" || note "WARNING: CuPy install failed; the GPU feature stage will fall back to CPU"
    fi
    in_env python -m pip install -q --no-deps -e "$TS2" || die "pip install -e failed"
    sha256sum "$TS2/environment.yml" | cut -d' ' -f1 > "$HOME_T/.environment.sha"
    say "Settings ($ENVSH)"
    if [ ! -f "$ENVSH" ]; then
        cat > "$ENVSH" <<EOF
# terrain-s2 settings for this machine (sourced by terrain_hpc.sh and the 'terrain-hpc' command)
export TERRAIN_DATA_DIR="\${TERRAIN_DATA_DIR:-$DATA_DEFAULT}"
export LINZ_API_KEY="\${LINZ_API_KEY:-}"           # set it here (Topo50 WFS sources)
export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-2}      # per worker process; workers run in parallel
export NUMBA_NUM_THREADS=\${NUMBA_NUM_THREADS:-2}
EOF
        note "written; put your LINZ_API_KEY in it"
    else
        note "kept (existing)"
    fi
    load_env
    mkdir -p "$TERRAIN_DATA_DIR"/{labels/manual,logs,partition}
    note "TERRAIN_DATA_DIR=$TERRAIN_DATA_DIR"
    mkdir -p "$HOME/.local/bin"
    cat > "$HOME/.local/bin/terrain-hpc" <<EOF
#!/usr/bin/env bash
# run a command in the terrain-s2 environment with this machine's settings, from the mirror
. "$ENVSH"; cd "$TS2" && exec "$MAMBA" run -r "$ENVROOT" -n "$ENVNAME" "\$@"
EOF
    chmod +x "$HOME/.local/bin/terrain-hpc"
    note "'terrain-hpc python scripts/...' runs anything in the environment (add ~/.local/bin to PATH)"
    check
}

sync_mirror() {
    git -C "$TS2" fetch -q origin "$TS2_BRANCH" || die "fetch failed"
    git -C "$TS2" checkout -q "$TS2_BRANCH" && git -C "$TS2" merge -q --ff-only "origin/$TS2_BRANCH" \
        || die "the mirror has diverged from GitHub: it must not be edited (re-clone it)"
    note "mirror at $(git -C "$TS2" log -1 --format='%h %cs %s' | cut -c1-100)"
}

update() {
    say "Update"
    sync_mirror
    # this script itself: the copy in ~/terrain is what runs, so refresh it from the mirror (a new file, so the
    # running shell keeps reading the old one); before 15 Oct 2026 'update' left it as installed
    tr -d '\r' < "$TS2/docs/wsl/terrain_hpc.sh" > "$HOME_T/terrain_hpc.sh.new" && mv "$HOME_T/terrain_hpc.sh.new" "$HOME_T/terrain_hpc.sh" \
        && note "script refreshed from the mirror ($HOME_T/terrain_hpc.sh)"
    local now; now=$(sha256sum "$TS2/environment.yml" | cut -d' ' -f1)
    if [ "$now" != "$(cat "$HOME_T/.environment.sha" 2>/dev/null)" ]; then
        note "environment.yml changed: updating the environment"
        "$MAMBA" install -r "$ENVROOT" -n "$ENVNAME" -y -f "$TS2/environment.yml" >/dev/null || die "update failed"
        echo "$now" > "$HOME_T/.environment.sha"
    fi
    in_env python -m pip install -q --no-deps -e "$TS2" || die "pip install -e failed"
    note "terrain-s2 $(in_env python -c 'import terrain_s2; print(terrain_s2.__version__)')"
}

check() {
    load_env
    say "Versions"
    in_env python -c 'import terrain_s2, numpy, rasterio, geopandas, pyflwdir; print("  terrain-s2", terrain_s2.__version__,
"| numpy", numpy.__version__, "| rasterio", rasterio.__version__, "| geopandas", geopandas.__version__, "| pyflwdir", pyflwdir.__version__)'
    say "GPU"
    in_env python - <<'EOF' || echo "  GPU: not usable (CPU only)"
try:
    import cupy as cp
    a = cp.arange(1_000_000, dtype=cp.float64)
    ok = float(a.sum()) == 499999500000.0
    name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"  CuPy {cp.__version__}: {name}; kernel {'ok' if ok else 'WRONG'}")
except Exception as e:
    print(f"  GPU: not usable ({type(e).__name__}: {e})"); raise SystemExit(1)
EOF
    say "Tests"
    (cd "$TS2" && in_env python -m pytest -q -p no:cacheprovider 2>&1 | tail -3)
    preflight
}

workers_for() {  # sized from memory: GB_PER_TILE per worker, 85 % of what WSL sees; at most cores / 2
    local mem cores w
    mem=$(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo)
    cores=$(cores)
    w=$(( mem * 85 / 100 / GB_PER_TILE ))
    [ "$w" -gt $(( cores / 2 )) ] && w=$(( cores / 2 ))
    [ "$w" -lt 1 ] && w=1
    echo "$w"
}

batch() {
    load_env
    local manifest="${1:-}"; shift || true
    [ -n "$manifest" ] || die "usage: batch <manifest> [run_tiles options]"
    local part="$TERRAIN_DATA_DIR/partition/nz_tiles.gpkg"
    [ -f "$part" ] || die "no partition at $part (run 'partition', or copy nz_tiles.gpkg there)"
    if [ "$(cores)" -lt 8 ]; then
        note "WARNING: WSL sees only $(cores) cores. Set processors= in %UserProfile%\\.wslconfig to the machine's"
        note "         logical processors (e.g. 32 or 64), run 'wsl --shutdown' in PowerShell, and reopen."
    fi
    local extra; mapfile -t extra < <(sizing "$@")
    note "options: ${extra[*]} (from $(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo) GB, $(cores) cores, $(gpu_mem_gb) GB of GPU memory)"
    note "in this terminal: closing it stops the batch ('start' runs it detached)"
    local log; log="$TERRAIN_DATA_DIR/logs/batch_$(date +%Y%m%d_%H%M%S).log"
    say "Batch: $manifest -> $log"
    (cd "$TS2" && in_env python scripts/run_tiles.py --manifest "$manifest" --partition "$part" "${extra[@]}") 2>&1 | tee "$log"
}

RUN="$HOME_T/run"                           # pid files and the supervisor log of a detached batch

sizing() {  # the run_tiles options for "$@" (workers, GPU workers), with a memory check
    local extra=("$@") w=0 g mem i
    mem=$(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo)
    if [[ " ${extra[*]} " != *" --workers "* ]]; then
        w=$(workers_for); g=0
        if command -v nvidia-smi >/dev/null 2>&1; then g=$(( $(gpu_mem_gb) / GB_GPU_PER_WORKER )); fi
        [ "$g" -gt "$w" ] && g=$w
        extra+=(--workers "$w" --gpu-workers "$g")
    else
        for i in "${!extra[@]}"; do [ "${extra[$i]}" = "--workers" ] && w="${extra[$((i + 1))]}"; done
    fi
    if [ $(( w * GB_PER_TILE )) -gt $(( mem * 85 / 100 )) ]; then
        echo "  WARNING: $w workers x ~$GB_PER_TILE GB is more than 85 % of the ${mem} GB WSL has: a worker (or the batch)" >&2
        echo "           may be killed for memory. $(( mem * 85 / 100 / GB_PER_TILE )) workers fit." >&2
    fi
    printf '%s\n' "${extra[@]}"
}

keep_awake() {  # Windows stays awake while this runs (WSL interop; nothing happens without it)
    command -v powershell.exe >/dev/null 2>&1 || return 0
    # shellcheck disable=SC2016
    powershell.exe -NoProfile -NonInteractive -Command '$s = Add-Type -MemberDefinition "[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint f);" -Name K -Namespace W -PassThru; while ($true) { [void]$s::SetThreadExecutionState(0x80000001); Start-Sleep 60 }' \
        >/dev/null 2>&1 &
    echo $! > "$RUN/awake.pid"
}

start() {
    load_env
    local manifest="${1:-}"; shift || true
    [ -n "$manifest" ] && [ -f "$manifest" ] || die "usage: start <manifest> [run_tiles options]"
    mkdir -p "$RUN"
    if [ -f "$RUN/batch.pid" ] && kill -0 "$(cat "$RUN/batch.pid")" 2>/dev/null; then
        die "a batch is already running (pid $(cat "$RUN/batch.pid")): 'status' or 'stop'"
    fi
    host_memory_check
    local opts; mapfile -t opts < <(sizing "$@")
    printf '%s\n' "$manifest" "${opts[@]}" > "$RUN/batch.args"
    setsid nohup bash "$0" _supervise >> "$RUN/supervisor.log" 2>&1 < /dev/null &
    echo $! > "$RUN/batch.pid"
    cache_guard "$(cat "$RUN/batch.pid")"
    say "Started (pid $(cat "$RUN/batch.pid")): $manifest ${opts[*]}"
    note "detached: closing this terminal does not stop it. 'status' for progress, 'stop' to end it."
    note "supervisor log: $RUN/supervisor.log; batch logs: $TERRAIN_DATA_DIR/logs/"
}

_supervise() {  # (internal) run the batch until every tile is done, at most 5 attempts
    load_env
    local args manifest attempt rc log
    mapfile -t args < "$RUN/batch.args"
    manifest="${args[0]}"
    keep_awake
    for attempt in 1 2 3 4 5; do
        echo "$(date '+%F %T') attempt $attempt: $manifest ${args[*]:1}"
        log="$TERRAIN_DATA_DIR/logs/batch_$(date +%Y%m%d_%H%M%S).log"
        (cd "$TS2" && in_env python scripts/run_tiles.py --manifest "$manifest" \
            --partition "$TERRAIN_DATA_DIR/partition/nz_tiles.gpkg" "${args[@]:1}") > "$log" 2>&1
        rc=$?
        echo "$(date '+%F %T') attempt $attempt ended: exit $rc; $(tail -1 "$log")"
        oom_report | sed 's/^/    /'
        if [ "$rc" -eq 0 ] || grep -q " 0 to run," "$log"; then break; fi   # all done (exit 1 = some tiles failed)
        sleep 60
    done
    [ -f "$RUN/awake.pid" ] && kill "$(cat "$RUN/awake.pid")" 2>/dev/null
    rm -f "$RUN/batch.pid" "$RUN/awake.pid"
    echo "$(date '+%F %T') finished"
}

oom_report() {
    local k
    k=$( { dmesg 2>/dev/null || sudo -n dmesg 2>/dev/null; } | grep -iE "out of memory|oom-kill|killed process" | tail -5)
    [ -n "$k" ] && echo "kernel: $k"
    echo "memory now: $(free -g | awk '/Mem:/ {print $3 " GB used, " $7 " GB available"}')"
}

status() {
    load_env
    local manifest="" last
    [ -f "$RUN/batch.args" ] && manifest=$(head -1 "$RUN/batch.args")
    say "Batch status"
    if [ -f "$RUN/batch.pid" ] && kill -0 "$(cat "$RUN/batch.pid")" 2>/dev/null; then
        note "running (supervisor pid $(cat "$RUN/batch.pid")): $manifest"
    else
        note "not running"
    fi
    if [ -n "$manifest" ] && [ -f "$manifest" ]; then
        local n ndone nfail t
        n=$(( $(wc -l < "$manifest") - 1 )); ndone=0; nfail=0
        while IFS=, read -r t _; do
            [ -f "$TERRAIN_DATA_DIR/runs/$t/done.json" ] && ndone=$((ndone + 1))
            [ -f "$TERRAIN_DATA_DIR/runs/$t/failed.json" ] && nfail=$((nfail + 1))
        done < <(tail -n +2 "$manifest")
        note "tiles: $ndone of $n done; $nfail with failed.json (retried by the next attempt)"
        note "finished since this batch started: $(find "$TERRAIN_DATA_DIR/runs" -mindepth 2 -maxdepth 2 -name done.json \
            -newer "$RUN/batch.args" 2>/dev/null | wc -l) (a rerun with new parameters leaves the old done.json until then)"
    fi
    last=$(ls -t "$TERRAIN_DATA_DIR"/logs/batch_*.log 2>/dev/null | head -1)
    [ -n "$last" ] && { note "last lines of $last:"; tail -5 "$last" | sed 's/^/    /'; }
    in_env python -c '
import json, sys
from pathlib import Path
p = sorted(x for x in (json.loads(f.read_text()).get("peak_rss_gb") for f in Path(sys.argv[1]).glob("*/done.json")) if x)
if p:
    print(f"  peak memory per tile: median {p[len(p) // 2]:.1f} GB, max {p[-1]:.1f} GB ({len(p)} tiles)")
' "$TERRAIN_DATA_DIR/runs"
    oom_report | sed 's/^/  /'
    [ -f "$RUN/supervisor.log" ] && { note "supervisor:"; tail -4 "$RUN/supervisor.log" | sed 's/^/    /'; }
}

stop() {
    if [ -f "$RUN/batch.pid" ] && kill -0 "$(cat "$RUN/batch.pid")" 2>/dev/null; then
        kill -- -"$(cat "$RUN/batch.pid")" 2>/dev/null || kill "$(cat "$RUN/batch.pid")"
        [ -f "$RUN/awake.pid" ] && kill "$(cat "$RUN/awake.pid")" 2>/dev/null
        rm -f "$RUN/batch.pid" "$RUN/awake.pid"
        note "stopped; finished tiles are kept, and 'start' resumes"
    else
        note "no batch running"
    fi
}

torch_install() {
    say "PyTorch (CUDA 12.8 build) for the patch CNN"
    in_env python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128 || die "pip install torch failed"
    in_env python -c 'import torch; print("  torch", torch.__version__, "| CUDA", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'
}

dataset() {
    load_env
    local manifests=() name=""
    while [ $# -gt 0 ]; do
        case "$1" in --name) name="$2"; shift 2 ;; *) manifests+=("$1"); shift ;; esac
    done
    [ -n "$name" ] && [ ${#manifests[@]} -gt 0 ] || die "usage: dataset <manifest...> --name N"
    say "Labels and dataset $name"
    (cd "$TS2" && for m in "${manifests[@]}"; do in_env python scripts/build_labels.py --manifest "$m"; done \
        && in_env python scripts/build_dataset.py --manifest "${manifests[@]}" --name "$name")
}

channels() {  # ML-3a steps 3-4: council channel lines snapped to the bed, then the label rasters (tiles with rasters)
    load_env
    [ $# -gt 0 ] || die "usage: channels <manifest...> [--workers N] [--rerun]"
    host_memory_check
    cache_guard $$
    local manifests=() opts=()
    while [ $# -gt 0 ]; do case "$1" in --*) opts+=("$@"); break ;; *) manifests+=("$1"); shift ;; esac; done
    (cd "$TS2" && in_env python scripts/snap_channels.py --manifest "${manifests[@]}" "${opts[@]}" \
        && in_env python scripts/build_channel_labels.py --manifest "${manifests[@]}" "${opts[@]}")
}

check_runs() {  # completeness and data issues of a manifest's bundles and runs (scripts/check_runs.py)
    load_env
    [ $# -gt 0 ] || die "usage: check-runs <manifest...> [--stage batch|channels|chips] [--dataset NAME] [--list ISSUE]"
    local manifests=() opts=()
    while [ $# -gt 0 ]; do case "$1" in --*) opts+=("$@"); break ;; *) manifests+=("$1"); shift ;; esac; done
    (cd "$TS2" && in_env python scripts/check_runs.py --manifest "${manifests[@]}" "${opts[@]}")
}

unet() {  # ML-3b: train and evaluate the channel U-Net (needs PyTorch: the 'torch' mode)
    load_env
    [ $# -ge 2 ] || die "usage: unet <dataset> <name> [train_unet options: --folds ..., --subsets ..., --epochs N]"
    host_memory_check
    cache_guard $$
    local ds="$1" name="$2"; shift 2
    (cd "$TS2" && in_env python scripts/train_unet.py --dataset "$ds" --name "$name" "$@")
}

predict() {  # ML-3b: the channel U-Net over whole tiles, and review sets (scripts/predict_channels.py)
    load_env
    [ $# -ge 3 ] || die "usage: predict <dataset> <name> <fold> [--pick COUNCIL... | --tiles T...] [--review]"
    local ds="$1" name="$2" fold="$3"; shift 3
    host_memory_check
    cache_guard $$
    (cd "$TS2" && in_env python scripts/predict_channels.py --dataset "$ds" --name "$name" --fold "$fold" "$@")
}

chips() {  # ML-3a step 5: training chips from the saved rasters and label rasters
    load_env
    [ $# -ge 2 ] || die "usage: chips <dataset name> <manifest...> [--workers N] [--per-tile N] [--rerun]"
    host_memory_check
    cache_guard $$
    local name="$1"; shift
    local manifests=() opts=()
    while [ $# -gt 0 ]; do case "$1" in --*) opts+=("$@"); break ;; *) manifests+=("$1"); shift ;; esac; done
    (cd "$TS2" && in_env python scripts/build_chips.py --name "$name" --manifest "${manifests[@]}" "${opts[@]}")
}

pipeline() {  # a plan of long steps, detached (scripts/pipeline.py)
    load_env
    local plan="${1:-}"; shift || true
    [ -n "$plan" ] || die "usage: pipeline <plan.yml> [--from STEP | --only STEP]"
    case "$plan" in /*) ;; *) plan="$TS2/$plan" ;; esac
    [ -f "$plan" ] || die "no plan at $plan"
    mkdir -p "$RUN"
    for f in batch.pid pipeline.pid; do
        if [ -f "$RUN/$f" ] && kill -0 "$(cat "$RUN/$f")" 2>/dev/null; then
            die "already running ($f: pid $(cat "$RUN/$f")): 'status', 'pipeline-status', 'stop' or 'pipeline-stop'"
        fi
    done
    host_memory_check
    printf '%s\n' "$plan" "$@" > "$RUN/pipeline.args"
    (cd "$TS2" && in_env python scripts/pipeline.py --plan "$plan" --dry-run) | sed 's/^/  /' | head -60
    setsid nohup bash "$0" _pipeline >> "$RUN/pipeline.log" 2>&1 < /dev/null &
    echo $! > "$RUN/pipeline.pid"
    cache_guard "$(cat "$RUN/pipeline.pid")"
    say "Started the plan (pid $(cat "$RUN/pipeline.pid")): $plan $*"
    note "detached: closing this terminal does not stop it. 'pipeline-status' for progress, 'pipeline-stop' to end it."
    note "log: $RUN/pipeline.log; each step's log and the summary: $TERRAIN_DATA_DIR/pipelines/<plan name>/"
}

_pipeline() {  # (internal) run the plan
    load_env
    local args; mapfile -t args < "$RUN/pipeline.args"
    keep_awake
    echo "$(date '+%F %T') plan ${args[*]}"
    (cd "$TS2" && in_env python scripts/pipeline.py --plan "${args[0]}" "${args[@]:1}")
    echo "$(date '+%F %T') plan ended: exit $?"
    [ -f "$RUN/awake.pid" ] && kill "$(cat "$RUN/awake.pid")" 2>/dev/null
    rm -f "$RUN/pipeline.pid" "$RUN/awake.pid"
}

pipeline_status() {
    load_env
    local plan name st
    plan=$(head -1 "$RUN/pipeline.args" 2>/dev/null) || die "no plan has been started"
    name=$(awk '/^name:/ {print $2; exit}' "$plan")
    st="$TERRAIN_DATA_DIR/pipelines/$name/status.json"
    if [ -f "$RUN/pipeline.pid" ] && kill -0 "$(cat "$RUN/pipeline.pid")" 2>/dev/null; then
        say "Plan $name: running (pid $(cat "$RUN/pipeline.pid"))"
    else
        say "Plan $name: not running"
    fi
    [ -f "$st" ] || { note "no status yet"; return; }
    in_env python - "$st" << 'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"  updated {s['updated']}; {s.get('free_gb')} GB free; running: {s.get('current') or '-'}")
for n, v in s["steps"].items():
    print(f"  {n:22s} {v.get('status'):12s} {v.get('minutes', '') or ''} {v.get('reason', '') or ''}")
cur = s.get("current")
if cur:
    lines = open(s["steps"][cur]["log"], errors="replace").read().splitlines()[-8:]
    print("\n  last lines of " + cur + ":\n    " + "\n    ".join(lines))
PY
}

pipeline_stop() {
    load_env
    [ -f "$RUN/pipeline.pid" ] || die "no plan is running"
    local pid; pid=$(cat "$RUN/pipeline.pid")
    pkill -TERM -s "$pid" 2>/dev/null || kill "$pid" 2>/dev/null
    [ -f "$RUN/awake.pid" ] && kill "$(cat "$RUN/awake.pid")" 2>/dev/null
    rm -f "$RUN/pipeline.pid" "$RUN/awake.pid"
    say "Stopped the plan (pid $pid); finished steps stay done: 'pipeline' with the same plan carries on"
}

labels() {
    load_env
    local src="${1:-}"
    [ -d "$src" ] || die "usage: labels <folder with the review files> (e.g. /mnt/d/eddie_data/labels/manual)"
    cp -v "$src"/*.geojson "$TERRAIN_DATA_DIR/labels/manual/" | sed 's/^/  /'
}

partition() {
    load_env
    say "Partition and coverage index (once per machine; or copy nz_tiles.gpkg from the other workstation)"
    local part="$TERRAIN_DATA_DIR/partition/nz_tiles.gpkg"
    (cd "$TS2" && in_env python scripts/partition.py --out "$part" \
        && in_env python scripts/tile_pool.py --partition "$part")
    note "the coverage index (sites/sources_coverage.geojson) comes with the repository; rebuild it with"
    note "terrain-hpc python scripts/index_sources.py --partition $part   (then commit it from the Windows clone)"
}

mode="${1:-setup}"; shift || true
case "$mode" in
    setup) setup ;;
    update) update ;;
    check) check ;;
    batch) batch "$@" ;;
    start) start "$@" ;;
    _supervise) _supervise ;;
    status) status ;;
    torch) torch_install ;;
    stop) stop ;;
    dataset) dataset "$@" ;;
    channels) channels "$@" ;;
    chips) chips "$@" ;;
    unet) unet "$@" ;;
    predict) predict "$@" ;;
    guard-setup) guard_setup ;;
    check-runs) check_runs "$@" ;;
    labels) labels "$@" ;;
    partition) partition ;;
    pipeline) pipeline "$@" ;;
    _pipeline) _pipeline ;;
    pipeline-status) pipeline_status ;;
    pipeline-stop) pipeline_stop ;;
    *) die "unknown mode '$mode' (setup | update | check | labels | partition | batch | start | status | stop | torch | dataset)" ;;
esac
