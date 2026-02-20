# FastTracker Development Guidelines for Claude

## ⚠️ CRITICAL DOCKER COMMANDS ⚠️

**NEVER use `docker compose restart` after rebuilding!**

After any code change (C++, Python, HTML, JavaScript):
```bash
# ✅ CORRECT:
docker compose build && docker compose down && docker compose up -d
# OR
docker compose build && docker compose up -d --force-recreate

# ❌ WRONG (doesn't use new image):
docker compose build && docker compose restart
```

**Why**: `restart` only restarts the existing container, it does NOT recreate it with the newly built image. Your changes will NOT be applied!

---

## 🚨 CRITICAL: FastTracker Requires GPU 🚨

**FastTracker is a GPU-accelerated application. It REQUIRES GPU to function properly.**

### Before Starting Container: Verify GPU Availability

```bash
# Check if host has GPU + CUDA driver:
nvidia-smi

# Expected output if GPU available:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 591.59       Driver Version: 591.59       CUDA Version: 13.1    |
# +-----------------------------------------------------------------------------+
# | GPU  Name                           ... |
# ...
```

### Mandatory Docker Run Flag

**✅ ALWAYS use `--gpus all` when starting FastTracker container:**

```bash
# CORRECT:
docker run -d --gpus all --name fasttracker -p 5001:5000 -v "$(pwd)/outputs:/app/outputs" fasttracker

# WRONG (container cannot access GPU → CUDA errors):
docker run -d --name fasttracker -p 5001:5000 -v "$(pwd)/outputs:/app/outputs" fasttracker
```

### Common Error Without GPU Flag

If you see this error:
```
CUDA error: CUDA driver version is insufficient for CUDA runtime version
```

**Before implementing CPU fallback or downgrading CUDA**:
1. ✅ Check `nvidia-smi` on host → If succeeds, GPU exists
2. ✅ Verify Docker run command includes `--gpus all`
3. ✅ Restart container with `--gpus all` flag

**Only implement CPU fallback if `nvidia-smi` fails on host.**

---

## 🚨 CRITICAL: NEVER Tell User "Changes Applied" Until After Rebuild 🚨

**WRONG Workflow ❌**:
1. Modify HTML/Python/C++ files
2. Tell user: "完了しました" or "変更を適用しました"
3. Start container ← **Container uses OLD image without changes!**
4. User reports: "なにも変化がない"

**CORRECT Workflow ✅**:
1. Modify files
2. **Rebuild Docker image**: `docker build -t fasttracker:latest .`
3. **Recreate container**: `docker stop X && docker rm X && docker run -d ...`
4. **Verify changes inside container**: `docker exec ... stat /app/...`
5. **ONLY THEN** tell user: "完了しました。ブラウザをリフレッシュしてください"

**Golden Rule**: If you modified ANY file that's copied into the Docker image (HTML, Python, C++, config), you MUST rebuild the image before telling the user changes are ready.

**Self-Check Before Responding to User**:
- [ ] Did I modify HTML/Python/C++ files?
- [ ] Did I rebuild the Docker image?
- [ ] Did I recreate the container (not just restart)?
- [ ] Did I verify the file timestamp inside container?
- [ ] Can I confidently say "changes are applied"?

If ANY checkbox is unchecked → **DO NOT tell user changes are ready!**

---

## 🚨 CRITICAL: Code Modification Workflow

When modifying C++ source code in this project, **ALWAYS** follow this exact workflow using TodoWrite:

### Mandatory Workflow (No Exceptions)

```
1. [in_progress] ソースコード修正を完了する
2. [pending] Docker強制リビルド (--no-cache) を実行
3. [pending] コンテナを完全再起動 (down && up)
4. [pending] バイナリ内容を検証（タイムスタンプ確認）
5. [pending] テストシミュレーション実行
6. [pending] 結果確認（CSV/ログで期待通りの動作を確認）
```

### Step-by-Step Instructions

#### Step 1: Source Code Modification
- Complete all code changes
- Use Read tool before Edit tool
- Mark as `completed` only when all files are modified

#### Step 2: Docker Rebuild (MANDATORY)

**For C++ source changes**:
```bash
docker compose build --no-cache 2>&1 | tee /tmp/docker_build.log | tail -30
```
- **ALWAYS use `--no-cache`** to avoid cache issues
- Verify build success: `grep "Built target fasttracker" /tmp/docker_build.log`

**For webapp-only changes (Python/HTML/JS)**:
```bash
docker compose build 2>&1 | tee /tmp/docker_build.log | tail -30
```
- No `--no-cache` needed (faster build)
- Still required because files are copied at build time

- Mark as `completed` only after confirming successful build

#### Step 3: Container Full Restart

**⚠️ CRITICAL: NEVER use `docker compose restart` ⚠️**

```bash
docker compose down && docker compose up -d
# OR
docker compose up -d --force-recreate
```

**Why `restart` doesn't work:**
- `docker compose restart` only restarts the EXISTING container (old image)
- It does NOT recreate the container with the newly built image
- Your code changes will NOT be reflected, even after rebuild

**Correct commands:**
- `docker compose down && docker compose up -d` - Stops and recreates container
- `docker compose up -d --force-recreate` - Forces container recreation

- Full down/up cycle to ensure new image is used
- Wait for container to start completely
- Mark as `completed` after container restart

#### Step 4: Binary Verification
```bash
docker exec fasttracker-fasttracker-1 ls -lh /app/fasttracker
docker exec fasttracker-fasttracker-1 stat -c '%y' /app/fasttracker
```
- Check timestamp is recent (within last few minutes)
- Optionally verify new code with `strings` command
- Mark as `completed` after timestamp verification

#### Step 5: Test Simulation

**IMPORTANT**: When working on TRACKER improvements, use the user's last simulation parameters unless explicitly instructed otherwise.

**How to find the user's last simulation parameters**:
1. Check CSV file timestamps: `ls -lth /home/aniah/FastTracker/*.csv | head -5`
2. Ask user: "What were the parameters of your last simulation?"
3. Check Web GUI session storage (if accessible)

**For TRACKER improvements, run simulation with user's parameters**:
```bash
# Example: Use parameters from user's last scenario
docker exec fasttracker-fasttracker-1 /app/fasttracker \
  --mode tracker \
  --scenario <user's scenario> \
  --duration <user's duration> \
  --framerate <user's framerate> \
  --launch-x <user's launch-x> --launch-y <user's launch-y> \
  --target-x <user's target-x> --target-y <user's target-y> \
  --sensor-x <user's sensor-x> --sensor-y <user's sensor-y> \
  --radar-max-range <user's max-range> \
  --false-alarm-rate <user's false-alarm-rate> \
  # ... other user-specified parameters
```

**For general testing (non-TRACKER improvements)**:
Run a test simulation with representative parameters:
```bash
docker exec fasttracker-fasttracker-1 /app/fasttracker \
  --mode tracker \
  --scenario single-ballistic \
  --duration 10 \
  --framerate 1 \
  --launch-x 0 --launch-y 0 \
  --target-x 20000 --target-y 20000 \
  --sensor-x 10000 --sensor-y 10000 \
  --radar-max-range 50000 \
  --false-alarm-rate 1e-7
```

- Mark as `completed` after simulation runs without errors
- For TRACKER improvements: **verify results match user's scenario expectations**

#### Step 6: Results Verification
```bash
docker exec fasttracker-fasttracker-1 cat /app/measurements.csv | head -20
docker exec fasttracker-fasttracker-1 cat /app/ground_truth.csv | head -10
```
- Verify CSV output matches expectations
- Check detection statistics in simulation output
- Confirm expected behavior (e.g., detections > 0, clutter generated)
- Mark as `completed` only after confirming expected results

### TodoWrite Usage Rules

1. **Create todo list at the START of any code modification task**
2. **Update status to `in_progress` before starting each step**
3. **Mark `completed` IMMEDIATELY after finishing each step**
4. **NEVER skip to next step until current step is marked `completed`**
5. **NEVER mark a step completed if verification fails**

### When This Workflow Applies

Execute this workflow whenever:
- Modifying C++ source files (`.cpp`, `.hpp`)
- Changing compilation flags or build configuration
- Adding new features that require testing
- Fixing bugs that need verification
- **TRACKER improvements**: tracking algorithm, data association, UKF parameters, etc.
  - **CRITICAL**: Always validate with user's last simulation parameters
  - Ask user for their last scenario if parameters are unclear
  - Compare results with user's expectations, not arbitrary test cases

### When to Skip Steps

**IMPORTANT**: Due to Docker containerization, most changes require rebuild.

You may skip the **full workflow** only for:
- Documentation updates (`.md` files) - no rebuild needed
- Configuration file changes (`.json`, `.yml`) that are read at runtime

**You CANNOT skip Steps 2-3 (rebuild + restart) for**:
- ✗ Python webapp changes (`app.py`, `index.html`) - **Rebuild required** (files copied at build time)
- ✗ C++ source changes - **Full workflow required** (compilation needed)
- ✗ Any file that's copied into Docker image - **Rebuild required**

### Web Application Changes (Python/HTML/JavaScript)

When modifying webapp files (`python/webapp/`):

**⚠️ CRITICAL: Use `down && up -d` or `--force-recreate`, NEVER just `restart` ⚠️**

**Required Steps**:
1. Make changes to files
2. **Rebuild container**: `docker compose build` (no `--no-cache` needed for webapp-only changes)
3. **Recreate container**: `docker compose down && docker compose up -d` or `docker compose up -d --force-recreate`
   - ❌ **DO NOT USE**: `docker compose restart` (doesn't use new image!)
   - ✅ **CORRECT**: `down && up -d` or `up -d --force-recreate`
4. **Clear browser cache**: Hard refresh (Ctrl+Shift+R) or open in incognito mode
5. Verify changes appear in browser

**Why rebuild is required**:
- Docker COPY instruction copies files at build time
- No volume mounts in docker-compose.yml (files are not live-mounted)
- Container restart alone does NOT update files inside container

**Quick command**:
```bash
docker compose build && docker compose down && docker compose up -d
```

**Verification**:
```bash
# Check file timestamp inside container
docker exec fasttracker-fasttracker-1 stat -c '%y' /app/python/webapp/templates/index.html

# Should be recent (within last few minutes)
```

## Finding User's Last Simulation Parameters

### Method 1: Check CSV File Timestamps
```bash
ls -lth /home/aniah/FastTracker/*.csv | head -10
```
- Most recent CSV files indicate last simulation run
- Typical files: `measurements.csv`, `ground_truth.csv`, `track_details.csv`, `results.csv`

### Method 2: Ask User Directly
When working on TRACKER improvements, always ask:
```
"最後のシミュレーションのパラメータを教えてください。特に以下を確認させてください：
- シナリオタイプ (scenario)
- 目標の発射位置と着弾位置 (launch/target coordinates)
- センサー位置 (sensor coordinates)
- レーダー最大探知距離 (max range)
- その他の重要なパラメータ"
```

### Method 3: Check Web GUI Logs
```bash
# Check recent container logs for parameter submissions
docker compose logs --tail 100 | grep -A 20 "runTracker"
```

### Method 4: Inspect CSV Content
```bash
# Ground truth reveals target trajectory parameters
head -20 /home/aniah/FastTracker/ground_truth.csv

# Results file may contain parameter summary
cat /home/aniah/FastTracker/results.csv
```

### What to Look For
Key parameters for TRACKER validation:
- **Scenario type**: ballistic, HGV, clustered, etc.
- **Target range**: distance from sensor to target
- **Target altitude**: especially for HGV scenarios
- **Sensor position**: affects detection geometry
- **Radar parameters**: max_range, FOV, antenna_boresight, search_elevation
- **Beam steering**: num_beams, beam_width, search_sector
- **Clutter settings**: false_alarm_rate
- **Tracking parameters**: Any custom UKF or IMM settings

### Default to User's Scenario
**Rule**: When user reports TRACKER issues (e.g., "誤警報が発生していない", "追尾リソースが割り当てられていない"), the fix MUST be validated with their exact scenario, not a generic test case.

## Common Issues and Solutions

### Issue: "Nothing has changed" after code modification
**Solution**: You forgot Step 2 (force rebuild) or Step 3 (full restart)
- Always run `--no-cache` rebuild for C++ changes
- Always do full `down && up` cycle
- For webapp changes: `docker compose build && docker compose down && docker compose up -d`

### Issue: "表示が変わっていない" (Web UI not updated after HTML/JS changes)
**Root Cause**: Container has old template files (Docker COPY happens at build time, not runtime)

**Solution**:
1. **Rebuild container**: `docker compose build` (copies new files into image)
2. **Restart container**: `docker compose down && docker compose up -d`
3. **Clear browser cache**: Hard refresh (Ctrl+Shift+R) or use incognito mode
4. **Verify**: `docker exec fasttracker-fasttracker-1 stat -c '%y' /app/python/webapp/templates/index.html`
   - Timestamp should be recent (within last few minutes)

**Why this happens**:
- Dockerfile uses `COPY python/webapp/ /app/python/webapp/` at build time
- No volume mounts → files are baked into image
- Simply restarting container does NOT update files

**Prevention**:
- Always rebuild after webapp changes: `docker compose build && docker compose up -d --force-recreate`
- Or add to docker-compose.yml for development:
  ```yaml
  volumes:
    - ./python/webapp:/app/python/webapp
  ```

### Issue: Tests pass locally but fail in container
**Solution**: Binary in container is outdated
- Verify Step 4 (binary timestamp)
- Re-run Steps 2-3

### Issue: Detection rate is 0%
**Root causes**:
1. Target outside sensor FOV (check antenna_boresight and FOV)
2. Target outside beam coverage (check beam steering setup)
3. SNR too low due to distance (check max_range and det_ref_range_m)
4. Search sector misconfigured (ensure search_center matches antenna_boresight)

## Project-Specific Context

### Architecture
- Multi-stage Docker build (builder + runtime)
- C++ backend with CUDA acceleration
- Python Flask web frontend
- CSV-based data output

### Key Files
- `src/simulation/radar_simulator.cpp` - Radar detection and clutter generation
- `src/main.cpp` - Main simulation loop and beam steering
- `python/webapp/app.py` - Web API and parameter handling
- `include/simulation/radar_simulator.hpp` - Radar parameter definitions

### Critical Parameters
- `antenna_boresight` - FOV center direction (atan2 coordinates: 0=East, π/2=North)
- `search_center` - Search beam center (should match antenna_boresight)
- `beam_directions_` - Active beam directions (empty = no beam steering)
- `snr_ref` - SNR at 1km reference distance (auto-computed via computeSnrRef())

### Detection Logic Flow
1. `isInFieldOfView()` - Check if target in FOV (antenna_boresight ± field_of_view/2)
2. `isOnBeam()` - Check if target in active beam (or return true if no beam steering)
3. `isDetectedSwerlingII()` - Check if SNR exceeds CFAR threshold

## 🚨 CRITICAL: Debugging Web GUI Parameter Issues

### NEVER Assume HTML Default Values are Current Settings

**WRONG Approach ❌**:
- Reading `index.html` and checking `value="0"` attributes
- Assuming default values in HTML template reflect current user settings
- Example mistake: Seeing `<input id="search-elevation" value="0">` and telling user "Search Elevation is 0°"

**CORRECT Approach ✅**:
1. **Ask user directly**: "現在のWeb GUIで設定されているパラメータ値を教えてください（Search Elevation、Auto Sectorなど）"
2. **Check actual simulation results**: Docker container's output files
3. **Inspect Docker logs**: Recent parameter submissions
4. **NEVER guess** based on HTML defaults

### File Location Confusion

**CRITICAL**: Web GUI simulations write to **Docker container**, not host filesystem!

**WRONG ❌**:
```bash
cat /home/aniah/FastTracker/results.csv  # This is OLD data from CLI runs
```

**CORRECT ✅**:
```bash
docker exec fasttracker-fasttracker-1 cat /app/results.csv  # Current Web GUI results
```

**File Locations**:
- **Host**: `/home/aniah/FastTracker/*.csv` - CLI simulation results (may be outdated)
- **Container**: `/app/*.csv` - Web GUI simulation results (current)
- **Always check container files** when debugging Web GUI issues

### Common Parameter Debugging Mistakes

#### Mistake 1: "Search Elevation is 0° because HTML says so"
```html
<!-- index.html default -->
<input id="search-elevation" value="0">
```
**Reality**: User may have changed it to 4° in the Web GUI
**Solution**: Ask user or check Docker logs

#### Mistake 2: "Auto Sector is ON because HTML says checked"
```html
<input id="auto-search-sector" checked>
```
**Reality**: User may have unchecked it
**Solution**: Ask user for current checkbox states

#### Mistake 3: "No new simulation ran because host CSV timestamp unchanged"
```bash
$ stat /home/aniah/FastTracker/results.csv
Modify: 2026-02-19 23:35:53  # Old timestamp
```
**Reality**: Web GUI writes to `/app/results.csv` inside container
**Solution**: Check `docker exec fasttracker-fasttracker-1 cat /app/results.csv`

#### Mistake 4: "I rebuilt the container but changes aren't applied"
```bash
# Wrong approach:
docker compose build
docker compose restart  # ❌ WRONG: Uses OLD container with OLD image
```
**Reality**: `docker compose restart` only restarts the existing container, it does NOT recreate it with the new image
**Solution**: ALWAYS use one of these after build:
```bash
docker compose down && docker compose up -d
# OR
docker compose up -d --force-recreate
```
**Verification**: Check file timestamp inside container
```bash
docker exec fasttracker-fasttracker-1 stat -c '%y' /app/python/webapp/templates/index.html
# Should be recent (within last few minutes)
```

### Verification Checklist for Parameter Issues

When user reports parameter-related issues (e.g., "Track Beams still 0"):

1. ✅ **Ask user**: "現在のパラメータ設定を確認させてください"
   - Search Elevation
   - Auto Sector checkbox state
   - Min/Max Elevation
   - Any other relevant parameters

2. ✅ **Check container files**:
   ```bash
   docker exec fasttracker-fasttracker-1 cat /app/results.csv | head -50
   docker exec fasttracker-fasttracker-1 cat /app/measurements.csv | wc -l
   docker exec fasttracker-fasttracker-1 cat /app/ground_truth.csv | head -20
   ```

3. ✅ **Verify target geometry**:
   ```bash
   # Calculate if search beams should detect target
   # Example: Target at el=0°, Search Elevation=4° → Detection fails!
   ```

4. ❌ **NEVER assume**:
   - HTML default values = current user settings
   - Host CSV files = latest Web GUI results
   - Auto-calculation functions are enabled (user may have disabled them)

### Real-World Example 1 (2026-02-20): Parameter Assumption Error

**User Issue**: "Track Beams still 0 after re-running simulation"

**My Mistake**:
- Read `index.html` line 802: `<input id="search-elevation" value="0">`
- Told user: "Search Elevation default is 0° (correct!)"
- **WRONG**: User had actually set it to **4°** in the Web GUI

**User Correction**:
- "Auto Sector is OFF, not ON"
- "Search Elevation is 4°, not 0°"
- "誤った情報を見ているのでは？" (Aren't you looking at wrong information?)

**Root Cause Analysis**:
- Target starts at elevation 0°, climbs to 1.56° by frame 50
- Search beams fixed at 4° elevation
- **Detection impossible** until target reaches ~3° (frame 70+)
- Result: Only 14 detections in 645 frames (2% detection rate)
- No confirmed tracks → No track beams allocated

**Correct Solution**:
- Ask user to set **Search Elevation = 0°**
- Verify with user before making claims about their settings

### Real-World Example 2 (2026-02-20): Forgot to Rebuild After HTML Changes

**User Issue**: "なにも変化がない" (Nothing has changed) after implementing Timeline altitude visualization

**My Mistake**:
1. Modified `index.html` to add altitude color coding to GT Lifetime background
2. Restarted Docker container with `docker run -d --name fasttracker ...`
3. **Told user changes were applied** ← **CRITICAL ERROR**
4. **FORGOT to rebuild Docker image** → Container still had old HTML file

**User Feedback**:
- "なにも変化がない" (Nothing has changed)

**Root Cause Analysis**:
- Modified `python/webapp/templates/index.html` on host filesystem
- Dockerfile uses `COPY python/webapp/ /app/python/webapp/` at **BUILD TIME**
- Started new container from **OLD IMAGE** (built before HTML changes)
- Container had old HTML file baked in → Changes invisible to user

**What Should Have Been Done**:
```bash
# CORRECT workflow:
docker stop fasttracker && docker rm fasttracker
docker build -t fasttracker:latest .               # ← REBUILD IMAGE
docker run -d --name fasttracker --gpus all -p 5000:5000 fasttracker:latest
```

**Lesson Learned**:
- **NEVER tell user changes are applied until AFTER rebuild + container recreation**
- Workflow for webapp changes:
  1. ✅ Modify HTML/Python files
  2. ✅ Rebuild Docker image: `docker build -t fasttracker:latest .`
  3. ✅ Recreate container: `docker stop X && docker rm X && docker run ...`
  4. ✅ Verify file timestamp: `docker exec ... stat /app/python/webapp/templates/index.html`
  5. ✅ **ONLY THEN** tell user changes are applied
- **Assume user will check immediately** → If rebuild is missing, user will report "no changes"
- **Prevention**: Always follow Steps 1-5 in order, mark each completed with TodoWrite

### Real-World Example 3 (2026-02-20): Missing `--gpus all` Flag → Wasted Effort on CPU Fallback

**User Issue**: "cudaのエラーが出た" → "CUDA driver version is insufficient for CUDA runtime version"

**My Initial Wrong Diagnosis**:
- Assumed: WSL2 environment has no GPU / CUDA driver too old
- Wasted effort implementing:
  1. GPU availability detection in C++ (`cudaGetDeviceCount()`)
  2. Automatic CPU-only fallback mode
  3. Error logging to `/tmp/fasttracker_gpu_error.log`
  4. Web UI error banner display
  5. Multiple Docker rebuilds (--no-cache)

**User's Critical Feedback**:
- "GPUが使えないと意味ないので使えるようにして" (If GPU can't be used, it's meaningless, so make it usable)

**Root Cause Analysis**:
- Host system **HAS GPU available**: NVIDIA RTX 4060, CUDA 13.1 driver (591.59)
- **Mistake**: Started container WITHOUT `--gpus all` flag:
  ```bash
  # ❌ WRONG (no GPU access):
  docker run -d --name fasttracker -p 5001:5000 -v "$(pwd)/outputs:/app/outputs" fasttracker

  # ✅ CORRECT (GPU enabled):
  docker run -d --gpus all --name fasttracker -p 5001:5000 -v "$(pwd)/outputs:/app/outputs" fasttracker
  ```
- Without `--gpus all`, container cannot access host GPU → CUDA initialization fails
- This is a **FastTracker GPU-accelerated application** → Without GPU, performance is severely degraded

**What Should Have Been Done**:
1. **ALWAYS check host GPU availability FIRST**:
   ```bash
   nvidia-smi  # Check if GPU + driver exist on host
   ```
   - If `nvidia-smi` succeeds → Host has GPU → **MUST use `--gpus all`**
   - If fails → No GPU → Then implement CPU fallback

2. **Default Docker run command for GPU apps**:
   ```bash
   docker run -d --gpus all --name fasttracker -p 5001:5000 -v "$(pwd)/outputs:/app/outputs" fasttracker
   ```

**Lesson Learned**:
- **🚨 CRITICAL: FastTracker is a GPU-accelerated application → ALWAYS start container with `--gpus all` 🚨**
- **Before implementing CPU fallback**: Check if host actually lacks GPU (run `nvidia-smi`)
- **Assumption failures**:
  - ❌ "WSL2 = no GPU" → WRONG! WSL2 supports CUDA via Windows driver
  - ❌ "CUDA error = driver too old" → WRONG! Error was because container couldn't access GPU
- **Correct diagnostic workflow**:
  1. User reports CUDA error
  2. Check `nvidia-smi` on host → If succeeds, GPU exists
  3. Check Docker run command → If missing `--gpus all`, add it
  4. **ONLY IF** no GPU on host → implement CPU fallback
- **Prevention checklist**:
  - [ ] Did I check `nvidia-smi` on host before assuming no GPU?
  - [ ] Did I verify Docker run command includes `--gpus all`?
  - [ ] Is this a GPU-dependent application? (FastTracker = YES)

**Standard Docker Run Command for FastTracker**:
```bash
# ✅ ALWAYS use this command (includes GPU support):
docker run -d \
  --gpus all \
  --name fasttracker \
  -p 5001:5000 \
  -v "$(pwd)/outputs:/app/outputs" \
  fasttracker

# Or with docker compose (ensure --gpus all equivalent in compose.yml):
docker compose up -d
```

## Automation Helper Scripts

### Full C++ Rebuild Script

For convenience, use this script for the complete workflow:

```bash
#!/bin/bash
# scripts/rebuild-and-test.sh

set -e

echo "=== Step 1: Force Rebuild ==="
docker compose build --no-cache

echo "=== Step 2: Restart Container ==="
docker compose down
docker compose up -d

echo "=== Step 3: Verify Binary ==="
docker exec fasttracker-fasttracker-1 stat -c '%y %n' /app/fasttracker

echo "=== Step 4: Run Test Simulation ==="
docker exec fasttracker-fasttracker-1 /app/fasttracker \
  --mode tracker --scenario single-ballistic --duration 10 \
  --framerate 1 --launch-x 0 --launch-y 0 \
  --target-x 20000 --target-y 20000 \
  --sensor-x 10000 --sensor-y 10000 \
  --radar-max-range 50000 --false-alarm-rate 1e-7

echo "=== Step 5: Check Results ==="
docker exec fasttracker-fasttracker-1 cat /app/measurements.csv | head -10

echo "=== All Steps Completed ==="
```

**Usage**: After C++ code modification, run `bash scripts/rebuild-and-test.sh`

### Quick Webapp Rebuild Script

For webapp-only changes (faster, no C++ compilation):

```bash
#!/bin/bash
# scripts/rebuild-webapp.sh

set -e

echo "=== Rebuilding Webapp (No C++ compilation) ==="
docker compose build

echo "=== Restarting Container ==="
docker compose down
docker compose up -d

echo "=== Verifying Template Update ==="
TEMPLATE_TIME=$(docker exec fasttracker-fasttracker-1 stat -c '%y' /app/python/webapp/templates/index.html)
echo "Template timestamp: $TEMPLATE_TIME"

echo "✅ Webapp Updated"
echo ""
echo "Next steps:"
echo "  1. Open http://localhost:5000"
echo "  2. Hard refresh browser (Ctrl+Shift+R)"
echo "  3. Verify changes appear"
echo ""
```

**Usage**: After HTML/Python changes, run `bash scripts/rebuild-webapp.sh`

---

**Remember**: These guidelines exist because skipping steps leads to wasted time debugging phantom issues. Always follow the workflow, always use TodoWrite, always verify results.
