#!/bin/bash
# Usage (as passed by jobsub_submit): run_lrs_vbr_pipeline.sh <manifest_pnfs_path> <output_dest_pnfs_dir> <ADCnum> <Channelnum> <MaxPulsenum> <voltage> <run> <window>

set -u             # treat unset variables as errors
set -o pipefail

INPUT_MANIFEST_PNFS="$1"
OUTPUT_DEST_PNFS="$2"
ADCNUM="$3"
CHANNELNUM="$4"
MAXPULSENUM="$5"
VOLTAGE="$6"
RUN="$7"
WINDOW="$8"

echo "=== Job starting on $(hostname) at $(date) ==="
echo "GRID_USER=${GRID_USER:-unset}  PROCESS=${PROCESS:-unset}"

cd "${_CONDOR_SCRATCH_DIR:?_CONDOR_SCRATCH_DIR is not set -- not running under condor?}"

stage_out() {
  local local_file="$1"
  local dest_dir="$2"
  local basename_f
  basename_f=$(basename "$local_file")

  ifdh cp -D "$local_file" "$dest_dir"
  local cp_status=$?

  if [ $cp_status -ne 0 ]; then
    echo "NOTE: ifdh cp reported exit status $cp_status for $basename_f -- verifying actual destination state" >&2
    if ifdh ls "${dest_dir}/${basename_f}" >/dev/null 2>&1; then
      echo "NOTE: ${basename_f} IS present at destination despite nonzero exit code -- treating as success" >&2
      return 0
    else
      echo "CONFIRMED: ${basename_f} is NOT present at destination -- real failure" >&2
      return $cp_status
    fi
  fi
  return 0
}

fail() {
  echo "FATAL: $1" >&2
  echo "--- Attempting to stage out any timing/diagnostic logs before exiting ---" >&2
  ifdh mkdir_p "$OUTPUT_DEST_PNFS" 2>/dev/null
  shopt -s nullglob
  for f in timing_step1_*.log timing_step2.log; do
    echo "  staging out $f" >&2
    stage_out "$f" "$OUTPUT_DEST_PNFS" || echo "  WARNING: could not stage out $f" >&2
  done
  shopt -u nullglob
  echo "=== Job FAILED at $(date) ==="
  exit 1
}

# Run a command under /usr/bin/time -v if available, capturing its verbose
# resource-usage report to $1; falls back gracefully if this container image
# doesn't ship /usr/bin/time. Propagates the wrapped command's exit status
# (including 128+N if killed by signal N -- e.g. 137 = SIGKILL/OOM).
run_with_timing() {
  local logfile="$1"; shift
  if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -v -o "$logfile" "$@"
  else
    echo "NOTE: /usr/bin/time not available in this container; timing log is minimal" > "$logfile"
    "$@"
    local rc=$?
    echo "exit_status=$rc" >> "$logfile"
    return $rc
  fi
}

# --- Minimal environment for ifdh (no dunesw/larsoft needed) ---
set +u
source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh || fail "could not source setup_dune.sh"
setup ifdhc || fail "could not set up ifdhc"
set -u

# --- Unpack + activate the packed Python environment shipped in the tarball ---
if [ ! -f "${INPUT_TAR_DIR_LOCAL}/myenv.tar.gz" ]; then
  fail "myenv.tar.gz not found at ${INPUT_TAR_DIR_LOCAL} -- check tarball contents"
fi
mkdir -p myenv
tar -xzf "${INPUT_TAR_DIR_LOCAL}/myenv.tar.gz" -C myenv || fail "could not unpack myenv.tar.gz"

set +u
source "myenv/bin/activate" || fail "could not activate packed venv"
set -u

export MPLBACKEND=Agg

python3 -c "import numpy, pandas, awkward, matplotlib, h5py, yaml, scipy, seaborn, statsmodels" \
  || fail "python environment is missing a required package"

SCRIPT_DIR="${INPUT_TAR_DIR_LOCAL}/Python_Scripts"

# --- Stage the manifest listing all HDF5 inputs for this run ---
LOCAL_MANIFEST="input_manifest.txt"
ifdh cp "$INPUT_MANIFEST_PNFS" "$LOCAL_MANIFEST" || fail "ifdh cp of input manifest failed"
[ -s "$LOCAL_MANIFEST" ] || fail "staged input manifest is missing or empty"

# --- Step 1: run once per HDF5 file listed in the manifest ---
STEP1_OUTPUTS=()
INDEX=0
while IFS= read -r HDF5_PNFS_PATH || [ -n "$HDF5_PNFS_PATH" ]; do
  [ -z "$HDF5_PNFS_PATH" ] && continue

  INDEX=$((INDEX + 1))
  LOCAL_HDF5="input_${INDEX}.hdf5"
  echo "--- Staging input file ${INDEX}: ${HDF5_PNFS_PATH} ---"
  ifdh cp "$HDF5_PNFS_PATH" "$LOCAL_HDF5" || fail "ifdh cp of ${HDF5_PNFS_PATH} failed"
  [ -s "$LOCAL_HDF5" ] || fail "staged file ${LOCAL_HDF5} is missing or empty"

  FILE_LIST="file_list_${INDEX}.txt"
  echo "$(pwd)/${LOCAL_HDF5}" > "$FILE_LIST"

  STEP1_OUT="step1_out_${INDEX}.npz"
  STEP1_TIMING_LOG="timing_step1_${INDEX}.log"
  echo "--- Step 1 (file ${INDEX}): Step1.py ---"
  run_with_timing "$STEP1_TIMING_LOG" python3 "${SCRIPT_DIR}/Step1.py" \
    --file_list "$FILE_LIST" \
    --output_file "$STEP1_OUT" \
    --ADCnum "$ADCNUM" \
    --Channelnum "$CHANNELNUM" \
    --MaxPulsenum "$MAXPULSENUM" \
    --voltage "$VOLTAGE" \
    --run "$RUN" \
    --window "$WINDOW"
  STATUS=$?
  echo "--- Step 1 (file ${INDEX}) resource summary: $(grep -E 'Maximum resident|Elapsed' "$STEP1_TIMING_LOG" 2>/dev/null | tr '\n' ' ') ---"
  echo "--- Disk usage after file ${INDEX}: $(du -sh . 2>/dev/null | cut -f1) ---"

  # Ship this file's timing log out now, not at the end -- survives even if a
  # LATER iteration gets killed by a resource limit
  ifdh mkdir_p "$OUTPUT_DEST_PNFS" 2>/dev/null
  stage_out "$STEP1_TIMING_LOG" "$OUTPUT_DEST_PNFS" || echo "WARNING: could not stage out $STEP1_TIMING_LOG" >&2

  [ $STATUS -eq 0 ] || fail "Step1.py exited with status $STATUS on file ${INDEX} (${HDF5_PNFS_PATH}) -- see ${STEP1_TIMING_LOG} (status 137 = killed by SIGKILL, most likely OOM)"
  [ -s "$STEP1_OUT" ] || fail "step 1 did not produce ${STEP1_OUT} for file ${INDEX}"

  STEP1_OUTPUTS+=("$STEP1_OUT")
  rm -f "$LOCAL_HDF5" "$FILE_LIST"
done < "$LOCAL_MANIFEST"

[ ${#STEP1_OUTPUTS[@]} -gt 0 ] || fail "no HDF5 files were processed -- check the manifest file"
echo "--- Step 1 complete: produced ${#STEP1_OUTPUTS[@]} npz file(s) ---"

# --- Step 2: fan-in over all step 1 npz outputs -> gain npz + pdf ---
STEP1_OUTPUT_MANIFEST="step1_output_manifest.txt"
: > "$STEP1_OUTPUT_MANIFEST"
for f in "${STEP1_OUTPUTS[@]}"; do
  echo "$(pwd)/${f}" >> "$STEP1_OUTPUT_MANIFEST"
done

echo "--- Step 2: Step2.py ---"
run_with_timing timing_step2.log python3 "${SCRIPT_DIR}/Step2.py" \
  --file_list "$STEP1_OUTPUT_MANIFEST" \
  --output_file step2_out.pdf \
  --gain_output step2_gain.npz \
  --adc "$ADCNUM" \
  --run "$RUN" \
  --bias_voltage "$VOLTAGE" \
  --window "$WINDOW"
STATUS=$?
echo "--- Step 2 resource summary: $(grep -E 'Maximum resident|Elapsed' timing_step2.log 2>/dev/null | tr '\n' ' ') ---"
[ $STATUS -eq 0 ] || fail "Step2.py exited with status $STATUS -- see timing_step2.log (status 137 = killed by SIGKILL, most likely OOM)"
[ -s step2_out.pdf ]   || fail "step 2 did not produce step2_out.pdf"
[ -s step2_gain.npz ]  || fail "step 2 did not produce step2_gain.npz"

# Step 3 not implemented yet -- intentionally omitted for now.

# --- Stage outputs out to dCache ---
echo "--- Staging outputs out to ${OUTPUT_DEST_PNFS} ---"
ifdh mkdir_p "$OUTPUT_DEST_PNFS" 2>/dev/null
stage_out step2_out.pdf   "$OUTPUT_DEST_PNFS" || fail "step2_out.pdf genuinely missing from destination after stage-out"
stage_out step2_gain.npz  "$OUTPUT_DEST_PNFS" || fail "step2_gain.npz genuinely missing from destination after stage-out"

echo "--- Staging out timing/diagnostic logs ---"
shopt -s nullglob
for f in timing_step1_*.log timing_step2.log; do
  stage_out "$f" "$OUTPUT_DEST_PNFS" || echo "WARNING: could not stage out $f" >&2
done
shopt -u nullglob

echo "=== Job completed successfully at $(date) ==="
exit 0