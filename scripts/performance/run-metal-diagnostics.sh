#!/usr/bin/env bash
set -euo pipefail

case "${BENCHMARK_PROFILE:-}" in
    prism|nova) ;;
    *) echo 'BENCHMARK_PROFILE must be prism or nova' >&2; exit 2 ;;
esac
mkdir -p benchmark-diagnostics

system_snapshot() {
    date -u
    vm_stat
    sysctl vm.swapusage
    pmset -g therm
}

run_case() {
    local revision=$1 label=$2 steps=$3 warmup=$4 recording=$5
    local output="benchmark-diagnostics/$BENCHMARK_PROFILE-$label"
    local options=()
    if [[ "$recording" == yes ]]; then options+=("-Ddeepj.benchmarkRecording=$output.jfr"); fi
    system_snapshot > "$output-system-before.txt"
    /usr/bin/time -l java -Xmx3g -XX:+ExitOnOutOfMemoryError \
        "-Xlog:gc*,safepoint:file=$output-gc.txt:time,uptime,level,tags" \
        -Ddeepj.benchmarkVariedData=true "${options[@]}" --enable-native-access=ALL-UNNAMED \
        -cp "$revision/target/classes:candidate/target/test-classes" \
        io.github.kirstenali.deepj.testing.AutogradTrainingBenchmark \
        "$BENCHMARK_PROFILE-probe" 1 "$steps" "$warmup" > "$output.txt" 2>&1
    system_snapshot > "$output-system-after.txt"
    tail -n 18 "$output.txt"
    if [[ "$recording" == yes ]]; then jfr summary "$output.jfr" > "$output-jfr-summary.txt"; fi
}

{
    sw_vers
    uname -m
    sysctl hw.memsize
    system_profiler SPDisplaysDataType
    git -C baseline rev-parse HEAD
    git -C candidate rev-parse HEAD
    shasum -a 256 baseline/target/classes/native/macos/libdeepj_metal_jni.dylib
    shasum -a 256 candidate/target/classes/native/macos/libdeepj_metal_jni.dylib
} > benchmark-diagnostics/environment.txt
cmp baseline/target/classes/native/macos/libdeepj_metal_jni.dylib \
    candidate/target/classes/native/macos/libdeepj_metal_jni.dylib

java -Xmx3g --enable-native-access=ALL-UNNAMED \
    -cp candidate/target/classes:candidate/target/test-classes \
    io.github.kirstenali.deepj.testing.MatmulBackwardCompare "$BENCHMARK_PROFILE" \
    > "benchmark-diagnostics/$BENCHMARK_PROFILE-matmul.txt" 2>&1

# Probe depths are reduced, not tensor widths, context, precision or kernel shapes.
# These diagnose GPU paths on a 7-GB VM, not full-model M4 speed equivalence.
run_case baseline 1-baseline 20 10 no
run_case candidate 2-candidate 20 10 no
run_case candidate 3-candidate 20 10 no
run_case baseline 4-baseline 20 10 no
run_case candidate profile-candidate 10 10 yes
run_case baseline profile-baseline 10 10 yes
