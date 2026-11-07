#!/usr/bin/env bash
# Usage: JAVA_HOME=... bash run.sh /absolute/deepj.jar /absolute/venv/bin/python /absolute/output audit|measure
set -euo pipefail
jar=$1
python=$2
output=$3
phase=$4
repo=$(cd "$(dirname "$0")/../../.." && pwd)
export VECLIB_MAXIMUM_THREADS=10
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ENABLE_MPS_FALLBACK=0
mkdir -p "$output/classes"
sources="$repo/src/test/java/io/github/kirstenali/deepj/nn/transformer/qwen3"
"$JAVA_HOME/bin/javac" -cp "$jar" -d "$output/classes" \
    "$sources/Qwen3TrainingFixture.java" "$sources/Qwen3TrainingAudit.java" "$sources/Qwen3TrainingBenchmark.java"

deepj() {
    local backend=$1 mode=$2 name=$3
    "$JAVA_HOME/bin/java" --enable-native-access=ALL-UNNAMED -Xmx1g -Xbatch -XX:CompileThresholdScaling=0.1 \
        -cp "$output/classes:$jar" io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3TrainingBenchmark \
        "$backend" "$mode" "$output/$name" > "$output/$name.log" 2>&1
    echo "Completed $name"
}

pytorch() {
    local backend=$1 name=$2
    shift 2
    "$python" "$repo/src/test/python/qwen3_training_benchmark.py" "$output/fixture" "$backend" \
        "$output/$name" "$@" > "$output/$name.log" 2>&1
    echo "Completed $name"
}

case "$phase" in
    audit)
        deepj cpu fixture fixture
        deepj cpu audit java-cpu-audit
        pytorch cpu torch-cpu-audit --audit
        deepj metal audit java-metal-audit
        pytorch mps torch-mps-audit --audit
        ;;
    measure)
        for backend in cpu metal; do
            reference=cpu
            if [[ "$backend" == metal ]]; then reference=mps; fi
            deepj "$backend" measure "java-$backend-1"
            pytorch "$reference" "torch-$reference-1"
            pytorch "$reference" "torch-$reference-2"
            deepj "$backend" measure "java-$backend-2"
        done
        ;;
    *) echo "Expected audit or measure" >&2; exit 2 ;;
esac
