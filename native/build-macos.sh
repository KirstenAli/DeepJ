#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
jdk_dir="${JAVA_HOME:-$(/usr/libexec/java_home)}"
output_dir="$project_dir/native/macos"
resource_dir="$project_dir/src/main/resources/native/macos"

mkdir -p "$output_dir" "$resource_dir"
xcrun clang++ -std=c++17 -O3 -DNDEBUG -fobjc-arc -dynamiclib \
  -I"$jdk_dir/include" -I"$jdk_dir/include/darwin" \
  -framework Foundation -framework Metal -framework MetalPerformanceShaders \
  "$project_dir/native/deepj_metal_jni.mm" -o "$output_dir/libdeepj_metal_jni.dylib"
cp "$output_dir/libdeepj_metal_jni.dylib" "$resource_dir/libdeepj_metal_jni.dylib"
