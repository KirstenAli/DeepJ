#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
jdk_dir="${JAVA_HOME:-$(/usr/libexec/java_home)}"
resource_dir="$project_dir/src/main/resources/native/macos"
mkdir -p "$resource_dir"
xcrun clang++ -arch arm64 -mmacosx-version-min=15.0 -std=c++17 -O3 -DNDEBUG -dynamiclib \
  -DACCELERATE_NEW_LAPACK \
  -I"$jdk_dir/include" -I"$jdk_dir/include/darwin" -framework Accelerate \
  "$project_dir/native/deepj_cpu_jni.cpp" -o "$resource_dir/libdeepj_cpu_jni.dylib"
