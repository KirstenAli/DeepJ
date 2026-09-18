#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

mvn org.apache.maven.plugins:maven-javadoc-plugin:3.11.2:javadoc
rsync -a --delete --exclude='javadoc.sh' --exclude='options' --exclude='packages' \
    target/reports/apidocs/ docs/api/
