#!/usr/bin/env bash
# Builds and runs the fork's unit-test programs against an existing build of this tree.
#
#   test/run_unit_tests.sh BUILD_DIR [GGML_INCLUDE_DIR] [OUT_DIR]
#
# BUILD_DIR is a CMake build with the default static libraries (libstable-diffusion.a and the ggml libraries
# under BUILD_DIR/ggml/src). GGML_INCLUDE_DIR defaults to ggml/include; OUT_DIR, where the programs and their
# logs go, defaults to BUILD_DIR/unit-tests. Each program is built as the header of its source file says. The
# exit code is 1 when any program does not build or does not pass, and its logs say why.
set -u

usage="usage: test/run_unit_tests.sh BUILD_DIR [GGML_INCLUDE_DIR] [OUT_DIR]"
[ $# -ge 1 ] || { echo "$usage" >&2; exit 2; }
B=$(cd "$1" 2>/dev/null && pwd) || { echo "no build directory: $1" >&2; exit 2; }
G=${2:-}
T=${3:-$B/unit-tests}
if [ -n "$G" ]; then
    G=$(cd "$G" 2>/dev/null && pwd) || { echo "no ggml include directory: $2" >&2; exit 2; }
fi
mkdir -p "$T" && T=$(cd "$T" && pwd) || exit 2
cd "$(dirname "$0")/.." || exit 2
G=${G:-$PWD/ggml/include}
[ -f "$G/ggml.h" ] || { echo "no ggml headers in $G" >&2; exit 2; }
[ -f "$B/libstable-diffusion.a" ] || { echo "no libstable-diffusion.a in $B: build the library first" >&2; exit 2; }

# The programs need the libraries CMake links into sd-cli, and those depend on the options of the build (WebP and
# WebM when their submodules are there, OpenMP...), so they are read from the build files.
cmake_link_libs() {
    local base libs="" item out=""
    if [ -f "$B/build.ninja" ]; then
        base=$B
        libs=$(awk '/^build bin\/sd-cli:/ { found = 1 } found && /^  LINK_LIBRARIES = / { sub(/^  LINK_LIBRARIES = /, ""); print; exit }' "$B/build.ninja")
    elif [ -f "$B/examples/cli/CMakeFiles/sd-cli.dir/link.txt" ]; then
        base=$B/examples/cli
        libs=$(sed -e 's/^.* -o [^ ]*sd-cli //' "$base/CMakeFiles/sd-cli.dir/link.txt")
    fi
    [ -n "$libs" ] || return 1
    for item in $libs; do
        case $item in
            -* | /*) out="$out $item" ;;
            *) out="$out $base/$item" ;;
        esac
    done
    echo "$out -pthread"
}
LIBS=$(cmake_link_libs) || LIBS="$B/libstable-diffusion.a $B/ggml/src/libggml.a $B/ggml/src/libggml-cpu.a $B/ggml/src/libggml-base.a -ldl -lm -fopenmp -pthread"
echo "linking with:$LIBS"
FL="-std=c++17 -O1 -DGGML_MAX_NAME=160 -Isrc -Isrc/core -Iinclude -Ithirdparty -I$G"
EX="examples/common/common.cpp examples/common/log.cpp examples/common/media_io.cpp examples/common/media_metadata.cpp examples/common/profile.cpp $T/zip.o"
cc -c thirdparty/zip.c -o "$T/zip.o" >"$T/zip.compile.log" 2>&1 || { echo "thirdparty/zip.c does not build, see $T/zip.compile.log" >&2; exit 1; }

failed=0
run() { # name, compile command
    local name=$1
    shift
    if [ ! -f "test/$name.cpp" ]; then
        printf 'FAIL  %-28s test/%s.cpp is missing\n' "$name" "$name"
        failed=$((failed + 1))
        return
    fi
    if ! eval "$@" >"$T/$name.compile.log" 2>&1; then
        printf 'FAIL  %-28s does not build, see %s\n' "$name" "$T/$name.compile.log"
        failed=$((failed + 1))
        return
    fi
    local start=$SECONDS
    timeout 600 "$T/$name" >"$T/$name.run.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        printf 'ok    %-28s %3d s\n' "$name" $((SECONDS - start))
    else
        printf 'FAIL  %-28s exit %d, see %s\n' "$name" $rc "$T/$name.run.log"
        tail -n 20 "$T/$name.run.log" | sed 's/^/      /'
        failed=$((failed + 1))
    fi
}

run backend_fit_plan_test "c++ $FL test/backend_fit_plan_test.cpp $LIBS -o $T/backend_fit_plan_test"
run model_loader_latency_test "c++ $FL test/model_loader_latency_test.cpp $LIBS -o $T/model_loader_latency_test"
run pe_imports_test "c++ -std=c++17 -g -fsanitize=address,undefined -Isrc test/pe_imports_test.cpp src/core/pe_imports.cpp -o $T/pe_imports_test"
run file_handle_pool_test "c++ -std=c++17 -g -fsanitize=address,undefined -Isrc -Iinclude -I$G -pthread test/file_handle_pool_test.cpp src/model_io/file_handle_pool.cpp -o $T/file_handle_pool_test"
run profile_test "c++ -std=c++17 -O1 -Iexamples -Iinclude -Ithirdparty -Isrc test/profile_test.cpp $EX $LIBS -o $T/profile_test"
run server_recovery_test "c++ -std=c++17 -O1 -Iexamples -Iexamples/server -Iinclude -Ithirdparty -Isrc test/server_recovery_test.cpp examples/server/recovery.cpp $EX $LIBS -o $T/server_recovery_test"
run api_robustness "c++ -std=c++17 -g -fsanitize=address,undefined -Iinclude test/api_robustness.cpp $LIBS -o $T/api_robustness"
run memory_guard_engine_test "c++ $FL test/memory_guard_engine_test.cpp $LIBS -o $T/memory_guard_engine_test"
run layer_split_fill_test "c++ $FL test/layer_split_fill_test.cpp $LIBS -o $T/layer_split_fill_test"
run device_health_test "c++ $FL test/device_health_test.cpp $LIBS -o $T/device_health_test"
run device_health_option_test "c++ -std=c++17 -O1 -Iexamples -Iinclude -Ithirdparty -Isrc test/device_health_option_test.cpp $EX $LIBS -o $T/device_health_option_test"
run server_guardian_test "c++ -std=c++17 -O1 -Iexamples -Iexamples/server -Iinclude -Ithirdparty -Isrc test/server_guardian_test.cpp examples/server/guardian.cpp examples/common/log.cpp $LIBS -o $T/server_guardian_test"

if [ $failed -eq 0 ]; then
    echo "all unit-test programs passed"
    exit 0
fi
echo "$failed unit-test programs failed"
exit 1
