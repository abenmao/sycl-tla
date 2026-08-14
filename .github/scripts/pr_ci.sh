#!/bin/bash
set -euo pipefail

# Shared PR CI stage interface. Workflows supply policy through environment
# variables and invoke one command per GitHub Actions step.

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
SSH_OPTS="${SSH_OPTS:--o ConnectTimeout=30 -o BatchMode=yes -o StrictHostKeyChecking=accept-new}"
CI_PLATFORM="${CI_PLATFORM:-}"
CI_BUILD_ID_TAG="${CI_BUILD_ID_TAG:-common}"
CI_UNIT_BUILD_TARGET="${CI_UNIT_BUILD_TARGET:-test/all}"
CI_EXAMPLE_BUILD_TARGET="${CI_EXAMPLE_BUILD_TARGET:-examples/all}"
CI_BENCHMARK_BUILD_TARGET="${CI_BENCHMARK_BUILD_TARGET:-benchmarks/all}"
CI_UNIT_TEST_REGEX="${CI_UNIT_TEST_REGEX:-test_unit}"
CI_EXAMPLE_TEST_REGEX="${CI_EXAMPLE_TEST_REGEX:-test_example}"
CI_BENCHMARK_TEST_REGEX="${CI_BENCHMARK_TEST_REGEX:-cbenchmarks}"
REMOTE_NINJA_GB_PER_JOB="${REMOTE_NINJA_GB_PER_JOB:-10}"
REMOTE_HEAVY_GB_PER_JOB="${REMOTE_HEAVY_GB_PER_JOB:-20}"
LOCAL_NINJA_GB_PER_JOB="${LOCAL_NINJA_GB_PER_JOB:-10}"
LOCAL_HEAVY_GB_PER_JOB="${LOCAL_HEAVY_GB_PER_JOB:-20}"
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
ONEAPI_COMPILER="${ONEAPI_COMPILER:-${ONEAPI_ROOT}/compiler/latest/bin/icpx}"
ONEAPI_FALLBACK_ROOT="${ONEAPI_FALLBACK_ROOT:-/opt/intel/oneapi-pytorch}"
CONTAINER_ONEAPI_ROOT="${CONTAINER_ONEAPI_ROOT:-/opt/intel/oneapi}"

require_positive_integer() {
    local name="$1" value="${!1:-}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${name} must be a positive integer, got '${value}'."
        return 1
    fi
}

validate_build_id_tag() {
    if ! [[ "$CI_BUILD_ID_TAG" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
        echo "ERROR: CI_BUILD_ID_TAG must contain only lowercase letters, numbers, '_' or '-'."
        return 1
    fi
}

write_output() {
    echo "$1" >> "${GITHUB_OUTPUT:?GITHUB_OUTPUT is required}"
}

write_env() {
    echo "$1" >> "${GITHUB_ENV:?GITHUB_ENV is required}"
}

run_build_script() {
    if [ -z "$CI_PLATFORM" ]; then
        echo "ERROR: CI_PLATFORM is required. Use bmg, pvc, or cri."
        return 1
    fi
    NINJA_JOBS="${NINJA_JOBS:-}" HEAVY_JOBS="${HEAVY_JOBS:-}" \
        bash "${WORKSPACE}/.github/scripts/ci_build.sh" "$CI_PLATFORM" "$@"
}

free_gb() {
    local user="$1"
    local host="$2"
    local output
    output=$(ssh ${SSH_OPTS} "${user}@${host}" \
        "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$output" ] && echo "$output" || echo -1
}

detect_strategy() {
    local cpu_threads ram_gb
    cpu_threads=$(nproc)
    ram_gb=$(awk '/MemTotal/{print int($2/1024/1024)}' /proc/meminfo)
    echo "Runner: ${cpu_threads} threads, ${ram_gb} GB RAM"
    if [ "${CI_FORCE_LOCAL:-false}" = "true" ]; then
        write_output "use_remote=false"
        echo "Strategy: LOCAL build (CI_FORCE_LOCAL=true)"
        return
    fi
    if [ -z "${REMOTE_BUILD_SERVERS:-}" ]; then
        write_output "use_remote=false"
        echo "Strategy: LOCAL build (no remote build servers configured)"
        return
    fi
    echo "Thresholds: >= ${REMOTE_CPU_THRESHOLD} threads AND >= ${REMOTE_RAM_THRESHOLD} GB RAM for local build"
    if [ "$cpu_threads" -ge "$REMOTE_CPU_THRESHOLD" ] && [ "$ram_gb" -ge "$REMOTE_RAM_THRESHOLD" ]; then
        write_output "use_remote=false"
        echo "Strategy: LOCAL build (${cpu_threads} threads >= ${REMOTE_CPU_THRESHOLD}, ${ram_gb} GB >= ${REMOTE_RAM_THRESHOLD})"
    else
        write_output "use_remote=true"
        echo "Strategy: REMOTE build (${cpu_threads} threads / ${ram_gb} GB below threshold)"
    fi
}

admit_build_server() {
    local checks=3 interval_min=15 admitted=false
    local name host user gate budget free
    local candidates="${REMOTE_BUILD_SERVERS:-}"

    if [ -n "$candidates" ]; then
        for i in $(seq 1 "$checks"); do
            while read -r name host user gate budget; do
                [ -z "$name" ] && continue
                if ! [[ "${gate:-}" =~ ^[0-9]+$ ]] || ! [[ "${budget:-}" =~ ^[1-9][0-9]*$ ]]; then
                    echo "::warning::Invalid remote build server candidate: ${name} ${host} ${user} ${gate} ${budget}"
                    continue
                fi
                free=$(free_gb "$user" "$host")
                echo "Check ${i}/${checks}: ${name} (${host}) MemAvailable=${free}GB (gate ${gate}GB, budget ${budget}GB)"
                if [ "${free:-0}" -ge "$gate" ]; then
                    echo "Selected build server: ${name} (${host}), budget ${budget}GB"
                    write_env "BUILD_SERVER_NAME=${name}"
                    write_env "BUILD_SERVER=${host}"
                    write_env "BUILD_SERVER_USER=${user}"
                    write_env "BUILD_MEM_BUDGET_GB=${budget}"
                    admitted=true
                    break
                fi
            done <<< "$(printf '%s\n' "$candidates" | tr ';' '\n')"
            [ "$admitted" = true ] && break
            if [ "$i" -lt "$checks" ]; then
                echo "All candidates below gate; waiting ${interval_min}min before recheck..."
                sleep $((interval_min * 60))
            fi
        done
    else
        echo "No remote build servers configured; using local build."
    fi
    write_output "admitted=${admitted}"
    if [ "$admitted" != true ] && [ -n "$candidates" ]; then
        echo "::warning::No build server admitted after capacity checks; falling back to local build."
    fi
}

cleanup_old_remote_builds() {
    ssh ${SSH_OPTS} "${BUILD_SERVER_USER}@${BUILD_SERVER}" "
        cutoff=\$(date -d '24 hours ago' +%s)
        for d in /home/${BUILD_SERVER_USER}/ci-builds/*/ /shared/builds/*/; do
            [ -d \"\$d\" ] || continue
            bname=\$(basename \"\$d\")
            [[ \"\$bname\" =~ ^[0-9]+-[a-z0-9_-]+-.+-[0-9]+\$ ]] || continue
            created=\$(stat -c %W \"\$d\" 2>/dev/null || echo 0)
            if [ \"\$created\" -le 0 ]; then
                created=\$(stat -c %Y \"\$d\" 2>/dev/null || echo 0)
            fi
            if [ \"\$created\" -gt 0 ] && [ \"\$created\" -lt \"\$cutoff\" ]; then
                sudo -n rm -rf \"\$d\"
            fi
        done
    " || true
}

reap_orphan_containers() {
    local reap_age_s=$(( (BUILD_STEP_TIMEOUT_MIN + ORPHAN_REAP_GRACE_MIN) * 60 ))
    ssh ${SSH_OPTS} "${BUILD_SERVER_USER}@${BUILD_SERVER}" '
        now=$(date +%s)
        for cid in $(docker ps -q --filter "name=cutlass-build-"); do
            started=$(docker inspect -f "{{.State.StartedAt}}" "$cid" 2>/dev/null)
            [ -z "$started" ] && continue
            started_s=$(date -d "$started" +%s 2>/dev/null || echo "$now")
            if [ $(( now - started_s )) -gt '"${reap_age_s}"' ]; then
                name=$(docker inspect -f "{{.Name}}" "$cid" 2>/dev/null)
                echo "Reaping orphan build container ${name#/} (age $(( now - started_s ))s)"
                docker rm -f "$cid" >/dev/null 2>&1 || true
            fi
        done
    ' || true
}

remote_build() {
    local build_id previous_build_id
    local runner_compiler oneapi_path match container_name NINJA_JOBS HEAVY_JOBS docker_rc
    validate_build_id_tag
    require_positive_integer BUILD_MEM_BUDGET_GB
    require_positive_integer REMOTE_NINJA_GB_PER_JOB
    require_positive_integer REMOTE_HEAVY_GB_PER_JOB
    build_id="${GITHUB_RUN_ID}-${CI_BUILD_ID_TAG}-${RUNNER_NAME}-${GITHUB_RUN_ATTEMPT}"
    previous_build_id="${GITHUB_RUN_ID}-${CI_BUILD_ID_TAG}-${RUNNER_NAME}-$((GITHUB_RUN_ATTEMPT - 1))"
    write_env "BUILD_ID=${build_id}"

    runner_compiler=$("$ONEAPI_COMPILER" --version | head -1)
    echo "Runner compiler: ${runner_compiler}"
    if ! ssh ${SSH_OPTS} "${BUILD_SERVER_USER}@${BUILD_SERVER}" "timeout 30 docker info > /dev/null 2>&1"; then
        echo "::warning::Build server unreachable or Docker not available."
        write_output "failure_type=infra"
        return 1
    fi

    reap_orphan_containers
    oneapi_path=$(readlink -f "$ONEAPI_ROOT")
    if ! match=$(ssh ${SSH_OPTS} "${BUILD_SERVER_USER}@${BUILD_SERVER}" "
        if [ -d '${oneapi_path}' ] && \
           '${oneapi_path}/compiler/latest/bin/icpx' --version 2>/dev/null | head -1 | grep -qF '${runner_compiler}'; then
            echo '${oneapi_path}'
        else
            for d in ${ONEAPI_FALLBACK_ROOT}/*/; do
                if [ -x \"\${d}compiler/latest/bin/icpx\" ] && \
                   \"\${d}compiler/latest/bin/icpx\" --version 2>/dev/null | head -1 | grep -qF '${runner_compiler}'; then
                    echo \"\${d%/}\"
                    break
                fi
            done
        fi
    "); then
        echo "::warning::SSH failed during compiler version check."
        write_output "failure_type=infra"
        return 1
    fi
    if [ -z "$match" ]; then
        echo "::warning::Build server has no oneAPI matching '${runner_compiler}'. Falling back to local build."
        write_output "failure_type=infra"
        return 1
    fi
    oneapi_path="$match"
    echo "Build server matching oneAPI found at: ${oneapi_path}"
    cleanup_old_remote_builds

    if ! ssh ${SSH_OPTS} "${BUILD_SERVER_USER}@${BUILD_SERVER}" "
        sudo -n rm -rf /home/${BUILD_SERVER_USER}/ci-builds/${previous_build_id} /shared/builds/${previous_build_id};
        mkdir -p /home/${BUILD_SERVER_USER}/ci-builds/${build_id}/source
    "; then
        echo "::warning::Failed to prepare remote build directory."
        write_output "failure_type=infra"
        return 1
    fi
    if ! rsync -a --delete --exclude='.git' -e "ssh ${SSH_OPTS}" \
        "${WORKSPACE}/" "${BUILD_SERVER_USER}@${BUILD_SERVER}:/home/${BUILD_SERVER_USER}/ci-builds/${build_id}/source/"; then
        echo "::warning::Failed to sync source to build server."
        write_output "failure_type=infra"
        return 1
    fi

    container_name="cutlass-build-${build_id}"
    kill_remote_container() {
        ssh ${SSH_OPTS} "${BUILD_SERVER_USER}@${BUILD_SERVER}" \
            "docker rm -f ${container_name} >/dev/null 2>&1" || true
    }
    trap kill_remote_container EXIT INT TERM
    NINJA_JOBS=$((BUILD_MEM_BUDGET_GB / REMOTE_NINJA_GB_PER_JOB))
    HEAVY_JOBS=$((BUILD_MEM_BUDGET_GB / REMOTE_HEAVY_GB_PER_JOB))
    [ "$NINJA_JOBS" -lt 1 ] && NINJA_JOBS=1
    [ "$HEAVY_JOBS" -lt 1 ] && HEAVY_JOBS=1
    set +e
    ssh ${SSH_OPTS} "${BUILD_SERVER_USER}@${BUILD_SERVER}" "
        docker run --rm --name ${container_name} \\
                    --entrypoint /workspace/bin/entrypoint.sh \\
          -v ${oneapi_path}:${CONTAINER_ONEAPI_ROOT}:ro \\
          -v /home/${BUILD_SERVER_USER}/ci-builds/${build_id}/source:/workspace/source:ro \\
          -v /shared/builds/${build_id}:/workspace/output \\
          -v /home/${BUILD_SERVER_USER}/tools/bmg-ocloc:/workspace/tools/bmg-ocloc:ro \\
          -v /home/${BUILD_SERVER_USER}/ci-builds/${build_id}/source/.github/docker/entrypoint.sh:/workspace/bin/entrypoint.sh:ro \\
          -e DISABLE_FLASH_ATTENTION=0 \\
          -e CI_PLATFORM=${CI_PLATFORM} \
          -e CI_BUILD_PROFILE=${CI_BUILD_PROFILE:-full} \
          -e CI_BUILD_TARGET_LIST='${CI_BUILD_TARGET_LIST:-}' \
          -e SYCL_TARGET=${SYCL_TARGET} \\
          -e ONEAPI_ROOT=${CONTAINER_ONEAPI_ROOT} \\
          -e BUILD_TARGETS=all \\
          -e NINJA_JOBS=${NINJA_JOBS} \\
          -e HEAVY_JOBS=${HEAVY_JOBS} \\
          -e REMOTE_BUILD_DIR=/tmp/cutlass-build-${build_id} \\
          -e REMOTE_SOURCE_DIR=${WORKSPACE} \\
          -e CMAKE_EXTRA_FLAGS='${CI_CMAKE_EXTRA_FLAGS:-}' \\
          cutlass-builder:latest
    "
    docker_rc=$?
    set -e
    trap - EXIT INT TERM
    if [ "$docker_rc" -ne 0 ]; then
        if [ "$docker_rc" -eq 42 ]; then
            echo "::error::Remote compilation failed (build error, exit code 42). Not falling back."
            write_output "failure_type=compile"
        else
            echo "::warning::Remote build infrastructure error (exit code ${docker_rc})."
            write_output "failure_type=infra"
        fi
        return 1
    fi
    if ! rsync -a -e "ssh ${SSH_OPTS}" \
        "${BUILD_SERVER_USER}@${BUILD_SERVER}:/shared/builds/${build_id}/build/" \
        "/tmp/cutlass-build-${build_id}/"; then
        echo "::warning::Failed to sync build output from build server."
        write_output "failure_type=infra"
        return 1
    fi
    write_env "BUILD_DIR=/tmp/cutlass-build-${build_id}"
    write_output "failure_type=none"
}

calculate_jobs() {
    local ram_gb budget_gb
    require_positive_integer LOCAL_NINJA_GB_PER_JOB
    require_positive_integer LOCAL_HEAVY_GB_PER_JOB
    ram_gb=$(awk '/MemTotal/{print int($2/1024/1024)}' /proc/meminfo)
    budget_gb=$((ram_gb * $1 / 100))
    NINJA_JOBS=$((budget_gb / LOCAL_NINJA_GB_PER_JOB))
    HEAVY_JOBS=$((budget_gb / LOCAL_HEAVY_GB_PER_JOB))
    [ "$NINJA_JOBS" -lt 1 ] && NINJA_JOBS=1
    [ "$HEAVY_JOBS" -lt 1 ] && HEAVY_JOBS=1
    [ "$NINJA_JOBS" -gt "$(nproc)" ] && NINJA_JOBS=$(nproc)
    echo "${2}: RAM=${ram_gb}GB budget=${budget_gb}GB -j${NINJA_JOBS} heavy=${HEAVY_JOBS}"
}

configure_local() {
    ensure_environment
    calculate_jobs "${LOCAL_BUILD_MEM_PCT}" "Local build"
    write_env "NINJA_JOBS=${NINJA_JOBS}"
    write_env "HEAVY_JOBS=${HEAVY_JOBS}"
    run_build_script configure
}

local_build() {
    ensure_environment
    calculate_jobs "${LOCAL_BUILD_MEM_PCT}" "Local build"
    write_env "NINJA_JOBS=${NINJA_JOBS}"
    write_env "HEAVY_JOBS=${HEAVY_JOBS}"
    local local_build_dir="${BUILD_DIR:-${WORKSPACE}/build}"
    if [ -d "$local_build_dir" ]; then
        echo "Cleaning local build directory: ${local_build_dir}"
        rm -rf -- "$local_build_dir"
    fi
    run_build_script all
}

fallback_notice() {
    echo "::warning::Fallback reason: ${FALLBACK_REASON:-Remote build did not complete.}"
    echo "Entering local build."
}

ensure_environment() {
    if [ "${CI_ENV_INITIALIZED:-false}" = "true" ]; then
        echo "CI environment already initialized; skipping setup."
        return 0
    fi

    echo "Initializing CI environment on demand..."
    source "${WORKSPACE}/.github/scripts/ci_build.sh"
    setup_environment "$CI_PLATFORM"
    if [ "$CI_PLATFORM" = "cri" ]; then
        source "${WORKSPACE}/.github/scripts/cri_device.sh"
        ensure_device
    fi
    export CI_ENV_INITIALIZED=true
    if [ -n "${GITHUB_ENV:-}" ]; then
        echo "CI_ENV_INITIALIZED=true" >> "$GITHUB_ENV"
    fi
}

build_target() {
    ensure_environment
    run_build_script build "$1"
}

verify_output() {
    local sample
    echo "BUILD_DIR: ${BUILD_DIR}"
    echo "=== Test executables ==="
    find "${BUILD_DIR}/test" -maxdepth 5 -name 'cutlass_test_unit_*' -type f 2>/dev/null | head -5 || true
    sample=$(find "${BUILD_DIR}/test" -maxdepth 5 -name 'cutlass_test_unit_*' -type f 2>/dev/null | head -1 || true)
    if [ -z "$sample" ]; then
        echo "::error::No test executables found in build output!"
        return 1
    fi
    echo "=== Sample executable: ${sample} ==="
    file "$sample"
    ldd "$sample" 2>&1 | head -20 || true
}

run_tests() {
    ensure_environment
    local build_path="${BUILD_DIR:-build}"
    local regex exclude=""
    local -a ctest_args
    if [ "$CI_PLATFORM" = "cri" ] && [ "${CI_BUILD_PROFILE:-full}" = "tiny" ]; then
        bash "${WORKSPACE}/.github/scripts/cri_device.sh" test "$1"
        return
    fi
    [[ "$build_path" = /* ]] || build_path="${WORKSPACE}/${build_path}"
    cd "$build_path"
    case "$1" in
        unit|ut)
            regex="$CI_UNIT_TEST_REGEX"
            exclude="${CI_UNIT_TEST_EXCLUDE:-}"
            ;;
        example|examples)
            regex="$CI_EXAMPLE_TEST_REGEX"
            exclude="${CI_EXAMPLE_TEST_EXCLUDE:-}"
            ;;
        benchmark|benchmarks|bm)
            regex="$CI_BENCHMARK_TEST_REGEX"
            if [ -n "${CI_BENCHMARK_TEST_EXCLUDE:-}" ]; then
                exclude="$CI_BENCHMARK_TEST_EXCLUDE"
            elif [ "${GPU:-}" = "PVC" ]; then
                exclude="fp8|config_file_bmg_prefill_bf16|prefill_legacy_xe_config_file_pvc"
            fi
            ;;
        *) echo "ERROR: Unknown test suite '$1'. Use unit, examples, or benchmarks."; return 1 ;;
    esac
    ctest_args=(-R "$regex" --output-on-failure)
    [ -n "$exclude" ] && ctest_args+=(-E "$exclude")
    ctest "${ctest_args[@]}"
}

cleanup_build() {
    local build_id="${BUILD_ID:-}"
    if [ "$CI_PLATFORM" = "cri" ]; then
        bash "${WORKSPACE}/.github/scripts/cri_device.sh" cleanup || true
    fi
    validate_build_id_tag
    if [ -n "$build_id" ] && [[ "$build_id" =~ ^[0-9]+-${CI_BUILD_ID_TAG}-.+-[0-9]+$ ]]; then
        rm -rf "/tmp/cutlass-build-${build_id}" || true
        ssh ${SSH_OPTS} "${BUILD_SERVER_USER}@${BUILD_SERVER}" \
            "docker rm -f cutlass-build-${build_id} >/dev/null 2>&1; sudo -n rm -rf /home/${BUILD_SERVER_USER}/ci-builds/${build_id} /shared/builds/${build_id}" || true
    else
        echo "Skipping cleanup: BUILD_ID '${build_id}' is empty or does not match expected pattern"
    fi
}

fail_compile() {
    echo "::error::Remote compilation failed due to a code error (not infrastructure). Review the 'Remote Build' step logs."
    return 1
}

usage() {
    cat <<'EOF'
Usage: pr_ci.sh <command> [argument]

Commands:
  detect              Select local or remote build from runner capacity.
    select-server       Select a build server and wait for capacity.
    remote-build        Build on the admitted remote server.
  fail-compile        Fail after a remote compiler error (no fallback).
    setup-environment   Export oneAPI and runtime environment for later steps.
  configure           Configure a strong local runner build.
    local-build         Configure and build CI_BUILD_TARGET_LIST or default targets.
  build <target>      Build one CMake target; examples use HEAVY_JOBS.
    verify-build        Verify synchronized remote build output.
  test <suite>        Run unit/ut, example(s), or benchmark(s)/bm tests.
    run-ut              Run unit tests.
    run-example         Run example tests.
    run-bm              Run benchmark tests.
  cleanup             Remove this run's remote and synchronized artifacts.
    fallback-notice     Report the fallback reason before the local build step.

Short aliases admit, remote, setup, and verify remain supported.
EOF
}

main() {
    local mode="${1:-help}"

    case "$mode" in
        detect) detect_strategy ;;
        admit|select-server) admit_build_server ;;
        remote|remote-build) remote_build ;;
        configure) configure_local ;;
        local-build) local_build ;;
        fallback-notice) fallback_notice ;;
        build) build_target "${2:?Usage: $0 build <test/all|examples/all|benchmarks/all>}" ;;
        setup|setup-environment) ensure_environment ;;
        verify|verify-build) verify_output ;;
        test) run_tests "${2:?Usage: $0 test <unit|examples|benchmarks>}" ;;
        run-ut) run_tests unit ;;
        run-example) run_tests examples ;;
        run-bm) run_tests benchmarks ;;
        cleanup) cleanup_build ;;
        fail-compile) fail_compile ;;
        help|-h|--help) usage ;;
        *) echo "ERROR: Unknown mode '$mode'."; usage; exit 1 ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi