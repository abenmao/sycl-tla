#!/bin/bash
set -euo pipefail

# Shared PR CI stage interface. Workflows supply policy through environment
# variables and invoke one command per GitHub Actions step.

CI_WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
CI_SSH_OPTIONS="${SSH_OPTS:--o ConnectTimeout=30 -o BatchMode=yes -o StrictHostKeyChecking=accept-new}"
CI_WORKFLOW_PLATFORM="${CI_PLATFORM:-}"
CI_RUN_ID_TAG="${CI_BUILD_ID_TAG:-common}"
CI_CONFIG_UNIT_TARGET="${CI_UNIT_BUILD_TARGET:-test/all}"
CI_CONFIG_EXAMPLE_TARGET="${CI_EXAMPLE_BUILD_TARGET:-examples/all}"
CI_CONFIG_BENCHMARK_TARGET="${CI_BENCHMARK_BUILD_TARGET:-benchmarks/all}"
CI_CONFIG_UNIT_TEST_REGEX="${CI_UNIT_TEST_REGEX:-test_unit}"
CI_CONFIG_EXAMPLE_TEST_REGEX="${CI_EXAMPLE_TEST_REGEX:-test_example}"
CI_CONFIG_BENCHMARK_TEST_REGEX="${CI_BENCHMARK_TEST_REGEX:-cbenchmarks}"
REMOTE_NINJA_MEMORY_GB_PER_JOB="${REMOTE_NINJA_GB_PER_JOB:-10}"
REMOTE_HEAVY_MEMORY_GB_PER_JOB="${REMOTE_HEAVY_GB_PER_JOB:-20}"
LOCAL_NINJA_MEMORY_GB_PER_JOB="${LOCAL_NINJA_GB_PER_JOB:-10}"
LOCAL_HEAVY_MEMORY_GB_PER_JOB="${LOCAL_HEAVY_GB_PER_JOB:-20}"
CI_ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
CI_ONEAPI_COMPILER="${ONEAPI_COMPILER:-${CI_ONEAPI_ROOT}/compiler/latest/bin/icpx}"
REMOTE_ONEAPI_FALLBACK_ROOT="${ONEAPI_FALLBACK_ROOT:-/opt/intel/oneapi-pytorch}"
DOCKER_ONEAPI_ROOT="${CONTAINER_ONEAPI_ROOT:-/opt/intel/oneapi}"

require_positive_integer() {
    local name="$1" value="${!1:-}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${name} must be a positive integer, got '${value}'."
        return 1
    fi
}

validate_build_id_tag() {
    if ! [[ "$CI_RUN_ID_TAG" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
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

resolve_runner_ssh_target() {
    local runner_host="${RUNNER_SSH_HOST:-}"
    local runner_user="${RUNNER_SSH_USER:-${USER:-}}"
    if [ -z "$runner_host" ]; then
        runner_host=$(hostname -f 2>/dev/null || hostname)
    fi
    if [ -z "$runner_user" ]; then
        runner_user=$(id -un)
    fi
    if [ -z "$runner_host" ] || [ -z "$runner_user" ]; then
        echo "::warning::Reverse SSH self-check requires a resolvable runner host and user. Set RUNNER_SSH_HOST and RUNNER_SSH_USER."
        return 1
    fi
    printf '%s@%s\n' "$runner_user" "$runner_host"
}

check_ssh_connection() {
    local server_name="$1"
    local server_user="$2"
    local server_host="$3"
    local runner_target reverse_command
    if ! runner_target=$(resolve_runner_ssh_target); then
        return 1
    fi

    printf -v reverse_command 'ssh %s %q true' "$CI_SSH_OPTIONS" "$runner_target"
    echo "Checking bidirectional SSH: runner -> ${server_user}@${server_host} -> ${runner_target}."
    if ! ssh ${CI_SSH_OPTIONS} "${server_user}@${server_host}" "$reverse_command" </dev/null; then
        echo "::warning::Bidirectional SSH self-check failed for build server ${server_name}: runner -> ${server_user}@${server_host} -> ${runner_target}. Verify SSH keys, sshd/authorized_keys, and network settings on both hosts."
        return 1
    fi
    echo "Bidirectional SSH self-check passed for build server ${server_name}."
}

find_build_server_oneapi_path() {
    local server_user="$1"
    local server_host="$2"
    local runner_compiler_version runner_oneapi_path

    if ! runner_compiler_version=$("$CI_ONEAPI_COMPILER" --version | head -1) || [ -z "$runner_compiler_version" ]; then
        return 1
    fi
    if ! runner_oneapi_path=$(readlink -f "$CI_ONEAPI_ROOT"); then
        return 1
    fi

    ssh ${CI_SSH_OPTIONS} "${server_user}@${server_host}" "
        matched_path=
        if [ -d '${runner_oneapi_path}' ] && \
           '${runner_oneapi_path}/compiler/latest/bin/icpx' --version 2>/dev/null | head -1 | grep -qF '${runner_compiler_version}'; then
            matched_path='${runner_oneapi_path}'
        else
            for d in ${REMOTE_ONEAPI_FALLBACK_ROOT}/*/; do
                if [ -x \"\${d}compiler/latest/bin/icpx\" ] && \
                   \"\${d}compiler/latest/bin/icpx\" --version 2>/dev/null | head -1 | grep -qF '${runner_compiler_version}'; then
                    matched_path=\"\${d%/}\"
                    break
                fi
            done
        fi
        if [ -z \"\$matched_path\" ]; then
            exit 1
        fi
        printf '%s\\n' \"\$matched_path\"
    " </dev/null
}

check_build_server_environment() {
    local server_name="$1"
    local server_user="$2"
    local server_host="$3"

    if ! check_ssh_connection "$server_name" "$server_user" "$server_host" >&2; then
        return 1
    fi
    if ! find_build_server_oneapi_path "$server_user" "$server_host" >/dev/null; then
        echo "::warning::Build server ${server_name} has no oneAPI matching the local compiler." >&2
        return 1
    fi
    echo "Compiler match succeeded for build server ${server_name}."
}

configure_build_environment() {
    local build_output_dir="${BUILD_DIR:-${CI_WORKSPACE}/build}"
    local build_ninja_jobs="${BUILD_NINJA_JOBS:-}"
    local build_heavy_jobs="${BUILD_HEAVY_JOBS:-}"
    local build_disable_flash_attention="${BUILD_DISABLE_FLASH_ATTENTION:-${DISABLE_FLASH_ATTENTION:-0}}"

    export BUILD_SOURCE_DIR="$CI_WORKSPACE"
    export BUILD_DIR="$build_output_dir"
    export BUILD_ONEAPI_ROOT="$CI_ONEAPI_ROOT"
    export BUILD_ONEAPI_SETUP_SCRIPT="${ONEAPI_SET_VARS_SCRIPT:-${CI_ONEAPI_ROOT}/setvars.sh}"
    export BUILD_NINJA_COMMAND="${NINJA_COMMAND:-ninja}"
    export BUILD_PROFILE="${CI_BUILD_PROFILE:-full}"
    export BUILD_TARGET_LIST="${CI_BUILD_TARGET_LIST:-}"
    export BUILD_CMAKE_EXTRA_FLAGS="${CI_CMAKE_EXTRA_FLAGS:-}"
    export BUILD_DISABLE_FLASH_ATTENTION="$build_disable_flash_attention"
    export BUILD_SYCL_TARGET="${SYCL_TARGET:-}"
    export BUILD_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
    export BUILD_CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
    export BUILD_IGC_VISA_OPTIONS="${IGC_VISAOptions:--perfmodel}"
    export BUILD_IGC_VECTOR_ALIAS_BB_THRESHOLD="${IGC_VectorAliasBBThreshold:-100000000000}"
    export BUILD_CC="${CC:-icx}"
    export BUILD_CXX="${CXX:-icpx}"
    export BUILD_CUTLASS_ENABLE_SYCL="${CUTLASS_ENABLE_SYCL:-ON}"
    export BUILD_CUTLASS_ENABLE_BENCHMARKS="${CUTLASS_ENABLE_BENCHMARKS:-ON}"
    export BUILD_CUTLASS_SYCL_PROFILING_ENABLED="${CUTLASS_SYCL_PROFILING_ENABLED:-ON}"
    export BUILD_CUTLASS_SYCL_RUNNING_CI="${CUTLASS_SYCL_RUNNING_CI:-ON}"
    export BUILD_UNIT_TARGET="$CI_CONFIG_UNIT_TARGET"
    export BUILD_EXAMPLE_TARGET="$CI_CONFIG_EXAMPLE_TARGET"
    export BUILD_BENCHMARK_TARGET="$CI_CONFIG_BENCHMARK_TARGET"
    export BUILD_RUNTIME_LIBRARY_PATHS="${CI_RUNTIME_LIBRARY_PATHS:-}"
    export BUILD_FLASH_ATTENTION_CMAKE_FILE="${FLASH_ATTENTION_CMAKE_FILE:-}"
    export BUILD_BMG_SYCL_TARGET="${BMG_SYCL_TARGET:-intel_gpu_bmg_g21}"
    export BUILD_PVC_SYCL_TARGET="${PVC_SYCL_TARGET:-intel_gpu_pvc}"
    export BUILD_CRI_SYCL_TARGET="${CRI_SYCL_TARGET:-intel_gpu_cri}"
    export BUILD_BMG_GRF="${BMG_GRF:-256}"
    export BUILD_PVC_GRF="${PVC_GRF:-256}"
    export BUILD_CRI_GRF="${CRI_GRF:-512}"
    export BUILD_NINJA_JOBS="$build_ninja_jobs"
    export BUILD_HEAVY_JOBS="$build_heavy_jobs"
}

run_build_script() {
    if [ -z "$CI_WORKFLOW_PLATFORM" ]; then
        echo "ERROR: CI_PLATFORM is required. Use bmg, pvc, or cri."
        return 1
    fi
    configure_build_environment
    bash "${CI_WORKSPACE}/.github/scripts/ci_build.sh" "$CI_WORKFLOW_PLATFORM" "$@"
}

get_remote_available_memory_gb() {
    local server_user="$1"
    local server_host="$2"
    local available_memory_gb
    available_memory_gb=$(ssh ${CI_SSH_OPTIONS} "${server_user}@${server_host}" \
        "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" </dev/null 2>/dev/null)
    [ -n "$available_memory_gb" ] && echo "$available_memory_gb" || echo -1
}

detect_strategy() {
    local runner_cpu_threads runner_memory_gb
    local local_cpu_threshold="${REMOTE_CPU_THRESHOLD:?REMOTE_CPU_THRESHOLD is required}"
    local local_memory_threshold_gb="${REMOTE_RAM_THRESHOLD:?REMOTE_RAM_THRESHOLD is required}"
    runner_cpu_threads=$(nproc)
    runner_memory_gb=$(awk '/MemTotal/{print int($2/1024/1024)}' /proc/meminfo)
    echo "Runner: ${runner_cpu_threads} threads, ${runner_memory_gb} GB RAM"
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
    echo "Thresholds: >= ${local_cpu_threshold} threads AND >= ${local_memory_threshold_gb} GB RAM for local build"
    if [ "$runner_cpu_threads" -ge "$local_cpu_threshold" ] && [ "$runner_memory_gb" -ge "$local_memory_threshold_gb" ]; then
        write_output "use_remote=false"
        echo "Strategy: LOCAL build (${runner_cpu_threads} threads >= ${local_cpu_threshold}, ${runner_memory_gb} GB >= ${local_memory_threshold_gb})"
    else
        write_output "use_remote=true"
        echo "Strategy: REMOTE build (${runner_cpu_threads} threads / ${runner_memory_gb} GB below threshold)"
    fi
}

admit_build_server() {
    local max_checks=3 check_interval_minutes=15 admitted=false
    local server_name server_host server_user admission_gate_gb build_budget_gb
    local available_memory_gb check_number
    local configured_candidates="${REMOTE_BUILD_SERVERS:-}"
    local reachable_candidates=""

    if [ -n "$configured_candidates" ]; then
        while read -r server_name server_host server_user admission_gate_gb build_budget_gb; do
            [ -z "$server_name" ] && continue
            if ! [[ "${admission_gate_gb:-}" =~ ^[0-9]+$ ]] || ! [[ "${build_budget_gb:-}" =~ ^[1-9][0-9]*$ ]]; then
                echo "::warning::Invalid remote build server candidate: ${server_name} ${server_host} ${server_user} ${admission_gate_gb} ${build_budget_gb}"
                continue
            fi
            if check_build_server_environment "$server_name" "$server_user" "$server_host"; then
                reachable_candidates+="${server_name} ${server_host} ${server_user} ${admission_gate_gb} ${build_budget_gb}"$'\n'
            fi
        done <<< "$(printf '%s\n' "$configured_candidates" | tr ';' '\n')"

        if [ -z "$reachable_candidates" ]; then
            echo "::warning::No remote build server passed the environment preflight; skipping capacity checks."
        else
            echo "Remote environment preflight passed; checking remote server capacity."
            for check_number in $(seq 1 "$max_checks"); do
                while read -r server_name server_host server_user admission_gate_gb build_budget_gb; do
                    [ -z "$server_name" ] && continue
                    available_memory_gb=$(get_remote_available_memory_gb "$server_user" "$server_host")
                    echo "Check ${check_number}/${max_checks}: ${server_name} (${server_host}) MemAvailable=${available_memory_gb}GB (gate ${admission_gate_gb}GB, budget ${build_budget_gb}GB)"
                    if [ "${available_memory_gb:-0}" -ge "$admission_gate_gb" ]; then
                        echo "Selected build server: ${server_name} (${server_host}), budget ${build_budget_gb}GB"
                        write_env "REMOTE_BUILD_SERVER_NAME=${server_name}"
                        write_env "REMOTE_BUILD_SERVER_HOST=${server_host}"
                        write_env "REMOTE_BUILD_SERVER_USER=${server_user}"
                        write_env "REMOTE_BUILD_MEMORY_BUDGET_GB=${build_budget_gb}"
                        admitted=true
                        break
                    fi
                done <<< "$reachable_candidates"
                [ "$admitted" = true ] && break
                if [ "$check_number" -lt "$max_checks" ]; then
                    echo "All candidates below gate; waiting ${check_interval_minutes}min before recheck..."
                    sleep $((check_interval_minutes * 60))
                fi
            done
        fi
    else
        echo "No remote build servers configured; using local build."
    fi
    write_output "admitted=${admitted}"
    if [ "$admitted" != true ] && [ -n "$configured_candidates" ]; then
        echo "::warning::No build server admitted after capacity checks; falling back to local build."
    fi
}

cleanup_old_remote_builds() {
    ssh ${CI_SSH_OPTIONS} "${REMOTE_BUILD_SERVER_USER}@${REMOTE_BUILD_SERVER_HOST}" "
        cutoff=\$(date -d '24 hours ago' +%s)
        for d in /home/${REMOTE_BUILD_SERVER_USER}/ci-builds/*/ /shared/builds/*/; do
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
    local build_timeout_minutes="${BUILD_STEP_TIMEOUT_MIN:?BUILD_STEP_TIMEOUT_MIN is required}"
    local orphan_grace_minutes="${ORPHAN_REAP_GRACE_MIN:?ORPHAN_REAP_GRACE_MIN is required}"
    local orphan_reap_age_seconds=$(( (build_timeout_minutes + orphan_grace_minutes) * 60 ))
    ssh ${CI_SSH_OPTIONS} "${REMOTE_BUILD_SERVER_USER}@${REMOTE_BUILD_SERVER_HOST}" '
        now=$(date +%s)
        for cid in $(docker ps -q --filter "name=cutlass-build-"); do
            started=$(docker inspect -f "{{.State.StartedAt}}" "$cid" 2>/dev/null)
            [ -z "$started" ] && continue
            started_s=$(date -d "$started" +%s 2>/dev/null || echo "$now")
            if [ $(( now - started_s )) -gt '"${orphan_reap_age_seconds}"' ]; then
                name=$(docker inspect -f "{{.Name}}" "$cid" 2>/dev/null)
                echo "Reaping orphan build container ${name#/} (age $(( now - started_s ))s)"
                docker rm -f "$cid" >/dev/null 2>&1 || true
            fi
        done
    ' || true
}

remote_build() {
    local remote_build_id previous_remote_build_id
    local docker_oneapi_mount_source docker_container_name docker_exit_code
    local docker_ninja_jobs docker_heavy_jobs
    local docker_disable_flash_attention="${DISABLE_FLASH_ATTENTION:-0}"
    local docker_platform="$CI_WORKFLOW_PLATFORM"
    local docker_build_profile="${CI_BUILD_PROFILE:-full}"
    local docker_build_target_list="${CI_BUILD_TARGET_LIST:-}"
    local docker_ninja_command="${NINJA_COMMAND:-ninja}"
    local docker_unit_target="$CI_CONFIG_UNIT_TARGET"
    local docker_example_target="$CI_CONFIG_EXAMPLE_TARGET"
    local docker_benchmark_target="$CI_CONFIG_BENCHMARK_TARGET"
    local docker_sycl_target="${SYCL_TARGET:-}"
    local docker_cmake_extra_flags="${CI_CMAKE_EXTRA_FLAGS:-}"
    local docker_runtime_library_paths="${CI_RUNTIME_LIBRARY_PATHS:-}"
    local docker_flash_attention_cmake_file="${FLASH_ATTENTION_CMAKE_FILE:-}"
    local docker_bmg_sycl_target="${BMG_SYCL_TARGET:-intel_gpu_bmg_g21}"
    local docker_pvc_sycl_target="${PVC_SYCL_TARGET:-intel_gpu_pvc}"
    local docker_cri_sycl_target="${CRI_SYCL_TARGET:-intel_gpu_cri}"
    local docker_bmg_grf="${BMG_GRF:-256}"
    local docker_pvc_grf="${PVC_GRF:-256}"
    local docker_cri_grf="${CRI_GRF:-512}"
    local docker_cmake_build_type="${CMAKE_BUILD_TYPE:-Release}"
    local docker_device_selector="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
    local docker_igc_visa_options="${IGC_VISAOptions:--perfmodel}"
    local docker_igc_vector_alias_bb_threshold="${IGC_VectorAliasBBThreshold:-100000000000}"
    local docker_cc="${CC:-icx}"
    local docker_cxx="${CXX:-icpx}"
    local docker_cutlass_enable_sycl="${CUTLASS_ENABLE_SYCL:-ON}"
    local docker_cutlass_enable_benchmarks="${CUTLASS_ENABLE_BENCHMARKS:-ON}"
    local docker_cutlass_sycl_profiling_enabled="${CUTLASS_SYCL_PROFILING_ENABLED:-ON}"
    local docker_cutlass_sycl_running_ci="${CUTLASS_SYCL_RUNNING_CI:-ON}"
    validate_build_id_tag
    require_positive_integer REMOTE_BUILD_MEMORY_BUDGET_GB
    require_positive_integer REMOTE_NINJA_MEMORY_GB_PER_JOB
    require_positive_integer REMOTE_HEAVY_MEMORY_GB_PER_JOB
    remote_build_id="${GITHUB_RUN_ID}-${CI_RUN_ID_TAG}-${RUNNER_NAME}-${GITHUB_RUN_ATTEMPT}"
    previous_remote_build_id="${GITHUB_RUN_ID}-${CI_RUN_ID_TAG}-${RUNNER_NAME}-$((GITHUB_RUN_ATTEMPT - 1))"
    write_env "REMOTE_BUILD_ID=${remote_build_id}"

    if ! ssh ${CI_SSH_OPTIONS} "${REMOTE_BUILD_SERVER_USER}@${REMOTE_BUILD_SERVER_HOST}" "timeout 30 docker info > /dev/null 2>&1"; then
        echo "::warning::Build server unreachable or Docker not available."
        write_output "failure_type=infra"
        return 1
    fi

    reap_orphan_containers
    if ! docker_oneapi_mount_source=$(find_build_server_oneapi_path "$REMOTE_BUILD_SERVER_USER" "$REMOTE_BUILD_SERVER_HOST"); then
        echo "::warning::Unable to resolve a compiler-matched oneAPI path on the selected build server."
        write_output "failure_type=infra"
        return 1
    fi
    echo "Using compiler-matched remote oneAPI at: ${docker_oneapi_mount_source}"
    cleanup_old_remote_builds

    if ! ssh ${CI_SSH_OPTIONS} "${REMOTE_BUILD_SERVER_USER}@${REMOTE_BUILD_SERVER_HOST}" "
        sudo -n rm -rf /home/${REMOTE_BUILD_SERVER_USER}/ci-builds/${previous_remote_build_id} /shared/builds/${previous_remote_build_id};
        mkdir -p /home/${REMOTE_BUILD_SERVER_USER}/ci-builds/${remote_build_id}/source
    "; then
        echo "::warning::Failed to prepare remote build directory."
        write_output "failure_type=infra"
        return 1
    fi
    if ! rsync -a --delete --exclude='.git' -e "ssh ${CI_SSH_OPTIONS}" \
        "${CI_WORKSPACE}/" "${REMOTE_BUILD_SERVER_USER}@${REMOTE_BUILD_SERVER_HOST}:/home/${REMOTE_BUILD_SERVER_USER}/ci-builds/${remote_build_id}/source/"; then
        echo "::warning::Failed to sync source to build server."
        write_output "failure_type=infra"
        return 1
    fi

    docker_container_name="cutlass-build-${remote_build_id}"
    kill_remote_container() {
        ssh ${CI_SSH_OPTIONS} "${REMOTE_BUILD_SERVER_USER}@${REMOTE_BUILD_SERVER_HOST}" \
            "docker rm -f ${docker_container_name} >/dev/null 2>&1" || true
    }
    trap kill_remote_container EXIT INT TERM
    docker_ninja_jobs=$((REMOTE_BUILD_MEMORY_BUDGET_GB / REMOTE_NINJA_MEMORY_GB_PER_JOB))
    docker_heavy_jobs=$((REMOTE_BUILD_MEMORY_BUDGET_GB / REMOTE_HEAVY_MEMORY_GB_PER_JOB))
    [ "$docker_ninja_jobs" -lt 1 ] && docker_ninja_jobs=1
    [ "$docker_heavy_jobs" -lt 1 ] && docker_heavy_jobs=1
    set +e
    ssh ${CI_SSH_OPTIONS} "${REMOTE_BUILD_SERVER_USER}@${REMOTE_BUILD_SERVER_HOST}" "
        docker run --rm --name ${docker_container_name} \\
                    --entrypoint /workspace/bin/entrypoint.sh \\
          -v ${docker_oneapi_mount_source}:${DOCKER_ONEAPI_ROOT}:ro \\
          -v /home/${REMOTE_BUILD_SERVER_USER}/ci-builds/${remote_build_id}/source:/workspace/source:ro \\
          -v /shared/builds/${remote_build_id}:/workspace/output \\
          -v /home/${REMOTE_BUILD_SERVER_USER}/tools/bmg-ocloc:/workspace/tools/bmg-ocloc:ro \\
          -v /home/${REMOTE_BUILD_SERVER_USER}/ci-builds/${remote_build_id}/source/.github/docker/entrypoint.sh:/workspace/bin/entrypoint.sh:ro \\
          -e DOCKER_DISABLE_FLASH_ATTENTION=${docker_disable_flash_attention} \\
          -e DOCKER_PLATFORM=${docker_platform} \\
          -e DOCKER_BUILD_PROFILE=${docker_build_profile} \\
          -e DOCKER_BUILD_TARGET_LIST='${docker_build_target_list}' \\
          -e DOCKER_NINJA_COMMAND=${docker_ninja_command} \\
          -e DOCKER_UNIT_TARGET=${docker_unit_target} \\
          -e DOCKER_EXAMPLE_TARGET=${docker_example_target} \\
          -e DOCKER_BENCHMARK_TARGET=${docker_benchmark_target} \\
          -e DOCKER_SYCL_TARGET=${docker_sycl_target} \\
          -e DOCKER_ONEAPI_ROOT=${DOCKER_ONEAPI_ROOT} \\
          -e DOCKER_TARGET_GROUP=all \\
          -e DOCKER_NINJA_JOBS=${docker_ninja_jobs} \\
          -e DOCKER_HEAVY_JOBS=${docker_heavy_jobs} \\
          -e DOCKER_HOST_BUILD_DIR=/tmp/cutlass-build-${remote_build_id} \\
          -e DOCKER_HOST_SOURCE_DIR=${CI_WORKSPACE} \\
          -e DOCKER_CMAKE_EXTRA_FLAGS='${docker_cmake_extra_flags}' \\
          -e DOCKER_RUNTIME_LIBRARY_PATHS='${docker_runtime_library_paths}' \\
          -e DOCKER_FLASH_ATTENTION_CMAKE_FILE='${docker_flash_attention_cmake_file}' \\
          -e DOCKER_BMG_SYCL_TARGET=${docker_bmg_sycl_target} \\
          -e DOCKER_PVC_SYCL_TARGET=${docker_pvc_sycl_target} \\
          -e DOCKER_CRI_SYCL_TARGET=${docker_cri_sycl_target} \\
          -e DOCKER_BMG_GRF=${docker_bmg_grf} \\
          -e DOCKER_PVC_GRF=${docker_pvc_grf} \\
          -e DOCKER_CRI_GRF=${docker_cri_grf} \\
          -e DOCKER_CMAKE_BUILD_TYPE=${docker_cmake_build_type} \\
          -e DOCKER_DEVICE_SELECTOR=${docker_device_selector} \\
          -e DOCKER_IGC_VISA_OPTIONS='${docker_igc_visa_options}' \\
          -e DOCKER_IGC_VECTOR_ALIAS_BB_THRESHOLD=${docker_igc_vector_alias_bb_threshold} \\
          -e DOCKER_CC=${docker_cc} \\
          -e DOCKER_CXX=${docker_cxx} \\
          -e DOCKER_CUTLASS_ENABLE_SYCL=${docker_cutlass_enable_sycl} \\
          -e DOCKER_CUTLASS_ENABLE_BENCHMARKS=${docker_cutlass_enable_benchmarks} \\
          -e DOCKER_CUTLASS_SYCL_PROFILING_ENABLED=${docker_cutlass_sycl_profiling_enabled} \\
          -e DOCKER_CUTLASS_SYCL_RUNNING_CI=${docker_cutlass_sycl_running_ci} \\
          cutlass-builder:latest
    "
    docker_exit_code=$?
    set -e
    trap - EXIT INT TERM
    if [ "$docker_exit_code" -ne 0 ]; then
        if [ "$docker_exit_code" -eq 42 ]; then
            echo "::error::Remote compilation failed (build error, exit code 42). Not falling back."
            write_output "failure_type=compile"
        else
            echo "::warning::Remote build infrastructure error (exit code ${docker_exit_code})."
            write_output "failure_type=infra"
        fi
        return 1
    fi
    if ! rsync -a -e "ssh ${CI_SSH_OPTIONS}" \
        "${REMOTE_BUILD_SERVER_USER}@${REMOTE_BUILD_SERVER_HOST}:/shared/builds/${remote_build_id}/build/" \
        "/tmp/cutlass-build-${remote_build_id}/"; then
        echo "::warning::Failed to sync build output from build server."
        write_output "failure_type=infra"
        return 1
    fi
    write_env "BUILD_DIR=/tmp/cutlass-build-${remote_build_id}"
    write_output "failure_type=none"
}

calculate_local_build_jobs() {
    local memory_budget_percent="$1"
    local log_label="$2"
    local runner_memory_gb build_memory_budget_gb
    require_positive_integer LOCAL_NINJA_MEMORY_GB_PER_JOB
    require_positive_integer LOCAL_HEAVY_MEMORY_GB_PER_JOB
    runner_memory_gb=$(awk '/MemTotal/{print int($2/1024/1024)}' /proc/meminfo)
    build_memory_budget_gb=$((runner_memory_gb * memory_budget_percent / 100))
    BUILD_NINJA_JOBS=$((build_memory_budget_gb / LOCAL_NINJA_MEMORY_GB_PER_JOB))
    BUILD_HEAVY_JOBS=$((build_memory_budget_gb / LOCAL_HEAVY_MEMORY_GB_PER_JOB))
    [ "$BUILD_NINJA_JOBS" -lt 1 ] && BUILD_NINJA_JOBS=1
    [ "$BUILD_HEAVY_JOBS" -lt 1 ] && BUILD_HEAVY_JOBS=1
    [ "$BUILD_NINJA_JOBS" -gt "$(nproc)" ] && BUILD_NINJA_JOBS=$(nproc)
    echo "${log_label}: RAM=${runner_memory_gb}GB budget=${build_memory_budget_gb}GB -j${BUILD_NINJA_JOBS} heavy=${BUILD_HEAVY_JOBS}"
}

configure_local() {
    ensure_environment
    calculate_local_build_jobs "${LOCAL_BUILD_MEM_PCT}" "Local build"
    write_env "BUILD_NINJA_JOBS=${BUILD_NINJA_JOBS}"
    write_env "BUILD_HEAVY_JOBS=${BUILD_HEAVY_JOBS}"
    run_build_script configure
}

local_build() {
    ensure_environment
    calculate_local_build_jobs "${LOCAL_BUILD_MEM_PCT}" "Local build"
    write_env "BUILD_NINJA_JOBS=${BUILD_NINJA_JOBS}"
    write_env "BUILD_HEAVY_JOBS=${BUILD_HEAVY_JOBS}"
    local local_build_dir="${BUILD_DIR:-${CI_WORKSPACE}/build}"
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
    if [ "${CI_ENVIRONMENT_INITIALIZED:-false}" = "true" ]; then
        echo "CI environment already initialized; skipping setup."
        return 0
    fi

    echo "Initializing CI environment on demand..."
    configure_build_environment
    source "${CI_WORKSPACE}/.github/scripts/ci_build.sh"
    setup_environment "$CI_WORKFLOW_PLATFORM"
    if [ "$CI_WORKFLOW_PLATFORM" = "cri" ]; then
        source "${CI_WORKSPACE}/.github/scripts/cri_device.sh"
        ensure_device
    fi
    export CI_ENVIRONMENT_INITIALIZED=true
    if [ -n "${GITHUB_ENV:-}" ]; then
        echo "CI_ENVIRONMENT_INITIALIZED=true" >> "$GITHUB_ENV"
    fi
}

build_target() {
    ensure_environment
    run_build_script build "$1"
}

verify_output() {
    local build_output_dir="${BUILD_DIR:?BUILD_DIR is required}"
    local sample_executable
    echo "BUILD_DIR: ${build_output_dir}"
    echo "=== Test executables ==="
    find "${build_output_dir}/test" -maxdepth 5 -name 'cutlass_test_unit_*' -type f 2>/dev/null | head -5 || true
    sample_executable=$(find "${build_output_dir}/test" -maxdepth 5 -name 'cutlass_test_unit_*' -type f 2>/dev/null | head -1 || true)
    if [ -z "$sample_executable" ]; then
        echo "::error::No test executables found in build output!"
        return 1
    fi
    echo "=== Sample executable: ${sample_executable} ==="
    file "$sample_executable"
    ldd "$sample_executable" 2>&1 | head -20 || true
}

run_tests() {
    ensure_environment
    local build_output_dir="${BUILD_DIR:-${CI_WORKSPACE}/build}"
    local test_regex exclude_regex=""
    local -a ctest_args
    if [ "$CI_WORKFLOW_PLATFORM" = "cri" ] && [ "${CI_BUILD_PROFILE:-full}" = "tiny" ]; then
        bash "${CI_WORKSPACE}/.github/scripts/cri_device.sh" test "$1"
        return
    fi
    [[ "$build_output_dir" = /* ]] || build_output_dir="${CI_WORKSPACE}/${build_output_dir}"
    cd "$build_output_dir"
    case "$1" in
        unit|ut)
            test_regex="$CI_CONFIG_UNIT_TEST_REGEX"
            exclude_regex="${CI_UNIT_TEST_EXCLUDE:-}"
            ;;
        example|examples)
            test_regex="$CI_CONFIG_EXAMPLE_TEST_REGEX"
            exclude_regex="${CI_EXAMPLE_TEST_EXCLUDE:-}"
            ;;
        benchmark|benchmarks|bm)
            test_regex="$CI_CONFIG_BENCHMARK_TEST_REGEX"
            if [ -n "${CI_BENCHMARK_TEST_EXCLUDE:-}" ]; then
                exclude_regex="$CI_BENCHMARK_TEST_EXCLUDE"
            elif [ "${GPU:-}" = "PVC" ]; then
                exclude_regex="fp8|config_file_bmg_prefill_bf16|prefill_legacy_xe_config_file_pvc"
            fi
            ;;
        *) echo "ERROR: Unknown test suite '$1'. Use unit, examples, or benchmarks."; return 1 ;;
    esac
    ctest_args=(-R "$test_regex" --output-on-failure)
    [ -n "$exclude_regex" ] && ctest_args+=(-E "$exclude_regex")
    ctest "${ctest_args[@]}"
}

cleanup_build() {
    local remote_build_id="${REMOTE_BUILD_ID:-}"
    if [ "$CI_WORKFLOW_PLATFORM" = "cri" ]; then
        bash "${CI_WORKSPACE}/.github/scripts/cri_device.sh" cleanup || true
    fi
    validate_build_id_tag
    if [ -n "$remote_build_id" ] && [[ "$remote_build_id" =~ ^[0-9]+-${CI_RUN_ID_TAG}-.+-[0-9]+$ ]]; then
        rm -rf "/tmp/cutlass-build-${remote_build_id}" || true
        ssh ${CI_SSH_OPTIONS} "${REMOTE_BUILD_SERVER_USER}@${REMOTE_BUILD_SERVER_HOST}" \
            "docker rm -f cutlass-build-${remote_build_id} >/dev/null 2>&1; sudo -n rm -rf /home/${REMOTE_BUILD_SERVER_USER}/ci-builds/${remote_build_id} /shared/builds/${remote_build_id}" || true
    else
        echo "Skipping cleanup: REMOTE_BUILD_ID '${remote_build_id}' is empty or does not match expected pattern"
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
    build <target>      Build one CMake target; examples use BUILD_HEAVY_JOBS.
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