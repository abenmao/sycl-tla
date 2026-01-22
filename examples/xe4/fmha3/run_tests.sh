# export L0SIM_GRITS_AUBLOAD_OPTS="-attr EU.XESIM 3 -sim_mode perf_mpu -enableFeature lscChangesForSystolic -enableDcGpgpuPrintMessage -enableFeature :stridedMatrixCopy -enableFeature tpSupportAttention -enablefeature :distributedSharedMemoryXe4      -cb_cfg hang_detect_enable true hang_threshold 20 hang_max_hang_cycles 1   -cb_cfg device_info_json_enable true -msglevel verbose -cb_cfg stats_enable true tg_interval 100 tg_cfg_filename TG_ArchTarget_xe4.txt -cb_cfg dump_stats_on_eu_thread_busy true staging_dump_stats_on_swtag true"
export L0SIM_GRITS_AUBLOAD_OPTS="-attr EU.XESIM 3 -sim_mode perf_mpu -enableFeature lscChangesForSystolic -enableDcGpgpuPrintMessage -enableFeature :stridedMatrixCopy -enableFeature tpSupportAttention -enablefeature :distributedSharedMemoryXe4      -cb_cfg hang_detect_enable true hang_threshold 20 hang_max_hang_cycles 1   -cb_cfg device_info_json_enable true -msglevel verbose -dtrace -cb_cfg stats_enable true tg_interval 100 tg_cfg_filename TG_ArchTarget_xe4.txt -cb_cfg dump_stats_on_eu_thread_busy true staging_dump_stats_on_swtag true"

# default case
# ./examples/xe4/fmha/test_sdpa_bfloat16 \
#     --batch=1 \
#     --num_heads=1 \
#     --seq_len_qo=512 \
#     --seq_len_kv=8192 \
#     --head_size_vo=128 \
#     --head_size_qk=128

# case 1
./examples/xe4/fmha/test_sdpa_bfloat16 \
    --batch=2 \
    --num_heads=1 \
    --seq_len_qo=1024 \
    --seq_len_kv=1024 \
    --head_size_vo=128 \
    --head_size_qk=128


# case 2
# ./examples/xe4/fmha/test_sdpa_bfloat16 \
#     --batch=4 \
#     --num_heads=16 \
#     --seq_len_qo=128 \
#     --seq_len_kv=128 \
#     --head_size_vo=128 \
#     --head_size_qk=128
