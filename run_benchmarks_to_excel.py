import subprocess
import re
import sys
import os
import pandas as pd
import time

BUILD_DIR = "./build/examples/06_bmg_flash_attention"
BINARY_TEMPLATE = "06_xe_fmha_fwd_{mode}_{dtype}_t_hdim{hdim}"

# Format: (mode, dtype, batch, head_dim, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, is_causal, is_varlen, comments)
TEST_CASES = [
    # #--- bfloat16 Cases ---
    ("prefill", "bfloat16", 64, 1, 96, 8, 4096, 4096, True, False, "BF16 Case 10 Prefill"),
    ("prefill", "bfloat16", 64, 1, 96, 8, 8192, 8192, True, False, "BF16 Case 10 Prefill"),
    ("prefill", "bfloat16", 64, 1, 96, 8, 16384, 16384, True, False, "BF16 Case 10 Prefill"),
    ("prefill", "bfloat16", 64, 1, 96, 8, 32768, 32768, True, False, "BF16 Case 10 Prefill"),

    # ("decode", "bfloat16", 1, 32, 8, 1, 4096, False, False, "BF16 Case 10"),

    # # --- MXFP8 Cases ---
    # ("prefill", "mx_float_e4m3", 1, 4, 4, 512, 512, False, False, "MXFP8 e4m3 Case 1 Prefill"),
    # 
    # ("prefill", "mx_float_e5m2", 1, 4, 4, 512, 512, False, False, "MXFP8 e5m2 Case 5 Prefill"),
    # # --- FP8 Cases ---
    # ("prefill", "float_e4m3", 1, 4, 4, 512, 512, False, False, "FP8 e4m3 Case 1 Prefill"),
]

def save_result(df):
    columns = [
        "#", "API Version", "Decode/Prefill", "data type", 
        "Batch", "NumHeads_q", "NumHeads_kv", "Seq Length QO", "Seq Length KV", 
        "Head Size QK", "Head Size VO", 
        "Causal Mask", "Variable SeqLen", 
        "CRI GB/s", "CRI TFlop/s", "CRI Time (ms)", 
        "CRI flops/clk", "CRI % of peak", 
        "Comments"
    ]
    
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]

    output_file = "benchmark_results.xlsx"
    try:
        df.to_excel(output_file, index=False, engine='openpyxl')
    except Exception as e:
        print(f"\nError saving Excel file: {e}")

def run_command(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return None

def parse_output(output, default_params):
    data = default_params.copy()
    
    int_patterns = {
        'Batch': r"Batch:\s+(\d+)",
        'NumHeads_q': r"NumHeads_q:\s+(\d+)",
        'NumHeads_kv': r"NumHeads_kv:\s+(\d+)",
        'Seq Length QO': r"Seq Length QO:\s+(\d+)",
        'Seq Length KV': r"Seq Length KV:\s+(\d+)",
        'Head Size QK': r"Head Size QK:\s+(\d+)",
        'Head Size VO': r"Head Size VO:\s+(\d+)",
    }
    for key, pattern in int_patterns.items():
        match = re.search(pattern, output)
        if match:
            data[key] = int(match.group(1))

    bool_patterns = {
        'Causal Mask': r"Causal Mask:\s+(\w+)",
        'Variable Sequence Length': r"Variable Sequence Length:\s+(\w+)",
    }
    for key, pattern in bool_patterns.items():
        match = re.search(pattern, output)
        if match:
            val = match.group(1).lower()
            data[key] = "Yes" if val == "true" else "No"

    perf_match = re.search(r"Performance:\s+([\d\.]+)\s+GB/s,\s+([\d\.]+)\s+TFlop/s,\s+([\d\.]+)\s+ms", output)
    if perf_match:
        data['CRI GB/s'] = float(perf_match.group(1))
        data['CRI TFlop/s'] = float(perf_match.group(2))
        data['CRI Time (ms)'] = float(perf_match.group(3))
    else:
        data['CRI GB/s'] = None
        data['CRI TFlop/s'] = None
        data['CRI Time (ms)'] = None

    return data

def main():
    results = []

    print(f"Starting Benchmark Run... ({len(TEST_CASES)} cases)")

    for idx, (mode, dtype, hdim, batch, nh_q, nh_kv, seq_qo, seq_kv, is_causal, is_varlen, comment) in enumerate(TEST_CASES, 1):
        
        binary_name = BINARY_TEMPLATE.format(mode=mode, dtype=dtype, hdim=hdim)
        binary_path = os.path.join(BUILD_DIR, binary_name)
        
        causal_flag = "--is_causal" if is_causal else ""
        varlen_flag = "--varlen" if is_varlen else ""
        
        cmd = (
            f"{binary_path} "
            f"--iterations=100 --batch={batch} --verify=0 "
            f"--num_heads_q={nh_q} --num_heads_kv={nh_kv} "
            f"--seq_len_qo={seq_qo} --seq_len_kv={seq_kv} "
            f"{causal_flag} {varlen_flag}"
        )

        print(f"[{idx}/{len(TEST_CASES)}] Running {comment}...")
        print(cmd)
        start_time = time.time()
        output = run_command(cmd)
        duration = time.time() - start_time
        print(output)
        print(f"Time taken: {duration} seconds")
        default_data = {
            'Batch': batch,
            'NumHeads_q': nh_q,
            'NumHeads_kv': nh_kv,
            'Seq Length QO': seq_qo,
            'Seq Length KV': seq_kv,
            'Head Size QK': 128, 
            'Head Size VO': 128,
            'Causal Mask': 'Yes' if is_causal else 'No',
            'Variable Sequence Length': 'Yes' if is_varlen else 'No'
        }

        row_data = {
            "#": idx,
            "API Version": "New",
            "Decode/Prefill": mode.capitalize(),
            "data type": dtype,
            "Comments": comment,
            "Time (s)": duration
        }

        if output:
            parsed = parse_output(output, default_data)
            row_data.update(parsed)
            row_data.update({
                "CRI flops/clk": None,
                "CRI % of peak": None,
            })
        else:
            row_data.update(default_data)
            row_data["Comments"] += " (Run Failed)"
            for k in ['CRI GB/s', 'CRI TFlop/s', 'CRI Time (ms)']:
                row_data[k] = None

        results.append(row_data)
        print(pd.DataFrame(results)[['NumHeads_q', 'NumHeads_kv', 'Seq Length QO', 'Seq Length KV', 'CRI TFlop/s']])

        df = pd.DataFrame(results)
        save_result(df)
    print(f"\nSuccess! Results saved to benchmark_results.xlsx")


if __name__ == "__main__":
    main()
