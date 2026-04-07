#!/usr/bin/env python3
import sys, csv
from pathlib import Path

def compare_perf(new_csv, baseline_csv, threshold=0.10):
    try:
        # Parse new CSV file and extract test names with their performance times
        # Format: {test_name: (time_in_us, line_number)}
        new_data = {}
        with Path(new_csv).open(encoding='utf-8') as f:
            for i, row in enumerate(csv.reader(f), 1):
                # Skip header row and rows with insufficient data
                if len(row) > 2 and row[2] and row[0] != 'test_name':
                    test_name = row[0]
                    time_value = float(row[2])
                    new_data[test_name] = (time_value, i)

        # Parse baseline CSV file with same format
        base_data = {}
        with Path(baseline_csv).open(encoding='utf-8') as f:
            for i, row in enumerate(csv.reader(f), 1):
                if len(row) > 2 and row[2] and row[0] != 'test_name':
                    test_name = row[0]
                    time_value = float(row[2])
                    base_data[test_name] = (time_value, i)

        # Find tests that exist in both datasets for comparison
        matched = []
        for test_name, (new_time, new_idx) in new_data.items():
            if test_name in base_data:
                baseline_time = base_data[test_name][0]
                matched.append((test_name, new_time, baseline_time, new_idx))

        # Identify new tests that don't have a baseline
        new_tests = []
        for test_name in new_data.keys():
            if test_name not in base_data:
                new_tests.append(test_name)

        # Report new tests without baseline
        if new_tests:
            print(f"[WARNING] {len(new_tests)} new test(s) found without baseline:")
            for test_name in new_tests:
                new_time, new_idx = new_data[test_name]
                print(f"  Test #{new_idx-1}: {test_name}")
                print(f"    Time: {new_time:.2f}us (no baseline for comparison)")
            print()  # Add blank line for readability

        # Identify performance regressions (tests that got slower beyond threshold)
        # Only flag as failure if new time is HIGHER (slower) than baseline + threshold
        failures = []
        for test_name, new_time, baseline_time, test_idx in matched:
            if new_time > baseline_time * (1 + threshold):
                degradation_pct = (new_time - baseline_time) / baseline_time * 100
                failures.append((test_name, new_time, baseline_time, degradation_pct, test_idx))

        # Identify performance improvements (tests that got faster)
        improvements = []
        for test_name, new_time, baseline_time, test_idx in matched:
            if new_time < baseline_time:
                improvement_pct = (baseline_time - new_time) / baseline_time * 100
                improvements.append((test_name, new_time, baseline_time, improvement_pct, test_idx))

        # Write/append to regression.csv with all test results
        output_dir = Path(__file__).parent.parent.parent / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)
        regression_csv_path = output_dir / 'regression.csv'

        # Read existing data if file exists to merge results
        existing_data = {}
        if regression_csv_path.exists():
            with regression_csv_path.open('r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 4:
                        # Store existing test results: {test_name: [baseline, new_val, status]}
                        existing_data[row[0]] = [row[1], row[2], row[3]]

        # Build complete results dictionary (merge existing + new)
        all_results = existing_data.copy()

        # Update with matched tests and their status
        for test_name, new_time, baseline_time, test_idx in matched:
            # Determine status based on performance change
            if test_name in [t[0] for t in failures]:
                status = 'REGRESSION'
            elif test_name in [t[0] for t in improvements]:
                status = 'IMPROVEMENT'
            else:
                status = 'PASSED'

            all_results[test_name] = [f'{baseline_time:.2f}', f'{new_time:.2f}', status]

        # Add new tests without baseline
        for test_name in new_tests:
            new_time, new_idx = new_data[test_name]
            all_results[test_name] = ['', f'{new_time:.2f}', 'NEW_TEST']

        # Write all results back to file
        with regression_csv_path.open('w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['name', 'baseline_value', 'new_value', 'status'])

            # Write all results (sorted by name for consistency)
            for test_name in sorted(all_results.keys()):
                baseline_val, new_val, status = all_results[test_name]
                writer.writerow([test_name, baseline_val, new_val, status])

        print(f"\n[INFO] Regression report written to: {regression_csv_path}")
        print(f"[INFO] Total tests in report: {len(all_results)} (added/updated {len(matched) + len(new_tests)} tests this run)")

        # Check if there are any regressions that exceed the threshold
        if failures:
            print(f"[FAIL] Performance regression detected (>{threshold*100}% degradation):")
            for test_name, new_val, base_val, deg, idx in failures:
                print(f"  Test #{idx-1}: {new_val:.2f}us vs {base_val:.2f}us ({deg:.1f}% slower)")
                print(f"    {test_name}")
            return 1

        # All tests passed - print summary
        print(f"[PASS] All {len(matched)} matched tests within {threshold*100}% of baseline")

        # Show detailed breakdown of all matched tests
        for test_name, new_time, baseline_time, test_idx in sorted(matched, key=lambda x: x[3]):
            diff = ((new_time - baseline_time) / baseline_time) * 100
            if diff < 0:
                print(f"  Test #{test_idx-1}: {new_time:.2f}us vs {baseline_time:.2f}us ({abs(diff):.1f}% faster) ✓")
            elif diff > 0:
                print(f"  Test #{test_idx-1}: {new_time:.2f}us vs {baseline_time:.2f}us (+{diff:.1f}% slower)")
            else:
                print(f"  Test #{test_idx-1}: {new_time:.2f}us vs {baseline_time:.2f}us (no change)")

        return 0
    except FileNotFoundError as e:
        print(f"[ERROR] CSV file not found: {e.filename}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"[ERROR] Invalid numeric value in CSV while processing '{new_csv}' or '{baseline_csv}': {e}", file=sys.stderr)
        return 1
    except IndexError as e:
        print(f"[ERROR] Invalid CSV format (missing columns) in '{new_csv}' or '{baseline_csv}': {e}", file=sys.stderr)
        return 1
    except csv.Error as e:
        print(f"[ERROR] Error parsing CSV file '{new_csv}' or '{baseline_csv}': {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(compare_perf(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.10))
