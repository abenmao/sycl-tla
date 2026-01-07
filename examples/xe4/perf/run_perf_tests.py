#!/usr/bin/env python3
"""
Kernel Performance Test Runner

This script runs individual kernel performance tests (GEMM, Flash Attention, etc.), captures their output,
extracts performance metrics from simulator output, and generates a CSV report.

Usage:
    python run_perf_tests.py --perf_binary ./kernel_perf --simulator_dir ~/sim_dir [--keep_json]
"""

import argparse
import subprocess
import os
import sys
import csv
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Global configuration
METRICS_PARSER_SCRIPT = "cobalt_json_time_anaylze.py"
SIMULATOR_JSON_FILE = "xesim_hltv.json"
PERF_TEST_RESULT_CSV = "perf_test_results.csv"
JSON_FILE_TIMEOUT_SECONDS = 20  # Max time to wait for JSON file generation
METRICS_PARSER_TIMEOUT_SECONDS = 10  # Max time to wait for metrics parser execution
TEST_EXECUTION_TIMEOUT_SECONDS = 6000  # Max time to wait for individual test execution

class PerfTestRunner:
    def __init__(self, perf_binary_path: str, simulator_dir: Optional[str], output_dir: Optional[str], keep_json: bool = False):
        self.perf_binary_path = Path(perf_binary_path).resolve()
        self.keep_json = keep_json
        self.csv_file = PERF_TEST_RESULT_CSV

        # Validate performance test binary
        if not self.perf_binary_path.exists():
            raise FileNotFoundError(f"Performance test binary not found: {self.perf_binary_path}")

        # Handle output_dir (can be None for list_tests_only)
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None

        # Handle simulator_dir (can be None for list_tests_only)
        if simulator_dir:
            self.simulator_dir = Path(simulator_dir).expanduser().resolve()
            self.json_file = self.simulator_dir / SIMULATOR_JSON_FILE

            if not self.simulator_dir.exists():
                raise FileNotFoundError(f"Simulator directory not found: {self.simulator_dir}")
        else:
            self.simulator_dir = None
            self.json_file = None

    def get_test_list(self, list_only: bool = False) -> List[str]:
        """Get list of all available gtests from the binary.

        Args:
            list_only: If True, print formatted test list and return empty list.
                      If False, return list of test names for execution.
        """
        try:
            if list_only:
                print(f"📋 Available Performance Tests from {self.perf_binary_path.name}:")
                print("=" * 60)
            else:
                print(f"Getting test list from {self.perf_binary_path}")

            result = subprocess.run(
                [str(self.perf_binary_path), "--gtest_list_tests"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                error_msg = f"Failed to get the list of tests (return code {result.returncode})"
                if result.stderr:
                    error_msg += f": {result.stderr}"
                assert False, error_msg

            # Parse gtest list output
            tests = []
            lines = result.stdout.split('\n')
            current_suite = ""
            test_count = 0

            for line in lines:
                if line.strip() == "":
                    continue
                if line.strip().endswith('...'):
                    line = line[:-3] # Remove trailing '...' which comes from long test names
                if line.strip().endswith('.'):
                    # Test suite line
                    current_suite = line.strip()[:-1]  # Remove trailing dot
                    if list_only:
                        print(f"\n{current_suite}:")
                elif line.startswith('  '):
                    # Test case line
                    test_case = line.strip()
                    if current_suite and test_case:
                        full_test_name = f"{current_suite}.{test_case}"
                        tests.append(full_test_name)
                        if list_only:
                            print(f"{full_test_name}")
                            test_count += 1
                else:
                    print("Warning: Unrecognized gtest test format:", line)

            if list_only:
                print(f"\n📊 Total: {test_count} tests available")
                return []
            else:
                assert tests, "No tests found in the binary"
                #print(f"Found {len(tests)} tests:")
                # for test in tests:
                #     print(f"  {test}")
                return tests

        except subprocess.TimeoutExpired:
            if list_only:
                print("❌ Timeout while listing tests")
                return []
            else:
                assert False, f"Timeout while getting test list"
        except Exception as e:
            if list_only:
                print(f"❌ Error listing tests: {e}")
                return []
            else:
                assert False, f"Error getting test list: {e}"

    def run_single_test(self, test_name: str) -> bool:
        """Run a single test and capture its output."""
        # Generate XML filename based on test name
        if '#' in test_name:
            # Handle parameterized test names with '#'
            xml_name = test_name.split('#')[0].strip()
            xml_name = xml_name.replace('/', '_').replace('.', '_')
        else:
            # Handle regular test names without '#'
            xml_name = test_name.replace('/', '_').replace('.', '_')

        xml_name = self.output_dir / f"{xml_name}_test_report.xml"
        print(f"📄 XML report will be saved as: {xml_name.name}")

        # clean test name
        clean_test_name = test_name.split('#')[0].strip()
        print(f"▶️ Running test (clean test name): {clean_test_name}")

        try:
            # delete existing JSON file before running test
            if self.json_file.exists():
                self.json_file.unlink()

            # Run the test
            cmd = [str(self.perf_binary_path), f"--gtest_filter=*{clean_test_name}", f"--gtest_output=xml:{xml_name}", "--gtest_color=no"]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Stream output to console with proper timeout handling
            try:
                # Use communicate() with timeout - this properly handles hanging processes
                stdout, _ = process.communicate(timeout=TEST_EXECUTION_TIMEOUT_SECONDS)

                # Print the captured output
                for line in stdout.splitlines():
                    print(line)

                success = process.returncode == 0
            except subprocess.TimeoutExpired:
                print(f"⚠️ Test {test_name} timed out after {TEST_EXECUTION_TIMEOUT_SECONDS} seconds")
                process.kill()
                # Get any partial output before killing
                try:
                    stdout, _ = process.communicate(timeout=1)
                    for line in stdout.splitlines():
                        print(line)
                except subprocess.TimeoutExpired:
                    pass
                success = False

            # Check if XML file was generated, if not, create a fallback XML
            xml_path = Path(xml_name)
            if not xml_path.exists():
                print(f"⚠️ XML file not found - gtest may have crashed. Generating fallback XML: {xml_name}")
                self._generate_fallback_xml(test_name, xml_name)
                success = False  # Mark as failure since gtest crashed

            if success:
                print(f"✅ Test {test_name} completed successfully")
            else:
                print(f"❌ Test {test_name} failed with return code {process.returncode}")

            return success

        except Exception as e:
            print(f"❌ Error running test {test_name}: {e}")
            return False

    def _generate_fallback_xml(self, test_name: str, xml_path: str):
        """Generate a fallback XML file when gtest crashes or doesn't produce output."""
        import xml.etree.ElementTree as ET
        from xml.dom import minidom

        # Parse test name to get suite and case
        if '.' in test_name:
            suite_name, case_name = test_name.split('.', 1)
        else:
            suite_name = "UnknownSuite"
            case_name = test_name

        # Create XML structure matching gtest format
        testsuites = ET.Element('testsuites')
        testsuites.set('tests', '1')
        testsuites.set('failures', '1')
        testsuites.set('disabled', '0')
        testsuites.set('errors', '0')
        testsuites.set('timestamp', time.strftime('%Y-%m-%dT%H:%M:%S'))
        testsuites.set('time', '0.001')
        testsuites.set('name', 'AllTests')

        testsuite = ET.SubElement(testsuites, 'testsuite')
        testsuite.set('name', suite_name)
        testsuite.set('tests', '1')
        testsuite.set('failures', '1')
        testsuite.set('disabled', '0')
        testsuite.set('errors', '0')
        testsuite.set('time', '0.001')

        testcase = ET.SubElement(testsuite, 'testcase')
        testcase.set('name', case_name)
        testcase.set('status', 'run')
        testcase.set('time', '0.001')
        testcase.set('classname', suite_name)

        failure = ET.SubElement(testcase, 'failure')
        failure.set('message', 'Unexpected test failure - gtest crashed or did not generate output')
        failure.set('type', 'UnexpectedFailure')

        # Write XML file with proper formatting
        rough_string = ET.tostring(testsuites, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        with open(xml_path, 'w') as f:
            f.write(pretty_xml)

    def extract_metrics(self, test_name: str) -> Dict[str, Optional[float]]:
        f"""Extract performance metrics using {METRICS_PARSER_SCRIPT}"""
        metrics = {
            'kernel_execution_time_us': None,
            'XeCore_Active_time_us': None
        }

        # check if METRICS_PARSER_SCRIPT exist in simulator_dir
        metrics_parser_path = self.simulator_dir / METRICS_PARSER_SCRIPT
        if not metrics_parser_path.exists():
            print(f"⚠️  Metrics parser script not found: {metrics_parser_path}")
            return metrics

        # Wait for JSON file to be generated by simulator (max 10 seconds)
        print(f"⏳ Waiting for simulator to generate {SIMULATOR_JSON_FILE}...")
        json_file_timeout = JSON_FILE_TIMEOUT_SECONDS  # seconds
        json_file_wait_interval = 0.5  # check every 500ms
        json_file_elapsed = 0

        while not self.json_file.exists() and json_file_elapsed < json_file_timeout:
            time.sleep(json_file_wait_interval)
            json_file_elapsed += json_file_wait_interval

        if not self.json_file.exists():
            print(f"⚠️  Timeout: {SIMULATOR_JSON_FILE} not generated within {json_file_timeout} seconds")
            return metrics

        #print(f"✅ {SIMULATOR_JSON_FILE} generated after {json_file_elapsed:.1f} seconds")

        try:
            # Run metrics parser to extract metrics
            metrics_parser_cmd = [
                'python', METRICS_PARSER_SCRIPT,
                '-hltv', str(self.json_file)
            ]

            print(f"Extracting metrics with: {' '.join(metrics_parser_cmd)}")
            result = subprocess.run(
                metrics_parser_cmd,
                capture_output=True,
                text=True,
                timeout=METRICS_PARSER_TIMEOUT_SECONDS,
                cwd=self.simulator_dir  # Run from simulator directory
            )

            if result.returncode != 0:
                print(f"⚠️  {METRICS_PARSER_SCRIPT} failed (return code {result.returncode})")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                return metrics


            # Parse the output to extract metrics
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                line = line.strip().lower()
                if 'kernel execution time:' in line and 'us' in line:
                    # Extract number from line like "kernels Execution Time: 401 us"
                    try:
                        value = float(line.split(':')[-1].split('(')[0].strip().replace('us', '').strip())
                        metrics['kernel_execution_time_us'] = value
                    except (ValueError, IndexError) as e:
                        print(f"⚠️  Failed to parse kernel execution time from: {line}")

                if 'xecore active time =' in line and 'us' in line:
                    # Extract number from line like "XeCore Active Time = 196 us"
                    try:
                        value = float(line.split('=')[-1].strip().replace('us', '').replace(')', '').strip())
                        metrics['xecore_active_time_us'] = value
                    except (ValueError, IndexError) as e:
                        print(f"⚠️  Failed to parse XeCore active time from: {line}")

                # print(f"Extracted metrics: {metrics}")

        except subprocess.TimeoutExpired:
            print(f"⚠️  Timeout while running {METRICS_PARSER_SCRIPT}")
        except Exception as e:
            print(f"⚠️  Error extracting metrics: {e}")

        return metrics

    def cleanup_json_file(self):
        """Remove the JSON file if keep_json is False."""
        if not self.keep_json and self.json_file.exists():
            try:
                self.json_file.unlink()
                print(f"🧹 Cleaned up {self.json_file}")
            except Exception as e:
                print(f"⚠️  Failed to clean up JSON file: {e}")

    def initialize_csv(self):
        """Initialize the CSV file with headers."""
        headers = [
            'test_name',
            'status',
            'kernel_execution_time_us',
            'xecore_active_time_us',
        ]

        with open(self.csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

        print(f"📊 Initialized CSV file: {self.csv_file}")

    def add_result_to_csv(self, test_name: str, success: bool, metrics: Dict[str, Optional[float]]):
        """Add test result to CSV file."""
        row = [
            test_name,
            'PASS' if success else 'FAIL',
            metrics.get('kernel_execution_time_us', ''),
            metrics.get('xecore_active_time_us', ''),
        ]

        with open(self.csv_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

        #print(f"📝 Added result to CSV: {test_name} -> {row[1]}")

    def run_single_specific_test(self, test_name: str):
        """Run a single specific test by name and generate the performance report."""
        print(f"Running Test: {test_name}")

        # Initialize CSV
        self.initialize_csv()

        print(f"\n🧪 Running test: {test_name}")

        # Run the test
        success = self.run_single_test(test_name)

        # Extract metrics
        metrics = self.extract_metrics(test_name)

        # Add to CSV
        self.add_result_to_csv(test_name, success, metrics)

        # Clean up JSON file
        self.cleanup_json_file()

        # Print summary
        print(f"\n{'='*60}")
        print("📋 TEST RESULT")
        print(f"{'='*60}")
        print(f"Test: {test_name}")
        if success:
            print(f"Result: PASSED ✅")
        else:
            print(f"Result: FAILED ❌")
        print(f"CSV Report: {self.csv_file}")
        print(f"{'='*60}")

    def run_all_tests(self, test_type_filter: Optional[str] = None):
        """Run all tests and generate the performance report.

        Args:
            test_type_filter: Optional filter to run only tests containing this string
        """
        print("Starting Kernel Performance Test Runner")
        if test_type_filter:
            print(f"Test Type Filter: {test_type_filter}")

        # Get test list
        tests = self.get_test_list()
        if not tests:
            print("❌ No tests found!")
            return

        # Filter tests by test_type if specified
        if test_type_filter:
            original_count = len(tests)
            tests = [test for test in tests if test_type_filter in test]
            filtered_count = len(tests)
            print(f"📋 Filtered tests: {filtered_count}/{original_count} tests match '{test_type_filter}'")

            if not tests:
                print(f"❌ No tests found matching filter '{test_type_filter}'!")
                return

        print(f"Running {len(tests)} tests...")

        # Initialize CSV
        self.initialize_csv()

        # Run each test
        passed = 0
        failed = 0

        for i, test_name in enumerate(tests, 1):
            print(f"\n🧪 Running test {i}/{len(tests)}: {test_name}")

            # Run the test
            success = self.run_single_test(test_name)

            # Extract metrics
            metrics = self.extract_metrics(test_name)

            # Add to CSV
            self.add_result_to_csv(test_name, success, metrics)

            # Clean up JSON file
            self.cleanup_json_file()

            # Update counters
            if success:
                passed += 1
            else:
                failed += 1

        # Print summary
        print(f"\n{'='*60}")
        print("📋 FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {len(tests)}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"CSV Report: {self.csv_file}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="""
🚀 Kernel Performance Test Runner for XE4 Hardware

This tool automates the execution of kernel performance tests (GEMM, Flash Attention, etc.)
on XE4 hardware using various libraries. It manages test execution, output collection,
metric extraction, and result reporting with comprehensive CSV output.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📚 USAGE EXAMPLES:

Basic Usage:
  # List available tests without running them
  python run_perf_tests.py --perf_binary ./build/gemm_perf --list_tests

  # Run all tests with real-time console output
  python run_perf_tests.py --perf_binary ./build/gemm_perf --simulator_dir ~/xe4_simulator --output_dir ./results --test_all

  # Run only L2 cache tests
  python run_perf_tests.py --perf_binary ./build/gemm_perf --simulator_dir ~/xe4_simulator --output_dir ./results --test_all --test_type l2

  # Run only L3 cache tests
  python run_perf_tests.py --perf_binary ./build/gemm_perf --simulator_dir ~/xe4_simulator --output_dir ./results --test_all --test_type l3

Development & Debugging:
  # Keep JSON files for manual analysis
  python run_perf_tests.py --perf_binary ./gemm_perf --simulator_dir ~/sim --output_dir ./results --test_all --keep_json

Single Test Execution:
  # List tests first to find test names
  python run_perf_tests.py --perf_binary ./gemm_perf --list_tests

  # Run specific test by name
  python run_perf_tests.py --perf_binary ./gemm_perf --simulator_dir ~/sim --output_dir ./results -t "GemmOperator/0"

  # Run single test with JSON file retention
  python run_perf_tests.py --perf_binary ./gemm_perf --simulator_dir ~/sim --output_dir ./results \
    -t "GemmOperator/1" --keep_json

Output Files:
  • CSV Report: perf_test_results.csv (always generated)
  • XML Reports: Individual test result XML files (always generated)
  • JSON Files: {SIMULATOR_JSON_FILE} (when --keep_json used)

Prerequisites:
  • Built performance test binary (GEMM, Flash Attention, etc.)
  • XE4 simulator environment with {METRICS_PARSER_SCRIPT} available
        """
    )

    parser.add_argument(
        '--perf_binary',
        required=True,
        metavar='PATH',
        help="""Path to the performance test executable binary.
                This should be built from performance test examples.
                Example: ./build/examples/xe4/perf_test/gemm_perf or ./build/flash_attention_perf"""
    )

    parser.add_argument(
        '--simulator_dir',
        required=False,
        metavar='DIR',
        help=f"""Path to the XE4 simulator directory containing {METRICS_PARSER_SCRIPT}.
                This script is used to extract performance metrics from
                simulator output files.
                Not required when using --list_tests."""
    )

    parser.add_argument(
        '--output_dir',
        required=False,
        metavar='DIR',
        help="""Path to the output directory where XML test reports will be generated.
                This directory will be created if it doesn't exist.
                Each test will generate an XML file with gtest results.
                Not required when using --list_tests."""
    )

    parser.add_argument(
        '--keep_json',
        action='store_true',
        help=f"""{SIMULATOR_JSON_FILE} is created for each test and gets overwritten.
                By default, this intermediate JSON files are deleted after test completion.
                Use this flag to retain it for manual analysis or debugging purposes."""
    )

    parser.add_argument(
        '-t', '--test',
        dest='single_test',
        metavar='TEST_NAME',
        help="""Run only the specified test instead of the full suite.
                Use the exact test name as shown by --gtest_list_tests.
                Examples: "GemmOperator/0", "GemmOperator/5"""
    )

    parser.add_argument(
        '--test_all',
        action='store_true',
        help="""Run all available tests in the test suite.
                This explicitly runs the complete test suite and generates
                performance metrics for every test. Use this flag to be
                explicit about running all tests."""
    )

    parser.add_argument(
        '--test_type',
        choices=['l2', 'l3'],
        metavar='TYPE',
        help="""Filter tests by type. Only run tests containing the specified
                type string in their test name. Valid values: 'l2' or 'l3'.
                Example: --test_type l2 will only run tests with 'l2' in their names."""
    )

    parser.add_argument(
        '--list_tests',
        action='store_true',
        help="""List all available tests and exit without running them.
                This shows the same output as running the binary with
                --gtest_list_tests flag. Useful for finding test names
                to use with the -t option."""
    )

    args = parser.parse_args()

    # Validate arguments
    if args.list_tests:
        # For listing tests, we only need the gemm_perf binary
        try:
            # Create a minimal runner just for listing tests
            runner = PerfTestRunner(args.perf_binary, None, None, args.keep_json)
            runner.get_test_list(list_only=True)
            return
        except Exception as e:
            print(f"❌ Error listing tests: {e}")
            sys.exit(1)

    # For running tests, both simulator_dir and output_dir are required
    if not args.simulator_dir:
        print("❌ Error: --simulator_dir is required to run GEMM performance tests")
        print("   Use --help for more information")
        sys.exit(1)

    if not args.output_dir:
        print("❌ Error: --output_dir is required to run GEMM performance tests")
        print("   This directory will contain XML test report files.")
        print("   Use --help for more information")
        sys.exit(1)

    # Ensure user specifies either single test or all tests
    if not args.single_test and not args.test_all:
        print("❌ Error: You must specify either -t/--test <TEST_NAME> or --test_all")
        print("   Use --list_tests to see available tests")
        print("   Use --help for more information")
        sys.exit(1)

    # Prevent conflicting test execution options
    if args.single_test and args.test_all:
        print("❌ Error: Cannot use both -t/--test and --test_all at the same time")
        print("   Choose either single test execution or full test suite")
        sys.exit(1)

    try:
        runner = PerfTestRunner(args.perf_binary, args.simulator_dir, args.output_dir, args.keep_json)

        if args.single_test:
            runner.run_single_specific_test(args.single_test)
        elif args.test_all:
            runner.run_all_tests(args.test_type)
        else:
            # This should not happen due to validation above
            print("❌ Error: No test execution mode specified")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()