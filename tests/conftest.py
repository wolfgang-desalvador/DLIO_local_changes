import os
import pytest

# Object-storage tests are disabled unless DLIO_OBJECT_STORAGE_TESTS=1 is set.
# Each object-storage test module also enforces this with a module-level
# pytest.skip(), so these tests are safe to collect without an object-storage
# endpoint — they simply skip.
#
# CI sets DLIO_OBJECT_STORAGE_TESTS=0 explicitly so the value is never missing
# from the build log.  Developers with a live endpoint set it to 1.
OBJECT_STORAGE_TESTS_ENABLED = os.environ.get("DLIO_OBJECT_STORAGE_TESTS", "0") == "1"

# Named output directory for all DLIO benchmark tests.
# Prevents DLIO from creating an ambiguous 'output/' folder in the working
# directory.  Override with DLIO_TEST_OUTPUT_DIR before running pytest:
#   DLIO_TEST_OUTPUT_DIR=my_results pytest dlio_benchmark/tests/
_DEFAULT_TEST_OUTPUT = 'dlio_test_output'
DLIO_TEST_OUTPUT_DIR = os.environ.get('DLIO_TEST_OUTPUT_DIR', _DEFAULT_TEST_OUTPUT)

# Set DLIO_OUTPUT_FOLDER so that _apply_env_overrides() in config.py picks it
# up for every DLIOBenchmark instantiation, including standalone calls that
# don't go through a test-local run_benchmark() wrapper.
os.environ.setdefault('DLIO_OUTPUT_FOLDER', DLIO_TEST_OUTPUT_DIR)

# Cap auto-sized thread counts during tests so the suite runs predictably on
# small CI runners (GitHub Actions: 2 cores) and doesn't saturate large dev
# machines.  Set DLIO_MAX_AUTO_THREADS in the environment to override.
os.environ.setdefault('DLIO_MAX_AUTO_THREADS', '2')


# HACK: to fix the reinitialization problem
def pytest_configure(config):
    config.is_dftracer_initialized = False
