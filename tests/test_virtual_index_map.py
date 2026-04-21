"""
Tests for VirtualIndexMap — the memory-efficient sample index map
that replaces the materialized dict in get_global_map_index().
 
Addresses: mlcommons/storage#329
"""
import sys
import os
import numpy as np
import pytest
 
# Add project root so we can import VirtualIndexMap standalone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
 
# We import VirtualIndexMap directly to avoid pulling in hydra/omegaconf deps
# during lightweight unit testing.  The class only depends on numpy + os.
import importlib
import ast
 
 
def _load_virtual_index_map():
    """
    Load VirtualIndexMap without importing the full config module
    (which requires hydra, omegaconf, mpi4py, etc.).
    """
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "dlio_benchmark", "utils", "config.py"
    )
    with open(config_path) as f:
        source = f.read()
 
    # Parse just the VirtualIndexMap class from the source
    tree = ast.parse(source)
    class_source_lines = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "VirtualIndexMap":
            class_source_lines = (node.lineno, node.end_lineno)
            break
 
    if class_source_lines is None:
        raise RuntimeError("VirtualIndexMap class not found in config.py")
 
    lines = source.split("\n")
    class_source = "\n".join(lines[class_source_lines[0] - 1 : class_source_lines[1]])
 
    # Create a minimal module environment
    env = {"np": np, "os": os, "__builtins__": __builtins__}
 
    # Mock StorageType enum
    class _MockStorageType:
        LOCAL_FS = "local_fs"
    env["StorageType"] = _MockStorageType
 
    exec(class_source, env)
    return env["VirtualIndexMap"]
 
 
VirtualIndexMap = _load_virtual_index_map()
 
 
# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------
 
 
class TestVirtualIndexMapCorrectness:
    """Verify VirtualIndexMap produces the same mapping as the old dict approach."""
 
    def _build_reference_dict(self, file_list, num_samples_per_file, start, end, seed=None):
        """Reproduce the old get_global_map_index logic exactly."""
        sample_list = np.arange(start, end + 1)
        if seed is not None:
            np.random.seed(seed)
            np.random.shuffle(sample_list)
        result = {}
        for i in range(len(sample_list)):
            g = int(sample_list[i])
            fi = g // num_samples_per_file
            si = g % num_samples_per_file
            result[g] = (os.path.abspath(file_list[fi]), si)
        return result
 
    def test_basic_mapping_no_shuffle(self):
        files = [f"/data/file_{i}.parquet" for i in range(10)]
        spf = 1000
        vmap = VirtualIndexMap(files, spf, 0, 9999, storage_type="local_fs")
        ref = self._build_reference_dict(files, spf, 0, 9999)
 
        for idx in [0, 1, 999, 1000, 5555, 9999]:
            assert vmap[idx] == ref[idx], f"Mismatch at idx={idx}: {vmap[idx]} != {ref[idx]}"
 
    def test_basic_mapping_with_shuffle(self):
        files = [f"/data/file_{i}.parquet" for i in range(10)]
        spf = 1000
        seed = 42
        vmap = VirtualIndexMap(files, spf, 0, 9999, shuffle_seed=seed, storage_type="local_fs")
        ref = self._build_reference_dict(files, spf, 0, 9999, seed=seed)
 
        for idx in ref:
            assert vmap[idx] == ref[idx], f"Mismatch at idx={idx}"
 
    def test_partial_range(self):
        """Simulate rank 1 of 4 (only a slice of the total samples)."""
        files = [f"/data/file_{i}.parquet" for i in range(10)]
        spf = 1000
        start, end = 2500, 4999
        vmap = VirtualIndexMap(files, spf, start, end, storage_type="local_fs")
 
        assert len(vmap) == 2500
        assert vmap[2500] == (os.path.abspath(files[2]), 500)
        assert vmap[4999] == (os.path.abspath(files[4]), 999)
 
    def test_items_iteration(self):
        """Verify .items() yields all entries with correct mappings."""
        files = [f"/data/f{i}.parquet" for i in range(5)]
        spf = 100
        vmap = VirtualIndexMap(files, spf, 0, 499, storage_type="local_fs")
 
        items_list = list(vmap.items())
        assert len(items_list) == 500
 
        for gidx, (fname, sidx) in items_list:
            expected_file = gidx // spf
            expected_sample = gidx % spf
            assert fname == os.path.abspath(files[expected_file])
            assert sidx == expected_sample
 
 
# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------
 
 
class TestVirtualIndexMapDeterminism:
    def test_shuffle_deterministic_same_seed(self):
        files = [f"/data/f{i}.parquet" for i in range(10)]
        v1 = VirtualIndexMap(files, 1000, 0, 9999, shuffle_seed=42)
        v2 = VirtualIndexMap(files, 1000, 0, 9999, shuffle_seed=42)
        assert np.array_equal(v1._sample_list, v2._sample_list)
 
    def test_shuffle_different_with_different_seed(self):
        files = [f"/data/f{i}.parquet" for i in range(10)]
        v1 = VirtualIndexMap(files, 1000, 0, 9999, shuffle_seed=42)
        v2 = VirtualIndexMap(files, 1000, 0, 9999, shuffle_seed=99)
        assert not np.array_equal(v1._sample_list, v2._sample_list)
 
    def test_no_shuffle_when_seed_is_none(self):
        files = [f"/data/f{i}.parquet" for i in range(10)]
        v = VirtualIndexMap(files, 1000, 0, 9999, shuffle_seed=None)
        expected = np.arange(0, 10000)
        assert np.array_equal(v._sample_list, expected)
 
 
# ---------------------------------------------------------------------------
# Interface / compatibility tests
# ---------------------------------------------------------------------------
 
 
class TestVirtualIndexMapInterface:
    def test_contains(self):
        files = [f"/data/f{i}.parquet" for i in range(5)]
        vmap = VirtualIndexMap(files, 100, 200, 499)
        assert 200 in vmap
        assert 499 in vmap
        assert 199 not in vmap
        assert 500 not in vmap
 
    def test_len(self):
        files = [f"/data/f{i}.parquet" for i in range(5)]
        vmap = VirtualIndexMap(files, 100, 0, 499)
        assert len(vmap) == 500
 
    def test_iter(self):
        files = [f"/data/f{i}.parquet" for i in range(2)]
        vmap = VirtualIndexMap(files, 100, 0, 199)
        indices = list(vmap)
        assert len(indices) == 200
 
    def test_repr(self):
        files = [f"/data/f{i}.parquet" for i in range(10)]
        vmap = VirtualIndexMap(files, 1000, 0, 9999)
        r = repr(vmap)
        assert "VirtualIndexMap" in r
        assert "10000" in r  # samples count
        assert "10" in r  # files count
 
    def test_non_local_fs_paths_not_resolved(self):
        files = ["bucket/prefix/file_0.parquet", "bucket/prefix/file_1.parquet"]
        vmap = VirtualIndexMap(files, 100, 0, 199, storage_type="s3")
        fname, _ = vmap[0]
        assert fname == "bucket/prefix/file_0.parquet"  # not os.path.abspath'd
 
 
# ---------------------------------------------------------------------------
# Memory tests
# ---------------------------------------------------------------------------
 
 
class TestVirtualIndexMapMemory:
    def test_memory_scales_linearly_with_samples(self):
        """Verify memory is O(N) with small constant (numpy array only)."""
        import tracemalloc
 
        files = [f"/data/f{i}.parquet" for i in range(100)]
        n_samples = 1_000_000
 
        tracemalloc.start()
        vmap = VirtualIndexMap(files, n_samples // 100, 0, n_samples - 1)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
 
        # numpy int64 array: 1M * 8 bytes = 8 MB
        # With overhead, should be well under 50 MB
        assert peak < 50_000_000, (
            f"Peak memory {peak / 1e6:.1f} MB for {n_samples} samples — "
            f"expected < 50 MB"
        )
 
    def test_no_materialized_dict_at_scale(self):
        """At 10M samples, a materialized dict would use ~2 GB.
        VirtualIndexMap should stay under 100 MB."""
        import tracemalloc
 
        files = [f"/data/f{i}.parquet" for i in range(100)]
        n_samples = 10_000_000
 
        tracemalloc.start()
        vmap = VirtualIndexMap(files, n_samples // 100, 0, n_samples - 1)
        _ = vmap[0]  # trigger any lazy init
        _ = vmap[n_samples - 1]
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
 
        # 10M * 8 bytes = 80 MB for numpy array
        assert peak < 150_000_000, (
            f"Peak memory {peak / 1e6:.1f} MB for {n_samples} samples — "
            f"expected < 150 MB (materialized dict would use ~2 GB)"
        )
 
 
# ---------------------------------------------------------------------------
# Samples sum verification
# ---------------------------------------------------------------------------
 
 
class TestSamplesSum:
    def test_samples_sum_matches_reference(self):
        """Verify numpy sum matches the old Python loop sum."""
        files = [f"/data/f{i}.parquet" for i in range(5)]
        spf = 100
        start, end = 0, 499
        seed = 42
 
        # Old approach
        sample_list = np.arange(start, end + 1)
        np.random.seed(seed)
        np.random.shuffle(sample_list)
        old_sum = sum(int(x) for x in sample_list)
 
        # New approach
        vmap = VirtualIndexMap(files, spf, start, end, shuffle_seed=seed)
        new_sum = int(np.sum(vmap._sample_list, dtype=np.int64))
 
        assert old_sum == new_sum
 
 
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
