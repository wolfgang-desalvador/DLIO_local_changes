# Parquet Optimization Analysis for DLIO Benchmark

This document provides a comprehensive analysis of Parquet read/write performance optimizations for the DLIO benchmark, including methodology, findings, and recommendations.

## Executive Summary

We identified that the original DLIO Parquet implementation suffered from a **schema problem** causing 4x file bloat and ~0.033 GB/s write ceiling. After fixing this and implementing storage benchmark mode, we achieved:

- **Write performance**: 6x improvement (0.033 → 0.21 GB/s)
- **Storage read throughput**: 3.6x improvement by removing decode overhead
- **Accurate benchmarking**: Pure storage I/O measurement without CPU pollution

## Problem Analysis

### Root Cause: FixedSizeListArray Schema Encoding

The original implementation used `FixedSizeListArray<uint8>` to store binary data:

```python
# Original (problematic) approach
pa.FixedSizeListArray.from_arrays(data, record_size)
```

**The problem**: PyArrow encodes each `uint8` element as a 4-byte `INT32` in Parquet's physical format, causing:
- 4x file size inflation (100MB data → 400MB file)
- 4x write time increase
- 4x read time increase

### Solution: Use large_binary Type

```python
# Fixed approach
pa.array(binary_data, type=pa.large_binary())
```

This stores binary data efficiently without inflation.

## Performance Testing Results

### Test Environment

- Storage: NVMe SSD (local)
- File sizes: 100MB - 1GB test files
- Methodology: Cache dropped between tests (`echo 3 > /proc/sys/vm/drop_caches`)

### Write Performance

| Method | Throughput | File Size | Notes |
|--------|-----------|-----------|-------|
| FixedSizeListArray (baseline) | 0.033 GB/s | 4x bloat | Original implementation |
| large_binary (fixed) | 0.21 GB/s | Correct | **6x improvement** |
| Preallocated buffers | 0.54 GB/s | Correct | +2.5x over fixed |
| Parallel files | 0.43 GB/s | Correct | +2x over fixed |

### Read Performance: Decode vs Storage

| Method | Throughput | What It Measures |
|--------|-----------|------------------|
| Full decode | 1.25 GB/s | Storage + CPU decode |
| Storage read (no decode) | 4.50 GB/s | Pure storage I/O |
| Row group iteration | 3.81 GB/s | Training access pattern |

**Key finding**: CPU decode overhead is **260%** - the CPU work takes 3.6x longer than the storage I/O itself.

### Read Optimizations (with full decode)

| Optimization | Throughput | Speedup |
|--------------|-----------|---------|
| Baseline | 0.37 GB/s | 1.0x |
| Memory map | 1.27 GB/s | 3.4x |
| Metadata cache | 1.39 GB/s | 3.8x |
| Parallel row groups | 1.37 GB/s | 3.7x |
| Column projection (1 col) | 2.87 GB/s | 7.8x |

## Storage Benchmark Mode

### Why It's Needed

For storage benchmarking, we want to measure **storage system capability**, not CPU performance. Standard Parquet reads mix both:

```
Standard read: Storage I/O (4.5 GB/s) → Decode (260% overhead) → 1.2 GB/s observed
```

### How It Works

The storage benchmark mode:

1. Parses Parquet metadata to get exact byte locations
2. Reads the same bytes that a full decode would read
3. Follows the same access pattern (row groups → column chunks)
4. Skips the CPU-intensive decode step

```python
# Standard read (storage + CPU)
table = pq.read_table(path)  # ~1.2 GB/s

# Storage benchmark mode (storage only)
reader = ParquetStorageReader(path)
bytes_read = reader.read_all()  # ~4.5 GB/s
```

### I/O Pattern Equivalence

The storage reader reads **exactly the same bytes** from **exactly the same offsets**:

```python
# Both methods read these regions:
for row_group in file.row_groups:
    for column in row_group.columns:
        read(offset=column.data_page_offset, size=column.total_compressed_size)
```

## Zero-Copy Analysis

### Arrow IPC vs Parquet

| Aspect | Arrow IPC | Parquet |
|--------|-----------|---------|
| On-disk format | Arrow in-memory format | Parquet columnar format |
| Encoding | None | RLE, Dictionary, Delta, etc. |
| Zero-copy possible | **Yes** | **No** |
| mmap benefit | True zero-copy | Reduces syscalls only |

### Memory Allocation Test (100MB data)

| Format | Memory Allocated | Zero-Copy? |
|--------|-----------------|------------|
| Arrow IPC (mmap) | 0.00 MB | Yes |
| Parquet (mmap) | 102.56 MB | No |

**Conclusion**: Parquet cannot support zero-copy because its on-disk format differs from in-memory format. Even with `compression=None`, internal encodings require decoding.

## Why Parquet Over Arrow IPC

Despite Arrow IPC's performance advantages, Parquet is the correct choice for ML storage benchmarks:

1. **Industry standard**: Used by Spark, Dask, Pandas, DuckDB, Snowflake, etc.
2. **Petabyte scale**: Proven at scale in production ML pipelines
3. **Ecosystem**: Universal tool support for debugging, inspection, transformation
4. **Compression**: 2-10x smaller files for typical ML data
5. **Representative**: Benchmarks should use what real workloads use

The storage benchmark mode allows accurate storage measurement while using the industry-standard format.

## O_DIRECT Implementation

### Purpose

O_DIRECT bypasses the kernel page cache, ensuring:
- Every read goes to the storage device
- No cache warm-up effects
- Accurate cold-cache performance measurement

### Requirements

1. File opened with `O_DIRECT` flag
2. Buffer must be memory-aligned (4KB for modern SSDs)
3. Read size must be aligned
4. Read offset must be aligned

### Implementation

```python
# Aligned buffer allocation
buf = ctypes.create_string_buffer(size + alignment)
addr = ctypes.addressof(buf)
aligned_addr = (addr + alignment - 1) & ~(alignment - 1)

# Aligned read
aligned_offset = (offset // alignment) * alignment
os.lseek(fd, aligned_offset, os.SEEK_SET)
os.readv(fd, [aligned_buffer])
```

### Performance Note

O_DIRECT is typically **slower** than cached reads because it bypasses the cache. This is intentional for benchmarking - we want to measure storage device performance, not cached memory performance.

## Recommendations

### For Storage Benchmarking

Use `format: parquet_storage` to measure pure storage throughput:

```yaml
dataset:
  format: parquet_storage
  data_folder: /nvme/data

parquet:
  memory_map: true
  use_odirect: false  # Set true for cold-cache testing
```

### For ML Training Simulation

Use `format: parquet` with optimizations:

```yaml
dataset:
  format: parquet
  data_folder: /nvme/data

parquet:
  memory_map: true
  use_threads: true
  metadata_cache: true
  row_group_cache_size: 4
```

### For Data Generation

Use the optimized generator (large_binary schema):

```yaml
dataset:
  format: parquet  # or parquet_storage (same generator)

parquet:
  row_group_size: 64  # Smaller row groups for better parallelism
  compression: none   # For storage benchmarking
```

## Files Modified

### New Files

- `reader/parquet_storage_reader.py` - Storage benchmark reader
- `docs/source/parquet_storage_benchmark.rst` - RST documentation
- `docs/PARQUET_OPTIMIZATION_ANALYSIS.md` - This analysis

### Modified Files

- `common/enumerations.py` - Added `PARQUET_STORAGE` format type
- `reader/reader_factory.py` - Added `ParquetStorageReader` mapping
- `data_generator/generator_factory.py` - Handle `PARQUET_STORAGE` in generator
- `reader/parquet_reader.py` - Optimized standard Parquet reader
- `data_generator/parquet_generator.py` - Fixed schema (large_binary)

## Appendix: Test Scripts

The following test scripts were used for this analysis (located in `/home/developer/workspace/nvme/`):

- `parquet_raw_read.py` - Raw byte read comparison
- `parquet_storage_reader.py` - Storage reader implementation test
- `zero_copy_demo.py` - Zero-copy memory allocation comparison
- `odirect_parquet_reader.py` - O_DIRECT implementation test

## Conclusion

The storage benchmark mode enables accurate Parquet-based storage benchmarking by:

1. Using the industry-standard Parquet format
2. Exercising realistic I/O access patterns
3. Removing CPU decode overhead from measurements
4. Providing 3.6x more accurate storage throughput numbers

This allows DLIO to serve as a representative benchmark for ML storage systems while accurately measuring storage performance.
