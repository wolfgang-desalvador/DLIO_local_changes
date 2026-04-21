Parquet Storage Benchmark Mode
==============================

This document describes the Parquet storage benchmark mode in DLIO, which
enables accurate storage system performance measurement by reading Parquet
files without CPU-intensive decoding overhead.

Overview
--------

When benchmarking storage systems for ML workloads, the goal is to measure
**storage throughput** - how fast data can be read from the storage system.
However, standard Parquet reads involve both storage I/O and CPU decoding:

.. code-block:: text

    Standard Parquet Read:
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │   Storage    │───>│   Decode     │───>│    Arrow     │
    │   I/O        │    │   (CPU)      │    │    Table     │
    └──────────────┘    └──────────────┘    └──────────────┘
        ~4.5 GB/s         ~260% overhead       ~1.2 GB/s

The decode step (RLE, dictionary, delta encoding) often takes **longer than
the storage I/O itself**, which pollutes storage benchmarks with CPU overhead.

Storage Benchmark Mode
----------------------

The ``parquet_storage`` format reads raw bytes from Parquet files following
the exact same I/O access pattern, but **skips decoding**:

.. code-block:: text

    Storage Benchmark Mode:
    ┌──────────────┐    ┌──────────────┐
    │   Storage    │───>│    Raw       │
    │   I/O        │    │    Bytes     │
    └──────────────┘    └──────────────┘
        ~4.5 GB/s        (no decode)

This provides:

- **Same I/O pattern**: Reads the exact same bytes from the same file offsets
- **Same access pattern**: Row groups, column chunks, sequential/random access
- **No CPU overhead**: Pure storage throughput measurement
- **Accurate benchmarking**: Measures what storage can actually deliver

Performance Comparison
----------------------

Testing on NVMe storage with a 1GB Parquet file:

=======================  ===========  ====================
Method                   Throughput   Measures
=======================  ===========  ====================
Full decode              1.25 GB/s    Storage + CPU
**Storage read**         **4.50 GB/s** **Storage only**
Row group iteration      3.81 GB/s    Training pattern
=======================  ===========  ====================

The decode overhead is **260%** - CPU work takes 3.6x longer than storage I/O.

Usage
-----

To use storage benchmark mode, set ``format: parquet_storage`` in your
workload configuration:

.. code-block:: yaml

    dataset:
      format: parquet_storage
      data_folder: /path/to/data

    parquet:
      memory_map: true      # Use mmap for efficient access
      use_odirect: false    # Set true to bypass page cache
      row_group_cache_size: 4

Configuration Options
---------------------

The following options can be configured under the ``parquet:`` section:

========================  ========  =========  ==========================================
Option                    Type      Default    Description
========================  ========  =========  ==========================================
memory_map                bool      true       Use memory-mapped file access
use_odirect               bool      false      Use O_DIRECT to bypass page cache
odirect_alignment         int       4096       Alignment for O_DIRECT reads (bytes)
row_group_cache_size      int       4          Number of row groups to cache
========================  ========  =========  ==========================================

O_DIRECT Mode
~~~~~~~~~~~~~

When ``use_odirect: true``, the reader bypasses the kernel page cache entirely.
This is useful for:

- Measuring true storage device throughput (not cached performance)
- Preventing benchmark warm-up effects from polluting results
- Testing storage systems under cold-cache conditions

Note: O_DIRECT requires Linux and may not work on all filesystems.

When to Use Each Mode
---------------------

Use ``parquet_storage`` (storage benchmark mode) when:

- Benchmarking storage system performance
- Comparing different storage backends (NVMe, HDD, network storage)
- Measuring I/O throughput without CPU interference
- Testing storage system limits

Use ``parquet`` (standard mode) when:

- Simulating actual ML training workloads
- Testing end-to-end data pipeline performance
- Measuring combined storage + compute throughput

Technical Details
-----------------

How It Works
~~~~~~~~~~~~

The storage reader performs these steps:

1. **Parse metadata**: Read Parquet footer to get file layout (row group
   locations, column chunk offsets, sizes)

2. **Build I/O map**: Create list of (offset, size) regions matching what
   a full decode would read

3. **Read raw bytes**: Read each region using mmap or direct I/O, following
   the same access pattern as PyArrow

4. **Skip decode**: Return after reading bytes - no RLE, dictionary, or
   other encoding is decoded

Why Parquet Cannot Support Zero-Copy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unlike Arrow IPC, Parquet **fundamentally cannot support zero-copy reads**:

.. code-block:: text

    Arrow IPC (zero-copy possible):
    - On-disk format = In-memory format
    - mmap'd file IS the Arrow buffer
    - 0 bytes allocated

    Parquet (zero-copy impossible):
    - On-disk format ≠ In-memory format
    - Data is encoded (RLE, dictionary, delta, bit-packing)
    - 100% of data must be allocated and decoded

Even with ``compression=None``, Parquet uses internal encodings that require
decoding. This is a fundamental format difference, not an implementation
choice.

Memory Allocation Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Testing with 100MB of float64 data:

===================  ==================  ==============
Format               Memory Allocated    Zero-Copy?
===================  ==================  ==============
Arrow IPC (mmap)     0.00 MB             Yes
Parquet (mmap)       102.56 MB           No
===================  ==================  ==============

The Parquet reader must allocate memory for decoding regardless of mmap usage.

Why Parquet is the Right Choice for ML Storage Benchmarks
---------------------------------------------------------

Despite Arrow IPC's performance advantages, **Parquet is the industry standard
for petabyte-scale ML datasets**:

- **Universal ecosystem support**: Spark, Dask, Pandas, DuckDB, etc.
- **Compression efficiency**: 2-10x smaller files for typical ML data
- **Column pruning**: Read only needed columns
- **Predicate pushdown**: Filter at storage level
- **Interoperability**: Works across all ML frameworks

A storage benchmark should use **the format that real workloads use**.
The storage benchmark mode allows measuring storage performance accurately
while using the industry-standard Parquet format.

Example Workloads
-----------------

DLRM Storage Benchmark
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    model: dlrm

    dataset:
      format: parquet_storage
      data_folder: /nvme/dlrm_data
      num_files_train: 24
      num_samples_per_file: 4718592
      record_length: 18944
      record_length_stdev: 0

    parquet:
      memory_map: true
      use_odirect: false

    reader:
      read_type: on_demand

Flux Storage Benchmark
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    model: flux

    dataset:
      format: parquet_storage
      data_folder: /nvme/flux_data
      num_files_train: 150
      num_samples_per_file: 288
      record_length: 2228224
      record_length_stdev: 0

    parquet:
      memory_map: true
      use_odirect: false

    reader:
      read_type: on_demand

Validating Results
------------------

To verify the storage benchmark mode reads the same bytes as a full decode:

.. code-block:: python

    import pyarrow.parquet as pq

    # Get expected byte count from metadata
    pf = pq.ParquetFile("test.parquet")
    expected_bytes = 0
    for i in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(i)
        for j in range(rg.num_columns):
            expected_bytes += rg.column(j).total_compressed_size

    # Storage reader reports same byte count
    from dlio_benchmark.reader.parquet_storage_reader import ParquetStorageReader
    reader = ParquetStorageReader(...)
    actual_bytes = reader.read_all()

    assert actual_bytes == expected_bytes

References
----------

- `Apache Parquet Format Specification <https://parquet.apache.org/docs/file-format/>`_
- `Apache Arrow IPC Format <https://arrow.apache.org/docs/format/Columnar.html>`_
- `MLPerf Storage Benchmark <https://github.com/mlcommons/storage>`_
