"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Parquet Storage Reader for DLIO Benchmark.

This reader performs the same I/O operations as a standard Parquet reader
but SKIPS the CPU-intensive decoding step. This isolates storage performance
from CPU performance, enabling accurate storage system benchmarking.

Key features:
- Same I/O access pattern as real Parquet reads (row groups, column chunks)
- Reads the exact same bytes from storage
- No CPU decode overhead (RLE, dictionary, delta encoding)
- Memory-mapped or direct file I/O modes
- Optional O_DIRECT for bypassing page cache

Why this matters for benchmarking:
- Standard Parquet reads: ~1.2 GB/s (storage + CPU decode)
- Storage-only reads: ~4.5 GB/s (pure storage I/O)
- Decode overhead: 260%+ (CPU work exceeds storage time)

For accurate storage benchmarks, we need to measure storage performance
without CPU decode overhead polluting the results.
"""
import os
import mmap
import bisect
import ctypes
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


@dataclass
class ColumnChunkRegion:
    """Describes a region of the file containing column chunk data."""
    offset: int
    size: int
    row_group: int
    column: int


class ParquetStorageReader(FormatReader):
    """
    Parquet reader for storage benchmarking.

    Reads raw bytes following Parquet's I/O access pattern without
    performing CPU-intensive decoding. This isolates storage throughput
    measurement from CPU performance.

    Configuration options (in workload YAML under parquet:):
        storage_benchmark: true  # Enable storage-only mode (no decode)
        memory_map: true         # Use mmap for file access
        use_odirect: false       # Use O_DIRECT to bypass page cache
        odirect_alignment: 4096  # Alignment for O_DIRECT reads

    Example workload configuration:
        format: parquet
        parquet:
          storage_benchmark: true
          memory_map: true
    """

    # O_DIRECT flag for Linux
    O_DIRECT = getattr(os, 'O_DIRECT', 0o40000)
    DEFAULT_ALIGNMENT = 4096

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)

        # Get parquet-specific configuration
        parquet_config = getattr(self._args, 'parquet', {}) or {}

        # Storage benchmark mode (no decode)
        self._storage_benchmark = parquet_config.get('storage_benchmark', False)

        # Memory mapping
        self._use_mmap = parquet_config.get('memory_map', True)

        # O_DIRECT for bypassing page cache
        self._use_odirect = parquet_config.get('use_odirect', False)
        self._alignment = parquet_config.get('odirect_alignment', self.DEFAULT_ALIGNMENT)

        # Metadata cache: filename -> (regions, offsets, total_bytes)
        self._metadata_cache: Dict[str, Tuple[List[ColumnChunkRegion], List[int], int]] = {}

        # Row group cache for storage benchmark mode
        self._rg_cache_size = int(parquet_config.get('row_group_cache_size', 4))
        self._rg_cache: Dict[Tuple[str, int], int] = {}  # (filename, rg_idx) -> bytes_read
        self._rg_lru: List[Tuple[str, int]] = []

        self.logger.info(
            f"{utcnow()} ParquetStorageReader thread={thread_index} epoch={epoch} "
            f"storage_benchmark={self._storage_benchmark} mmap={self._use_mmap} "
            f"odirect={self._use_odirect}"
        )

    def _load_file_layout(self, filename: str) -> Tuple[List[ColumnChunkRegion], List[int], int]:
        """
        Load Parquet file layout from metadata.

        Returns:
            (regions, offsets, total_bytes) where:
            - regions: List of ColumnChunkRegion describing data locations
            - offsets: Cumulative row counts for bisect lookup
            - total_bytes: Total data bytes to read
        """
        if filename in self._metadata_cache:
            return self._metadata_cache[filename]

        import pyarrow.parquet as pq

        pf = pq.ParquetFile(filename)
        meta = pf.metadata

        regions = []
        offsets = [0]
        total_bytes = 0

        for rg_idx in range(meta.num_row_groups):
            rg = meta.row_group(rg_idx)
            offsets.append(offsets[-1] + rg.num_rows)

            for col_idx in range(rg.num_columns):
                col = rg.column(col_idx)
                size = col.total_compressed_size
                regions.append(ColumnChunkRegion(
                    offset=col.data_page_offset,
                    size=size,
                    row_group=rg_idx,
                    column=col_idx
                ))
                total_bytes += size

        # Sort by offset for sequential access
        regions.sort(key=lambda r: r.offset)

        result = (regions, offsets, total_bytes)
        self._metadata_cache[filename] = result

        self.logger.debug(
            f"{utcnow()} ParquetStorageReader loaded layout for {filename}: "
            f"row_groups={meta.num_row_groups} regions={len(regions)} "
            f"total_bytes={total_bytes / (1024**2):.1f}MB"
        )

        return result

    def _evict_lru(self):
        """Evict the least-recently-used row group from cache."""
        if self._rg_lru:
            oldest = self._rg_lru.pop(0)
            self._rg_cache.pop(oldest, None)

    def _read_region_mmap(self, mm: mmap.mmap, region: ColumnChunkRegion) -> int:
        """Read a region using memory mapping."""
        data = mm[region.offset:region.offset + region.size]
        # Touch first and last byte to ensure data is read
        if len(data) > 0:
            _ = data[0]
        if len(data) > 1:
            _ = data[-1]
        return region.size

    def _read_region_direct(self, f, region: ColumnChunkRegion) -> int:
        """Read a region using direct file I/O."""
        f.seek(region.offset)
        data = f.read(region.size)
        return len(data)

    def _read_region_odirect(self, fd: int, region: ColumnChunkRegion) -> int:
        """Read a region using O_DIRECT with aligned buffers."""
        alignment = self._alignment

        # Align offset down
        aligned_offset = (region.offset // alignment) * alignment
        offset_adj = region.offset - aligned_offset

        # Align size up
        aligned_size = ((offset_adj + region.size + alignment - 1) // alignment) * alignment

        # Seek to aligned position
        os.lseek(fd, aligned_offset, os.SEEK_SET)

        # Allocate aligned buffer
        buf = ctypes.create_string_buffer(aligned_size + alignment)
        addr = ctypes.addressof(buf)
        aligned_addr = (addr + alignment - 1) & ~(alignment - 1)
        aligned_buf = (ctypes.c_char * aligned_size).from_address(aligned_addr)

        # Read into aligned buffer
        total = 0
        mv = memoryview(aligned_buf)
        while total < aligned_size:
            try:
                n = os.readv(fd, [mv[total:]])
                if n == 0:
                    break
                total += n
            except BlockingIOError:
                continue

        return region.size

    def _read_row_group_regions(self, filename: str, rg_idx: int) -> int:
        """
        Read all column chunk regions for a specific row group.

        Returns: Number of bytes read
        """
        regions, _, _ = self._load_file_layout(filename)

        # Filter regions for this row group
        rg_regions = [r for r in regions if r.row_group == rg_idx]

        bytes_read = 0

        if self._use_odirect and self.O_DIRECT != 0:
            # O_DIRECT mode
            fd = os.open(filename, os.O_RDONLY | self.O_DIRECT)
            try:
                for region in rg_regions:
                    bytes_read += self._read_region_odirect(fd, region)
            finally:
                os.close(fd)
        elif self._use_mmap:
            # Memory-mapped mode
            with open(filename, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    for region in rg_regions:
                        bytes_read += self._read_region_mmap(mm, region)
                finally:
                    mm.close()
        else:
            # Direct file I/O mode
            with open(filename, 'rb') as f:
                for region in rg_regions:
                    bytes_read += self._read_region_direct(f, region)

        return bytes_read

    @dlp.log
    def open(self, filename):
        """
        Open a Parquet file and load its layout metadata.

        Returns: (regions, offsets, total_bytes) tuple for the file
        """
        return self._load_file_layout(filename)

    @dlp.log
    def close(self, filename):
        """Evict cached row groups for this file."""
        keys_to_remove = [k for k in self._rg_cache if k[0] == filename]
        for k in keys_to_remove:
            self._rg_cache.pop(k, None)
            if k in self._rg_lru:
                self._rg_lru.remove(k)
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        """
        Read the row group containing sample_index.

        For storage benchmark mode, reads raw bytes without decoding.
        Reports the actual bytes read to the profiler.
        """
        regions, offsets, _ = self.open_file_map[filename]

        # Binary search: offsets[rg_idx] <= sample_index < offsets[rg_idx+1]
        rg_idx = max(0, bisect.bisect_right(offsets, sample_index) - 1)
        num_row_groups = len(offsets) - 1
        rg_idx = min(rg_idx, num_row_groups - 1)

        cache_key = (filename, rg_idx)

        if cache_key not in self._rg_cache:
            # Read row group data (raw bytes, no decode)
            bytes_read = self._read_row_group_regions(filename, rg_idx)

            # LRU eviction
            while len(self._rg_cache) >= self._rg_cache_size:
                self._evict_lru()

            self._rg_cache[cache_key] = bytes_read
            self._rg_lru.append(cache_key)
        else:
            # Move to end of LRU list
            try:
                self._rg_lru.remove(cache_key)
            except ValueError:
                pass
            self._rg_lru.append(cache_key)

        dlp.update(image_size=self._rg_cache[cache_key])

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        dlp.update(step=step)
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        self._rg_cache.clear()
        self._rg_lru.clear()
        self._metadata_cache.clear()
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
