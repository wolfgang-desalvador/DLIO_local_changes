"""
Optimized Parquet reader for local and network filesystems.

Key optimizations over the original reader:
- Memory-mapped file access for reduced syscall overhead
- Metadata caching to avoid repeated footer reads across epochs
- Row group caching with LRU eviction
- Column projection with per-column read flag
- Threading support for parallel column decompression

Reads parquet files via pyarrow directly. Each file is opened by reading its
footer (column + row-group metadata), then individual row groups are fetched on
demand as DLIO requests specific sample indices. Row groups are cached with an
LRU bound so consecutive samples from the same row group cost only one read.

Configuration (under dataset.parquet in the DLIO YAML):
  memory_map:           true  # use mmap for file access (default: true)
  use_threads:          true  # parallel column decompression (default: true)
  metadata_cache:       true  # cache file footers across open/close (default: true)
  row_group_cache_size: 4     # max row groups held in memory per reader thread
  columns:                    # column specs; only columns with read: true are loaded
    - name: feature1
      read: true
    - name: label
      read: true
"""
import bisect

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


class ParquetReader(FormatReader):
    """
    Optimized Parquet reader for local/network filesystems.

    Uses row-group-granular access with caching. Opens files with memory
    mapping for efficient access. Row groups are cached with LRU eviction.

    DLIO's FormatReader protocol:
      open(filename)            → returns (ParquetFile, cumulative_offsets)
      get_sample(filename, idx) → bisect-locates the row group, fetches if not
                                  cached, updates dlp metrics with byte count
      close(filename)           → evicts row-group cache entries for that file
      next() / read_index()     → delegate to FormatReader base class
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)

        # Get parquet-specific configuration from the parsed YAML
        parquet_columns = getattr(self._args, 'parquet_columns', []) or []

        # Memory mapping (enabled by default for performance)
        self._use_mmap = bool(getattr(self._args, 'parquet_memory_map', True))

        # Threading for parallel column reads
        self._use_threads = bool(getattr(self._args, 'parquet_use_threads', True))

        # Column selection — extract names where read flag is True (default)
        if parquet_columns:
            self._columns = [
                c.get('name') for c in parquet_columns
                if isinstance(c, dict) and c.get('read', True)
            ]
            if not self._columns:
                self._columns = None  # Read all if no readable columns specified
        else:
            self._columns = None

        # Row-group cache configuration
        self._rg_cache_size = int(getattr(self._args, 'parquet_row_group_cache_size',
                                          getattr(self._args, 'parquet_row_group_size', 4)))
        self._rg_cache: dict = {}   # (filename, rg_idx) → compressed_bytes
        self._rg_lru: list = []     # insertion-order LRU key list

        # Metadata cache to avoid repeated footer reads across open/close cycles
        self._metadata_cache: dict = {}  # filename → (ParquetFile, offsets)
        self._use_metadata_cache = bool(getattr(self._args, 'parquet_metadata_cache', True))

        self.logger.info(
            f"{utcnow()} ParquetReader thread={thread_index} epoch={epoch} "
            f"mmap={self._use_mmap} threads={self._use_threads} "
            f"columns={self._columns} rg_cache_size={self._rg_cache_size}"
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _evict_lru(self):
        """Evict the least-recently-used row group from the cache."""
        if self._rg_lru:
            oldest = self._rg_lru.pop(0)
            self._rg_cache.pop(oldest, None)

    # ── FormatReader interface ────────────────────────────────────────────────

    @dlp.log
    def open(self, filename):
        """
        Open a Parquet file and read its footer metadata.

        Returns (ParquetFile, cumulative_offsets) where offsets[i] is the
        first row index of row group i, and offsets[-1] is total row count.
        Uses metadata cache when enabled to avoid repeated footer reads.
        """
        # Check metadata cache first
        if self._use_metadata_cache and filename in self._metadata_cache:
            return self._metadata_cache[filename]

        import pyarrow.parquet as pq

        pf = pq.ParquetFile(filename, memory_map=self._use_mmap)
        meta = pf.metadata

        # Build cumulative row offsets [0, rg0_rows, rg0+rg1_rows, ...]
        offsets = [0]
        for i in range(meta.num_row_groups):
            offsets.append(offsets[-1] + meta.row_group(i).num_rows)

        self.logger.debug(
            f"{utcnow()} ParquetReader.open {filename} "
            f"row_groups={meta.num_row_groups} total_rows={offsets[-1]}"
        )

        result = (pf, offsets)

        # Cache metadata if enabled
        if self._use_metadata_cache:
            self._metadata_cache[filename] = result

        return result

    @dlp.log
    def close(self, filename):
        """Evict cached row groups for this file to free memory."""
        keys_to_remove = [k for k in self._rg_cache if k[0] == filename]
        for k in keys_to_remove:
            self._rg_cache.pop(k, None)
            if k in self._rg_lru:
                self._rg_lru.remove(k)
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        """
        Read the row group containing sample_index and update I/O metrics.

        Uses bisect to locate the row group in O(log N), fetches from disk if
        not already cached. Reports compressed row-group bytes to the profiler.
        Actual row data is discarded — DLIO uses self._args.resized_image.
        """
        pf, offsets = self.open_file_map[filename]

        # Binary search: offsets[rg_idx] <= sample_index < offsets[rg_idx+1]
        rg_idx = max(0, bisect.bisect_right(offsets, sample_index) - 1)
        rg_idx = min(rg_idx, pf.metadata.num_row_groups - 1)

        cache_key = (filename, rg_idx)
        if cache_key not in self._rg_cache:
            # Read row group from disk — this is the measured I/O
            pf.read_row_group(
                rg_idx,
                columns=self._columns,
                use_threads=self._use_threads,
            )

            rg_meta = pf.metadata.row_group(rg_idx)
            compressed_bytes = sum(
                rg_meta.column(c).total_compressed_size
                for c in range(rg_meta.num_columns)
            )

            while len(self._rg_cache) >= self._rg_cache_size:
                self._evict_lru()

            self._rg_cache[cache_key] = compressed_bytes
            self._rg_lru.append(cache_key)
        else:
            # Move to end (most recently used)
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
