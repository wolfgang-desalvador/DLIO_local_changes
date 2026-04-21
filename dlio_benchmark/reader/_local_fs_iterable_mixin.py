"""
_LocalFSIterableMixin — parallel prefetch for local-filesystem iterable readers.

WHY THIS EXISTS — PARITY WITH _S3IterableMixin
===============================================
DLIO is a storage benchmark. FormatReader.next() always yields
``self._args.resized_image`` — a single pre-allocated dummy tensor. The actual
decoded file bytes are NEVER used. They are consulted for exactly one thing:
the ``image_size`` metric inside ``dlp.update(image_size=N)``.

Without this mixin, local-FS readers open and decode files ONE AT A TIME inside
the next() loop (queue depth = 1). The S3 iterable readers pre-fetch ALL files
in parallel before the iteration starts (queue depth = N). This is a structural
parity violation — local-FS benchmarks look slower than they physically should
be, making cross-backend comparisons invalid.

This mixin gives local-FS readers the same pre-fetch pattern as _S3IterableMixin:

1. Before next(): parallel-read all assigned files via ThreadPoolExecutor
2. Store only the raw byte count per file (never decode numpy/PIL/h5py)
3. During next() / get_sample(): dict lookup → telemetry → return resized_image

I/O IS FULLY MEASURED
=====================
The full read() of each file still happens inside _localfs_prefetch_all().
Only the decode step (np.load, PIL.open, h5py.File) is skipped — that decode
is pure CPU overhead that has nothing to do with storage bandwidth.

USAGE PATTERN
=============
Subclass from BOTH the format-specific parent AND this mixin::

    class ImageReader(_OriginalImageReader, _LocalFSIterableMixin):
        @dlp.log_init
        def __init__(self, dataset_type, thread_index, epoch):
            super().__init__(dataset_type, thread_index, epoch)
            self._localfs_init()

        @dlp.log
        def open(self, filename):
            return self._local_cache.get(filename, 0)

        @dlp.log
        def get_sample(self, filename, sample_index):
            dlp.update(image_size=self._local_cache.get(filename, 0))

        def next(self):
            self._localfs_prefetch_all()
            for batch in super().next():
                yield batch
"""
from concurrent.futures import ThreadPoolExecutor

from dlio_benchmark.utils.utility import utcnow


class _LocalFSIterableMixin:
    """
    Mixin providing parallel local-filesystem prefetch for iterable readers.

    Do NOT instantiate directly. Mix in alongside a FormatReader subclass;
    call ``_localfs_init()`` from the subclass ``__init__`` after
    ``super().__init__()``.
    """

    def _localfs_init(self) -> None:
        """
        Initialise mixin state.

        Sets:
          - ``self._local_cache`` (dict: filename → int byte count)
        """
        self._local_cache: dict = {}   # filename → int (raw byte count only)

    def _read_local_bytes(self, path: str) -> int:
        """Read a local file and return its byte count. No decode."""
        with open(path, 'rb') as fh:
            return len(fh.read())

    def _localfs_prefetch_all(self) -> None:
        """
        Collect all files assigned to this thread and prefetch them in parallel.

        Call at the top of ``next()`` before the iteration loop. Deduplicates
        filenames while preserving order (a multi-sample file may appear many
        times in the thread's file_map entries).
        """
        thread_entries = self.file_map.get(self.thread_index, [])
        seen = set()
        paths = []
        for _, filename, _ in thread_entries:
            if filename not in seen:
                seen.add(filename)
                paths.append(filename)

        if not paths:
            return

        self.logger.info(
            f"{utcnow()} {self.__class__.__name__} thread={self.thread_index} "
            f"prefetching {len(paths)} local files (parallel)"
        )

        n_workers = min(64, len(paths))
        cache = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for path, byte_count in zip(paths, pool.map(self._read_local_bytes, paths)):
                cache[path] = byte_count
        self._local_cache = cache

    def _localfs_ensure_cached(self, filename: str) -> None:
        """Fetch a single file on demand if not already in the cache."""
        if filename not in self._local_cache:
            self._local_cache[filename] = self._read_local_bytes(filename)
