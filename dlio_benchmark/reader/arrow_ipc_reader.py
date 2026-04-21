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

Arrow IPC Reader for DLIO Benchmark.

Arrow IPC files store data in the same format as Arrow's in-memory
representation, enabling zero-copy reads when memory-mapped.
"""
import bisect

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile, utcnow

dlp = Profile(MODULE_DATA_READER)


class ArrowIPCReader(FormatReader):
    """
    Arrow IPC (Feather v2) reader with memory-mapping support.

    Arrow IPC enables zero-copy reads via memory mapping since the file
    format matches the in-memory representation.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)

        # Get configuration
        parquet_config = getattr(self._args, 'parquet', {}) or {}
        self._use_mmap = parquet_config.get('memory_map', True)

        # Column selection
        columns_config = parquet_config.get('columns', [])
        if columns_config:
            self._columns = [
                c.get('name') for c in columns_config
                if c.get('read', True)
            ]
            if not self._columns:
                self._columns = None
        else:
            self._columns = None

        # Batch cache
        self._batch_cache_size = int(parquet_config.get('row_group_cache_size', 4))
        self._batch_cache = {}
        self._batch_lru = []

        # Metadata cache
        self._metadata_cache = {}
        self._use_metadata_cache = parquet_config.get('metadata_cache', True)

        self.logger.info(
            f"{utcnow()} ArrowIPCReader thread={thread_index} epoch={epoch} "
            f"mmap={self._use_mmap} columns={self._columns}"
        )

    def _evict_lru(self):
        """Evict least-recently-used batch from cache."""
        if self._batch_lru:
            oldest = self._batch_lru.pop(0)
            self._batch_cache.pop(oldest, None)

    @dlp.log
    def open(self, filename):
        """
        Open an Arrow IPC file.

        Returns (reader, cumulative_offsets) where offsets map sample indices
        to record batches.
        """
        import pyarrow as pa

        # Check metadata cache
        if self._use_metadata_cache and filename in self._metadata_cache:
            return self._metadata_cache[filename]

        if self._use_mmap:
            source = pa.memory_map(filename, 'r')
        else:
            source = filename

        reader = pa.ipc.open_file(source)

        # Build cumulative offsets for record batches
        offsets = [0]
        for i in range(reader.num_record_batches):
            # Get batch metadata without reading data
            batch = reader.get_batch(i)
            offsets.append(offsets[-1] + batch.num_rows)

        self.logger.debug(
            f"{utcnow()} ArrowIPCReader.open {filename} "
            f"batches={reader.num_record_batches} total_rows={offsets[-1]}"
        )

        result = (reader, offsets)

        if self._use_metadata_cache:
            self._metadata_cache[filename] = result

        return result

    @dlp.log
    def close(self, filename):
        """Evict cached batches for this file."""
        keys_to_remove = [k for k in self._batch_cache if k[0] == filename]
        for k in keys_to_remove:
            self._batch_cache.pop(k, None)
            if k in self._batch_lru:
                self._batch_lru.remove(k)
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        """
        Read the record batch containing sample_index.

        Uses bisect for O(log N) batch lookup with caching.
        """
        reader, offsets = self.open_file_map[filename]

        # Binary search for batch index
        batch_idx = max(0, bisect.bisect_right(offsets, sample_index) - 1)
        batch_idx = min(batch_idx, reader.num_record_batches - 1)

        cache_key = (filename, batch_idx)

        if cache_key not in self._batch_cache:
            # Read batch
            batch = reader.get_batch(batch_idx)
            batch_bytes = batch.nbytes

            # LRU eviction
            while len(self._batch_cache) >= self._batch_cache_size:
                self._evict_lru()

            self._batch_cache[cache_key] = batch_bytes
            self._batch_lru.append(cache_key)
        else:
            # Move to end of LRU
            try:
                self._batch_lru.remove(cache_key)
            except ValueError:
                pass
            self._batch_lru.append(cache_key)

        dlp.update(image_size=self._batch_cache[cache_key])

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        dlp.update(step=step)
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        self._batch_cache.clear()
        self._batch_lru.clear()
        self._metadata_cache.clear()
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
