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
"""
from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.reader._local_fs_iterable_mixin import _LocalFSIterableMixin
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_DATA_READER)


class NPZReader(FormatReader, _LocalFSIterableMixin):
    """
    Reader for NPZ files.

    Uses _LocalFSIterableMixin to prefetch all assigned files in parallel
    before the iteration loop. np.load decode is skipped because only the
    raw byte count is needed for image_size telemetry.
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        self._localfs_init()

    @dlp.log
    def open(self, filename):
        super().open(filename)
        return self._local_cache.get(filename, 0)

    @dlp.log
    def close(self, filename):
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)
        byte_count = self.open_file_map.get(filename, 0)
        dlp.update(image_size=byte_count)

    def next(self):
        self._localfs_prefetch_all()
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        dlp.update(step=step)
        filename, _ = self.global_index_map[image_idx]
        self._localfs_ensure_cached(filename)
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True

