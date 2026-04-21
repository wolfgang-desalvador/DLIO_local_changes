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

Arrow IPC Generator for DLIO Benchmark.

Arrow IPC (Feather v2) is an uncompressed columnar format where bytes on
disk match the in-memory format. This eliminates encoding/decoding overhead
but produces larger files than compressed Parquet.

Used for comparison benchmarks against Parquet.
"""
import os
import numpy as np
import pyarrow as pa

from dlio_benchmark.common.enumerations import Compression
from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import Profile, progress, gen_random_tensor
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR

dlp = Profile(MODULE_DATA_GENERATOR)

# Map compression to Arrow IPC options
COMPRESSION_MAP = {
    Compression.NONE: None,
    Compression.LZ4: 'lz4',
    Compression.ZSTD: 'zstd',
}

# NumPy dtype mapping
NP_DTYPE_MAP = {
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint32': np.uint32,
    'uint64': np.uint64,
    'int8': np.int8,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
}

PA_SCALAR_MAP = {
    'uint8': pa.uint8(),
    'uint16': pa.uint16(),
    'uint32': pa.uint32(),
    'uint64': pa.uint64(),
    'int8': pa.int8(),
    'int16': pa.int16(),
    'int32': pa.int32(),
    'int64': pa.int64(),
    'float16': pa.float16(),
    'float32': pa.float32(),
    'float64': pa.float64(),
}


class ArrowIPCGenerator(DataGenerator):
    """
    Arrow IPC (Feather v2) data generator.

    Similar API to ParquetGenerator but writes Arrow IPC format.
    """

    def __init__(self):
        super().__init__()
        # Get parquet/arrow config (shared config section)
        parquet_config = getattr(self._args, 'parquet', {}) or {}
        self.parquet_columns = parquet_config.get('columns', [])

    def _build_schema(self, record_size=None):
        """Build PyArrow schema from configuration."""
        if not self.parquet_columns:
            return pa.schema([('data', pa.large_binary())])

        fields = []
        for col_spec in self.parquet_columns:
            name = col_spec.get('name', 'data')
            dtype = col_spec.get('dtype', 'float32')
            size = col_spec.get('size', 1)

            pa_scalar = PA_SCALAR_MAP.get(dtype)

            if pa_scalar is not None:
                if size == 1:
                    fields.append(pa.field(name, pa_scalar))
                else:
                    fields.append(pa.field(name, pa.large_binary()))
            elif dtype == 'binary':
                fields.append(pa.field(name, pa.large_binary()))
            elif dtype == 'string':
                fields.append(pa.field(name, pa.string()))
            else:
                fields.append(pa.field(name, pa.large_binary()))

        return pa.schema(fields)

    def _generate_column_data(self, col_spec, num_samples, rng):
        """Generate data for a single column."""
        name = col_spec.get('name', 'data')
        dtype = col_spec.get('dtype', 'float32')
        size = col_spec.get('size', 1)

        np_dtype = NP_DTYPE_MAP.get(dtype)
        pa_scalar = PA_SCALAR_MAP.get(dtype)

        if np_dtype is not None and pa_scalar is not None:
            if size == 1:
                data = gen_random_tensor(shape=(num_samples,), dtype=np_dtype, rng=rng)
                return name, pa.array(data, type=pa_scalar)
            else:
                bytes_per_sample = size * np.dtype(np_dtype).itemsize
                binary_data = []
                for _ in range(num_samples):
                    arr = gen_random_tensor(shape=(size,), dtype=np_dtype, rng=rng)
                    binary_data.append(arr.tobytes())
                return name, pa.array(binary_data, type=pa.large_binary())

        if dtype == 'binary':
            binary_data = [rng.bytes(size) for _ in range(num_samples)]
            return name, pa.array(binary_data, type=pa.large_binary())

        if dtype == 'string':
            ints = rng.integers(0, 2**31, size=num_samples)
            strings = [f"s_{v}" for v in ints]
            return name, pa.array(strings, type=pa.string())

        binary_data = [rng.bytes(size) for _ in range(num_samples)]
        return name, pa.array(binary_data, type=pa.large_binary())

    def _generate_legacy_data(self, record_size, num_samples, rng):
        """Generate legacy single-column binary data."""
        binary_data = []
        for _ in range(num_samples):
            data = gen_random_tensor(shape=(record_size,), dtype=np.uint8, rng=rng)
            binary_data.append(data.tobytes())
        return {'data': pa.array(binary_data, type=pa.large_binary())}

    @dlp.log
    def generate(self):
        """Generate Arrow IPC files."""
        super().generate()

        np.random.seed(10 + self.my_rank)
        rng = np.random.default_rng(seed=10 + self.my_rank)

        dim = self.get_dimension(self.total_files_to_generate)
        compression = COMPRESSION_MAP.get(self.compression, None)

        # IPC write options
        if compression:
            options = pa.ipc.IpcWriteOptions(compression=compression)
        else:
            options = pa.ipc.IpcWriteOptions()

        for i in dlp.iter(range(self.my_rank, int(self.total_files_to_generate), self.comm_size)):
            progress(i + 1, self.total_files_to_generate, "Generating Arrow IPC Data")

            out_path_spec = self.storage.get_uri(self._file_list[i])

            # Calculate record size
            dim_raw = dim[2 * i]
            if isinstance(dim_raw, list):
                dim1 = int(dim_raw[0])
                dim2 = int(dim_raw[1]) if len(dim_raw) > 1 else 1
            else:
                dim1 = int(dim_raw)
                dim2 = int(dim[2 * i + 1])
            record_size = dim1 * dim2

            # Build schema
            schema = self._build_schema(record_size)

            # Generate column data
            if self.parquet_columns:
                columns = {}
                for col_spec in self.parquet_columns:
                    name, array = self._generate_column_data(col_spec, self.num_samples, rng)
                    columns[name] = array
            else:
                columns = self._generate_legacy_data(record_size, self.num_samples, rng)

            # Create table
            table = pa.table(columns, schema=schema)

            # Ensure parent directory exists
            parent_dir = os.path.dirname(out_path_spec)
            if parent_dir and not out_path_spec.startswith('s3://'):
                os.makedirs(parent_dir, exist_ok=True)

            # Write Arrow IPC file
            with pa.ipc.new_file(out_path_spec, schema, options=options) as writer:
                writer.write_table(table)

        np.random.seed()
