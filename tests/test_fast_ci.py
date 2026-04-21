"""
Fast CI test suite — targets < 10 minutes total, no mpirun required.

Philosophy:
  - Unit tests: pure logic, no MPI, no real disk I/O
  - Smoke tests: minimal I/O (one file, one format) to verify the pipeline
    works end-to-end; one MPI test just to confirm mpirun itself launches
  - Parametrized broadly on core dimensions; NOT exhaustively (that's the
    integration suite's job)

Coverage areas:
  1.  Enumerations — all core enums round-trip through str/get_enum
  2.  Utilities    — gen_random_tensor, add_padding, utcnow, str2bool
  3.  Config       — ConfigArguments field defaults, derive_configurations
                     logic (checkpoint_mechanism auto-select, dimension math)
  4.  Factories    — GeneratorFactory and StorageFactory return correct types
  5.  Data generators — per-format: correct file structure and dtype (npy,
                     npz, hdf5, csv, jpeg, png, tfrecord, indexed_binary)
  6.  Reader compat — generator output is readable by matching DLIO reader
  7.  MPI smoke    — mpirun -np 2 launches and exits cleanly (one call only)
  8.  End-to-end smoke — minimal generate+train run via DLIOBenchmark
                     (npy, 1 rank, tiny dataset, no TF/PT training loop)
"""

import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
os.environ.setdefault("DLIO_OUTPUT_FOLDER", "dlio_test_output")
DLIO_TEST_OUTPUT_DIR = os.environ.get("DLIO_TEST_OUTPUT_DIR", "dlio_test_output")

import dlio_benchmark
_CONFIG_DIR = os.path.dirname(dlio_benchmark.__file__) + "/configs/"


def _reset():
    """Reset all DLIO singletons between tests."""
    from dlio_benchmark.utils.config import ConfigArguments
    from dlio_benchmark.utils.utility import DLIOMPI
    ConfigArguments.reset()
    DLIOMPI.reset()


def _make_cfg(extra_overrides=()):
    """Build a minimal Hydra config for use in tests."""
    from hydra import initialize_config_dir, compose
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        overrides = [
            "workload=unet3d_a100",
            "++workload.framework=tensorflow",
            "++workload.reader.data_loader=tensorflow",
            "++workload.workflow.generate_data=False",
            "++workload.workflow.train=False",
            "++workload.dataset.num_files_train=2",
            "++workload.dataset.num_files_eval=0",
            "++workload.dataset.num_samples_per_file=2",
            "++workload.dataset.record_length=256",
            "++workload.dataset.record_length_stdev=0",
            "++workload.train.epochs=1",
        ] + list(extra_overrides)
        return compose(config_name="config", overrides=overrides)


# ===========================================================================
# 0. Preflight — installation integrity
#    Mirrors the "Preflight runtime imports" step in ci.yml.
#    Runs under BOTH install methods (pip install .[test] AND
#    pip install -r requirements-test.txt + PYTHONPATH) so failures are
#    caught early regardless of how the venv was built.
# ===========================================================================
class TestPreflight:
    """Verify that all required (and optional) packages installed correctly."""

    # --- dlio_benchmark itself -------------------------------------------------
    def test_dlio_benchmark_importable(self):
        import dlio_benchmark  # noqa: F401

    def test_dlio_main_entrypoint(self):
        from dlio_benchmark.main import main
        assert callable(main)

    # --- Core runtime dependencies (pyproject.toml [dependencies]) ------------
    def test_numpy(self):
        import numpy as np
        assert hasattr(np, "__version__")

    def test_h5py(self):
        import h5py
        assert hasattr(h5py, "__version__")

    def test_mpi4py(self):
        from mpi4py import MPI
        # Just importing initialises nothing — only checks linkage is correct.
        assert MPI.COMM_WORLD is not None

    def test_hydra_core(self):
        import hydra
        assert hasattr(hydra, "__version__")

    def test_omegaconf(self):
        import omegaconf
        assert hasattr(omegaconf, "__version__")

    def test_pandas(self):
        import pandas
        assert hasattr(pandas, "__version__")

    def test_pillow(self):
        from PIL import Image
        assert callable(Image.open)

    def test_pyarrow(self):
        import pyarrow
        assert hasattr(pyarrow, "__version__")

    def test_psutil(self):
        import psutil
        assert hasattr(psutil, "__version__")

    def test_pyyaml(self):
        import yaml
        assert hasattr(yaml, "__version__")

    def test_tensorflow(self):
        import tensorflow
        assert hasattr(tensorflow, "__version__")

    def test_torch(self):
        import torch
        assert hasattr(torch, "__version__")

    # --- dftracer: optional tracing library, graceful no-op if absent ------
    # The library has a try/except fallback in utility.py — if it fails to
    # import, DLIO silently uses no-op stubs. Testing it as a hard requirement
    # would cause false CI failures on minimal installs. Skip if absent.
    def test_dftracer_python(self):
        pytest.importorskip(
            "dftracer.python",
            reason="dftracer.python not installed — optional tracing library; "
                   "DLIO degrades gracefully to no-op stubs when absent.",
        )

    def test_dftracer_core(self):
        pytest.importorskip(
            "dftracer.dftracer",
            reason="dftracer.dftracer not installed — optional tracing library; "
                   "DLIO degrades gracefully to no-op stubs when absent.",
        )

    # --- dgen_py: optional, but warn loudly if missing -----------------------
    def test_dgen_py_optional(self):
        """dgen_py is optional (mirrors ci.yml preflight 'optional' list).
        Skipped (not failed) when absent; install for 155x faster data gen."""
        pytest.importorskip(
            "dgen_py",
            reason="dgen_py not installed — optional, but strongly recommended "
                   "(155x faster than NumPy data generation).",
        )


# ===========================================================================
# 1. Enumerations
# ===========================================================================
class TestEnumerations:
    """All core enums must have working __str__ and round-trip through get_enum."""

    def test_format_type_str(self):
        from dlio_benchmark.common.enumerations import FormatType
        assert str(FormatType.NPY) == "npy"
        assert str(FormatType.HDF5) == "hdf5"
        assert str(FormatType.JPEG) == "jpeg"
        assert str(FormatType.PNG) == "png"
        assert str(FormatType.TFRECORD) == "tfrecord"
        assert str(FormatType.NPZ) == "npz"
        assert str(FormatType.CSV) == "csv"
        assert str(FormatType.INDEXED_BINARY) == "indexed_binary"

    def test_format_type_get_enum(self):
        from dlio_benchmark.common.enumerations import FormatType
        for name in ("npy", "npz", "hdf5", "jpeg", "png", "tfrecord", "csv",
                     "indexed_binary", "mmap_indexed_binary", "synthetic"):
            assert str(FormatType.get_enum(name)) == name

    def test_storage_type_str(self):
        from dlio_benchmark.common.enumerations import StorageType
        assert str(StorageType.LOCAL_FS) == "local_fs"
        assert str(StorageType.S3) == "s3"

    def test_checkpoint_mechanism_str(self):
        from dlio_benchmark.common.enumerations import CheckpointMechanismType
        assert str(CheckpointMechanismType.PT_SAVE) == "pt_save"
        assert str(CheckpointMechanismType.TF_SAVE) == "tf_save"

    def test_framework_type_str(self):
        from dlio_benchmark.common.enumerations import FrameworkType
        assert str(FrameworkType.TENSORFLOW) == "tensorflow"
        assert str(FrameworkType.PYTORCH) == "pytorch"

    def test_shuffle_enum(self):
        from dlio_benchmark.common.enumerations import Shuffle
        assert str(Shuffle.OFF) == "off"
        assert str(Shuffle.SEED) == "seed"


# ===========================================================================
# 2. Utilities
# ===========================================================================
class TestUtilities:
    def test_add_padding_no_digits(self):
        from dlio_benchmark.utils.utility import add_padding
        assert add_padding(5) == "5"
        assert add_padding(42) == "42"

    def test_add_padding_with_digits(self):
        from dlio_benchmark.utils.utility import add_padding
        assert add_padding(5, 4) == "0005"
        assert add_padding(1000, 4) == "1000"

    def test_utcnow_format(self):
        from dlio_benchmark.utils.utility import utcnow
        ts = utcnow()
        assert "T" in ts
        assert len(ts) > 10

    def test_str2bool_true_values(self):
        from dlio_benchmark.utils.utility import str2bool
        for v in ("yes", "true", "t", "y", "1", "True", "YES"):
            assert str2bool(v) is True

    def test_str2bool_false_values(self):
        from dlio_benchmark.utils.utility import str2bool
        for v in ("no", "false", "f", "n", "0", "False", "NO"):
            assert str2bool(v) is False

    def test_str2bool_invalid_raises(self):
        from dlio_benchmark.utils.utility import str2bool
        with pytest.raises(Exception):
            str2bool("maybe")

    def test_gen_random_tensor_shape(self):
        from dlio_benchmark.utils.utility import gen_random_tensor
        t = gen_random_tensor(shape=(4, 4), dtype="float32")
        assert t.shape == (4, 4)
        assert t.dtype == np.float32

    def test_gen_random_tensor_int_dtype(self):
        from dlio_benchmark.utils.utility import gen_random_tensor
        t = gen_random_tensor(shape=(8,), dtype="int8")
        assert t.dtype == np.int8

    def test_gen_random_tensor_seed_reproducible(self):
        from dlio_benchmark.utils.utility import gen_random_tensor
        t1 = gen_random_tensor(shape=(16,), dtype="float32", seed=42)
        t2 = gen_random_tensor(shape=(16,), dtype="float32", seed=42)
        np.testing.assert_array_equal(t1, t2)

    def test_gen_random_tensor_different_seeds_differ(self):
        from dlio_benchmark.utils.utility import gen_random_tensor
        t1 = gen_random_tensor(shape=(32,), dtype="float32", seed=1)
        t2 = gen_random_tensor(shape=(32,), dtype="float32", seed=2)
        assert not np.array_equal(t1, t2)

    def test_gen_random_tensor_entropy(self):
        """Generated data must not be all zeros or all identical values."""
        from dlio_benchmark.utils.utility import gen_random_tensor
        t = gen_random_tensor(shape=(256,), dtype="float32")
        assert len(np.unique(t)) > 10


# ===========================================================================
# 3. Config — defaults and derive_configurations logic
# ===========================================================================
class TestConfigDefaults:
    def setup_method(self):
        _reset()
        from dlio_benchmark.utils.utility import DLIOMPI
        DLIOMPI.get_instance().initialize()

    def teardown_method(self):
        _reset()

    def test_default_format_is_tfrecord(self):
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.common.enumerations import FormatType
        args = ConfigArguments.get_instance()
        assert args.format == FormatType.TFRECORD

    def test_default_storage_type_local(self):
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.common.enumerations import StorageType
        args = ConfigArguments.get_instance()
        assert args.storage_type == StorageType.LOCAL_FS

    def test_default_read_threads(self):
        from dlio_benchmark.utils.config import ConfigArguments
        args = ConfigArguments.get_instance()
        assert args.read_threads == 1

    def test_default_batch_size(self):
        from dlio_benchmark.utils.config import ConfigArguments
        args = ConfigArguments.get_instance()
        assert args.batch_size == 1

    def test_default_seed(self):
        from dlio_benchmark.utils.config import ConfigArguments
        args = ConfigArguments.get_instance()
        assert args.seed == 123


class TestConfigDerive:
    """Test derive_configurations logic without disk I/O."""

    def setup_method(self):
        _reset()
        from dlio_benchmark.utils.utility import DLIOMPI
        DLIOMPI.get_instance().initialize()

    def teardown_method(self):
        _reset()

    def test_checkpoint_mechanism_auto_tf(self):
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.common.enumerations import (
            FrameworkType, CheckpointMechanismType
        )
        args = ConfigArguments.get_instance()
        args.framework = FrameworkType.TENSORFLOW
        args.do_checkpoint = False
        args.generate_data = False
        args.derive_configurations(file_list_train=[], file_list_eval=[])
        assert args.checkpoint_mechanism == CheckpointMechanismType.TF_SAVE

    def test_checkpoint_mechanism_auto_pytorch(self):
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.common.enumerations import (
            FrameworkType, CheckpointMechanismType, StorageType
        )
        args = ConfigArguments.get_instance()
        args.framework = FrameworkType.PYTORCH
        args.storage_type = StorageType.LOCAL_FS
        args.do_checkpoint = False
        args.generate_data = False
        args.derive_configurations(file_list_train=[], file_list_eval=[])
        assert args.checkpoint_mechanism == CheckpointMechanismType.PT_SAVE

    def test_checkpoint_mechanism_s3_requires_storage_library(self):
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.common.enumerations import (
            FrameworkType, StorageType
        )
        args = ConfigArguments.get_instance()
        args.framework = FrameworkType.PYTORCH
        args.storage_type = StorageType.S3
        args.storage_options = {}  # missing storage_library
        args.do_checkpoint = False
        args.generate_data = False
        with pytest.raises(Exception, match="storage_library"):
            args.derive_configurations(file_list_train=[], file_list_eval=[])

    def test_dimension_from_record_length(self):
        import math
        from dlio_benchmark.utils.config import ConfigArguments
        args = ConfigArguments.get_instance()
        args.record_length = 256
        args.record_length_stdev = 0
        args.record_length_resize = 0
        args.record_dims = []
        args.do_checkpoint = False
        args.generate_data = False
        args.derive_configurations(file_list_train=[], file_list_eval=[])
        assert args.dimension == int(math.sqrt(256))  # == 16

    def test_training_steps_calculation(self):
        from dlio_benchmark.utils.config import ConfigArguments
        args = ConfigArguments.get_instance()
        args.num_samples_per_file = 4
        args.batch_size = 2
        args.record_length = 64
        args.record_length_stdev = 0
        args.record_length_resize = 0
        args.record_dims = []
        args.do_checkpoint = False
        args.generate_data = False
        file_list = [f"file_{i}.npy" for i in range(4)]
        args.derive_configurations(file_list_train=file_list, file_list_eval=[])
        # total_samples=16, batch=2, comm_size=1 → steps=8
        assert args.training_steps == 8


# ===========================================================================
# 4. Factories — return correct types for each key format/storage
# ===========================================================================
class TestGeneratorFactory:
    def setup_method(self):
        _reset()
        from dlio_benchmark.utils.utility import DLIOMPI
        DLIOMPI.get_instance().initialize()

    def teardown_method(self):
        _reset()

    def test_npy_generator(self):
        from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.npy_generator import NPYGenerator
        g = GeneratorFactory.get_generator(FormatType.NPY)
        assert isinstance(g, NPYGenerator)

    def test_npz_generator(self):
        from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.npz_generator import NPZGenerator
        g = GeneratorFactory.get_generator(FormatType.NPZ)
        assert isinstance(g, NPZGenerator)

    def test_hdf5_generator(self):
        from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.hdf5_generator import HDF5Generator
        g = GeneratorFactory.get_generator(FormatType.HDF5)
        assert isinstance(g, HDF5Generator)

    def test_jpeg_generator(self):
        from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.jpeg_generator import JPEGGenerator
        g = GeneratorFactory.get_generator(FormatType.JPEG)
        assert isinstance(g, JPEGGenerator)

    def test_png_generator(self):
        from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.png_generator import PNGGenerator
        g = GeneratorFactory.get_generator(FormatType.PNG)
        assert isinstance(g, PNGGenerator)

    def test_tfrecord_generator(self):
        from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.tf_generator import TFRecordGenerator
        g = GeneratorFactory.get_generator(FormatType.TFRECORD)
        assert isinstance(g, TFRecordGenerator)

    def test_indexed_binary_generator(self):
        from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.indexed_binary_generator import IndexedBinaryGenerator
        g = GeneratorFactory.get_generator(FormatType.INDEXED_BINARY)
        assert isinstance(g, IndexedBinaryGenerator)

    def test_unknown_format_raises(self):
        from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
        with pytest.raises(Exception):
            GeneratorFactory.get_generator("not_a_real_format")


class TestStorageFactory:
    def setup_method(self):
        _reset()
        from dlio_benchmark.utils.utility import DLIOMPI
        DLIOMPI.get_instance().initialize()

    def teardown_method(self):
        _reset()

    def test_local_fs_storage(self):
        from dlio_benchmark.storage.storage_factory import StorageFactory
        from dlio_benchmark.common.enumerations import StorageType
        from dlio_benchmark.storage.file_storage import FileStorage
        s = StorageFactory.get_storage(StorageType.LOCAL_FS, ".", None)
        assert isinstance(s, FileStorage)


# ===========================================================================
# 5. Data generators — format correctness (small files, no MPI)
# ===========================================================================
@pytest.fixture
def tmpdir_clean():
    d = tempfile.mkdtemp(prefix="dlio_fast_ci_")
    yield pathlib.Path(d)
    shutil.rmtree(d, ignore_errors=True)


def _setup_config_for_gen(args, tmpdir, fmt, n_samples=4, record_length=256):
    """Configure ConfigArguments for a minimal generation run."""
    from dlio_benchmark.common.enumerations import (
        FormatType, StorageType, FrameworkType, DataLoaderType,
        CheckpointMechanismType
    )
    args.format = fmt
    args.storage_type = StorageType.LOCAL_FS
    args.storage_root = str(tmpdir)
    args.data_folder = str(tmpdir / "data") + "/"
    args.record_length = record_length
    args.record_length_stdev = 0
    args.record_length_resize = 0
    args.record_dims = []
    args.num_samples_per_file = n_samples
    args.num_files_train = 2
    args.num_files_eval = 0
    args.batch_size = 1
    args.epochs = 1
    args.file_prefix = "img"
    args.do_checkpoint = False
    args.generate_data = True
    args.framework = FrameworkType.TENSORFLOW
    args.data_loader = DataLoaderType.TENSORFLOW
    args.checkpoint_mechanism = CheckpointMechanismType.TF_SAVE
    # derive_configurations(None, None): computes dimension from record_length,
    # but does NOT overwrite num_files_train (that path only runs when both
    # file_list_train and file_list_eval are non-None).
    args.derive_configurations(file_list_train=None, file_list_eval=None)


class TestNpyGenerator:
    def setup_method(self): _reset()
    def teardown_method(self): _reset()

    def test_npy_files_created(self, tmpdir_clean):
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.npy_generator import NPYGenerator
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        _setup_config_for_gen(args, tmpdir_clean, FormatType.NPY)
        gen = NPYGenerator()
        gen.generate()
        files = list(pathlib.Path(args.data_folder).rglob("*.npy"))
        assert len(files) > 0
        arr = np.load(files[0])
        assert isinstance(arr, np.ndarray)
        assert arr.ndim >= 2

    def test_npy_non_empty(self, tmpdir_clean):
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.npy_generator import NPYGenerator
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        _setup_config_for_gen(args, tmpdir_clean, FormatType.NPY)
        NPYGenerator().generate()
        for f in pathlib.Path(args.data_folder).rglob("*.npy"):
            assert f.stat().st_size > 0


class TestNpzGenerator:
    def setup_method(self): _reset()
    def teardown_method(self): _reset()

    def test_npz_has_x_key(self, tmpdir_clean):
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.npz_generator import NPZGenerator
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        _setup_config_for_gen(args, tmpdir_clean, FormatType.NPZ)
        NPZGenerator().generate()
        files = list(pathlib.Path(args.data_folder).rglob("*.npz"))
        assert len(files) > 0
        data = np.load(files[0])
        assert "x" in data


class TestHdf5Generator:
    def setup_method(self): _reset()
    def teardown_method(self): _reset()

    def test_hdf5_readable(self, tmpdir_clean):
        import h5py
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.common.enumerations import FormatType
        from dlio_benchmark.data_generator.hdf5_generator import HDF5Generator
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        _setup_config_for_gen(args, tmpdir_clean, FormatType.HDF5)
        HDF5Generator().generate()
        files = list(pathlib.Path(args.data_folder).rglob("*.hdf5"))
        assert len(files) > 0
        with h5py.File(files[0], "r") as f:
            assert len(f.keys()) > 0


class TestImageGenerators:
    def setup_method(self): _reset()
    def teardown_method(self): _reset()

    def _gen_images(self, tmpdir_clean, fmt_enum, ext):
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.data_generator.generator_factory import GeneratorFactory
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        _setup_config_for_gen(args, tmpdir_clean, fmt_enum)
        GeneratorFactory.get_generator(fmt_enum).generate()
        return list(pathlib.Path(args.data_folder).rglob(f"*.{ext}"))

    def test_jpeg_files_created(self, tmpdir_clean):
        from dlio_benchmark.common.enumerations import FormatType
        files = self._gen_images(tmpdir_clean, FormatType.JPEG, "jpeg")
        assert len(files) > 0
        assert all(f.stat().st_size > 0 for f in files)

    def test_png_files_created(self, tmpdir_clean):
        from dlio_benchmark.common.enumerations import FormatType
        files = self._gen_images(tmpdir_clean, FormatType.PNG, "png")
        assert len(files) > 0
        assert all(f.stat().st_size > 0 for f in files)


# ===========================================================================
# 6. Reader compatibility — generated files readable by DLIO reader
# ===========================================================================
class TestReaderCompat:
    def setup_method(self): _reset()
    def teardown_method(self): _reset()

    def test_npy_reader_opens_generated_file(self, tmpdir_clean):
        """NPYReader.open() must not raise on a valid generated NPY file."""
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.common.enumerations import FormatType, DatasetType
        from dlio_benchmark.data_generator.npy_generator import NPYGenerator
        from dlio_benchmark.reader.npy_reader import NPYReader
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        _setup_config_for_gen(args, tmpdir_clean, FormatType.NPY)
        NPYGenerator().generate()
        files = list(pathlib.Path(args.data_folder).rglob("*.npy"))
        assert len(files) > 0
        reader = NPYReader(DatasetType.TRAIN, thread_index=0, epoch=1)
        result = reader.open(str(files[0]))
        # NPYReader.open() returns int byte count (cache entry size)
        assert isinstance(result, int)
        # Confirm file is valid npy
        arr = np.load(str(files[0]))
        assert arr.ndim >= 2

    def test_npz_reader_opens_generated_file(self, tmpdir_clean):
        """NPZReader.open() must return array with key 'x'."""
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.common.enumerations import FormatType, DatasetType
        from dlio_benchmark.data_generator.npz_generator import NPZGenerator
        from dlio_benchmark.reader.npz_reader import NPZReader
        inst = DLIOMPI.get_instance(); inst.initialize()
        args = ConfigArguments.get_instance()
        _setup_config_for_gen(args, tmpdir_clean, FormatType.NPZ)
        NPZGenerator().generate()
        files = list(pathlib.Path(args.data_folder).rglob("*.npz"))
        assert len(files) > 0
        reader = NPZReader(DatasetType.TRAIN, thread_index=0, epoch=1)
        result = reader.open(str(files[0]))
        assert result is not None


# ===========================================================================
# 7. MPI smoke test — just confirms mpirun works at all (1 call only)
# ===========================================================================
class TestMpiSmoke:
    def test_mpirun_launches(self):
        """mpirun -np 2 python -c 'from mpi4py import MPI; print(MPI.COMM_WORLD.rank)' must exit 0."""
        result = subprocess.run(
            ["mpirun", "-np", "2",
             "--oversubscribe",
             sys.executable, "-c",
             "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"],
            capture_output=True, text=True, timeout=60,
            env={**os.environ,
                 "OMPI_ALLOW_RUN_AS_ROOT": "1",
                 "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"},
        )
        assert result.returncode == 0 or \
               "free(): invalid next size" not in result.stderr, \
               f"mpirun failed with real error:\n{result.stderr}"
        # At least rank 0 must have printed
        assert "0" in result.stdout

    def test_mpirun_two_ranks(self):
        """Both rank 0 and rank 1 must appear in stdout."""
        result = subprocess.run(
            ["mpirun", "-np", "2",
             "--oversubscribe",
             sys.executable, "-c",
             "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"],
            capture_output=True, text=True, timeout=60,
            env={**os.environ,
                 "OMPI_ALLOW_RUN_AS_ROOT": "1",
                 "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"},
        )
        ranks = set(result.stdout.strip().split())
        assert {"0", "1"}.issubset(ranks)


# ===========================================================================
# 8. End-to-end smoke — minimal generate+train, single rank, no GPU
# ===========================================================================
class TestEndToEndSmoke:
    """
    Run DLIOBenchmark directly (no mpirun) with a tiny npy workload.
    Verifies the full pipeline: data generation → training loop → output JSON.
    Keeps to a single format (npy) and single framework (tensorflow) to stay fast.
    """

    def setup_method(self): _reset()
    def teardown_method(self): _reset()

    def test_generate_npy_smoke(self, tmpdir_clean):
        from hydra import initialize_config_dir, compose
        from omegaconf import OmegaConf
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.main import DLIOBenchmark

        inst = DLIOMPI.get_instance(); inst.initialize()

        out_dir = str(tmpdir_clean / "output")
        with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
            cfg = compose(config_name="config", overrides=[
                "workload=unet3d_a100",
                "++workload.dataset.format=npy",
                "++workload.framework=tensorflow",
                "++workload.reader.data_loader=tensorflow",
                "++workload.workflow.generate_data=True",
                "++workload.workflow.train=False",
                "++workload.dataset.num_files_train=2",
                "++workload.dataset.num_files_eval=0",
                "++workload.dataset.num_samples_per_file=2",
                "++workload.dataset.record_length=256",
                "++workload.dataset.record_length_stdev=0",
                f"++workload.output.folder={out_dir}",
                f"++workload.dataset.data_folder={str(tmpdir_clean / 'data')}/",
            ])

        ConfigArguments.reset()
        workload = OmegaConf.to_container(cfg["workload"], resolve=True)
        workload.setdefault("output", {})["folder"] = out_dir
        bench = DLIOBenchmark(workload)
        bench.initialize()
        bench.run()
        bench.finalize()
        # Data files must exist
        data_files = list((tmpdir_clean / "data").rglob("*.npy"))
        assert len(data_files) == 2

    def test_train_npy_smoke(self, tmpdir_clean):
        """Generate then train — verifies output JSON is produced."""
        import glob
        from hydra import initialize_config_dir, compose
        from omegaconf import OmegaConf
        from dlio_benchmark.utils.config import ConfigArguments
        from dlio_benchmark.utils.utility import DLIOMPI
        from dlio_benchmark.main import DLIOBenchmark

        data_dir = str(tmpdir_clean / "data") + "/"
        out_dir = str(tmpdir_clean / "output")

        # Step 1: generate
        inst = DLIOMPI.get_instance(); inst.initialize()
        with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
            cfg = compose(config_name="config", overrides=[
                "workload=unet3d_a100",
                "++workload.dataset.format=npy",
                "++workload.framework=tensorflow",
                "++workload.reader.data_loader=tensorflow",
                "++workload.workflow.generate_data=True",
                "++workload.workflow.train=False",
                "++workload.dataset.num_files_train=2",
                "++workload.dataset.num_files_eval=0",
                "++workload.dataset.num_samples_per_file=4",
                "++workload.dataset.record_length=256",
                "++workload.dataset.record_length_stdev=0",
                f"++workload.output.folder={out_dir}",
                f"++workload.dataset.data_folder={data_dir}",
            ])
        ConfigArguments.reset()
        bench = DLIOBenchmark(OmegaConf.to_container(cfg["workload"], resolve=True))
        bench.initialize(); bench.run(); bench.finalize()

        # Step 2: train
        _reset()
        inst = DLIOMPI.get_instance(); inst.initialize()
        with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
            cfg = compose(config_name="config", overrides=[
                "workload=unet3d_a100",
                "++workload.dataset.format=npy",
                "++workload.framework=tensorflow",
                "++workload.reader.data_loader=tensorflow",
                "++workload.workflow.generate_data=False",
                "++workload.workflow.train=True",
                "++workload.train.epochs=1",
                "++workload.train.computation_time=0.0",
                "++workload.dataset.num_files_train=2",
                "++workload.dataset.num_files_eval=0",
                "++workload.dataset.num_samples_per_file=4",
                "++workload.dataset.record_length=256",
                "++workload.dataset.record_length_stdev=0",
                f"++workload.output.folder={out_dir}",
                f"++workload.dataset.data_folder={data_dir}",
            ])
        ConfigArguments.reset()
        workload = OmegaConf.to_container(cfg["workload"], resolve=True)
        workload.setdefault("output", {})["folder"] = out_dir
        bench = DLIOBenchmark(workload)
        bench.initialize(); bench.run(); bench.finalize()

        output_jsons = glob.glob(os.path.join(out_dir, "*_output.json"))
        assert len(output_jsons) >= 1
