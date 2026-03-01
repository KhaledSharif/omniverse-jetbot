"""Adapter layer for reading/writing demonstration data in NPZ and HDF5 formats.

Provides:
- open_demo(filepath) — unified reader dispatching on file extension
- NpzDemoData / Hdf5DemoData — dict-like wrappers for both formats
- HDF5DemoWriter — incremental writer with O(delta) appends
- convert_npz_to_hdf5() — migration utility for existing NPZ files

HDF5 layout mirrors the flat NPZ layout: per-step datasets (observations, actions,
rewards, dones, costs) plus per-episode metadata datasets (episode_starts,
episode_lengths, episode_returns, episode_success). File-level metadata is stored
as HDF5 root attributes.
"""

import os
import numpy as np

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


# ---------------------------------------------------------------------------
# Unified reader
# ---------------------------------------------------------------------------

def open_demo(filepath: str) -> 'DemoData':
    """Open a demo file (.npz or .hdf5) and return a dict-like reader.

    Args:
        filepath: Path to .npz or .hdf5 file

    Returns:
        DemoData subclass supporting data['key'], 'key' in data, and close()
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.npz':
        return NpzDemoData(filepath)
    elif ext in ('.hdf5', '.h5'):
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required to read HDF5 demo files. Install with: pip install h5py")
        return Hdf5DemoData(filepath)
    else:
        raise ValueError(f"Unsupported demo file extension: {ext} (expected .npz or .hdf5)")


# ---------------------------------------------------------------------------
# DemoData base / NPZ wrapper
# ---------------------------------------------------------------------------

class DemoData:
    """Base class for demo data readers."""

    def __getitem__(self, key):
        raise NotImplementedError

    def __contains__(self, key):
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class NpzDemoData(DemoData):
    """Wraps np.load() result with a consistent interface."""

    def __init__(self, filepath: str):
        self._data = np.load(filepath, allow_pickle=True)

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def close(self):
        if hasattr(self._data, 'close'):
            self._data.close()


class _MetadataWrapper:
    """Wraps an HDF5 attrs dict so that .item() returns a plain Python dict,
    matching the np.array(dict, dtype=object).item() convention used by NPZ."""

    def __init__(self, attrs_dict: dict):
        self._dict = attrs_dict

    def item(self):
        return self._dict


class Hdf5DemoData(DemoData):
    """Reads HDF5 demo files with the same dict-like interface as NpzFile."""

    def __init__(self, filepath: str):
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required to read HDF5 demo files")
        self._h5 = h5py.File(filepath, 'r')

    def __getitem__(self, key):
        if key == 'metadata':
            return _MetadataWrapper(dict(self._h5.attrs))
        return self._h5[key][:]

    def __contains__(self, key):
        if key == 'metadata':
            return len(self._h5.attrs) > 0
        return key in self._h5

    def close(self):
        if self._h5:
            self._h5.close()
            self._h5 = None


# ---------------------------------------------------------------------------
# HDF5DemoWriter — incremental O(delta) writer
# ---------------------------------------------------------------------------

class HDF5DemoWriter:
    """Incremental HDF5 writer for demo recording.

    Manages an open h5py.File with resizable, chunked datasets. Only the
    delta since the last flush is written to disk, giving O(delta) cost
    regardless of total dataset size.
    """

    # Dataset specifications: (name, dtype, is_per_step, chunk_rows, compress)
    _STEP_DATASETS = [
        ('observations', np.float32, 256, 'lzf'),
        ('actions',      np.float32, 256, 'lzf'),
        ('rewards',      np.float32, 256, 'lzf'),
        ('dones',        np.bool_,   256, None),
        ('costs',        np.float32, 256, 'lzf'),
    ]
    _EP_DATASETS = [
        ('episode_starts',  np.int64,   64, None),
        ('episode_lengths', np.int64,   64, None),
        ('episode_returns', np.float32, 64, None),
        ('episode_success', np.bool_,   64, None),
    ]

    def __init__(self, filepath: str, obs_dim: int, action_dim: int,
                 chunk_rows: int = 256, compression: str = 'lzf',
                 image_shape: tuple = None):
        """Open or create an HDF5 demo file for incremental writing.

        If the file already exists, it is opened in append mode and cursors
        are set to the current dataset sizes (for resume support).

        Args:
            filepath: Path to .hdf5 file
            obs_dim: Observation dimension
            action_dim: Action dimension
            chunk_rows: Chunk size for per-step datasets
            compression: Compression filter ('lzf', 'gzip', or None)
            image_shape: Optional (H, W, C) shape for camera images. When set,
                creates a /images dataset with gzip compression for uint8 frames.
        """
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required for HDF5 demo writing. Install with: pip install h5py")

        self.filepath = filepath
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._image_shape = image_shape

        existing = os.path.exists(filepath)
        self._h5 = h5py.File(filepath, 'a')

        if existing and 'observations' in self._h5:
            # Resume: set cursors to existing data size
            self._step_cursor = self._h5['observations'].shape[0]
            self._ep_cursor = self._h5['episode_starts'].shape[0] if 'episode_starts' in self._h5 else 0
        else:
            self._step_cursor = 0
            self._ep_cursor = 0
            self._create_datasets(chunk_rows, compression)

    def _create_datasets(self, chunk_rows: int, compression: str):
        """Create resizable datasets in the HDF5 file."""
        for name, dtype, cr, comp in self._STEP_DATASETS:
            if name == 'observations':
                shape = (0, self.obs_dim)
                maxshape = (None, self.obs_dim)
                chunks = (min(chunk_rows, cr), self.obs_dim)
            elif name == 'actions':
                shape = (0, self.action_dim)
                maxshape = (None, self.action_dim)
                chunks = (min(chunk_rows, cr), self.action_dim)
            else:
                shape = (0,)
                maxshape = (None,)
                chunks = (min(chunk_rows, cr),)

            # Use the specified compression, or override per-dataset
            ds_comp = comp if comp else compression if compression else None
            # dones don't benefit much from compression
            if name == 'dones':
                ds_comp = None

            self._h5.create_dataset(
                name, shape=shape, maxshape=maxshape, dtype=dtype,
                chunks=chunks, compression=ds_comp,
            )

        for name, dtype, cr, comp in self._EP_DATASETS:
            self._h5.create_dataset(
                name, shape=(0,), maxshape=(None,), dtype=dtype,
                chunks=(min(chunk_rows, cr),), compression=comp,
            )

        # Optional images dataset for camera recordings
        if self._image_shape is not None:
            h, w, c = self._image_shape
            self._h5.create_dataset(
                'images',
                shape=(0, h, w, c),
                maxshape=(None, h, w, c),
                dtype=np.uint8,
                chunks=(min(32, chunk_rows), h, w, c),
                compression='gzip',
                compression_opts=4,
            )

    def append_steps(self, observations, actions, rewards, dones, costs):
        """Append new step data to datasets.

        Args:
            observations: array-like (N, obs_dim)
            actions: array-like (N, action_dim)
            rewards: array-like (N,)
            dones: array-like (N,)
            costs: array-like (N,)
        """
        obs = np.asarray(observations, dtype=np.float32)
        act = np.asarray(actions, dtype=np.float32)
        rew = np.asarray(rewards, dtype=np.float32)
        don = np.asarray(dones, dtype=np.bool_)
        cos = np.asarray(costs, dtype=np.float32)

        n = len(obs)
        if n == 0:
            return

        for ds_name, arr in [('observations', obs), ('actions', act),
                              ('rewards', rew), ('dones', don), ('costs', cos)]:
            ds = self._h5[ds_name]
            old_len = ds.shape[0]
            ds.resize(old_len + n, axis=0)
            ds[old_len:old_len + n] = arr

        self._step_cursor += n

    def append_images(self, images):
        """Append camera images to the /images dataset.

        Args:
            images: array-like (N, H, W, C) uint8
        """
        if 'images' not in self._h5:
            return
        imgs = np.asarray(images, dtype=np.uint8)
        n = len(imgs)
        if n == 0:
            return
        ds = self._h5['images']
        old_len = ds.shape[0]
        ds.resize(old_len + n, axis=0)
        ds[old_len:old_len + n] = imgs

    def append_episode(self, start: int, length: int, ep_return: float, success: bool):
        """Append one episode metadata row.

        Args:
            start: Step index where this episode starts
            length: Number of steps in this episode
            ep_return: Total episode return
            success: Whether the episode was successful
        """
        for ds_name, value in [('episode_starts', np.int64(start)),
                                ('episode_lengths', np.int64(length)),
                                ('episode_returns', np.float32(ep_return)),
                                ('episode_success', np.bool_(success))]:
            ds = self._h5[ds_name]
            old_len = ds.shape[0]
            ds.resize(old_len + 1, axis=0)
            ds[old_len] = value

        self._ep_cursor += 1

    def truncate_to(self, n_steps: int):
        """Truncate per-step datasets to n_steps rows.

        Used by abandon_episode() to discard partial episode data that was
        already flushed to disk.

        Args:
            n_steps: Number of steps to keep
        """
        for name, _, _, _ in self._STEP_DATASETS:
            ds = self._h5[name]
            if ds.shape[0] > n_steps:
                ds.resize(n_steps, axis=0)
        # Also truncate images if present
        if 'images' in self._h5:
            ds = self._h5['images']
            if ds.shape[0] > n_steps:
                ds.resize(n_steps, axis=0)
        self._step_cursor = n_steps

    def set_metadata(self, metadata_dict: dict):
        """Store metadata as HDF5 root attributes.

        Args:
            metadata_dict: Dict of metadata key/value pairs
        """
        for key, value in metadata_dict.items():
            self._h5.attrs[key] = value

    def flush(self):
        """Flush HDF5 buffers to disk."""
        if self._h5:
            self._h5.flush()

    def close(self):
        """Flush and close the HDF5 file."""
        if self._h5:
            self._h5.flush()
            self._h5.close()
            self._h5 = None

    @property
    def step_cursor(self):
        """Number of steps currently on disk."""
        return self._step_cursor

    @property
    def ep_cursor(self):
        """Number of episodes currently on disk."""
        return self._ep_cursor


# ---------------------------------------------------------------------------
# NPZ → HDF5 converter
# ---------------------------------------------------------------------------

def convert_npz_to_hdf5(npz_path: str, hdf5_path: str):
    """Convert an existing NPZ demo file to HDF5 format.

    Args:
        npz_path: Path to source .npz file
        hdf5_path: Path to destination .hdf5 file
    """
    if not HDF5_AVAILABLE:
        raise ImportError("h5py is required for HDF5 conversion. Install with: pip install h5py")

    data = np.load(npz_path, allow_pickle=True)

    obs = data['observations']
    actions = data['actions']
    obs_dim = obs.shape[1] if obs.ndim == 2 else 0
    action_dim = actions.shape[1] if actions.ndim == 2 else 0

    writer = HDF5DemoWriter(hdf5_path, obs_dim=obs_dim, action_dim=action_dim)

    # Write step data
    rewards = data['rewards']
    dones = data['dones'] if 'dones' in data else np.zeros(len(obs), dtype=bool)
    costs = data['costs'] if 'costs' in data else np.zeros(len(obs), dtype=np.float32)
    writer.append_steps(obs, actions, rewards, dones, costs)

    # Write episode data
    ep_starts = data['episode_starts']
    ep_lengths = data['episode_lengths']
    ep_returns = data['episode_returns'] if 'episode_returns' in data else np.zeros(len(ep_starts), dtype=np.float32)
    ep_success = data['episode_success'] if 'episode_success' in data else np.zeros(len(ep_starts), dtype=bool)

    for i in range(len(ep_starts)):
        writer.append_episode(
            int(ep_starts[i]), int(ep_lengths[i]),
            float(ep_returns[i]), bool(ep_success[i])
        )

    # Write metadata
    if 'metadata' in data:
        metadata = data['metadata'].item()
        if isinstance(metadata, dict):
            writer.set_metadata(metadata)

    writer.close()
    print(f"Converted {npz_path} -> {hdf5_path}")
    print(f"  {len(obs)} steps, {len(ep_starts)} episodes")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert NPZ demo files to HDF5')
    parser.add_argument('input', help='Input .npz file')
    parser.add_argument('output', nargs='?', default=None, help='Output .hdf5 file (default: same name with .hdf5)')
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = os.path.splitext(args.input)[0] + '.hdf5'

    convert_npz_to_hdf5(args.input, output)
