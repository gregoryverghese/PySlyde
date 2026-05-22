from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional, Sequence, Union

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

SUPPORTED_SOURCES: tuple[str, ...] = ("disk", "lmdb", "rocksdb")

PathLike = Union[str, Path]
SourceType = Literal["disk", "lmdb", "rocksdb"]
PCAMethod = Literal["incremental", "full"]


@dataclass
class PCAConfig:
    """
    Configuration for PCA fitting in PySlyde.

    Attributes:
        n_components:
            Number of principal components to retain.

            If None:
                - for full PCA, scikit-learn keeps min(n_samples, n_features)
                - for incremental PCA, the effective number of components can
                  depend on the first fitted batch, which is usually undesirable

            For reproducible dimensionality reduction, it is recommended to set
            this explicitly.

        method:
            PCA fitting method.

            Supported values:
                - "incremental": fit IncrementalPCA in batches
                - "full": fit standard PCA in memory

        batch:
            Number of feature rows to accumulate before each `partial_fit`
            call when `method="incremental"`.

            Ignored when `method="full"`.

            Must be greater than 0 for incremental PCA.

        skip_all_zero:
            Whether to discard feature rows that are entirely zero.

            This is useful for excluding empty/background feature vectors.

        sort_paths:
            Whether to sort discovered file or database paths.

            When enabled, feature data is processed in a consistent and deterministic
            order, improving reproducibility across runs and systems.

            When disabled, the processing order depends on filesystem traversal and
            may vary between runs.

            This option applies when loading features from disk, LMDB, or RocksDB
            via `fit_pca(...)`. It has no effect when fitting from parser output
            via `fit_pca_from_parser(...)`.

        verbose:
            Whether to print simple progress messages during fitting.
    """

    n_components: Optional[int] = None
    method: PCAMethod = "incremental"
    batch: int = 10000
    skip_all_zero: bool = True
    sort_paths: bool = True
    verbose: bool = False


def _iter_valid_feature_rows(
    arrays: Iterable[np.ndarray],
    *,
    skip_all_zero: bool = True,
) -> Iterator[np.ndarray]:
    """
    Normalize incoming arrays into 2D feature matrices and optionally
    remove all-zero rows.

    Each yielded item may be either:
        - a single feature vector of shape (d,)
        - a matrix of feature vectors of shape (n, d)

    Yields:
        np.ndarray: A 2D array of shape (n, d).

    Raises:
        ValueError:
            If an array is neither 1D nor 2D, or if feature dimensionality is
            inconsistent across inputs.
    """
    feature_dim: Optional[int] = None

    for arr in arrays:
        mat = np.asarray(arr)

        if mat.ndim == 1:
            mat = mat[np.newaxis, :]
        elif mat.ndim != 2:
            raise ValueError(f"Expected 1D or 2D feature array, got shape {mat.shape}")

        if feature_dim is None:
            feature_dim = mat.shape[1]
        elif mat.shape[1] != feature_dim:
            raise ValueError(
                f"Inconsistent feature dimension: expected {feature_dim}, "
                f"got {mat.shape[1]}"
            )

        if skip_all_zero:
            mat = mat[~np.all(mat == 0, axis=1)]

        if mat.shape[0] == 0:
            continue

        yield mat


def _discover_feature_files(
    *,
    root: PathLike = ".",
    pattern: str,
    paths: Optional[Sequence[PathLike]] = None,
    recursive: bool = True,
    sort_paths: bool = True,
) -> list[Path]:
    """
    Resolve explicit paths or discover matching paths under a root directory.

    If `paths` is provided, `root` and `pattern` are ignored.
    """
    if paths is not None:
        resolved = [Path(p) for p in paths]
    else:
        root = Path(root)
        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        resolved = [p for p in iterator if p.is_file()]

    if sort_paths:
        resolved = sorted(resolved)

    return resolved


def _is_lmdb_dir(path: Path) -> bool:
    """
    Return True if the path appears to be an LMDB database directory.
    """
    return (
        path.is_dir()
        and (path / "data.mdb").is_file()
        and (path / "lock.mdb").is_file()
    )


def _is_rocksdb_dir(path: Path) -> bool:
    """
    Return True if the path appears to be a RocksDB database directory.

    This uses RocksDB structural markers rather than directory naming.
    """
    if not path.is_dir():
        return False

    has_current = (path / "CURRENT").is_file()
    has_manifest = any(p.name.startswith("MANIFEST") for p in path.iterdir())

    return has_current and has_manifest


def _discover_database_dirs(
    *,
    source: Literal["lmdb", "rocksdb"],
    root: PathLike = ".",
    pattern: Optional[str] = None,
    paths: Optional[Sequence[PathLike]] = None,
    recursive: bool = True,
    sort_paths: bool = True,
) -> list[Path]:
    """
    Resolve explicit database paths or discover database directories under a root.

    Behavior:
        - if `paths` is provided, use those directly
        - else if `pattern` is provided, match database directory names using it
        - else discover database directories by internal file structure
    """
    if paths is not None:
        resolved = [Path(p) for p in paths]
    else:
        root = Path(root)

        if pattern is not None:
            iterator = root.rglob(pattern) if recursive else root.glob(pattern)
            resolved = [p for p in iterator if p.is_dir()]
        else:
            iterator = root.rglob("*") if recursive else root.glob("*")

            if source == "lmdb":
                resolved = [p for p in iterator if _is_lmdb_dir(p)]
            elif source == "rocksdb":
                resolved = [p for p in iterator if _is_rocksdb_dir(p)]
            else:
                raise ValueError(f"Unsupported data source: {source}")

    if sort_paths:
        resolved = sorted(resolved)

    return resolved


def _iter_feature_arrays_from_disk(
    *,
    root: PathLike = ".",
    pattern: Optional[str] = None,
    paths: Optional[Sequence[PathLike]] = None,
    recursive: bool = True,
    sort_paths: bool = True,
) -> Iterator[np.ndarray]:
    """
    Yield feature arrays from .npy files on disk.

    If `pattern` is None, defaults to "*.npy".
    """
    file_paths = _discover_feature_files(
        root=root,
        pattern="*.npy" if pattern is None else pattern,
        paths=paths,
        recursive=recursive,
        sort_paths=sort_paths,
    )

    for path in file_paths:
        yield np.load(path)


def _iter_feature_arrays_from_lmdb(
    *,
    root: PathLike = ".",
    pattern: Optional[str] = None,
    paths: Optional[Sequence[PathLike]] = None,
    recursive: bool = True,
    sort_paths: bool = True,
) -> Iterator[np.ndarray]:
    """
    Yield feature arrays from one or more LMDB databases.

    If `pattern` is None, LMDB database directories are discovered by structure
    (directories containing data.mdb and lock.mdb).
    """
    from pyslyde.io.lmdb_io import LMDBRead

    db_paths = _discover_database_dirs(
        source="lmdb",
        root=root,
        pattern=pattern,
        paths=paths,
        recursive=recursive,
        sort_paths=sort_paths,
    )

    for db_path in db_paths:
        reader = LMDBRead(str(db_path))
        for key in reader.get_keys():
            yield reader.read_image(key)


def _iter_feature_arrays_from_rocksdb(
    *,
    root: PathLike = ".",
    pattern: Optional[str] = None,
    paths: Optional[Sequence[PathLike]] = None,
    recursive: bool = True,
    sort_paths: bool = True,
) -> Iterator[np.ndarray]:
    """
    Yield feature arrays from one or more RocksDB databases.

    If `pattern` is None, RocksDB database directories are discovered by
    structure (e.g. CURRENT + MANIFEST*).
    """
    try:
        from pyslyde.io.rocksdb_io import RocksDBRead
    except Exception as e:
        raise ImportError(
            "RocksDB support is required for source='rocksdb', but could not be imported.\n"
            "Install it via:\n\n"
            "    pip install -e .[rocksdb]\n"
        ) from e

    db_paths = _discover_database_dirs(
        source="rocksdb",
        root=root,
        pattern=pattern,
        paths=paths,
        recursive=recursive,
        sort_paths=sort_paths,
    )

    for db_path in db_paths:
        reader = RocksDBRead(str(db_path))
        for key in reader.get_keys():
            arr = reader.read_image(key)
            if arr is not None:
                yield arr


def _iter_feature_arrays_from_parser(
    feature_generator: Iterable[tuple[tuple[int, int], np.ndarray]],
) -> Iterator[np.ndarray]:
    """
    Yield feature arrays from parser.extract_features().
    """
    for _, feature_vec in feature_generator:
        yield feature_vec


def _resolve_feature_arrays(
    *,
    source: SourceType,
    root: PathLike = ".",
    pattern: Optional[str] = None,
    paths: Optional[Sequence[PathLike]] = None,
    recursive: bool = True,
    sort_paths: bool = True,
) -> Iterator[np.ndarray]:
    """
    Resolve a feature-array iterator from the requested storage source.
    """
    if source == "disk":
        return _iter_feature_arrays_from_disk(
            root=root,
            pattern=pattern,
            paths=paths,
            recursive=recursive,
            sort_paths=sort_paths,
        )

    if source == "lmdb":
        return _iter_feature_arrays_from_lmdb(
            root=root,
            pattern=pattern,
            paths=paths,
            recursive=recursive,
            sort_paths=sort_paths,
        )

    if source == "rocksdb":
        return _iter_feature_arrays_from_rocksdb(
            root=root,
            pattern=pattern,
            paths=paths,
            recursive=recursive,
            sort_paths=sort_paths,
        )

    raise ValueError(
        f"Unsupported data source: {source}. "
        f"Supported sources are: {', '.join(repr(s) for s in SUPPORTED_SOURCES)}."
    )


def _fit_pca_from_arrays(
    arrays: Iterable[np.ndarray],
    *,
    config: PCAConfig,
):
    """
    Fit PCA from an iterable of feature arrays.

    Supports both standard PCA and IncrementalPCA.

    Returns:
        PCA or IncrementalPCA:
            Fitted PCA model.

    Raises:
        ValueError:
            If no valid feature rows are found, or if configuration is invalid.
    """
    if config.n_components is not None and config.n_components <= 0:
        raise ValueError("n_components must be > 0 or None")

    rows = _iter_valid_feature_rows(arrays, skip_all_zero=config.skip_all_zero)

    if config.method == "full":
        mats = list(rows)
        if not mats:
            raise ValueError("No valid feature rows found to fit PCA.")

        X = np.vstack(mats)

        if config.verbose:
            print(f"Fitting full PCA on {X.shape[0]} rows")

        model = PCA(n_components=config.n_components)
        model.fit(X)
        return model

    if config.method == "incremental":
        if config.batch <= 0:
            raise ValueError("batch must be > 0 for incremental PCA")

        model = IncrementalPCA(n_components=config.n_components)
        batch_parts: list[np.ndarray] = []
        rows_in_batch = 0
        total_rows = 0

        def flush() -> None:
            nonlocal batch_parts, rows_in_batch, total_rows
            if not batch_parts:
                return

            X = np.vstack(batch_parts)

            if config.verbose:
                print(f"partial_fit on {X.shape[0]} rows")

            model.partial_fit(X)
            total_rows += X.shape[0]
            batch_parts = []
            rows_in_batch = 0

        for mat in rows:
            batch_parts.append(mat)
            rows_in_batch += mat.shape[0]

            if rows_in_batch >= config.batch:
                flush()

        flush()

        if total_rows == 0:
            raise ValueError("No valid feature rows found to fit PCA.")

        return model

    raise ValueError(f"Unknown PCA method: {config.method}")


def fit_pca(
    *,
    source: SourceType,
    config: Optional[PCAConfig] = None,
    root: PathLike = ".",
    pattern: Optional[str] = None,
    paths: Optional[Sequence[PathLike]] = None,
    recursive: bool = True,
):
    """
    Fit PCA from stored feature vectors.

    Args:
        source:
            Storage source containing the feature vectors.

            Supported values:
                - "disk": feature arrays stored as .npy files
                - "lmdb": one or more LMDB databases
                - "rocksdb": one or more RocksDB databases

        config:
            PCA fitting configuration. If None, defaults to `PCAConfig()`.

        root:
            Root directory under which files or database directories are discovered.

        pattern:
            Optional pattern used to select feature containers under `root`.

            Behavior by source:
                - disk:
                    pattern matches feature files; defaults to "*.npy" if omitted
                - lmdb:
                    pattern matches LMDB database directory names
                - rocksdb:
                    pattern matches RocksDB database directory names

            If omitted for LMDB or RocksDB, database directories are discovered
            by their internal file structure.

        paths:
            Optional explicit file or database paths.

            If provided, these are used directly and `root` / `pattern` are ignored.

        recursive:
            Whether discovery under `root` should recurse into subdirectories.

    Returns:
        PCA or IncrementalPCA:
            Fitted PCA model.
    """
    config = PCAConfig() if config is None else config

    arrays = _resolve_feature_arrays(
        source=source,
        root=root,
        pattern=pattern,
        paths=paths,
        recursive=recursive,
        sort_paths=config.sort_paths,
    )

    return _fit_pca_from_arrays(arrays, config=config)


def fit_pca_from_parser(
    feature_generator: Iterable[tuple[tuple[int, int], np.ndarray]],
    *,
    config: Optional[PCAConfig] = None,
):
    """
    Fit PCA directly from `parser.extract_features(...)` output.

    Args:
        feature_generator:
            Generator yielding `((x, y), feature_vec)` tuples.

        config:
            PCA fitting configuration. If None, defaults to `PCAConfig()`.

    Returns:
        PCA or IncrementalPCA:
            Fitted PCA model.
    """
    config = PCAConfig() if config is None else config
    arrays = _iter_feature_arrays_from_parser(feature_generator)
    return _fit_pca_from_arrays(arrays, config=config)
