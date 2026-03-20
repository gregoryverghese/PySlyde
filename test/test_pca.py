import numpy as np
import pytest
from sklearn.decomposition import PCA, IncrementalPCA

from pyslyde.util.pca import PCAConfig, fit_pca, fit_pca_from_parser

pytestmark = pytest.mark.pca


def test_fit_pca_from_parser_full_mode():
    """
    fit_pca_from_parser should fit a standard PCA model from parser-like
    feature output.
    """
    features = [
        ((0, 0), np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        ((1, 0), np.array([2.0, 3.0, 4.0], dtype=np.float32)),
        ((2, 0), np.array([3.0, 4.0, 5.0], dtype=np.float32)),
        ((3, 0), np.array([4.0, 5.0, 6.0], dtype=np.float32)),
    ]

    config = PCAConfig(n_components=2, method="full")
    model = fit_pca_from_parser(features, config=config)

    assert isinstance(model, PCA)
    assert model.components_.shape == (2, 3)
    assert model.n_components_ == 2


def test_fit_pca_disk_incremental_mode(tmp_path):
    """
    fit_pca should discover .npy files on disk and fit IncrementalPCA.
    """
    arr1 = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    arr2 = np.array([[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    np.save(tmp_path / "a.npy", arr1)
    np.save(tmp_path / "b.npy", arr2)

    config = PCAConfig(n_components=2, method="incremental", batch=2)
    model = fit_pca(source="disk", root=tmp_path, config=config)

    assert isinstance(model, IncrementalPCA)
    assert model.components_.shape == (2, 3)
    assert model.n_components == 2


def test_fit_pca_disk_recursive_discovery(tmp_path):
    """
    fit_pca should recursively discover .npy feature files.
    """
    subdir = tmp_path / "nested"
    subdir.mkdir()

    arr = np.array([[1.0, 0.0, 2.0], [2.0, 1.0, 3.0]], dtype=np.float32)
    np.save(subdir / "features.npy", arr)

    model = fit_pca(
        source="disk",
        root=tmp_path,
        config=PCAConfig(n_components=1, method="full"),
    )

    assert model.components_.shape == (1, 3)


def test_fit_pca_skips_zero_rows(tmp_path):
    """
    All-zero rows should be removed when skip_all_zero=True.
    """
    arr1 = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    arr2 = np.array([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]], dtype=np.float32)

    np.save(tmp_path / "a.npy", arr1)
    np.save(tmp_path / "b.npy", arr2)

    model = fit_pca(
        source="disk",
        root=tmp_path,
        config=PCAConfig(n_components=1, method="full", skip_all_zero=True),
    )

    assert model.components_.shape == (1, 3)


def test_fit_pca_invalid_n_components(tmp_path):
    """
    n_components must be > 0 or None.
    """
    arr = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    np.save(tmp_path / "a.npy", arr)

    with pytest.raises(ValueError, match="n_components must be > 0 or None"):
        fit_pca(
            source="disk",
            root=tmp_path,
            config=PCAConfig(n_components=0, method="full"),
        )


def test_fit_pca_invalid_batch_for_incremental(tmp_path):
    """
    batch must be > 0 for incremental PCA.
    """
    arr = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    np.save(tmp_path / "a.npy", arr)

    with pytest.raises(ValueError, match="batch must be > 0 for incremental PCA"):
        fit_pca(
            source="disk",
            root=tmp_path,
            config=PCAConfig(n_components=1, method="incremental", batch=0),
        )
