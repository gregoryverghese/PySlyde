# Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Development tooling setup

Clone the repository and install the necessary development extras.

```sh
pip install -e ".[dev,docs]"
```

### Testing

To run the tests, use `pytest`.
It can automatically find tests within the repo, or you can pass it the directory containing the tests.

```sh
pytest test/
```

Some integration tests require additional resources (network access,
large models, or external datasets) and are skipped by default unless
explicitly configured.

#### Feature extractor integration tests

`test/test_feature_extractor_integration.py` contains opt-in integration
tests that exercise real model loading and a forward pass for supported
feature extractors. Some tests may download model weights from Hugging Face and/or require substantial memory, so they are skipped by default unless enabled explicitly.

Enable these tests with network access by setting:

``` sh
RUN_NETWORK_TESTS=1 pytest test/test_feature_extractor_integration.py
```

##### Gated Hugging Face models

Some models are gated on Hugging Face. If the weights are not already
present in your local cache, you must provide a Hugging Face token:

``` sh
RUN_NETWORK_TESTS=1 HUGGINGFACE_TOKEN=<your_token> pytest test/test_feature_extractor_integration.py
```

If a gated model is already cached locally, the integration tests are
designed to load it without requiring `HUGGINGFACE_TOKEN`.

##### Memory thresholds (OOM avoidance)

Large foundation models can trigger OS-level out-of-memory (OOM) kills
during loading or inference. To improve reliability, the integration
tests perform dynamic checks for available CPU RAM and (when CUDA is
available) free GPU VRAM, and will skip tests when resources are
insufficient.

You may override the default resource thresholds:

``` sh
RUN_NETWORK_TESTS=1 MIN_CPU_AVAIL_GB=24 MIN_FREE_VRAM_GB=24 pytest test/test_feature_extractor_integration.py
```

**Notes:**

-   `MIN_FREE_VRAM_GB` applies only when CUDA is available.
-   These environment variables apply only to
    `test/test_feature_extractor_integration.py`.

#### Slide integration tests

`test/test_slide_integration.py` contains opt-in integration tests for
the slide processing pipeline, including whole-slide image
loading, annotation parsing, mask generation, region extraction, and
artifact saving.

These tests require external fixture data (whole-slide images and
annotation files) and will be skipped unless integration data is
configured.

##### Required fixture structure

The integration data directory must have the following structure:

    PARENT/
    ├── annotations/
    │   ├── asap.xml
    │   ├── name.csv
    │   ├── geojson.json or .geojson
    │   ├── imagej.xml
    │   └── qupath.json
    └── wsi/
        └── wsi.ndpi

**Notes:**

-   The `annotations` directory may contain additional files.
-   Tests select the first matching file for each supported format.
-   Supported whole-slide image formats include `.svs`, `.ndpi`, `.tif`,
    `.tiff`, `.ome.tif`, and `.ome.tiff`.

##### Providing integration data

The fixture root is resolved in the following order of precedence (highest to lowest):

1.  **Local directory or archive**

``` sh
PYSLYDE_IT_DATA_DIR=/path/to/PARENT pytest test/test_slide_integration.py
```

You may also provide a compressed archive:

``` sh
PYSLYDE_IT_DATA_DIR=/path/to/fixtures.zip pytest test/test_slide_integration.py
```

2.  **Remote archive URL**

``` sh
PYSLYDE_IT_DATA_URL=<archive_url> pytest test/test_slide_integration.py
```

Optional integrity verification:

``` sh
PYSLYDE_IT_DATA_URL=<archive_url> PYSLYDE_IT_DATA_SHA256=<sha256> pytest test/test_slide_integration.py
```

3. **Default remote archive**
If neither is provided, tests will default to `DEFAULT_DATA_URL` set in `conftest.py` 

4. If neither is set, the tests skip automatically with explanations.

##### Temporary directory

You may optionally specify a temporary directory for writing outputs:

``` sh
pytest test/test_slide_integration.py --basetemp=<temp_dir>
```

**Notes:**

- Supported compressed archive formats include `.zip`, `.tar`, `.tar.gz`, and `.tgz`.
- Google Drive URLs are supported. The test suite automatically uses the
  `gdown` dependency to handle downloads.

### Linting and formatting

We use [`ruff`](https://docs.astral.sh/ruff/) to lint and format code.

To lint files in the current directory, run:

```sh
ruff check 
```

This will report any errors, e.g. unused imports.
You can run `ruff check --fix` to automatically fix errors where possible.

To format files in the current directory, run:

```sh
ruff format
```

This will automatically update formatting where needed.
If you just want to see the suggested changes, run `ruff format --diff`.
