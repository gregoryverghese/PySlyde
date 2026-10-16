# Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Development tooling setup

Clone the repository and install the necessary development extras.

```sh
pip install -e ".[all,dev,docs]"
```

### Testing

Run the tests from the repository root. The three most common cases are:

```sh
# 1. Fast: unit tests only (mocked, no network)
pytest -m "not integration and not models"

# 2. Everything that needs no special set-up or access: unit, integration
#    and open-model tests (network needed on the first run)
RUN_NETWORK_TESTS=1 pytest -m "not gated and not bigmem"

# 3. Everything: also gated and very large models, plus the tutorial notebooks
RUN_NETWORK_TESTS=1 HUGGINGFACE_TOKEN=<token> pytest
pytest --nbmake docs/examples/*.ipynb
```

See [Running the tests](https://pyslyde.readthedocs.io/en/latest/testing.html) for what each marker
(`integration`, `models`, `gated`, `bigmem`) covers, the environment variables
that control them, and how to run the integration tests on your own data.

### Documentation

The documentation (Sphinx, in `docs/`) is built with warnings as errors in CI
and on Read the Docs. With the `docs` extra installed (see above), build it
locally with:

```sh
sphinx-build -W --keep-going -b html docs docs/_build/html
```

The tutorial notebooks in `docs/examples/` are rendered with their stored
outputs. After changing one, re-execute it top to bottom before committing,
e.g. with `jupyter execute --inplace docs/examples/<name>.ipynb`.

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
