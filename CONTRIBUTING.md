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
