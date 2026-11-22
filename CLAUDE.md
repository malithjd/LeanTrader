# LeanTrader: notes for Claude

The product spec is the source of truth: https://claude.ai/artifact/8hYeY46idxvC2Ls3ipPGLe

## Dependencies (check on every pull request)

- Every third-party package that a `.py` file imports must be listed in `requirements.txt` or `requirements-ml.txt`. Standard-library modules and this repo's own modules (`dating`, `universe`, `prices`, ...) don't count.
- `requirements.txt` is for core packages: data pulls, features, rankers, tests. `requirements-ml.txt` is for GPU and language-model packages: torch, transformers, peft, bitsandbytes, accelerate, datasets, vllm.
- Every line needs a minimum version, `package>=X.Y`, set to the version the author ran (`pip show <package>` prints it).
- If a pull request adds an import of a package that isn't listed, or adds a requirements line without a version, flag it. Name the file and the import, say which requirements file it belongs in, and give the line to add (`package>=X.Y`, with X.Y from the author's `pip show`).
- The pip name can differ from the import name: `sklearn` is `scikit-learn`, `yaml` is `pyyaml`, `bs4` is `beautifulsoup4`, `PIL` is `pillow`.
