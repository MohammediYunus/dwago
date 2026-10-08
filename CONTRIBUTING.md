# Contributing

Bug reports, focused fixes, and documentation improvements are welcome.

## Set up and test

Use Python 3.10 or newer and a virtual environment:

```sh
git clone https://github.com/MohammediYunus/dwago.git
cd dwago
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,lexical,mcp]"
python -m pytest
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
The test extras include the optional MCP dependency so its tests run too.
The suite uses local fixtures and needs no model downloads or API keys.
GitHub Actions runs the same tests on Python 3.10 through 3.14 on Linux,
and Python 3.12 on Windows and macOS.
Optional embedding backends are not exercised by this workflow.

## Report a problem

Include the command you ran, the Python and dwago versions, expected and
actual behavior, and a small reproducible example. For an incorrect graph
edge, a few files showing the imports are usually enough. Remove secrets
and private source code from examples and logs before sharing them.

## Send a change

Keep each pull request focused on one problem. For a behavior change, add
a regression test that fails before the fix and passes afterward, then run
the full suite. Explain the problem, the resulting behavior, and the checks
you actually ran. An issue is helpful for larger changes, but a small fix
can go straight to a pull request.
