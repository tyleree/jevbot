"""The `jevbot` command line (DESIGN.md section 14).

`jevbot.cli.main` is the typer root (console script `jevbot = jevbot.cli.main:app`). Every area's sub-app lives in the file owned by
the package that implements the area (`data_cmds`, `jev_cmds`, `backtest_cmds`, `baselines_cmds`, `eval_cmds`, `leakage_cmds`,
`paper_cmds`, `record_cmds`, `doctor`); the root registers them lazily from a static list, so this package imports nothing here.
"""
