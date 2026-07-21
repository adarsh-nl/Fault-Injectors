"""
Command-line entry points.

Nothing in `common.py` makes a decision. It reads resolved config and
assembles the modules that implement the paper. If a value is not in the
config, it is not configurable.

Entry points
------------
train      train one model (either track) under clean conditions
evaluate   score a checkpoint under one condition
benchmark  run a whole fault sweep and write the results bundle

All three take positional `key=value` config overrides and a `--config` path,
following the lgcpbench CLI convention (parse_known_args, `main(argv)->int`).
"""
