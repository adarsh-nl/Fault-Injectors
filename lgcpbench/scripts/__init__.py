"""CLI entry points for LGCP-Bench.

    python -m lgcpbench.scripts.benchmark   # clean run + fault sweep
    python -m lgcpbench.scripts.evaluate    # one condition
    python -m lgcpbench.scripts.simulate    # network-only latency sweep

All three take positional config overrides, so no source edit is ever
required to change an experiment.
"""
