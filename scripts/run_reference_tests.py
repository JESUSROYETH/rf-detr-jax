"""Run a dedicated reference selection, failing if any selected check skips."""

from __future__ import annotations

import sys

import pytest


class RequireExecutedChecks:
    def __init__(self):
        self.skipped = set()

    def pytest_runtest_logreport(self, report):
        if report.skipped:
            self.skipped.add(report.nodeid)

    def pytest_collectreport(self, report):
        if report.skipped:
            self.skipped.add(report.nodeid)


def main():
    checks = RequireExecutedChecks()
    status = pytest.main(sys.argv[1:], plugins=[checks])
    if checks.skipped:
        print("Reference checks did not execute:\n" + "\n".join(sorted(checks.skipped)), file=sys.stderr)
        return 1
    return int(status)


if __name__ == "__main__":
    raise SystemExit(main())
