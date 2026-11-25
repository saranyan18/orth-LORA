"""Run model-generated Python against unit tests in a subprocess (timeout + memory/CPU limits on POSIX).

WARNING: this is NOT a security sandbox.  Run evaluations inside a container / throwaway VM if you
do not trust the model output (Colab is fine).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from typing import List


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    if m:
        return m.group(1)
    return text.split("###")[0]


def _limits():  # pragma: no cover - POSIX only
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30, 2 * 2**30))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))


def run_tests(code: str, tests: List[str], setup: str = "", timeout: int = 10) -> bool:
    program = "\n".join([setup or "", code, *tests]) + "\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            r = subprocess.run([sys.executable, "-I", path], capture_output=True, timeout=timeout, cwd=d,
                               preexec_fn=_limits if os.name == "posix" else None)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False
