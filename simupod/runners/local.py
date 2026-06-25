"""Run phsolver as a local subprocess.

The solver streams JSON-lines events on stdout (NUMERICS.md section 7), e.g.
``{"event": "progress", "step": 100, ...}`` and on failure
``{"event": "error", "reason": "divergence"}``; outputs land in the output
directory as monitor binaries plus ``manifest.json`` (simupod.data).
"""

import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional, Union

from ..components import Simulation
from ..data import SimulationData
from .progress import default_renderer

_STDERR_TAIL_CHARS = 4000


class SolverRunError(RuntimeError):
    """phsolver could not be found, failed, or reported an error event."""

    def __init__(self, message: str, *, returncode: Optional[int] = None,
                 stderr_tail: Optional[str] = None):
        text = message
        if returncode is not None:
            text += f" (exit code {returncode})"
        if stderr_tail:
            text += "\n--- stderr (tail) ---\n" + stderr_tail
        super().__init__(text)
        self.returncode = returncode
        self.stderr_tail = stderr_tail


def _as_executable(path: Union[str, Path]) -> Optional[Path]:
    p = Path(path)
    return p if p.is_file() and os.access(p, os.X_OK) else None


def find_solver(solver_path: Union[str, Path, None] = None) -> Optional[Path]:
    """Locate the phsolver binary: explicit argument, then $SIMUPOD_SOLVER
    (legacy $PHOTONHUB_SOLVER still accepted), then PATH, then the in-repo default
    build directory. An explicit argument or environment override that does not
    exist is an error, not a fallthrough. Returns None only when nothing is
    configured and no binary is found."""
    if solver_path is not None:
        p = _as_executable(solver_path)
        if p is None:
            raise SolverRunError(f"solver_path is not an executable file: {solver_path}")
        return p
    env = os.environ.get("SIMUPOD_SOLVER") or os.environ.get("PHOTONHUB_SOLVER")
    if env:
        p = _as_executable(env)
        if p is None:
            raise SolverRunError(f"$SIMUPOD_SOLVER is not an executable file: {env}")
        return p
    on_path = shutil.which("phsolver")
    if on_path:
        return Path(on_path)
    # repo root / build / phsolver, for in-tree development checkouts
    return _as_executable(Path(__file__).resolve().parents[3] / "build" / "phsolver")


def _device_args(device: Union[str, None]) -> list:
    """Validate a device selector and turn it into the phsolver ``--device``
    flag, or ``[]`` when unset (the solver then defaults to CPU). Accepts
    ``"cpu"``, ``"gpu"``, or ``"gpu:N"`` (N a device index) — the engine CLI's
    grammar (engine/src/main/phsolver.cpp). Rejected client-side so a typo fails
    here with a clear message rather than at the solver."""
    if device is None:
        return []
    d = device.strip()
    ok = d in ("cpu", "gpu") or (
        d.startswith("gpu:") and d[4:].isdigit() and d[4:] != "")
    if not ok:
        raise SolverRunError(
            f"invalid device {device!r}: expected 'cpu', 'gpu', or 'gpu:N'")
    return ["--device", d]


def run_local(
    sim: Simulation,
    output_dir: Union[str, Path, None] = None,
    solver_path: Union[str, Path, None] = None,
    progress: Optional[Callable[[dict], None]] = None,
    timeout: Optional[float] = None,
    device: Union[str, None] = None,
    quiet: bool = False,
    log_file: Union[str, Path, None] = None,
) -> SimulationData:
    """Run ``phsolver run sim.json --output <dir>`` and load the results.

    ``progress`` (if given) receives every parsed JSON-lines event dict as it
    arrives, and takes over the human surface. When ``progress`` is ``None`` a
    default live status line (field decay vs. the shutoff threshold, phase,
    stability, throughput, ETA) is rendered to stderr; pass ``quiet=True`` to
    silence it. Either way the child runs with ``--progress none`` so Python is
    the only thing drawing the status line.
    ``log_file`` (if given) is forwarded to ``phsolver --log-file`` so the engine
    mirrors the full JSON-lines event stream (start/progress/done/error — field
    decay, phase, stability, throughput) to that path *as it runs*. The engine
    writes it directly, so the record survives even if this process is killed or
    crashes; it is independent of ``progress``/``quiet``. The parent directory is
    created if needed.
    ``timeout`` (seconds) kills the solver and raises. Outputs go to
    ``output_dir`` (created if needed) or a fresh persistent temp directory.
    ``device`` selects the backend — ``"cpu"`` (default when ``None``), ``"gpu"``,
    or ``"gpu:N"``; it is passed to ``phsolver --device``. GPU ``ModeSource``
    injection is wired in the engine (NUMERICS §18); its numerical GPU↔CPU
    equivalence is gated by ``GpuEquivalence.ModeSourceSceneMatchesCpu`` and is
    pending hardware verification (CI compiles the HIP path but does not run it).
    """
    solver = find_solver(solver_path)
    if solver is None:
        raise SolverRunError(
            "phsolver binary not found: pass solver_path=, set $SIMUPOD_SOLVER, "
            "put phsolver on PATH, or build the engine "
            "(cmake -S engine -B build && cmake --build build)"
        )

    out_dir = Path(output_dir) if output_dir is not None else Path(
        tempfile.mkdtemp(prefix="photonhub-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "sim.json"
    # to_wire_json (not a raw model_dump_json) so the canonical wire rules
    # apply — notably the omission of an unset pml_num_layers, which keeps
    # Phase-0-style specs consumable by schema-1.0 phsolver binaries.
    spec_path.write_text(sim.to_wire_json() + "\n", encoding="utf-8")

    cmd = [str(solver), "run", str(spec_path), "--output", str(out_dir),
           "--progress", "none"]
    cmd += _device_args(device)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd += ["--log-file", str(log_path)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)

    # Human-readable status: a caller-supplied callback owns the surface; else
    # render a default live status line to stderr unless silenced. The child is
    # told --progress none regardless, so Python is the single human surface.
    renderer = default_renderer() if (progress is None and not quiet) else None

    stderr_chunks: list = []
    stderr_thread = threading.Thread(
        target=lambda: stderr_chunks.append(proc.stderr.read()), daemon=True)
    stderr_thread.start()

    timed_out = threading.Event()
    watchdog = None
    if timeout is not None:
        def _kill():
            timed_out.set()
            proc.kill()
        watchdog = threading.Timer(timeout, _kill)
        watchdog.daemon = True
        watchdog.start()

    error_event = None
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # non-JSON chatter is tolerated, never fatal
            if not isinstance(event, dict):
                continue
            if progress is not None:
                progress(event)
            elif renderer is not None:
                renderer(event)
            if event.get("event") == "error":
                error_event = event
        returncode = proc.wait()
    finally:
        if watchdog is not None:
            watchdog.cancel()
        proc.stdout.close()
        stderr_thread.join(timeout=5.0)
        proc.stderr.close()

    stderr_tail = ("".join(c for c in stderr_chunks if c))[-_STDERR_TAIL_CHARS:]

    if timed_out.is_set():
        raise SolverRunError(f"phsolver timed out after {timeout} s and was killed",
                             stderr_tail=stderr_tail)
    if error_event is not None:
        raise SolverRunError(
            f"solver reported an error: {error_event.get('reason', error_event)}",
            returncode=returncode, stderr_tail=stderr_tail)
    if returncode != 0:
        raise SolverRunError("phsolver exited with an error",
                             returncode=returncode, stderr_tail=stderr_tail)

    # "Solver lies" guard: exit 0 with a missing/malformed manifest or .bin
    # is still a solver failure — surface it as SolverRunError so callers
    # have a single exception surface, not raw FileNotFoundError/ValueError.
    try:
        return SimulationData(out_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        raise SolverRunError(
            f"phsolver exited cleanly but its outputs are unreadable: {e}",
            returncode=returncode, stderr_tail=stderr_tail) from e
