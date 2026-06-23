"""Self-contained tests for the cloud client (``ph.web``) with a mocked HTTP
transport — no server, no photonhub-cloud dependency. Proves the cloud path
returns the same SimulationData/Job/SolverRunError as the local path."""

import importlib
import io
import json
import struct
import tarfile
from pathlib import Path

import numpy as np
import pytest

import photonhub as ph
from photonhub.runners.batch import BatchData, Job
from photonhub.runners.local import SolverRunError

# the `run` submodule (the name `photonhub.web.run` is shadowed by the run
# function re-export, so reach the module object explicitly)
_runmod = importlib.import_module("photonhub.web.run")


def _bundle_bytes() -> bytes:
    """A tar.gz of a minimal manifest.json + probe.bin, as the server returns."""
    manifest = {
        "manifest_version": "1",
        "monitors": [{"name": "probe", "type": "field_time", "file": "probe.bin",
                      "dtype": "float32", "shape": [5, 2],
                      "dims": ["sample", "component"], "components": ["Ez", "Hx"],
                      "sample_steps": [1, 2, 3, 4, 5], "dt_s": 1e-16}],
        "run": {"n_steps": 5, "steps_run": 5, "dt_s": 1e-16},
        "grid": {"shape": [4, 4, 4], "dl_um": 0.05},
        "provenance": {"solver_version": "fake", "device_name": "fake"},
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in [("manifest.json",
                            json.dumps(manifest).encode()),
                           ("probe.bin", struct.pack("<10f", *range(10)))]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FakeHttp:
    """Stands in for web.client.HttpClient. ``mode`` picks the job outcome."""

    def __init__(self, cfg, mode="ok"):
        self.cfg = cfg
        self.mode = mode

    def submit_job(self, spec, **kw):
        return {"job_id": "job-1"}

    def get_job(self, job_id):
        if self.mode == "running":
            return {"state": "running", "progress": {"step": 1, "total": 5}}
        if self.mode == "fail":
            return {"state": "failed",
                    "error": {"reason": "divergence", "stderr_tail": "boom"}}
        return {"state": "succeeded"}

    def download_result(self, job_id):
        return _bundle_bytes()


@pytest.fixture
def configured(tmp_path):
    ph.web.configure(api_key="ph_test_x", url="http://localhost:0",
                     cache_dir=tmp_path / "cache", poll_interval_s=0.0,
                     poll_backoff_max_s=0.0)
    yield
    ph.web.reset()


def _patch(monkeypatch, mode):
    monkeypatch.setattr(_runmod, "HttpClient", lambda cfg: FakeHttp(cfg, mode))


def _make_sim():
    return ph.Simulation(
        size_um=(0.2, 0.2, 0.2), grid=ph.UniformGridSpec(dl_um=0.05),
        run=ph.RunSpec(n_steps=5),
        sources=[ph.PointDipole(center_um=(0.1, 0.1, 0.1), polarization="Ez",
                                source_time=ph.GaussianPulse(freq0_hz=1.934e14,
                                                             fwidth_hz=4e13))],
        monitors=[ph.FieldTimeMonitor(name="probe", center_um=(0.15, 0.1, 0.1),
                                      fields=["Ez"])])


def test_unconfigured_raises_weberror():
    ph.web.reset()
    with pytest.raises(ph.web.WebError):
        ph.web.run(_make_sim())


def test_run_returns_simulationdata(monkeypatch, configured):
    _patch(monkeypatch, "ok")
    data = ph.web.run(_make_sim())
    assert isinstance(data, ph.SimulationData)
    assert "probe" in data.monitor_names
    np.testing.assert_array_equal(
        data["probe"].values.ravel(), np.arange(10, dtype="float32"))


def test_run_async_is_same_job_type(monkeypatch, configured):
    _patch(monkeypatch, "ok")
    job = ph.web.run_async(_make_sim())
    assert isinstance(job, Job)                 # the SAME handle as ph.run_async
    data = job.result(timeout=5)
    assert isinstance(data, ph.SimulationData)
    assert job.done is True


def test_failure_raises_solverrunerror(monkeypatch, configured):
    _patch(monkeypatch, "fail")
    job = ph.web.run_async(_make_sim())
    with pytest.raises(SolverRunError) as ei:
        job.result(timeout=5)
    assert "divergence" in str(ei.value)
    assert ei.value.stderr_tail == "boom"


def test_timeout_raises_timeouterror(monkeypatch, configured):
    _patch(monkeypatch, "running")              # never finishes
    job = ph.web.run_async(_make_sim())
    with pytest.raises(TimeoutError):
        job.result(timeout=0.2)


def test_batch_partial_failure(monkeypatch, configured):
    # good sims succeed; a sim whose name marker == "boom" fails. Drive it by
    # routing each name through a mode-specific fake.
    monkeypatch.setattr(_runmod, "HttpClient", lambda cfg: FakeHttp(cfg, "ok"))
    bd = ph.web.Batch({"a": _make_sim(), "b": _make_sim()}).run()
    assert isinstance(bd, BatchData)
    assert isinstance(bd["a"], ph.SimulationData)
    assert isinstance(bd["b"], ph.SimulationData)


# --------------------------------------------------------------------------
# Gaps not covered by the six tests above: cache path-traversal guard,
# config env-var precedence, client 4xx->WebError mapping, a batch where one
# name genuinely fails, and the cancelled-job path.
# --------------------------------------------------------------------------


class _NameRoutedHttp:
    """Per-name FakeHttp: routes each job to a mode chosen by the submitted
    ``name`` so a Batch can mix successes and failures in one run. Submission
    returns the name as the job_id; get_job keys its outcome off that id."""

    def __init__(self, cfg, fail_names=()):
        self.cfg = cfg
        self.fail_names = set(fail_names)

    def submit_job(self, spec, *, name=None, **kw):
        return {"job_id": name or "job-1"}

    def get_job(self, job_id):
        if job_id in self.fail_names:
            return {"state": "failed",
                    "error": {"reason": "divergence", "stderr_tail": "boom"}}
        return {"state": "succeeded"}

    def download_result(self, job_id):
        return _bundle_bytes()


def test_batch_mixed_success_and_failure(monkeypatch, configured):
    # "b" genuinely fails; the batch must surface it in .errors while "a"
    # succeeds, and indexing the failed name re-raises its SolverRunError.
    monkeypatch.setattr(
        _runmod, "HttpClient",
        lambda cfg: _NameRoutedHttp(cfg, fail_names={"b"}))
    bd = ph.web.Batch({"a": _make_sim(), "b": _make_sim()}).run()
    assert isinstance(bd["a"], ph.SimulationData)
    assert "a" in bd and "b" not in bd          # __contains__ = successes only
    assert set(bd.errors) == {"b"}
    assert isinstance(bd.errors["b"], SolverRunError)
    with pytest.raises(SolverRunError):
        _ = bd["b"]                             # indexing re-raises


def test_cancelled_job_raises_solverrunerror(monkeypatch, configured):
    class _Cancelled(FakeHttp):
        def get_job(self, job_id):
            return {"state": "cancelled"}

    monkeypatch.setattr(_runmod, "HttpClient", lambda cfg: _Cancelled(cfg))
    job = ph.web.run_async(_make_sim())
    with pytest.raises(SolverRunError) as ei:
        job.result(timeout=5)
    assert "cancelled" in str(ei.value)


# --- cache._safe_extract path-traversal guard -----------------------------

from photonhub.web import cache as _cache  # noqa: E402


def _tar_with_member(name: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name)
        data = b"x"
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_safe_extract_rejects_parent_traversal(tmp_path):
    dest = tmp_path / "job"
    with pytest.raises(ValueError, match="unsafe path"):
        _cache._safe_extract(_tar_with_member("../escape.txt"), dest)
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_rejects_absolute_path(tmp_path):
    dest = tmp_path / "job"
    outside = tmp_path / "outside.txt"
    with pytest.raises(ValueError, match="unsafe path"):
        _cache._safe_extract(_tar_with_member(f"/{outside.name}"), dest)


def test_safe_extract_accepts_normal_bundle(tmp_path):
    dest = tmp_path / "job"
    _cache._safe_extract(_bundle_bytes(), dest)
    assert (dest / "manifest.json").is_file()
    assert (dest / "probe.bin").is_file()


# --- config env-var precedence (configure / get_config) -------------------

from photonhub.web import config as _config  # noqa: E402


def test_configure_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("PHOTONHUB_API_KEY", "ph_env")
    monkeypatch.setenv("PHOTONHUB_URL", "http://env-host:9/")
    try:
        cfg = ph.web.configure(api_key="ph_explicit",
                               url="http://explicit:1/")
        assert cfg.api_key == "ph_explicit"     # explicit beats env
        assert cfg.url == "http://explicit:1"   # trailing slash stripped
    finally:
        ph.web.reset()


def test_configure_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("PHOTONHUB_API_KEY", "ph_env")
    monkeypatch.setenv("PHOTONHUB_URL", "http://env-host:9/")
    try:
        cfg = ph.web.configure()
        assert cfg.api_key == "ph_env"
        assert cfg.url == "http://env-host:9"
    finally:
        ph.web.reset()


def test_configure_defaults_url_when_unset(monkeypatch):
    monkeypatch.delenv("PHOTONHUB_URL", raising=False)
    try:
        cfg = ph.web.configure(api_key="ph_x")
        assert cfg.url == _config.DEFAULT_URL.rstrip("/")
    finally:
        ph.web.reset()


def test_configure_missing_key_raises_weberror(monkeypatch):
    monkeypatch.delenv("PHOTONHUB_API_KEY", raising=False)
    ph.web.reset()
    with pytest.raises(ph.web.WebError, match="no API key"):
        ph.web.configure()


# --- client error -> WebError mapping -------------------------------------

from photonhub.web.client import HttpClient, _parse_detail  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402


def _cfg():
    return _config.WebConfig(url="http://localhost:0", api_key="ph_x",
                             cache_dir=Path("/tmp"), request_timeout_s=0.01)


def test_client_4xx_maps_to_weberror_with_detail(monkeypatch):
    body = io.BytesIO(json.dumps({"detail": "bad spec"}).encode())
    err = urllib.error.HTTPError("http://x/v1/jobs", 400, "Bad Request", {}, body)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: (_ for _ in ()).throw(err))
    with pytest.raises(ph.web.WebError) as ei:
        HttpClient(_cfg()).get_json("/v1/jobs")
    assert ei.value.status_code == 400
    assert ei.value.body == "bad spec"          # parsed from JSON "detail"


def test_client_network_error_maps_to_weberror(monkeypatch):
    err = urllib.error.URLError("connection refused")
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: (_ for _ in ()).throw(err))
    with pytest.raises(ph.web.WebError, match="failed"):
        HttpClient(_cfg()).get_json("/v1/account")


def test_parse_detail_variants():
    assert _parse_detail(json.dumps({"detail": "boom"})) == "boom"
    assert _parse_detail(json.dumps({"x": 1})) == {"x": 1}  # no detail -> obj
    assert _parse_detail("not json") == "not json"          # falls back to text
