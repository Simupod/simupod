"""Schema generation: emit/check CLI and sanity of the produced JSON Schema."""

import json
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

import photonhub.schema as phs


def _run(args, env):
    return subprocess.run([sys.executable, "-m", "photonhub.schema", *args],
                          capture_output=True, text=True, env=env)


def test_emit_writes_valid_json_schema(tmp_path, subprocess_env):
    target = tmp_path / "simulation_v1.json"
    proc = _run(["emit", str(target)], subprocess_env)
    assert proc.returncode == 0, proc.stderr
    schema = json.loads(target.read_text())

    assert schema["title"] == "Simulation"
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    for key in ("schema_version", "size_um", "grid", "run", "background",
                "pml_num_layers", "structures", "boundaries", "sources",
                "monitors"):
        assert key in schema["properties"], key
    for model in ("UniformGridSpec", "RunSpec", "GaussianPulse", "PointDipole",
                  "PlaneWave", "FieldTimeMonitor", "FieldSnapshotMonitor",
                  "FieldDftMonitor", "FluxMonitor", "Background", "Boundaries",
                  "Medium", "Box", "Sphere", "Structure"):
        assert model in schema["$defs"], model
    # discriminated unions carry the discriminator on the wire 'type' key
    assert schema["properties"]["monitors"]["items"]["discriminator"]["propertyName"] == "type"
    assert schema["properties"]["sources"]["items"]["discriminator"]["propertyName"] == "type"
    geometry = schema["$defs"]["Structure"]["properties"]["geometry"]
    assert geometry["discriminator"]["propertyName"] == "type"
    assert set(schema["required"]) == {"size_um", "grid", "run", "sources"}


def test_check_mode_passes_then_fails_on_drift(tmp_path, subprocess_env):
    target = tmp_path / "simulation_v1.json"
    assert _run(["emit", str(target)], subprocess_env).returncode == 0
    assert _run(["check", str(target)], subprocess_env).returncode == 0

    target.write_text(target.read_text() + "\n// drift\n")
    proc = _run(["check", str(target)], subprocess_env)
    assert proc.returncode == 1
    assert "stale" in proc.stderr

    target.unlink()
    assert _run(["check", str(target)], subprocess_env).returncode == 1


def test_committed_schema_is_in_sync():
    committed = REPO_ROOT / "schemas" / "simulation_v1.json"
    if not committed.is_file():
        pytest.skip("not running from a repo checkout")
    assert committed.read_text(encoding="utf-8") == phs.schema_text(), (
        "schemas/simulation_v1.json is stale; run 'python -m photonhub.schema emit'")


def test_schema_is_as_strict_as_the_implementations():
    """The published schema must not be looser than what pydantic and the
    engine enforce: a third-party producer validating against it must not be
    able to emit specs both implementations reject."""
    schema = json.loads(phs.schema_text())

    # Source polarization: electric components only (both source kinds).
    for model in ("PointDipole", "PlaneWave"):
        pol = schema["$defs"][model]["properties"]["polarization"]
        assert pol["enum"] == ["Ex", "Ey", "Ez"]

    # Plane wave: per-axis tangential-polarization restriction is published
    # (NUMERICS.md section 13), so producers cannot emit a longitudinal
    # polarization the engine rejects.
    all_of = schema["$defs"]["PlaneWave"]["allOf"]
    by_axis = {b["if"]["properties"]["axis"]["const"]:
               b["then"]["properties"]["polarization"]["enum"] for b in all_of}
    assert by_axis == {"x": ["Ey", "Ez"], "y": ["Ex", "Ez"], "z": ["Ex", "Ey"]}

    # Structures (NUMERICS.md sections 9-10): ordered list of Structure;
    # medium bounds match the engine's validator.
    assert schema["properties"]["structures"]["items"] == {
        "$ref": "#/$defs/Structure"}
    medium = schema["$defs"]["Medium"]["properties"]
    assert medium["permittivity"]["minimum"] == 1.0
    assert medium["conductivity_s_per_m"]["minimum"] == 0.0
    for item in schema["$defs"]["Box"]["properties"]["size_um"]["prefixItems"]:
        assert item["exclusiveMinimum"] == 0
    assert schema["$defs"]["Sphere"]["properties"]["radius_um"][
        "exclusiveMinimum"] == 0

    # CPML (NUMERICS.md section 11) + adiabatic absorber (section 21): every
    # boundary axis accepts "pml" and "absorber"; both layer counts are bounded
    # below.
    for axis in ("x", "y", "z"):
        assert schema["$defs"]["Boundaries"]["properties"][axis]["enum"] == [
            "periodic", "pec", "pml", "absorber"]
    assert schema["properties"]["pml_num_layers"]["minimum"] == 4
    assert schema["properties"]["absorber_num_layers"]["minimum"] == 4

    # DFT monitors (NUMERICS.md section 12): non-empty positive freqs_hz.
    for model in ("FieldDftMonitor", "FluxMonitor"):
        freqs = schema["$defs"][model]["properties"]["freqs_hz"]
        assert freqs["minItems"] == 1
        assert freqs["items"]["exclusiveMinimum"] == 0
    for item in schema["$defs"]["FieldDftMonitor"]["properties"]["size_um"][
            "prefixItems"]:
        assert item["minimum"] == 0  # plane/line/point regions are legal

    # run: exactly one of run_time_s / n_steps, with null-as-absent
    # semantics matching both the pydantic runtime and the engine parser.
    one_of = schema["$defs"]["RunSpec"]["oneOf"]
    assert {"run_time_s"} in [set(b["required"]) for b in one_of]
    assert {"n_steps"} in [set(b["required"]) for b in one_of]

    # int32 bounds mirror the engine's as_int range checks.
    n_steps = schema["$defs"]["RunSpec"]["properties"]["n_steps"]
    assert {"maximum": 2**31 - 1, "minimum": 1, "type": "integer"} in n_steps["anyOf"]

    # Monitor names: filename rule published as an ECMA pattern (enforcement
    # is the AfterValidator; pydantic's Rust regex engine lacks lookahead).
    for model in ("FieldTimeMonitor", "FieldSnapshotMonitor",
                  "FieldDftMonitor", "FluxMonitor"):
        name = schema["$defs"][model]["properties"]["name"]
        assert name["minLength"] == 1
        assert name["pattern"] == "^(?!\\.{1,2}$)[^/\\\\]+$"
