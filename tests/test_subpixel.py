"""Subpixel-smoothing flag (NUMERICS.md §16, schema 1.4.0): additive-optional
on the wire — omitted when unset (byte-identical documents), present when set,
and strictly round-tripping. The engine's smoothing math is covered by the
C++ test_subpixel.cpp; here we only pin the client/wire surface.
"""

import json

import simupod as ph

from conftest import make_sim


def test_default_is_tensor_on_for_nondispersive():
    # D2 (NUMERICS.md §16): a non-dispersive sim with no explicit subpixel choice
    # defaults to subpixel-ON with the diagonal-KFJ "tensor" average (Tidy3D's
    # subpixel-on posture), and the resolved value is serialized explicitly on
    # the wire (the engine field default is off).
    sim = make_sim()
    assert sim.subpixel is True
    assert sim.subpixel_method == "tensor"
    wire = sim.to_wire_dict()
    assert wire["subpixel"] is True
    assert wire["subpixel_method"] == "tensor"


def test_default_falls_back_to_off_for_dispersive():
    # D2 auto-fallback: a dispersive (Lorentz) scene defaults to subpixel OFF
    # (the subpixel × ADE late-time instability) when not set explicitly.
    pole = ph.LorentzPole(resonance_frequency_hz=2.0e14, delta_eps=1.5)
    box = ph.Structure(
        geometry=ph.Box(center_um=(0.1, 0.1, 0.1), size_um=(0.1, 0.1, 0.1)),
        medium=ph.Medium(permittivity=2.0, lorentz=pole),
    )
    sim = make_sim(structures=[box])
    assert sim.subpixel is False
    assert "subpixel" not in sim.to_wire_dict()


def test_explicit_subpixel_false_is_respected_when_unset_default_would_enable():
    # An explicit choice always wins over the D2 default.
    sim = make_sim(subpixel=False)
    assert sim.subpixel is False
    assert sim.to_wire_dict()["subpixel"] is False


def test_wire_ingestion_does_not_apply_the_construction_default():
    # Round-trip fidelity: a document that OMITS subpixel means the engine
    # default (off); from_wire_json must NOT flip it to the D2 construction
    # default, or older docs would stop round-tripping byte-identically.
    doc = json.loads(make_sim(subpixel=False).to_wire_json())
    doc.pop("subpixel", None)
    doc.pop("subpixel_method", None)
    back = ph.Simulation.from_wire_json(json.dumps(doc))
    assert back.subpixel is False
    assert "subpixel" not in back.to_wire_dict()


def test_set_true_appears_on_the_wire():
    sim = make_sim(subpixel=True)
    assert sim.subpixel is True
    wire = sim.to_wire_dict()
    assert wire["subpixel"] is True


def test_set_false_explicitly_still_round_trips():
    # Explicitly set (even to the default) -> in model_fields_set -> serialized,
    # so an explicit choice survives the wire.
    sim = make_sim(subpixel=False)
    wire = json.loads(sim.to_wire_json())
    assert wire["subpixel"] is False
    back = ph.Simulation.from_wire_json(sim.to_wire_json())
    assert back.subpixel is False


def test_from_wire_parses_subpixel():
    sim = make_sim(subpixel=True)
    back = ph.Simulation.from_wire_json(sim.to_wire_json())
    assert back.subpixel is True
    assert back.schema_version == "1.12.0-alpha.1"


def test_wire_ingestion_is_strict_about_bool():
    # from_wire_json uses strict typing to match the engine's nlohmann parse
    # (spec_io.cpp requires a JSON boolean), so a string/number in the wire is
    # rejected — even though lax construction would coerce "yes"/1 to True.
    import pytest
    from pydantic import ValidationError

    base = json.loads(make_sim().to_wire_json())
    for bad in ("true", 1):
        doc = dict(base, subpixel=bad)
        with pytest.raises(ValidationError):
            ph.Simulation.from_wire_json(json.dumps(doc))


# --- subpixel_method selector (schema 1.7.0, NUMERICS.md §16.5) ---------------


def test_subpixel_method_default_is_tensor_when_auto_enabled():
    # D2: when the construction default enables subpixel (non-dispersive), the
    # method it selects is the diagonal-KFJ "tensor" (not the isotropic volume).
    sim = make_sim()
    assert sim.subpixel_method == "tensor"
    assert sim.to_wire_dict()["subpixel_method"] == "tensor"


def test_subpixel_method_tensor_appears_on_the_wire():
    sim = make_sim(subpixel=True, subpixel_method="tensor")
    assert sim.subpixel_method == "tensor"
    wire = sim.to_wire_dict()
    assert wire["subpixel_method"] == "tensor"


def test_subpixel_method_round_trips_from_wire():
    sim = make_sim(subpixel=True, subpixel_method="tensor")
    back = ph.Simulation.from_wire_json(sim.to_wire_json())
    assert back.subpixel_method == "tensor"


def test_subpixel_method_rejects_unknown_value():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        make_sim(subpixel=True, subpixel_method="bogus")
