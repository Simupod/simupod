"""Adiabatic absorber boundary (NUMERICS.md §21) — model, wire, helper."""

import json

import pytest
from pydantic import ValidationError

import photonhub as ph

from conftest import make_sim


def test_default_boundary_is_pml_on_all_faces():
    # Schema 1.12 flipped the user-facing default from periodic to PML.
    b = ph.Boundaries()
    assert (b.x, b.y, b.z) == ("pml", "pml", "pml")


def test_absorber_is_a_valid_boundary_kind_and_round_trips():
    sim = make_sim(boundaries=ph.Boundaries(x="periodic", y="periodic",
                                            z="absorber"))
    assert sim.boundaries.z == "absorber"
    wire = json.loads(sim.to_wire_json())
    assert wire["boundaries"]["z"] == "absorber"
    back = ph.Simulation.from_wire_json(sim.to_wire_json())
    assert back.boundaries.z == "absorber"


def test_absorber_knobs_default_and_round_trip():
    sim = make_sim()
    assert sim.absorber_num_layers == 40
    assert sim.absorber_m == 3.0
    # Additive-optional: unset knobs are omitted from the wire (1.12 forward-compat).
    wire = json.loads(sim.to_wire_json())
    assert "absorber_num_layers" not in wire
    assert "absorber_m" not in wire
    # Explicitly set -> emitted and round-tripped.
    sim2 = make_sim(absorber_num_layers=60, absorber_m=2.0)
    wire2 = json.loads(sim2.to_wire_json())
    assert wire2["absorber_num_layers"] == 60
    assert wire2["absorber_m"] == 2.0
    back = ph.Simulation.from_wire_json(sim2.to_wire_json())
    assert back.absorber_num_layers == 60
    assert back.absorber_m == 2.0


def test_absorber_knob_bounds():
    assert make_sim(absorber_num_layers=4).absorber_num_layers == 4
    with pytest.raises(ValidationError):
        make_sim(absorber_num_layers=3)
    assert make_sim(absorber_m=1.0).absorber_m == 1.0
    with pytest.raises(ValidationError):
        make_sim(absorber_m=0.5)


def test_with_absorber_sets_all_faces():
    sim = make_sim().with_absorber()
    assert (sim.boundaries.x, sim.boundaries.y, sim.boundaries.z) == (
        "absorber", "absorber", "absorber")
    assert sim.absorber_num_layers == 40
    sim2 = make_sim().with_absorber(num_layers=64)
    assert sim2.absorber_num_layers == 64
    assert sim2.boundaries.x == "absorber"


def test_absorber_allowed_on_symmetry_far_face():
    # A symmetry axis needs a non-periodic far face; absorber is now allowed
    # alongside pml/pec (one-sided, NUMERICS.md §20/§21).
    sim = make_sim(
        boundaries=ph.Boundaries(x="periodic", y="absorber", z="periodic"),
        symmetry=(0, -1, 0),
    )
    assert sim.boundaries.y == "absorber"
    assert sim.symmetry == (0, -1, 0)
