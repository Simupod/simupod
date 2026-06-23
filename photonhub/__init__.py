"""PhotonHub Python client — build simulation specs, run the solver, load results.

>>> import photonhub as ph
>>> sim = ph.Simulation(...)
>>> data = ph.run_local(sim)
>>> probe = data["probe"]  # xarray.DataArray
"""

from .components import (
    SCHEMA_VERSION,
    Background,
    Boundaries,
    Box,
    Cylinder,
    FieldDftMonitor,
    FieldSnapshotMonitor,
    FieldTimeMonitor,
    FluxMonitor,
    GaussianPulse,
    LorentzPole,
    Medium,
    ModeSource,
    PlaneWave,
    PointDipole,
    PolySlab,
    RunSpec,
    Simulation,
    Sphere,
    Structure,
    GradedAxisCoords,
    GradedGridSpec,
    UniformGridSpec,
    auto_grid,
)
from . import library
from .cost import CostEstimate, estimate_cost
from .data import SimulationData
from .gds import GdsLayer, import_gds, read_gds_cell_names
from .hdf5 import convert_to_hdf5
from .runners import (
    Batch,
    BatchData,
    Job,
    SolverRunError,
    find_solver,
    run_async,
    run_local,
)
from . import web
from . import inverse_design
from .inverse_design import (
    DesignRegion,
    GradientResult,
    ModePower,
    OptimizeResult,
    ParametricResult,
    PointIntensity,
    assemble_gradient,
    optimize,
    optimize_parametric,
    value_and_gradient,
)

__version__ = "0.0.1"

__all__ = [
    "Background",
    "Batch",
    "BatchData",
    "Boundaries",
    "Box",
    "CostEstimate",
    "Cylinder",
    "FieldDftMonitor",
    "FieldSnapshotMonitor",
    "FieldTimeMonitor",
    "FluxMonitor",
    "GaussianPulse",
    "GdsLayer",
    "import_gds",
    "Job",
    "LorentzPole",
    "Medium",
    "ModeSource",
    "library",
    "PlaneWave",
    "PointDipole",
    "PolySlab",
    "read_gds_cell_names",
    "RunSpec",
    "SCHEMA_VERSION",
    "Simulation",
    "web",
    "SimulationData",
    "SolverRunError",
    "Sphere",
    "Structure",
    "GradedAxisCoords",
    "GradedGridSpec",
    "UniformGridSpec",
    "auto_grid",
    "convert_to_hdf5",
    "estimate_cost",
    "find_solver",
    "run_async",
    "run_local",
    # inverse design (adjoint topology optimization)
    "inverse_design",
    "DesignRegion",
    "PointIntensity",
    "ModePower",
    "GradientResult",
    "OptimizeResult",
    "ParametricResult",
    "value_and_gradient",
    "assemble_gradient",
    "optimize",
    "optimize_parametric",
    "__version__",
]
