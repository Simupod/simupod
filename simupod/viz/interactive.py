"""``interactive_preview()`` — a Jupyter scrubber for a :class:`Simulation`:
slide the cut plane through the domain and watch the cross-section update live,
with cut-axis / grid / scene-vs-ε toggles. The "see it before you run it" moment,
interactive but fully local (no web app).

ipywidgets is the optional ``simupod[viz]`` extra and is imported LAZILY, so the
core SDK never depends on it. The frame rendering (:func:`render_slice`) is a
plain matplotlib call — pure and testable headless; ``interactive_preview`` just
wires it to widgets.
"""

_NO_IPYWIDGETS = (
    "interactive_preview requires ipywidgets (the interactive viz extra) and a "
    "Jupyter/IPython environment. Install it with:\n    pip install simupod[viz]"
)


def _require_ipywidgets():
    try:
        import ipywidgets as widgets  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without ipywidgets
        raise ImportError(_NO_IPYWIDGETS) from exc
    return widgets


def render_slice(sim, axis: str, value: float, *, eps: bool = True,
                 grid: bool = True, ax=None):
    """Render one preview frame: the meshed-ε heatmap (``eps=True``) or the
    analytic scene (``eps=False``) at ``axis=value``, optionally with the grid
    overlay. Returns the matplotlib ``Axes`` (pure — no widgets)."""
    if eps:
        return sim.plot_eps(**{axis: value}, ax=ax, grid=grid)
    return sim.plot(**{axis: value}, ax=ax, grid=grid)


def interactive_preview(sim, *, axis: str = "z", eps: bool = True,
                        grid: bool = True):
    """Build a Jupyter widget that scrubs the cut plane through ``sim``.

    A cut-axis dropdown, a position slider (auto-ranged to the realized domain),
    and ε / grid toggles drive a live cross-section. Returns the ipywidgets
    container, which displays inline as a cell's last expression. Requires the
    ``simupod[viz]`` extra and a notebook."""
    widgets = _require_ipywidgets()
    import matplotlib.pyplot as plt

    realized = sim._realized_um()

    def _axis_max(a: str) -> float:
        return float(realized["xyz".index(a)])

    m0 = _axis_max(axis)
    axis_dd = widgets.Dropdown(options=["x", "y", "z"], value=axis,
                               description="cut")
    pos = widgets.FloatSlider(min=0.0, max=m0, value=m0 / 2.0, step=m0 / 100.0,
                              description="µm", continuous_update=False,
                              readout_format=".3f")
    eps_cb = widgets.Checkbox(value=eps, description="meshed ε")
    grid_cb = widgets.Checkbox(value=grid, description="grid")
    out = widgets.Output()

    def render(*_):
        from IPython.display import display
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6.0, 4.5))
            render_slice(sim, axis_dd.value, pos.value,
                         eps=eps_cb.value, grid=grid_cb.value, ax=ax)
            display(fig)        # inline-backend friendly; no Agg "cannot show" warning
            plt.close(fig)

    def on_axis(_change):
        m = _axis_max(axis_dd.value)
        pos.max = m
        pos.step = m / 100.0
        pos.value = m / 2.0
        render()

    axis_dd.observe(on_axis, names="value")
    for w in (pos, eps_cb, grid_cb):
        w.observe(render, names="value")

    render()
    return widgets.VBox([widgets.HBox([axis_dd, pos]),
                         widgets.HBox([eps_cb, grid_cb]), out])
