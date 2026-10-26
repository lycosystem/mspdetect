"""Visual checks and manual correction of planes (requires ``pip install mspdetect[viz]``).

- :func:`qc_figure`: static figure with axial and coronal slices, e.g. to save as PNG.
- :func:`viewer`: axial and coronal views with sliders (Jupyter).
- :func:`view_result`: the viewer for one patient of a batch run, loaded via its config.
- :func:`adjust_plane`: correct a plane by hand and save the correction with a note.

Images are shown in radiological convention: the patient's right on the viewer's left.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mspdetect.io import Case
from mspdetect.plane import Plane

# Grey-level window of the CT (HU); a lower upper limit makes the image brighter
WINDOW_HU = (-250, 1150)
# Muted colours for lines on the CT; planes and structures use separate colours.
# Every plane gets its own colour (they repeat only beyond 12 planes).
PLANE_COLORS = (
    "#d62728",  # red
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#bcbd22",  # olive
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#8c564b",  # brown
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#c5b0d5",  # lavender
    "#f7b6d2",  # light pink
)
STRUCTURE_COLORS = ("#2ca02c", "#3f9abe", "#f0e442", "#7f7f7f")  # green, light green, yellow, grey
# Legend text size: a size in points or "small", "medium", "large", ...
LEGEND_FONTSIZE = "medium"
# Voxels shown around the Body
ZOOM_MARGIN = 5
# Largest manual rotation (degrees) and shift (mm) offered by adjust_plane
MAX_ROTATION_DEG = 15.0
MAX_SHIFT_MM = 20.0


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Plotting requires matplotlib: pip install 'mspdetect[viz]'") from e
    return plt


def _widgets():
    try:
        import ipywidgets
        from IPython.display import display
    except ImportError as e:
        raise ImportError(
            "Interactive views require ipywidgets: pip install 'mspdetect[viz]'"
        ) from e
    return ipywidgets, display


def _planes_and_labels(planes, labels) -> tuple[list[Plane], list[str]]:
    planes = [planes] if isinstance(planes, Plane) else list(planes)
    labels = list(labels) if labels is not None else [f"plane {i}" for i in range(len(planes))]
    if len(labels) != len(planes):
        raise ValueError(f"{len(planes)} planes but {len(labels)} labels")
    return planes, labels


def _plane_line(case: Case, plane: Plane, axis: int, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Voxel coordinates ``(x, other)`` of the plane intersected with a slice.

    ``axis`` is the fixed axis (2: axial slice, 1: coronal slice); returns positions
    along array axis 0 (left-right) and the remaining in-plane axis.
    """
    other = 1 if axis == 2 else 2
    m = case.affine[:3, :3].T @ plane.normal
    c = case.affine[:3, 3] @ plane.normal - plane.offset
    along = np.arange(case.image.shape[other], dtype=float)
    x = -(m[other] * along + m[axis] * index + c) / m[0]
    return x, along


def _extent(
    case: Case,
    zoom_slices: tuple[int, int] | None = None,
    margin: int = ZOOM_MARGIN,
) -> list[tuple[int, int]]:
    """Voxel range per axis to display: the Body (plus a margin) or the whole image.

    With ``zoom_slices = (start, stop)``, the left-right and anterior-posterior range
    is that of the Body within these slices only (e.g. excluding the shoulders).
    """
    full = [(0, n - 1) for n in case.image.shape]
    body = case.masks.get("Body")
    if body is None or not body.any():
        return full
    ranges = []
    for axis in range(3):
        region = body
        if zoom_slices is not None and axis != 2:
            region = body[:, :, max(0, zoom_slices[0]) : zoom_slices[1]]
        other = tuple(a for a in range(3) if a != axis)
        idx = np.flatnonzero(region.any(axis=other))
        if idx.size == 0:
            ranges.append(full[axis])
            continue
        n = case.image.shape[axis]
        ranges.append((max(0, idx[0] - margin), min(n - 1, idx[-1] + margin)))
    return ranges


def _show_slice(
    ax,
    case: Case,
    planes: Sequence[Plane],
    labels: Sequence[str],
    axis: int,
    index: int,
    structures: Sequence[str] = (),
    extent: list[tuple[int, int]] | None = None,
    marker: int | None = None,
    styles: Sequence[str] | None = None,
) -> None:
    """Draw an axial (``axis=2``) or coronal (``axis=1``) slice with planes and contours.

    Axes are in mm from the image corner (voxel index times voxel size).
    ``marker`` (voxel index) draws a dashed horizontal line, e.g. the position of the
    other view.
    """
    other = 1 if axis == 2 else 2
    sx, so = case.voxel_size[0], case.voxel_size[other]
    extent = extent or [(0, n - 1) for n in case.image.shape]
    image = np.take(case.image, index, axis=axis)
    ax.imshow(
        image.T,
        cmap="gray",
        origin="lower",
        vmin=WINDOW_HU[0],
        vmax=WINDOW_HU[1],
        extent=[-0.5 * sx, (image.shape[0] - 0.5) * sx, -0.5 * so, (image.shape[1] - 0.5) * so],
    )
    x_mm = np.arange(image.shape[0]) * sx
    y_mm = np.arange(image.shape[1]) * so
    for i, name in enumerate(structures):
        mask = np.take(case.masks[name], index, axis=axis)
        if mask.any():
            color = STRUCTURE_COLORS[i % len(STRUCTURE_COLORS)]
            ax.contour(
                x_mm, y_mm, mask.T.astype(float), levels=[0.5], colors=[color], linewidths=1.5
            )
            ax.plot([], [], color=color, lw=1.5, label=name)  # legend entry
    styles = styles or ["-"] * len(planes)
    for i, (plane, label) in enumerate(zip(planes, labels, strict=True)):
        x, along = _plane_line(case, plane, axis, index)
        color = PLANE_COLORS[i % len(PLANE_COLORS)]
        ax.plot(x * sx, along * so, styles[i], color=color, lw=1.5, label=label)
    if marker is not None:
        ax.axhline(marker * so, color="yellow", lw=1, ls="--")
    ax.set_xlim(extent[0][1] * sx, extent[0][0] * sx)  # patient right on the viewer's left
    ax.set_ylim(extent[other][0] * so, extent[other][1] * so)
    ax.set_aspect("equal")
    position = index * case.voxel_size[axis]
    if axis == 2:
        ax.set_title(f"Axial z={index} (z={position:.1f} mm)")
        ax.set_ylabel("Y (coronal) mm")
    else:
        ax.set_title(f"Coronal y={index} (y={position:.1f} mm)")
        ax.set_ylabel("Z (axial) mm")
    ax.set_xlabel("X (sagittal) mm")
    ax.text(0.02, 0.5, "R", transform=ax.transAxes, color="w", va="center")
    ax.text(0.95, 0.5, "L", transform=ax.transAxes, color="w", va="center")


def _legend(ax, loc: str) -> None:
    """Legend on the image, with a light background."""
    style = {"fontsize": LEGEND_FONTSIZE, "framealpha": 0.8}
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc=loc, **style)


def _default_slices(case: Case) -> tuple[int, int]:
    """Initial axial and coronal slice: through the GTVp if loaded, else the centre."""
    if "GTVp" in case.masks and case.masks["GTVp"].any():
        idx = np.argwhere(case.masks["GTVp"])
        centre = idx.mean(axis=0).round().astype(int)
        return int(centre[2]), int(centre[1])
    return case.image.shape[2] // 2, case.image.shape[1] // 2


def _draw_views(case, planes, labels, structures, z, y, styles=None, title=None, zoom_slices=None):
    plt = _pyplot()
    extent = _extent(case, zoom_slices)
    fig, (ax_axial, ax_coronal) = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    common = {"structures": structures, "extent": extent, "styles": styles}
    _show_slice(ax_axial, case, planes, labels, axis=2, index=z, marker=y, **common)
    _show_slice(ax_coronal, case, planes, labels, axis=1, index=y, marker=z, **common)
    _legend(ax_axial, "upper right")
    fig.suptitle(title or case.case_id)
    plt.show()


def qc_figure(
    case: Case,
    planes: Plane | Sequence[Plane],
    slices: Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
    structures: Sequence[str] = (),
):
    """Axial slices and one coronal slice of a case with the plane(s) drawn in.

    Parameters
    ----------
    case
        The case.
    planes
        One plane (e.g. ``result.plane``) or several to compare.
    slices
        Axial slice indices; default: three slices through the ``GTVp`` if loaded,
        otherwise through the scan.
    labels
        Legend labels for the planes.
    structures
        Structures whose contours are drawn.

    Returns
    -------
    matplotlib.figure.Figure

    """
    plt = _pyplot()
    planes, labels = _planes_and_labels(planes, labels)
    if slices is None:
        if "GTVp" in case.masks and case.masks["GTVp"].any():
            z = np.flatnonzero(case.masks["GTVp"].any(axis=(0, 1)))
            slices = np.linspace(z[0], z[-1], 3).round().astype(int)
        else:
            n = case.image.shape[2]
            slices = [n // 4, n // 2, 3 * n // 4]

    extent = _extent(case, (int(min(slices)), int(max(slices)) + 1))
    fig, axes = plt.subplots(
        1, len(slices) + 1, figsize=(4 * (len(slices) + 1), 4.5), layout="constrained"
    )
    for ax, z in zip(axes, slices, strict=False):
        _show_slice(ax, case, planes, labels, 2, int(z), structures, extent)
    y = _default_slices(case)[1]
    _show_slice(axes[-1], case, planes, labels, 1, y, structures, extent)
    if len(planes) > 1 or structures:
        _legend(axes[0], "lower left")
    fig.suptitle(case.case_id)
    return fig


def viewer(
    case: Case,
    planes: Plane | Sequence[Plane],
    labels: Sequence[str] | None = None,
    structures: Sequence[str] | None = None,
    zoom_slices: tuple[int, int] | None = None,
):
    """Axial and coronal views with sliders to scroll through the case (Jupyter).

    Parameters
    ----------
    case
        The case.
    planes
        One plane or several to compare, drawn as lines.
    labels
        Legend labels for the planes.
    structures
        Structures whose contours are drawn; default: all loaded masks except Body.
    zoom_slices
        Slices ``(start, stop)`` whose Body extent sets the zoom of both views, e.g.
        ``result.crop_slices``; default: the Body over the whole scan.

    """
    widgets, display = _widgets()
    planes, labels = _planes_and_labels(planes, labels)
    if structures is None:
        structures = [name for name in case.masks if name != "Body"]
    z0, y0 = _default_slices(case)
    z = widgets.IntSlider(z0, 0, case.image.shape[2] - 1, description="axial")
    y = widgets.IntSlider(y0, 0, case.image.shape[1] - 1, description="coronal")

    def draw(z, y):
        _draw_views(case, planes, labels, structures, z, y, zoom_slices=zoom_slices)

    out = widgets.interactive_output(draw, {"z": z, "y": y})
    box = widgets.VBox([widgets.HBox([z, y]), out])
    display(box)
    return box


def view_result(case_name: str, config, structures: Sequence[str] | None = None):
    """:func:`viewer` for one patient of a batch run.

    Loads the CT, the structures of the metrics and the crop reference, and the
    saved planes: the first start, the best start (if different) and a manual
    correction (if any).

    Parameters
    ----------
    case_name
        Patient directory name, as in the result tables.
    config
        The :class:`~mspdetect.config.BatchConfig` or the path to its YAML file.
    structures
        Structures to load and draw; default: metrics structures, target and crop
        reference (as far as they are listed in ``data.structures``).

    """
    from mspdetect.config import BatchConfig
    from mspdetect.io import load_case
    from mspdetect.results import load_manual_plane, load_result

    if not isinstance(config, BatchConfig):
        config = BatchConfig.from_yaml(config)
    if structures is None:
        names = [*config.metrics.structures, config.metrics.target, config.detection.crop.reference]
        structures = list(dict.fromkeys(n for n in names if n is not None))
    files = {n: config.data.structures[n] for n in structures if n in config.data.structures}
    case = load_case(
        config.data.root / case_name,
        image=config.data.image,
        structures={**files, **_body_file(config)},
        case_id=case_name,
    )
    result_dir = config.output / case_name
    result = load_result(result_dir)
    planes, labels = [result.first_start_plane], ["first start"]
    if result.best_start != 0:
        planes.append(result.plane)
        labels.append(f"best (start {result.best_start})")
    manual = load_manual_plane(result_dir)
    if manual is not None:
        planes.append(manual.plane)
        labels.append("manual")
    return viewer(
        case,
        planes,
        labels,
        [n for n in structures if n in case.masks],
        zoom_slices=result.crop_slices,
    )


def _body_file(config) -> dict:
    """The Body structure (used to zoom the views), if configured."""
    return {"Body": config.data.structures["Body"]} if "Body" in config.data.structures else {}


def _rotation(axis: int, angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    i, j = [k for k in range(3) if k != axis]
    rot = np.eye(3)
    rot[i, i], rot[i, j], rot[j, i], rot[j, j] = c, -s, s, c
    return rot


def adjusted_plane(
    plane: Plane,
    pivot,
    axial_deg: float = 0.0,
    coronal_deg: float = 0.0,
    shift_mm: float = 0.0,
) -> Plane:
    """Plane rotated about a pivot point and shifted along its normal.

    Parameters
    ----------
    plane
        Plane to adjust.
    pivot
        World point (mm) about which the plane is rotated; projected onto the plane.
    axial_deg
        Rotation about the superior axis (seen as a rotation in axial slices).
    coronal_deg
        Rotation about the anterior axis (seen as a rotation in coronal slices).
    shift_mm
        Shift along the normal, positive towards the patient's right.

    """
    pivot = np.asarray(pivot, dtype=float)
    on_plane = pivot - plane.signed_distance(pivot) * plane.normal
    normal = _rotation(2, axial_deg) @ _rotation(1, coronal_deg) @ plane.normal
    normal = normal * np.sign(normal[0])  # keep pointing to the patient's right
    return Plane(normal, normal @ on_plane + shift_mm)


def adjust_plane(
    case: Case, result, result_dir: str | Path, structures: Sequence[str] | None = None
):
    """Correct the plane of a case by hand and save the correction (Jupyter).

    Sliders rotate the plane in the axial and coronal view (about the centre of the
    crop used for detection) and shift it along its normal, starting from the plane
    of the first start (or from a previously saved correction). "Save correction"
    writes ``manual.json`` to ``result_dir`` together with the required note, the date
    and the user; the detection results are not changed. :func:`mspdetect.batch.run_metrics`
    uses the correction instead of the detected plane.

    Parameters
    ----------
    case
        The case, e.g. loaded with :func:`~mspdetect.io.load_case`.
    result
        Its detection result, e.g. ``load_result(result_dir)``.
    result_dir
        Directory of the result, where ``manual.json`` is written.
    structures
        Structures whose contours are drawn; default: all loaded masks except Body.

    """
    from mspdetect.results import load_manual_plane, save_manual_plane

    widgets, display = _widgets()
    if structures is None:
        structures = [name for name in case.masks if name != "Body"]
    base = result.first_start_plane
    centre = (np.array(result.crop.start) + np.array(result.crop.stop) - 1) / 2
    pivot = case.voxel_to_world(centre[None])[0]

    previous = load_manual_plane(result_dir)
    start = previous.adjustment if previous is not None else {}

    def slider(name, value, limit, step, description):
        return widgets.FloatSlider(
            value=float(start.get(name, value)),
            min=-limit,
            max=limit,
            step=step,
            description=description,
            continuous_update=False,
            style={"description_width": "initial"},
        )

    axial = slider("axial_deg", 0.0, MAX_ROTATION_DEG, 0.1, "rotation axial (°)")
    coronal = slider("coronal_deg", 0.0, MAX_ROTATION_DEG, 0.1, "rotation coronal (°)")
    shift = slider("shift_mm", 0.0, MAX_SHIFT_MM, 0.25, "shift (mm, + = right)")
    z0, y0 = _default_slices(case)
    z = widgets.IntSlider(z0, 0, case.image.shape[2] - 1, description="axial")
    y = widgets.IntSlider(y0, 0, case.image.shape[1] - 1, description="coronal")
    note = widgets.Text(
        value=previous.note if previous is not None else "",
        placeholder="Reason for the correction (required)",
        description="note",
        layout=widgets.Layout(width="60%"),
    )
    save = widgets.Button(description="Save correction", button_style="warning")
    status = widgets.HTML(
        f"Existing correction from {previous.created} by {previous.user}" if previous else ""
    )

    def current() -> Plane:
        return adjusted_plane(base, pivot, axial.value, coronal.value, shift.value)

    def draw(axial_deg, coronal_deg, shift_mm, z, y):
        corrected = adjusted_plane(base, pivot, axial_deg, coronal_deg, shift_mm)
        _draw_views(
            case,
            [base, corrected],
            ["detected (first start)", "corrected"],
            structures,
            z,
            y,
            styles=["--", "-"],
            zoom_slices=result.crop_slices,
        )

    def on_save(_):
        try:
            adjustment = {
                "axial_deg": axial.value,
                "coronal_deg": coronal.value,
                "shift_mm": shift.value,
            }
            path = save_manual_plane(result_dir, current(), note.value, base, adjustment)
        except (ValueError, OSError) as e:
            status.value = f"<b style='color:red'>Not saved: {e}</b>"
        else:
            status.value = f"Saved to {path}"

    save.on_click(on_save)
    out = widgets.interactive_output(
        draw,
        {"axial_deg": axial, "coronal_deg": coronal, "shift_mm": shift, "z": z, "y": y},
    )
    box = widgets.VBox(
        [
            widgets.HBox([axial, coronal, shift]),
            widgets.HBox([z, y]),
            out,
            widgets.HBox([note, save]),
            status,
        ],
    )
    display(box)
    return box
