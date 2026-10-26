Inputs, outputs and conventions
===============================

Input data
----------

One directory per patient with NIfTI files (``.nii`` or ``.nii.gz``, searched recursively):
the CT in Hounsfield units and, optionally, binary structure masks on the **same voxel grid**
as the CT (checked on loading). Files are identified by name without extension
(case-insensitive), e.g. ``CT.nii.gz``, ``GTVp.nii.gz``.

Only the CT is required. Some structure names have a special meaning:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Name
     - Used for
   * - ``GTVp`` (or any name set as ``crop.reference``)
     - selecting the slices used to fit the plane
   * - ``Body``
     - masking out the couch and, with ``lateral_extent_mm``, centring the shoulder crop
   * - ``Mandible``
     - extending the slices to the lower mandible if the reference ends below it
       (``include_mandible``)

See :doc:`api/data` for detailed information on data loading.


Outputs
-------

Per patient, in ``<output>/<patient>/``:

- ``msp.json``: plane, objective value, settings, package version, crop and a summary of each
  optimisation start.
- ``details.npz``: grid search (``grid_planes``, ``grid_values``, ``grid_deltas``) and, per start
  ``NN``, the plane at every iteration (``startNN_trajectory``) with the Huber and quadratic
  objective (``startNN_huber``, ``startNN_quadratic``). Planes are rows ``[nx, ny, nz, offset_mm]``.
  Use :func:`mspdetect.results.load_result` to read both files.
- ``manual.json`` (only after a manual correction, see :func:`mspdetect.viz.adjust_plane`): the
  corrected plane, the detected plane it replaces, a note, date, user and package version.

Tables in ``<output>/``: ``msp_results.csv`` (one row per patient; ``manual_correction`` marks
patients with a ``manual.json``) and ``failures.csv``.
``mspdetect metrics`` reads the planes from ``<output>/`` and writes ``metrics.csv`` and
``metrics_failures.csv`` to ``metrics.output`` if set (e.g. to compare several metric settings for
the same planes), otherwise to ``<output>/``.


Metrics
-------

For every structure in ``metrics.structures``: ``<name>_left_cc`` and ``<name>_right_cc``, the
volume on each side of the plane (counted by voxel centres). ``metrics.volume_region`` sets
which part of the structures is counted:

- ``whole`` (default): the whole structures. For long structures such as the spinal cord this
  includes parts far from where the plane was fitted, where neck flexion or rotation moves
  them off the plane.
- ``slab``: only the slices used to detect the plane, i.e. the cranial-caudal range of
  ``detection.crop`` (``fixed_size`` or ``margins`` around the reference, mandible rule,
  ``slice_range``), with the full left-right and anterior-posterior extent. The columns
  ``slab_start`` and ``slab_stop`` give the slice range ``[start, stop)`` (canonical orientation,
  counted from inferior). In Python: ``compute_metrics(..., slices=result.crop_slices)``.

The target metrics always use the whole target structure. For the ``target`` structure
(default ``GTVp``), prefixed with its name:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Definition
   * - ``side``
     - side with the larger volume (``left``, ``right``, ``central`` if equal)
   * - ``crosses_midline``
     - the structure has voxels on both sides
   * - ``distance_msp_mm``
     - if crossing: how far the structure extends into the side with the smaller volume (> 0);
       otherwise minus the distance from the plane to the closest voxel (< 0)
   * - ``lateral_extent_mm``
     - extent perpendicular to the plane
   * - ``left_max_mm``, ``right_max_mm``
     - largest distance from the plane on each side (0 if none)
   * - ``max_diameter_mm``
     - largest distance between two voxels of the structure

With ``n_starts > 1``, ``metrics.plane`` selects the plane: ``first_start`` (default) uses the
optimisation started from the grid point with the lowest initial value, ``best`` the lowest
final quadratic loss over all starts, and ``all_starts`` computes the metrics for every start.
``metrics.csv`` has one row per patient and evaluated start ("long" format), with the columns
``start`` (index of the start), ``is_first_start`` and ``is_best``. To compare the starts:

.. code-block:: python

   df = pd.read_csv("metrics.csv")
   df.groupby("case_id")["GTVp_distance_msp_mm"].agg(["min", "max", "std"])  # spread per patient
   df[df.is_first_start]  # first start only

In Python: ``result.starts[i].final`` is the plane of start ``i``;
``result.select_plane("first_start")`` or ``result.plane`` (= ``"best"``).


Conventions
-----------

- All volumes are reoriented to RAS+ on loading. Planes are stored in **world coordinates**
  (scanner RAS+ in mm, as given by the NIfTI affine): the set of points ``w`` with
  ``normal · w = offset``.
- The normal points to the patient's **right**. Signed distances are positive on the patient's
  right and negative on the left, independent of how the images were stored.
- Volumes are in cc (= cm³ = ml), distances in mm and angles in degrees. ``azimuth_deg`` and
  ``polar_deg`` are the spherical angles of the normal; a plane aligned with the scanner's
  sagittal plane has azimuth 0° and polar angle 90°.

See :mod:`mspdetect.plane` for the plane representation.
