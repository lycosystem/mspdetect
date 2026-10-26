Visualization
=============

In a Jupyter notebook:

.. code-block:: python

   from mspdetect.viz import adjust_plane, qc_figure, view_result, viewer

   # axial and coronal views with sliders for one patient of a batch run: first start,
   # best start (if different) and manual correction (if any), with structure contours
   view_result("patient01", "config.yaml")

   viewer(case, [plane_a, plane_b], labels=["a", "b"])  # any case and planes
   qc_figure(case, result.plane).savefig("qc.png")  # static figure

**Manual correction.** If the detection failed for a patient, the plane can be corrected by hand:

.. code-block:: python

   import mspdetect as msp

   result_dir = "results/patient01"
   adjust_plane(case, msp.load_result(result_dir), result_dir)

Sliders rotate the plane in the axial and coronal view (about the centre of the detection crop)
and shift it left-right, starting from the detected plane (first start). "Save correction"
requires a note and writes ``manual.json``; the detection results are not changed, and deleting
``manual.json`` restores the detected plane. Wherever a correction exists, ``mspdetect metrics``
uses it instead of the detected plane(s) (one row, ``plane_source = manual``, ``start = manual``).

See :doc:`api/visualization` for all options of :func:`~mspdetect.viz.view_result`,
:func:`~mspdetect.viz.viewer`, :func:`~mspdetect.viz.qc_figure` and
:func:`~mspdetect.viz.adjust_plane`.
