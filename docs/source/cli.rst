MSPdetect CLI
==================

The ``mspdetect`` command processes many cases at once. All settings are read from
a YAML configuration file; a typical run is:

.. code-block:: bash

   mspdetect check-config config.yaml   # validate and print all settings (incl. defaults)
   mspdetect run config.yaml            # detect the plane of all cases
   mspdetect metrics config.yaml        # compute metrics from the detected planes

``metrics`` reads the planes saved by ``run``, so it can be repeated with other
metric settings without detecting the planes again.

Behaviour
---------

- Cases with an existing result computed with the same detection settings are
  skipped; set ``overwrite: true`` to recompute them.
- A failing case does not stop the run; it is listed in ``failures.csv``
  (``metrics_failures.csv`` for ``metrics``). The exit code is 1 if any case failed.
- ``-v`` shows debug messages.

Configuration
-------------

Example with all options (see :doc:`configuration` for their meaning):

.. literalinclude:: ../../examples/config.example.yaml
   :language: yaml

Command help
------------

.. program-output:: mspdetect --help
