``rheon.polars.stream``
=======================

.. automodule:: rheon.polars.stream
   :no-members:

.. autodata:: rheon.polars.stream.ROLE

Columns made before a bank
--------------------------

Windows over the clock as columns, targets whose values arrive late, series that tick at their own times brought onto one clock, and bars that close when a running quantity reaches a threshold.

.. entries:: stream

A run's states as one file
--------------------------

The states of a run's stages, a bank's among them, saved as one file written in full or not at all, and each stage's state read back.

.. entries:: stream-checkpoint
