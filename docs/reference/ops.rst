``rheon.polars.ops``
====================

.. automodule:: rheon.polars.ops
   :no-members:

Window operators
----------------

Polars expressions over a stream's clock that look back or ahead.

.. entries:: ops

Solving for an unknown
----------------------

The root of an equation of the row, solved row by row in the same run as
the window operators.

.. entries:: ops-solve

Seasonality
-----------

Sines and cosines of each row's phase in a fixed or a calendar period, as
features of a regression: plain Polars expressions with no state. And STL
and MSTL, statsmodels' decompositions of each group's last rows into trend,
seasons and remainder, read a few rows late, where they no longer move.

.. entries:: ops-seasonal

Basis expansions
----------------

Clamped B-splines, QR-projected natural cubic splines, piecewise-linear hinges
and polynomial interaction products: plain Polars expressions with no state
that stream directly into any model in a bank.

.. entries:: ops-basis
