WakeyWakey Documentation
========================

WakeyWakey is a lightweight wake word detection package optimized for deployment from microcontrollers to Raspberry Pi.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api

Installation
============

Install WakeyWakey from PyPI:

.. code-block:: bash

   pip install wakeywakey

Quick Start
===========

.. code-block:: python

   import wakeywakey
   
   # Quick detection
   detector = wakeywakey.quick_detect("model.pth", threshold=0.7)
   detector.start_detection()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 