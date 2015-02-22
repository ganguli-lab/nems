.. nems documentation master file, created by
   sphinx-quickstart on Tue Feb  3 19:41:44 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

======================
Neural Encoding Models
======================

Neural encoding models are models that predict the neural response to an external stimulus. This package provides utilities
for training (or fitting), testing, and visualizing neural encoding models.

The starting point is the ``NeuralEncodingModel`` class and its subclasses. For example, the ``LNLN`` subclass takes in
a set of data (stimuli and neural responses) and initializes a model object that can be used to fit parameters of a two-layer
linear-nonlinear cascade model (LN-LN) to the given data, and test those parameters on a (randomly held-out) subset of the data.

.. note:: This package is under active development. Please refer to these pages or the `nems`_ Github repository for the latest updates.

.. _nems: https://github.com/ganguli-lab/nems/


Contents:

.. toctree::
   :maxdepth: 1

   install
   quickstart
   api
   changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

