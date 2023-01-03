.. SystemTools documentation master file, created by
   sphinx-quickstart on Tue Jan  3 09:21:05 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SystemTools Documentation
=======================================
SystemTools provides python software for iterative tasks, analyzing and visualizing CxSystem2 simulations. The CxSystem2 is a high-level simulation tool housed at `CxSystem repository <https://github.com/VisualNeuroscience-UH/CxSystem2>`_  and `documented <https://cxsystem2.readthedocs.io/en/latest>`_ at readthedocs.org.

The iterative tasks include creation of anatomical and physiological csv files, running simulations, and analyzing the results. 

The main analysis tools include calculating the coherence, Granger causality, transfer entropy and normalized error between input and output matrices.

The main visualization tools include 
   * plotting single simulation results (spikes, monitored states, input-to-output coherence, etc.)
   * comparing multiple simulations (coherence, Granger causality, transfer entropy, normalized error, etc.) 
   * plotting 
      * analysis results of 1D or 2D array simulations.
      * any parametric data against each other 
      * categorical plots of parametric data

SystemTools is in active research use by the `Visual Neuroscience group <https://www.helsinki.fi/en/researchgroups/visual-neuroscience/>`_ at the University of Helsinki. Active use implies active evolution of the code.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
