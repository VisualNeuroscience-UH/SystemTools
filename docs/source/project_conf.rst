.. _project_conf:


Reference for the :mod:`project_conf_module` script file
========================================================

Use keyword substring "file" in filenames, and "folder" in folder names to assert that they are turned into pathlib objects. Path structure is assumed to be root_path/project/experiment/output_folder.

Some lines are marked: :code:`### Housekeeping ###. Do not comment out the next two lines.` . If you comment such lines, the `conf` will not function properly.

.. _abbreviations:

Abbreviations
-------------

.. csv-table::
   :header: Abbreviation, Description
   :widths: 15, 75
   :delim: |

   ana  | analysis
   ci   | current injection
   col  | column
   coll | collated, collected
   conn | connections
   full | full absolute path
   start| startpoint
   param| parameter


.. _module-level-attributes:

Module-level Attributes
-----------------------

These attributes are set in the :mod:`project_conf_module` script file. They provide the context for running the methods below the :code:´if __name__ == "__main__":´ block. 

.. attribute:: root_path

   The root path in different operating systems. (str)

.. attribute:: project

   The name of the project. (str)

.. attribute:: experiment

   The name of the current experiment. (str)

.. attribute:: path

   Path structure is assumed to be root_path/project/experiment/output_folder. (str)

.. attribute:: input_folder

   The name of the input folder. (str)

.. attribute:: matlab_workspace_file

   The name of the Matlab workspace file. (str)

.. attribute:: conn_skeleton_file_in

   The name of the connection skeleton file in. This should house correct neuron groups and unit quantities (str)

.. attribute:: conn_file_out

   The name of the connection file out. (str)

.. attribute:: input_filename

   The name of the input file. (str)

.. attribute:: startpoint_csv_folder

   The name of the startpoint CSV folder. (str) For use with jupyter notebook simulations. 

.. attribute:: startpoint

   The startpoint. (str) In FCN22 project the valid startpoints include Comrade, Bacon and HiFi.

.. attribute:: parameter

   The parameter(str). In FCN22 project the valid parameters include C, gL, VT, EL and delay

.. attribute:: output_folder

   The name of the output folder. (str)

.. attribute:: t_idx_start

   The index of the start time sample. (int) The start index cuts the beginning of time samples. Use 0, None or integer btw [0 N_time_samples]. 

.. attribute:: t_idx_end

   The index of the end time sample. (int) The end index of time samples. If negative, will count from the end.

.. attribute:: NG_output

   The name of the output neuron group. (str) In FCN22 project the default value is "NG3_L4_SS_L4" 

.. attribute:: to_spa_dict: 
    
    Data context for multiple analyzes and visualizations (dict). This dictionary has the following keys:
        - startpoints: List of strings representing the startpoints of the analyzes.
        - parameters: List of strings representing the parameters of the analyzes.
        - analyzes: List of strings representing the types of analyzes to be performed. Available analyzes include 'Coherence', 'Granger Causality', 'GC as TE', 'Transfer Entropy', 'Simulation Error', 'Excitatory Firing Rate', 'Inhibitory Firing Rate', 'Euclidean Distance'
    
    If you give to_spa_dict = None, only single files will be handled.

:example to_spa_dict:

.. code-block:: python

    {
        "startpoints": ["Comrad", "Bacon", "HiFi"],
        "parameters": ["C", "gL", "VT", "EL", "delay"],
        "analyzes": [
            "Coherence",
            "Granger Causality",
            "Transfer Entropy",
            "Simulation Error",
            "Excitatory Firing Rate",
            "Inhibitory Firing Rate",
        ],
    }

.. attribute:: profile

   Flag for profiling executed methods (bool)

|

Methods below the :code:´if __name__ == “__main__”:´ block.
------------------------------------------------------------


Project files & Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: construction.connection_translation_module.ConnectionTranslator.replace_conn

.. autofunction:: construction.connection_translation_module.ConnectionTranslator.create_control_conn

.. autofunction:: construction.connection_translation_module.ConnectionTranslator.create_current_injection

.. autofunction:: viz.viz_module.Viz.show_conn


Compile cluster run metadata parts and transfer data to project folder. 
metapath = "[path to]/cluster_metadata_[timestamp].pkl"

.. autofunction:: project.project_utilities_module.ProjectUtilities.cluster_metadata_compiler_and_data_transfer

.. autofunction:: project.project_utilities_module.ProjectUtilities.multiple_cluster_metadata_compiler_and_data_transfer

.. autofunction:: analysis.analysis_module.Analysis.optimal_value_analysis

.. autofunction:: viz.viz_module.Viz.show_optimal_value_analysis

.. autofunction:: viz.viz_module.Viz.show_iter_optimal_value_analysis

.. autofunction:: analysis.analysis_module.Analysis.describe_optimal_values
   
Analysis & Viz, single files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Show input and data. In the methods below, you can set :code:`results_filename=file_to_display` to avoid writing the the same filename or None to multiple places. If you then set :code:`file_to_display = None`, the functions selects the most recent data file in the output_folder.

file_to_display has the format "[full path to file ] [prefix]_results_[timestamp].gz"

.. autofunction:: viz.viz_module.Viz.show_spikes

.. autofunction:: viz.viz_module.Viz.show_readout_on_input

.. autofunction:: viz.viz_module.Viz.show_analog_results


Analysis & Viz, array runs, single startpoint, single analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _attr_ref1:

.. attribute:: save_figure_with_arrayidentifier

If specified, the displayed array analysis figures are saved as arrayIdentifier_analysis_identifier.svg in the specified folder. The value should be a string containing the array identifier.

.. _attr_ref2:

.. attribute:: save_figure_to_folder

If specified, the displayed array analysis figures are saved in the specified folder. The value should be a string containing the name of the folder.

.. autofunction:: analysis.analysis_module.Analysis.analyze_IxO_array

.. autofunction:: analysis.analysis_module.Analysis.analyze_arrayrun

.. _meth_ref1:

.. autofunction:: viz.viz_module.Viz.show_analyzed_arrayrun


Viz, single array runs, multiple analyzes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: viz.viz_module.Viz.system_polar_bar


Analyze & Viz, array runs, multiple iterations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`save_figure_with_arrayidentifier <attr_ref1>`, :ref:`save_figure_to_folder <attr_ref2>` and :ref:`show_analyzed_arrayrun <meth_ref1>` above for reference.

.. autofunction:: viz.viz_module.Viz.show_xy_plot

.. autofunction:: viz.viz_module.Viz.show_IxO_conf_mtx


Analyze & Viz, array runs, multiple iterations, multiple paths
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: viz.viz_module.Viz.show_catplot


Automated csv generation, simulation and analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. attribute:: create_csvs

A flag indicating whether to create csv files. (0 or 1)

.. attribute:: run_simulation

A flag indicating whether to run simulations. (0 or 1)

.. attribute:: run_analysis

A flag indicating whether to run analysis. (0 or 1)

.. attribute:: run_optimal_analysis

A flag indicating whether to run optimal analysis. (0 or 1)

.. attribute:: parallel_analysis

A flag indicating whether to run the analysis in parallel. (bool)

.. attribute:: analysis_type

The type of analysis to run. (str)
    # Two parallel analyzes exist (None for non-parallel):
    # 'full_IxO' : create .gz with all inputs vs all outputs, then csv:s.
    # Valid IxO analyzes include Coherence, TransferEntropy, GrCaus, NormError
    # 'matched_IO' : create csv:s from matched input to output pairs. All analyzes are valid.

.. attribute:: create_log_file

A flag indicating whether to create a log file. (bool)


.. attribute:: optimal_value_delays

A list of delays for optimal value analysis. (list)

Example: :code:`optimal_value_delays = [0, 1000, 10]` creates optimal value analysis for delays 0, 10, 20, ..., 1000.


.. attribute:: n_iters

The number of iterations to run. (int)

.. attribute:: iter_start_idx

The starting index for iterations. (int)

This is useful e.g. if you need to rerun a failed analysis.

.. attribute:: pad_zeros

The number of padded zeros in iterated folder names. (int)
Example: :code:`pad_zeros = 2`

.. attribute:: iter_idx_list

A list of indices for iterations. (list or None)

Build this manually if you want to rerun only a subset of iterations.

.. attribute:: input_fname_prefix

The prefix of the input file name. (str)
Example: :code:`input_fname_prefix = "noise_210916_SingleSpike"`

.. attribute:: input_fname_ci_suffix

The suffix of the input file name for ci. (str)

Example: :code:`input_fname_ci_suffix = "_ci.mat"`

.. attribute:: time_ids

A dictionary mapping names to time IDs. (dict)

Example:

.. code-block::  

   time_ids = {"Comrad": "221122", "Bacon": "221122", "HiFi": "221122"}

.. attribute:: anat_update_dict

A dictionary containing updates for the anatomy configuration. (dict)

Example:
.. code-block::  

   anat_update_dict = {
      "workspace_path": f"{path}",
      "import_connections_from": f"../in/connections_SingleSpike_ci.gz",
      "run_in_cluster": 0, # Available only for pre-configured clusters
      "cluster_job_file_path": "../csc_puhti_sv.job",
   }

.. attribute:: in_folder_full

The path to the input folder. (pathlib object, str)

.. attribute:: phys_update_dict

A dictionary containing updates for the physiology configuration. (dict)

Example for dummy dict for dev and debugging:

.. code-block::  

   phys_update_dict = {
      "current_injection": [
         ["base_ci_path", nan, f"r'{in_folder_full}'", "", "variable"],
         [
               "ci_filename",
               nan,
               f"'{input_fname_prefix}{input_fname_ci_suffix}'",
               "",
               "variable",
         ],
      ],
      "C": [
         ["L4_CI_BC", "C", "{ 30.0 & 50.0  }", " * pF", "key"],
         ["L4_CI_SS", "C", "{ 30.0 & 50.0  }", " * pF", "key"],
      ],
      "gL": [
         ["L4_CI_BC", "gL", "{ 1.0 | 3.0 | 1.0 }", " * nS", "key"],
         ["L4_CI_SS", "gL", "{ 0.5 | 2.5 | 1.0 }", " * nS", "key"],
      ],
   }


.. autofunction:: project.project_manager_module.ProjectManager.build_iterator

.. autofunction:: project.project_manager_module.ProjectManager.run_iterator