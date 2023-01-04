.. _project_conf:


Reference for :mod:`data_io`
============================

.. autofunction:: data_io.data_io_module.DataIO.listdir_loop

.. autofunction:: data_io.data_io_module.DataIO.most_recent

.. autofunction:: data_io.data_io_module.DataIO.parse_path

.. autofunction:: data_io.data_io_module.DataIO.get_data

.. autofunction:: data_io.data_io_module.DataIO.get_csv_as_df

.. autofunction:: data_io.data_io_module.DataIO.read_input_matfile



Reference for the :mod:`project_conf_module` script file
========================================================

Use keyword substring "file" in filenames, and "folder" in folder names to assert that they are turned into pathlib objects. Path structure is assumed to be root_path/project/experiment/output_folder

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


Analysis & Viz, single files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Show input and data. In the methods below, you can set :code:`results_filename=file_to_display` to avoid writing the the same filename or None to multiple places. If you then set :code:`file_to_display = None`, the functions selects the most recent data file in the output_folder.

file_to_display has the format "[full path to file ] [prefix]_results_[timestamp].gz"

.. autofunction:: viz.viz_module.Viz.show_spikes

.. autofunction:: viz.viz_module.Viz.show_readout_on_input

.. autofunction:: viz.viz_module.Viz.show_analog_results