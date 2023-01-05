.. _introduction:

Introduction
============

SystemTools was developed to simplify and automate CxSystem2 array simulations, analysis of the massive amounts of simulations (e.g. 30k by one run) and visualization of the data and analyzed results at various levels of detail. It is built to be a tool for researcher and the software design aims at *simple, maintainable, extensible* and *error-free* code. 

A ProjectManager (*PM*) instance is created at the beginning of execution. The *PM* is a facade for the rest of the code and necessary dependencies between objects are injected at the construction of the *PM*. Because complexity is hidden behind the *PM*, the *conf* is should remain simple and easy to understand. This has been tested with a few students and the feedback has been positive. For new projects, in practice, new modules are developed in parallel with the *conf* script. The *conf* script is then updated to use the new modules.


.. _ref published results:

Use for viewing published work
------------------------------
Published work is available as `jupyter notebooks <https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Notebook%20Basics.html>`_ . The notebooks are pre-configured and self-explanatory and contain the code to reproduce the figures in the projects.

Navigate to project folder, for example:

    :code:`cd SystemTools/scripts/FCN22`

and run the jupyter lab or notebook:

    :code:`jupyter lab` or :code:`jupyter notebook`

Finally, open the notebook of interest and run the cells. 

The `scripts` folder contains the following project:

FCN22
^^^^^

This folder contains jupyter notebooks for creating figures for the paper entitled **Biophysical Parameters Control Signal Transfer in Spiking Network** by *Tomás Garnier Artiñano, Vafa Andalibi, Iiris Atula, Matteo Maestri and Simo Vanni*. The manuscript is currently (3rd Jan 2022) under revision for Frontiers in Computational Neuroscience.


.. _ref research:

Use for research
----------------
The interface for researchers is the project_conf_module.py (*conf* below) script, under project folder. Instead of GUI or command line tools, such configuration script is easiest to use and maintain. After changing the parameters in the project_conf_module.py, you always run :code:`python project/project_conf_module.py`. The top of the file contain imports and the path and contextual parameters, such as experiment name, data and analyses to work on, etc. The methods to execute are protected by an :code:`if __name__ == '__main__'` condition. 

Above :code:`if __name__ == '__main__'`

* Change the parameters in the *conf* file to suit your needs

Below :code:`if __name__ == '__main__'`

* Remove comment character (hash sign) to activate a *conf* statement

* Example for visualizing spikes as a rasterplot for a single simulation:
    
    .. code-block:: console

        PM.viz.show_spikes(results_filename=file_to_display, savefigname="")
    
    * PM is a ProjectManager instance
    * viz is a Viz (visualization) instance, attached to PM at construction
    * show_spikes is a viz method showing rasterplot of spikes
    * results_filename is searched from project folders, if :code:`None`, the latest results file is used
    * if savefigname is not empty string :code:`""`, the figure is saved to the project folder

See :ref:`project configuration script reference <project_conf>` for more information on the *conf* file.

