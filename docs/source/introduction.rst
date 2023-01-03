.. _introduction:

Introduction
============

SystemTools was developed to simplify and automate CxSystem2 array simulations, analysis of the massive amounts of simulations (e.g. 30k by one run) and visualization of the data and analyzed results at various levels of detail. It is built to be a tool for researcher. 

The interface is simple: run :code:`python project/project_conf_module.py`, the configuration script for the project. The top of the file contain imports and the path and contextual parameters, such as experiment name, data and analyses to work on, etc. The methods to execute are protected by an :code:`if __name__ == '__main__'`` condition. 

Use for research
----------------

* Remove comment character hash sign "#"" to activate a  *conf* statement
    #### Example
    * PM.viz.show_spikes(results_filename=file_to_display, savefigname="")
        * PM is a ProjectManager instance
        * viz is a Viz (visualization) instance
        * show_spikes is a viz method showing rasterplot of spikes

*  Run it as `python project/project_conf_module.py`


Use for viewing published work
------------------------------

The `scripts` folder contains the following projects:

FCN22
^^^^^

This folder contains jupyter notebooks for creating figures for the paper entitled **Biophysical Parameters Control Signal Transfer in Spiking Network** by *Tomás Garnier Artiñano, Vafa Andalibi, Iiris Atula, Matteo Maestri and Simo Vanni*. The manuscript is currently (3rd Jan 2022) under revision for Frontiers in Computational Neuroscience.
<br><br>