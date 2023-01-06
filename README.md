# SystemTools

This repository provides a set of tools for CxSystem2 data analysis, visualization and iterative tasks. 

The main files are in the `project` directory:

* `project/project_conf_module.py`: *conf* below, provides the primary interface to work with SystemTools.
* `project/project_manager_module.py` provides a facade design pattern for the rest of the code.


## Use for viewing published work

The `scripts` folder contains the following projects:

### FCN22

This folder contains jupyter notebooks for creating figures for the paper entitled **Biophysical Parameters Control Signal Transfer in Spiking Network** by *Tomás Garnier Artiñano, Vafa Andalibi, Iiris Atula, Matteo Maestri and Simo Vanni*. The manuscript is currently (3rd Jan 2022) under revision for Frontiers in Computational Neuroscience.


## Use for research
*conf* module is a script, with context (paths etc.) parameters at the top, and executable statements at the bottom.

#### Example *conf* statement
    * PM.viz.show_spikes(results_filename=file_to_display, savefigname="")
        * PM is a ProjectManager instance
        * viz is a Viz (visualization) instance
        * show_spikes is a viz method showing rasterplot of spikes

*  Run it as `python project/project_conf_module.py`
<br>

## Environment Setup and Installation

* Install Python3 (version 3.9 or higher) in your operating system:
  The jupyter notebook files were generated with python 3.10 running on Windows 10 and Ubuntu 22.04, and any python installation working on these environments are viable options.
  * **Windows Users**: There are a few options you can use if your main OS is windows. We strongly recommend Windows Subsystem for Linux (WSL2) shell which is one of the easiest way to get access to an integrated linux environment in Windows. A great tutorial on how to install it is available in [Ubuntu website](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview). 
  * **Linux & Mac Users**: You can skip to the next step.  

* Check python version in your shell. In this example the version is 3.11 which fulfills the requirement: 
    ```
    $ python3 --version
    Python 3.11.0rc1
    ```
* Make sure `virtualenv` is also installed in your environment: 
    ```
    python3 -m pip install --user virtualenv
    ```
* Create a virtual environment for this script:
    ```
    $ python3 -m venv ~/FCN2
    ```
    
    or for Windows powershell
    ```
    python3 -m venv $home/FCN2 
    ```
* Activate the virtual environment:
    ```
    $ source ~/FCN2/bin/activate
    ```
    
    or for Windows powershell
    ```
    .\FCN2\Scripts\Activate
    ```
* Go to your SystemTools folder and install the requirements:
    ```
    $ pip install -r requirements.txt
    ```
* Navigate to the `FCN22` folder and run the jupyter lab:
    ```
    $ cd SystemTools/scripts/FCN22
    $ jupyter lab --no-browser

    [truncated]
    To access the server, open this file in a browser:
            file:///home/username/.local/share/jupyter/runtime/jpserver-7983-open.html
        Or copy and paste one of these URLs:
            http://localhost:8888/...
        or http://127.0.0.1:8888/...
    ```
* At this point, by clicking on one of the links in the output, you should have access to the notebook.
<br>

## Documentation

You can create the html documentation locally with Sphinx using a theme provided by Read the Docs. 

* Start terminal and activate the python environment FCN2
* Go to `docs` folder.
* Type 
    ```
    make html
    ```
This creates `build/html` folder, where the index.html (double click or open from browser) is the root documentation folder. 

<!---
You can access the documentation of the SystemTools at 
[cxsystem2.readthedocs.io](https://cxsystem2.readthedocs.io).
-->
<br>

## Citation

If you use SystemTools for your work, we kindly ask you to cite any of our related article:

```
Garnier Artiñano, T., Andalibi, V., Atula, I., Maestri, M., Vanni, S., Biophysical Parameters Control Signal Transfer in Spiking Network, 
```
<!---
2023, Frontiers in Computational Neuroscience
-->

<br>

## Support

You are encouraged to use <a href="https://github.com/VisualNeuroscience-UH/SystemTools/issues" target="_blank">Github's Issues</a> to report bugs, or request enhancements or new features.
<br>

## Team

In alphabetical order:
- [Vafa Andalibi](https://www.andalibi.me)
- [Simo Vanni](https://scholar.google.fi/citations?user=nRiUf30AAAAJ&hl=en)
<br>

## License

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

- **[GPL3-Clause](https://www.gnu.org/licenses/gpl-3.0.en.html)**