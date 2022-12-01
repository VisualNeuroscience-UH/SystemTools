# SystemTools

This repository provides a set of tools for CxSystem2 data analysis 
and visualization tasks. 

The main files are in the `project` directory:

* `project/project_conf_module.py`: provides the primary interface to work with SystemTools.
* `project/project_manager_module.py` provides a facade design pattern for the rest of the code.

## Scripts 

The `scripts` folder contains research scripts and tools that are created using CxSystem2:

### FCN2

This folder contains the notebook which includes the analysis used in the paper entitled [TBD].

#### Environment Setup

* Install Python3 (any version higher than 3.5) in your operating system:
  * **Windows Users**: There are a few options you can use if your main OS is windows. We highly recommend Windows Subsystem for Linux (WSL2) shell which is one of the easiest way to get access to an integrated linux environment in Windows. A great tutorial on how to install it is available in [Ubuntu website](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview).
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
* Activate the virtual environment:
    ```
    $ source ~/FCN2/bin/activate
    ```
* Install the requirements:
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
            http://localhost:8888/lab?token=e0537f8d814c3f274296b4c40d0357502796b86444234937
        or http://127.0.0.1:8888/lab?token=e0537f8d814c3f274296b4c40d0357502796b86444234937
    ```
* At this point, by clicking on one of the links in the output, you should have access to the notebook.