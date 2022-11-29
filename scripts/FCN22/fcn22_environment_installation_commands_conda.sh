# Commands to build virtual environment and download repositories for 
# Tomas Garnier Artinano, Vafa Andalibi, Iiris Atula, Matteo Maestri & Simo Vanni. 
# Biophysical parameters control signal transfer in spiking network 
# Fronties in Computational Neuroscience (under review)

# Creating FCN22 environment with conda package manager.

conda create -y --name FCN22 python=3.10
conda install -y --name FCN22 -c conda-forge brian2
conda install -y  --name FCN22 matplotlib jupyter
conda install -y  --name FCN22 seaborn
conda install -y  --name FCN22 colour-science
conda install -y  --name FCN22 scikit-learn
conda install -y  --name FCN22 pytest

conda activate FCN22

pip install pyinform

# For running simulations you need CxSystem2 and for analysing and
# visualizing the results and scripting long simulations you need SystemTools.

git clone https://github.com/VisualNeuroscience-UH/CxSystem2.git
pip install -U -e CxSystem2/
# Confirm installation: command cxsystem2 at terminal should give usage.
# Installation can be tested by command pytest in the CxSystem2 folder

git clone https://github.com/VisualNeuroscience-UH/SystemTools.git
# Installation can be tested by command pytest in the SystemTools folder

