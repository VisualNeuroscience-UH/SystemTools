# For running simulations you need CxSystem2 and for analysing and
# visualizing the results and scripting long simulations you need SystemTools.
# Creating FCN22 environment and downloading and installation of the 
# necessary git repositories, run the following commands at the folder 
# where you want to install the CxSystem2 and SystemTools git repos:

conda create -y --name FCN22 python=3.10
conda install -y --name FCN22 -c conda-forge brian2
conda install -y  --name FCN22 matplotlib jupyter
conda install -y  --name FCN22 seaborn
conda install -y  --name FCN22 colour-science
conda install -y  --name FCN22 scikit-learn
conda install -y  --name FCN22 pytest

conda activate FCN22

pip install pyinform

git clone https://github.com/VisualNeuroscience-UH/CxSystem2.git
#cd CxSystem2
pip install -U -e CxSystem2/
# confirm installation command cxsystem2 should give usage

#cd .. # back to git repos root
git clone https://github.com/VisualNeuroscience-UH/SystemTools.git
