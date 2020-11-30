# Some Thoughts in Deep Learning venv and libraries

## Install libraries below in the venv to allow 

pip install numpy==1.19.3 (1.19.4 will not work with Windows 2004)

pip install tf-nightly, pandas, matplotlib, keras

## If you want to use Intel or AMD graphic card to accelerate the training, plaindML is recommanded. 

### plaindML Windows installation

1 First install Chocolatey by starting an Administrator PowerShell and running:

Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

2 Install Python:

choco install -y python git vcredist2015

3 Switch to an unprivileged PowerShell to install and set up PlaidML with Keras

pip install -U plaidml-keras
plaidml-setup

4 PlaidML should now be installed! You can test the installation by running MobileNet in plaidbench.

pip install plaidml-keras plaidbench
plaidbench keras mobilenet