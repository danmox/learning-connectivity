# learning-connectivity

Code for training and testing CNNs for learning (algebraic) connectivity maximizing network configurations.

## Installation

Clone the repository and initializes submodules:
```bash
git clone --recursive https://github.com/danmox/learning-connectivity.git
```
It is highly recommended that this project be used in conjunction with a [virtual environment](https://docs.python.org/3/tutorial/venv.html). Python dependencies can be installed with:
```bash
pip install -r requirements.txt
```
Note that the project requires Python 3.

## Datasets

New datasets can be generated using `hdf5_dataset_utils.py`. Existing datasets for task agent teams of 2-6 agents can be downloaded from [here](https://drive.google.com/drive/folders/12P8N_Bfu0LSYAsLbeFRTSH9yt2MVObeo?usp=sharing).

## Usage

Each of the python files in the top level directory comes with a commandline interface. See the help string for usage instructions:
```bash
python <file.py> -h
```
