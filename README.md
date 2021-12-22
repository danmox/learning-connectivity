# learning-connectivity

Code for training and testing CNNs for learning (algebraic) connectivity maximizing network configurations.

## Installation

Clone the repository and initialize submodules:
```bash
git clone --recursive https://github.com/danmox/learning-connectivity.git
```
It is highly recommended that this project be used in conjunction with a [virtual environment](https://docs.python.org/3/tutorial/venv.html). Python dependencies can be installed with:
```bash
pip3 install -r requirements.txt
```
Note that the project requires Python 3.

## Usage

Each of the python files in the top level directory comes with a commandline interface. See the help string for usage instructions:
```bash
python3 <file.py> -h
```

## Datasets

New datasets can be generated using `hdf5_dataset_utils.py` (e.g. `python3 hdf5_dataset_utils.py generate --scale 2 100000 3` to generate 100,000 256x256 images with 3 task agents). Existing datasets for task agent teams of 2-6 agents can be downloaded [here](https://drive.google.com/drive/folders/12P8N_Bfu0LSYAsLbeFRTSH9yt2MVObeo?usp=sharing).

## Training

To train a ConvAEModel_px256_nf128_8x8kern model on the 256px datasets for 20 epochs:
```bash
python3 cnn.py train ConvAEModel_px256_nf128_8x8kern data/256px/256_connectivity_100000s_*.hdf5 --epochs 20
```
Other possible network models to train can be found in `models.py`.

## Results

A variety of different tests can be run on trained networks with `cnn_results.py`. See the help string for more information. All tests can be run with the `run_tests` script.
