# learning-connectivity

Code for training and testing CNNs for learning (algebraic) connectivity maximizing network configurations as outlined in our paper [Learning Connectivity-Maximizing Network Configurations](https://ieeexplore.ieee.org/abstract/document/9695258). To cite this work use the following BibTeX entry:
```bibtex
@article{mox2022learning,
  author={Mox, Daniel and Kumar, Vijay and Ribeiro, Alejandro},
  journal={IEEE Robotics and Automation Letters},
  title={Learning Connectivity-Maximizing Network Configurations},
  year={2022},
  volume={7},
  number={2},
  pages={5552-5559},
  doi={10.1109/LRA.2022.3146524}
}
```

## Installation

Clone the repository and initialize submodules:
```bash
git clone https://github.com/danmox/learning-connectivity.git
```
It is highly recommended that this project be used in conjunction with a [virtual environment](https://docs.python.org/3/tutorial/venv.html). Python dependencies can be installed with:
```bash
pip install -r requirements.txt
```
**Note that the project requires Python 3.**

## Usage

Each of the python files in the top level directory comes with a commandline interface. See the help string for usage instructions:
```bash
python <file.py> -h
```

## Datasets

New datasets can be generated using `hdf5_dataset_utils.py` (e.g. `python hdf5_dataset_utils.py generate --scale 2 100000 3` to generate 100,000 256x256 image pairs with 3 task agents in the input image). Existing datasets for task agent teams of 2-6 agents can be downloaded [here](https://drive.google.com/drive/folders/12P8N_Bfu0LSYAsLbeFRTSH9yt2MVObeo?usp=sharing).

## Models / Training

Limited pre-trained models can be found in the `models` directory. The model used in the paper is `ConvAEModel_px256_nf128_8x8kern__256_2t2t3t3t4t5t6t__valloss_4.199e-04_epoch_14__20211202-054934.ckpt`.

To train a new ConvAEModel_px256_nf128_8x8kern model on the 256px datasets (assuming they have been downloaded to `data/256px`) directory for 20 epochs:
```bash
python cnn.py train ConvAEModel_px256_nf128_8x8kern data/256px/256_connectivity_100000s_*.hdf5 --epochs 20
```
Other possible network models to train can be found in `models.py`.

## Results

A variety of different tests can be run on trained networks with `cnn_results.py`. See the help string for more information. All tests can be run with the `run_tests` script.
