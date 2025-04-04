# Implicit Grasp Diffusion: Bridging the Gap between Dense Prediction and Sampling-based Grasping
Accepted by Conference on Robot Learning 2024 (CoRL 2024)

Paper Link: https://proceedings.mlr.press/v270/song25b.html
Project Link: https://renaud-detry.net/research/2022-kuleuven-neurobotics/

<img width="918" alt="image" src="https://github.com/user-attachments/assets/23850428-0d1a-4bab-a45c-b0c1dc272aa8">


## Introduction


This paper aims to bridge the gap between dense prediction and sampling-based methods. We propose a novel framework named Implicit Grasp Diffusion (IGD) that leverages implicit neural representations to extract expressive local features, and that generates grasps by sampling from diffusion models conditioned on these local features. We evaluated our model on a clutter removal task in both simulated and real-world environments. The experimental results have demonstrated the high grasp accuracy, strong noise robustness, and multi-modal grasp modeling of the proposed method.

If you find our work useful in your research, please consider [citing](#citing).

## Installation

1. Create a conda environment with python=3.8

2. Install [ROS](https://wiki.ros.org/ROS/Installation).

3. Install [pytorch](https://pytorch.org/get-started/previous-versions/). We use Pytorch 1.13.1 with CUDA 11.7.

4. Install packages list in [requirements.txt](requirements.txt). Then install `torch-scatter` following [here](https://github.com/rusty1s/pytorch_scatter), based on `pytorch` version and `cuda` version. (PS: if there is an error about sklearn when installing open3d, you can export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True)

5. Go to the root directory and install the project locally using `pip`

```
pip install -e .
```

6. Build ConvONets dependents by running `python scripts/convonet_setup.py build_ext --inplace`.

7. We use the same data as [GIGA](https://github.com/UT-Austin-RPL/GIGA.git). You can download the [data](https://utexas.box.com/s/h3ferwjhuzy6ja8bzcm3nu9xq1wkn94s), then unzip and place the data folder under the repo's root. 

## Self-supervised Data Generation

### Raw synthetic grasping trials

Pile scenario:

```bash
python scripts/generate_data_parallel.py --scene pile --object-set pile/train --num-grasps 4000000 --num-proc 40 --save-scene ./data/pile/data_pile_train_random_raw_4M
```

Packed scenario:
```bash
python scripts/generate_data_parallel.py --scene packed --object-set packed/train --num-grasps 4000000 --num-proc 40 --save-scene ./data/pile/data_packed_train_random_raw_4M
```

Please run `python scripts/generate_data_parallel.py -h` to print all options.

### Data clean and processing

First clean and balance the data using:

```bash
python scripts/clean_balance_data.py /path/to/raw/data
```

Then construct the dataset (add noise):

```bash
python scripts/construct_dataset_parallel.py --num-proc 40 --single-view --add-noise dex /path/to/raw/data /path/to/new/data
```

### Save occupancy data

Sampling occupancy data on the fly can be very slow and block the training, so I sample and store the occupancy data in files beforehand:

```bash
python scripts/save_occ_data_parallel.py /path/to/raw/data 100000 2 --num-proc 40
```

Please run `python scripts/save_occ_data_parallel.py -h` to print all options.


## Training

### Train IGD

Run:

```bash
python scripts/train_igd.py --dataset /path/to/new/data --dataset_raw /path/to/raw/data
```

## Simulated grasping

Run:

```bash
python scripts/sim_grasp_multiple.py --num-view 1 --object-set (packed/test | pile/test) --scene (packed ｜ pile) --num-rounds 100 --sideview --add-noise dex --force --best --model /path/to/model --type igd --result-path /path/to/result
```

This command will run the experiment with each seed specified in the arguments.

Run `python scripts/sim_grasp_multiple.py -h` to print a complete list of optional arguments.

## Pre-generated data

Data generation is very costly. So it'd be better to use pre-generated data. Because the occupancy data takes too much space (over 100G), we do not upload the occupancy data, you can generate them following the instruction in this [section](#save-occupancy-data). This generation won't take too long time.

| Scenario | Raw data | Processed data |
| ----------- | ----------- | ----------- |
| Pile | [link](https://utexas.box.com/s/w1abs6xfe8d2fo0h9k4bxsdgtnvuwprj) | [link](https://utexas.box.com/s/l3zpzlc1p6mtnu7ashiedasl2m3xrtg2) |
| Packed | [link](https://utexas.box.com/s/roaozwxiikr27rgeauxs3gsgpwry7gk7) | [link](https://utexas.box.com/s/h48jfsqq85gt9u5lvb82s5ft6k2hqdcn) |

## Pretrained model

Packed scene: https://drive.google.com/file/d/1kdukKdpIa4_r06l-DUJxuUn_w14eneTs/view?usp=drive_link

Pile scene: https://drive.google.com/file/d/1utAwzIDBO_awws9h0I-DYw330Y4D-kHu/view?usp=drive_link


## Related Repositories

1. Our code is largely based on [VGN](https://github.com/ethz-asl/vgn) and [GIGA](https://github.com/UT-Austin-RPL/GIGA.git).

2. We use [ConvONets](https://github.com/autonomousvision/convolutional_occupancy_networks) as our backbone.

## Citing
```
@inproceedings{song2024b,
  author = {Song, Pinhao and Li, Pengteng and Detry, Renaud},
  title = {Implicit Grasp Diffusion: Bridging the Gap between Dense Prediction and Sampling-based Grasping},
  year = {2024},
  booktitle = {Conference on Robot Learning},
}
```


