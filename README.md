# GR00T-H

[![License](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Model License](https://img.shields.io/badge/Model-NVIDIA%20Open%20Model%20License-green.svg)](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/nvidia/GR00T-H)
[![Open-H Dataset](https://img.shields.io/badge/Dataset-Open--H-orange)](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)

A healthcare robotics variant of [GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T), post-trained on the [Open-H dataset](open_h/README.md) for multi-embodiment surgical and healthcare robot autonomy across 16 robot platforms and 34+ institutions.

<p align="center">
  <img src="media/gr00t-h-header.png" width="800" alt="GR00T-H Header"/>
</p>

## Overview

GR00T-H post-trains the GR00T N1.7 vision-language-action (VLA) foundation model on surgical robot data from multiple institutions and robot platforms simultaneously. Each institution records data differently: different robots, coordinate conventions, frame rates, camera setups, and state/action representations. GR00T-H addresses this by defining per-embodiment modality configs that convert each dataset into a common representation (`REL_XYZ_ROT6D` for EEF poses) while sharing the core VLA backbone.

The primary additions over upstream Isaac-GR00T live in [`open_h/`](open_h/README.md):

- Per-embodiment modality configs converting 16 healthcare robot datasets to a common action representation
- Multi-embodiment training config and dataset preparation tooling
- Extensions to the data pipeline, including clutch-aware filtering, motion scaling, and step filtering

For general robotics use cases, the upstream [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) project is a better starting point.

## News

- **[June 2026]** Released [GR00T-H-N1.7](https://huggingface.co/nvidia/GR00T-H-N1.7), a commercially usable GR00T-H post-trained on the same Open-H data under the NVIDIA Open Model License.
- **[March 2026]** Released GR00T-H with a [pretrained checkpoint](https://huggingface.co/nvidia/GR00T-H) and the [Open-H dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment)

## Model Variants

| Model | Base Model | Params | Branch | HuggingFace | License |
|-------|------------|--------|------------|-------------|---------|
| GR00T-H-N1.7 | [GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) | 3B | [main](https://github.com/NVIDIA-Medtech/GR00T-H/tree/main) | [Weights](https://huggingface.co/nvidia/GR00T-H-N1.7) | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| GR00T-H | [GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) | 3B | [GR00T-H-N1.6](https://github.com/NVIDIA-Medtech/GR00T-H/tree/GR00T-H-N1.6) | [Weights](https://huggingface.co/nvidia/GR00T-H) | [NVIDIA OneWay Noncommercial License](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf) |

## Quick Start

### Installation

```bash
git clone --recurse-submodules git@github.com:NVIDIA-Medtech/GR00T-H.git
cd GR00T-H
uv sync --python 3.10
uv pip install -e .
```

The top-level `uv sync` environment targets Linux x86_64 dGPU systems. For Jetson Orin, Jetson Thor, or DGX Spark, use the platform-specific setup instructions in `scripts/deployment/README.md`.

If `flash-attn` was not built during `uv sync`, install it manually:

```bash
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
```

For containerized setup, see the [Docker Setup Guide](docker/README.md).

### Inference

```bash
uv run python scripts/deployment/standalone_inference_script.py \
  --model-path nvidia/GR00T-H-N1.7 \
  --dataset-path <INSERT_CMR_VERSIUS_DATA_PATH> \
  --embodiment-tag CMR_VERSIUS \
  --traj-ids 0 1 2 \
  --inference-mode pytorch \
  --action-horizon 8
```

For full inference options including TensorRT, see the [inference guide](scripts/deployment/README.md).

### Finetuning on Open-H Embodiments

```bash
uv run torchrun --nproc_per_node=8 --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-H-N1.7 \
    --dataset-path <INSERT_CMR_VERSIUS_DATA_PATH> \
    --embodiment-tag CMR_VERSIUS \
    --num-gpus 8 \
    --global-batch-size 32 \
    --max-steps 20000 \
    --output-dir /path/to/output
```

See [open_h/README.md](open_h/README.md) for a deeper dive on finetuning, multi-embodiment training, and dataset preparation.

## Open-H Dataset

<p align="center">
  <img src="media/open-h-collage.jpg" width="800" alt="Open-H Dataset"/>
</p>

The [Open-H dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment) comprises 16 healthcare robot embodiments across 34+ institutions, stored in [LeRobot](https://github.com/huggingface/lerobot) format. See [open_h/embodiments/README.md](open_h/embodiments/README.md) for the full embodiment comparison table.

## Documentation

| Guide | Description |
|-------|-------------|
| [Open-H Overview](open_h/README.md) | GR00T-H additions, embodiment configs, dataset preparation, training |
| [Embodiment Comparison](open_h/embodiments/README.md) | All 16 embodiments: dimensions, cameras, action formats |
| [Action Configuration](open_h/docs/action_configuration.md) | `REL_XYZ_ROT6D`, rotation formats, adding new embodiments |
| [Data Preparation](open_h/docs/data_preparation.md) | Stats pipeline, temporal statistics, troubleshooting |
| [Inference Guide](scripts/deployment/README.md) | Inference options, TensorRT, server-client architecture |
| [Policy API](getting_started/policy.md) | Observation/action formats, batched inference, environment integration |
| [Finetuning Guide](getting_started/finetune_new_embodiment.md) | Custom embodiment finetuning tutorial |
| [Hardware Recommendations](getting_started/hardware_recommendation.md) | GPU and deployment hardware guidance |
| [Docker Setup](docker/README.md) | Containerized environment setup |

## Base Model

GR00T-H-N1.7 builds on [GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T), a 3B-parameter VLA model with a Cosmos-Reason2-2B / Qwen3-VL backbone and diffusion transformer action head. The GR00T-H-N1.7 checkpoint is [`nvidia/GR00T-H-N1.7`](https://huggingface.co/nvidia/GR00T-H-N1.7).

<details>
<summary>Base model architecture</summary>

<p align="center">
<img src="media/model-architecture.png" width="800" alt="GR00T N1.7 Architecture"/>
</p>

</details>

## License

| Component | License |
|-----------|---------|
| Source code | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| GR00T-H-N1.7 model weights | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

## Citation

```bibtex
@article{openh2026,
  title={Open-H-Embodiment: A Large-Scale Dataset for Enabling Foundation Models in Medical Robotics},
  author={Nelson, Nigel and Chen, Juo-Tung and Haworth, Jesse and Chen, Xinhao and Zbinden, Lukas and Huang, Dianye and Abdelaal, Alaa Eldin and Arezzo, Alberto and Acar, Ayberk and Alambeigi, Farshid and Ammirati, Carlo Alberto and Ao, Yunke and Aranda Rodriguez, Pablo David and Atar, Soofiyan and Ballo, Mattia and Barnes, Noah and Barontini, Federica and Binkiewicz, Filip and Black, Peter and Bodenstedt, Sebastian and Borgioli, Leonardo and Budjak, Nikola and Calm{\'e}, Benjamin and Carrillo, Fabio and Cavalcanti, Nicola and Chen, Changwei and Chen, Haoxin and Chen, Sihang and Chen, Qihan and Chen, Zhongyu and Chen, Ziyang and Cheng, Shing Shin and Cheng, Meiqing and Cheng, Min and Chiu, Zih-Yun Sarah and Chu, Xiangyu and Correa-Gallego, Camilo and Dagnino, Giulio and Deguet, Anton and Delgado, Jacob and DeLong, Jonathan C. and Deng, Kaizhong and Dimitrakakis, Alexander and Ding, Qingpeng and Ding, Hao and Distefano, Giovanni and Donoho, Daniel and Duan, Anqing and Esposito, Marco and Farritor, Shane and Fayad, Jad and Fayad, Zahi and Ferradosa, Mario and Filicori, Filippo and Finn, Chelsea and F{\"u}rnstahl, Philipp and Ge, Jiawei and Giannarou, Stamatia and Giralt Ludevid, Xavier and Giraud, Frederic and Godbole, Aditya Amit and Goldberg, Ken and Goldenberg, Antony and Granero Marana, Diego and Guo, Xiaoqing and Haidegger, Tam{\'a}s and Hailey, Evan and Hansen, Pascal and Hao, Ziyi and Hari, Kush and Hayashi, Kengo and Hawkins, Jonathon and Haworth, Shelby and Hellig, Ortrun and Herrell, S. Duke and Hong, Zhouyang and Howe, Andrew and Hu, Junlei and Hu, Zhaoyang Jacopo and Jain, Ria and Rafiee Javazm, Mohammad and Ji, Howard and Ji, Rui and Ji, Jianmin and Jiang, Zhongliang and Jones, Dominic and Jopling, Jeffrey and Jordan, Britton and Ju, Ran and Kam, Michael and Kang, Luoyao and Kang, Fausto and Kapuria, Siddhartha and Kazanzides, Peter and Kiehler, Sonika and Kilmer, Ethan and Kim, Ji Woong (Brian) and Korzeniowski, Przemys{\l}aw and Kuchi, Chandra and Kumar, Nithesh and Kuntz, Alan and Lavagno, Federico and Lee, Yu Chung and Lee, Hao-Chih and Li, Hang and Li, Zhen and Liang, Xiao and Lin, Xinxin and Lin, Jinsong and Liu, Chang and Liu, Fei and Liu, Pei and Liu, Yun-hui and Liuchen, Wanli and Luk{\'a}cs, Eszter and Mann, Sareena and Mannas, Miles and Marinelli, Brett and Martyniak, Sabina and Marzola, Francesco and Mazza, Lorenzo and Mei, Xueyan and Morais, Maria Clara and Muratore, Luigi and Narayanaswamy, Chetan Reddy and Naskr{\k{e}}t, Micha{\l} and Navarro-Alarcon, David and Neary, Cyrus and Ng, Chi Kit and Nguan, Christopher and Noonan, David and Oh, Ki Hwan and Olesch, Tom Christian and Okamura, Allison M. and Opfermann, Justin and Pescio, Matteo and Pham, Doan Xuan Viet and Porras, Tito and Ren, Hongliang and Rodriguez Jimenez, Ariel and Rodriguez y Baena, Ferdinando and Salcudean, Septimiu E. and Sathya, Asmitha and Satish, Preethi and Seenivasan, Lalithkumar and Shao, Jiaqi and Shen, Yiqing and Sheng, Yu and Shi, Lucy XiaoYang and Soul{\'e}, Zoe and Speidel, Stefanie and Su, Mingwu and Su, Jianhao and Sunmola, Idris and Tak{\'a}cs, Krist{\'o}f and Tang, Yunxi and Thornycroft, Patrick and Tian, Yu and Thompson, Jordan and Turkcan, Mehmet K. and Unberath, Mathias and Valdastri, Pietro and Vives, Carlos and Vuong, Quan and Wagner, Martin and Wang, Farong and Wang, Wei and Wang, Lidian and Wang, Chung-Pang and Wang, Guankun and Wang, Junyi and Wang, Erqi and Wang, Ziyi and Watts, Tanner and Wein, Wolfgang and Wu, Yimeng and Wu, Zijian and Wu, Hongjun and Wu, Luohong and Wu, Jie Ying and Wu, Junlin and Wu, Victoria and Wu, Kaixuan and W{\'o}jcikowski, Mateusz and Xiao, Yunye and Xiao, Nan and Xie, Wenxuan and Yang, Hao and Yang, Tianqi and Yang, Yinuo and Ye, Menglong and Yeung, Ryan S. and Yilmaz, Nural and Yin, Chim Ho and Yip, Michael and Younis, Rayan and Yu, Chenhao and Zaman, Sayem Nazmuz and Zefran, Milos and Zhang, Han and Zhang, Yuelin and Zhang, Yidong and Zhang, Yanyong and Zhang, Xuyang and Zhang, Yameng and Zhang, Joyce and Zhong, Ning and Zhou, Peng and Zhou, Haoying and Zuo, Xiuli and Navab, Nassir and Azizian, Mahdi and Huver, Sean D. and Krieger, Axel},
  year={2026},
  url={https://open-h.github.io}
}
```

## Resources

- [Open-H Paper](https://arxiv.org/abs/2604.21017) - Open-H dataset and GR00T-H paper
- [Open-H Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment) - Multi-embodiment healthcare robot dataset
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) - Upstream base model repository
- [GR00T N1 Paper](https://research.nvidia.com/labs/lpr/publication/gr00tn1_2025/) - Research paper
- [NVIDIA MedTech Open Models](https://github.com/NVIDIA-Medtech)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
