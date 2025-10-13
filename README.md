<div align="center">
<h1>DreamSwapV: Mask-guided Subject Swapping for Any Customized Video Editing </h1> 

<a href='https://arxiv.org/abs/2508.14465'><img src='https://img.shields.io/badge/ArXiv-2508.14465-red'></a>
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/victor-thu/DreamSwapV)
</div>

## 📅 TODO
- [x] Page and video demo
- [x] Model weights
- [x] Inference code
- [ ] DreamSwapV-Benchmark

## 💻 Getting Started
### 📑 Requirements
```
pip install requirements.txt
```
The implementation is tested under python 3.9, as well as pytorch 2.5.1 and torchvision 0.20.1+cu124. We recommend equivalent pytorch version for stable performance.
We also recommend installing flash-attn-3 for faster inference. We have provided a [precompiled wheel](https://huggingface.co/victor-thu/DreamSwapV/blob/main/flash_attn_3-3.0.0b1-cp311-cp311-linux_x86_64.whl) built in our test environment, which you can install using the following command:
```
pip install flash_attn_3-3.0.0b1-cp311-cp311-linux_x86_64.whl
```
If you encounter installation issues, please check whether your environment fully matches our test environment, or compile your own flash-attn-3 wheel compatible with your environment.

### ⭐️ Model Preparation

#### DreamSwapV ckpt

You can download DreamSwapV ckpt from https://huggingface.co/victor-thu/DreamSwapV to any folder and pass it through args.checkpoint.

#### TrackingSAM ckpt

TrackingSAM ckpt path is ./utils/sam/ckpt, you should download the following ckpts to this folder:

SAM model, the default model is [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth).

DeAOT/AOT model, the default model is [R50_DeAOTL_PRE_YTB_DAV.pth](https://drive.google.com/file/d/1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ/view).

Grounding-Dino model, the default model is [groundingdino_swint_ogc](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth).

#### DWPose and 3D Hamer ckpt

DWPose ckpt path is ./utils/dwpose/ckpts, you should download the following ckpt to this folder:

onnx_det model, the default model is [yolox_l.onnx](https://huggingface.co/yzd-v/DWPose/blob/main/yolox_l.onnx).

onnx_pose model, the default model is [dw-ll_ucoco_384.onnx](https://huggingface.co/yzd-v/DWPose/blob/main/dw-ll_ucoco_384.onnx).

3D Hamer ckpt path is ./utils/hamer/, you should download the following ckpt to this folder:

Hamer model, the default model is [hamer_demo_data.tar.gz](https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT). After downloading this tar.gz, you can unzip it to get a _DATA folder (./utils/hamer/_DATA).

#### Wan2.1 VAE ckpt

The default Wan2.1 VAE ckpt will be automaticly downloaded to your huggingface cache folder, you can also manually download it from [Wan2.1_VAE.pth](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/blob/main/Wan2.1_VAE.pth).

### 💪 Inference

We provide scripts for single video inference and benchmark batch inference:

```
# single inference
python inference.py --video your_mp4_path --first_mask the_first_frame_mask_of_your_mp4 --ref the_reference_you_want_to_inject
                    --checkpoint your_dreamswapv_ckpt_path --output_dir ./outputs --device cuda:0 --save_debug


# benchmark inference
python inference_batch.py --bench_root your_benchmark_path --checkpoint your_dreamswapv_ckpt_path --output_dir ./outputs --save_debug
```

### 🎉 Acknowledgements
We would like to thank the contributors to the following repositories, for their open research.

* TrackingSAM - https://github.com/z-x-yang/Segment-and-Track-Anything/
* DWPose - https://github.com/IDEA-Research/DWPose/
* Hamer - https://github.com/geopavlakos/hamer
* Wan2.1 - https://github.com/Wan-Video/Wan2.1

Licenses for borrowed code and third-party dependencies can be found in [code_licenses.md](https://code.byted.org/wangweitao.owl/DreamSwapV/blob/main/code_licenses.md) and [third_party.txt](https://code.byted.org/wangweitao.owl/DreamSwapV/blob/main/third_party.txt) file. 

### License
The project is licensed under the [Apache-2.0 license](https://code.byted.org/wangweitao.owl/DreamSwapV/blob/main/LICENSE.txt). To utilize or further develop this project for commercial purposes through proprietary means, permission must be granted by us (as well as the owners of any borrowed code).

### Citations
Please consider citing the related paper(s) in your publications if it helps your research.
```
@article{wang2025dreamswapv,
  title={DreamSwapV: Mask-guided Subject Swapping for Any Customized Video Editing},
  author={Wang, Weitao and Wang, Zichen and Shen, Hongdeng and Lu, Yulei and Fan, Xirui and Wu, Suhui and Zhang, Jun and Wang, Haoqian and Zhang, Hao},
  journal={arXiv preprint arXiv:2508.14465},
  year={2025}
}
@article{cheng2023segment,
  title={Segment and track anything},
  author={Cheng, Yangming and Li, Liulei and Xu, Yuanyou and Li, Xiaodi and Yang, Zongxin and Wang, Wenguan and Yang, Yi},
  journal={arXiv preprint arXiv:2305.06558},
  year={2023}
}
@inproceedings{yang2023effective,
  title={Effective whole-body pose estimation with two-stages distillation},
  author={Yang, Zhendong and Zeng, Ailing and Yuan, Chun and Li, Yu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4210--4220},
  year={2023}
}
@inproceedings{pavlakos2024reconstructing,
  title={Reconstructing hands in 3d with transformers},
  author={Pavlakos, Georgios and Shan, Dandan and Radosavovic, Ilija and Kanazawa, Angjoo and Fouhey, David and Malik, Jitendra},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9826--9836},
  year={2024}
}
@article{wan2025wan,
  title={Wan: Open and advanced large-scale video generative models},
  author={Wan, Team and Wang, Ang and Ai, Baole and Wen, Bin and Mao, Chaojie and Xie, Chen-Wei and Chen, Di and Yu, Feiwu and Zhao, Haiming and Yang, Jianxiao and others},
  journal={arXiv preprint arXiv:2503.20314},
  year={2025}
}
```
