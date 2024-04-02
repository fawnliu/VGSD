# Multi-view Dynamic Reflection Prior for Video Glass Surface Detection (AAAI 2024)

[AAAI 2024] Official code release of our paper "Multi-view Dynamic Reflection Prior for Video Glass Surface Detection"


### Dataest

The training and testing dataset is available at [Google](https://drive.google.com/drive/folders/1QsdYI5Gwi-rKKwGgdE7GFTjhRO4-wIiI?usp=sharing). 


### Evaluation
Download the predicted results from the [link](https://github.com/fawnliu/VGSD/releases/download/1.0/pred.zip) and run the following command to evaluate the results.

```bash
python eval.py -pred ../results/pred  -gt ../VGD_dataset/test
```

### Inference
Download the trained model from the [VGSD.pth](https://github.com/fawnliu/VGSD/releases/download/1.0/VGSD.pth) and run the following command to generate the predicted results.

```bash
python infer.py -pred ../results/ -exp ../checkpoints/VGSD.pth 
```

### Training
1. Please inference the reflection maps for the glass regions in the training dataset (192 videos) from [SIRR](https://github.com/zdlarr/Location-aware-SIRR).
2. Download backbone weights from the [resnext_101_32x4d.pth](https://github.com/fawnliu/VGSD/releases/download/1.0/resnext_101_32x4d.pth) and run `train.py` to train the model.


### Contact
If you have any questions, please feel free to contact me via `fawnliu2333@gmail.com`.

### Citation

```bibtex
@inproceedings{liu2024multi,
  title={Multi-View Dynamic Reflection Prior for Video Glass Surface Detection},
  author={Liu, Fang and Liu, Yuhao and Lin, Jiaying and Xu, Ke and Lau, Rynson WH},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={4},
  pages={3594--3602},
  year={2024}
}
```