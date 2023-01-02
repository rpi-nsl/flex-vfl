# Flexible Vertical Federated Learning

This directory builds off of the [Multimodal-Infomax repo](https://github.com/declare-lab/Multimodal-Infomax) to simulate a vertical federated learning setting and run Flex-VFL.

## Running Flex-VFL with CMU-MOSEI dataset

### Dataset

Download the CMU-MOSI and CMU-MOSEI dataset from [Google Drive](https://drive.google.com/drive/folders/1djN_EkrwoRLUt7Vq_QfNZgCl_24wBiIK?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1Wxo4Bim9JhNmg8265p3ttQ) (extraction code: g3m2). Place them under the folder `Multimodal-Infomax/datasets`

### Dependencies

Set up the environment (need conda prerequisite)
```
conda env create -f environment.yml
conda activate MMIM
```

### Running Flex-VFL

To run all experiments sequentially:
```.bash
python run_sbatch.py
```

To plot existing results:
```.bash
python plot_time.py 
python plot_adapt.py 
```

### Citation
This directory is an extension of the Momentum Contrast (MoCo) with Alignment and Uniformity Losses for C-VFL: [github.com/SsnL/moco_align_uniform](https://github.com/SsnL/moco_align_uniform). Please cite the following paper if you use this work in your research:
```bibtex
@article{han2021improving,
  title={Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis},
  author={Han, Wei and Chen, Hui and Poria, Soujanya},
  journal={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2021}
}
```
