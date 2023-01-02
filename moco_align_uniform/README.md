# Flexible Vertical Federated Learning

This directory builds off of the [Momentum Contrast (MoCo) with Alignment and Uniformity Losses repo](https://github.com/SsnL/moco_align_uniform) to simulate a vertical federated learning setting and run Flex-VFL.

## Running Flex-VFL with the ModelNet40 dataset and ImageNet dataset

### Datasets

The ModelNet40 dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/1YaGWesl9DyYNoE8Pfe80EmqHkoJ0XlKU/view?usp=sharing).
The dataset must be placed in a folder named 'view' in the this directory. 

The ImageNet dataset can be downloaded from [image-net.org](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
The data must be placed in a folder named 'imagenet', then preprocessed with:
```.bash
python create_imagenet_subset.py imagenet imagenet100 
```

### Running Flex-VFL
To run all experiments sequentially:
```.bash
python run_sbatch.py
```

To plot existing results:
```.bash
python plot_time.py 
python plot_time_mvcnn.py 
python plot_time_mvcnn_adapt.py 
```


## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
