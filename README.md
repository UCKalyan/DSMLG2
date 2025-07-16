# 3D UNET Project for BraTS 2020 Brain Tumor Analysis

This project provides a complete pipeline for brain tumor analysis on the BraTS 2020 dataset, including 3D classification of tumors as benign or malignant, and 2D/3D segmentation of tumor sub-regions.

## Project Structure

```bash
brats_unet3d_project/
├── config/
│   └── config.yaml
├── data/
│   ├── brats2020_preprocess.py
│   ├── slicer.py
│   └── dataset_loader.py
├── models/
│   ├── unet2d.py
│   ├── unet3d.py
│   └── classifier3d.py
├── training/
│   ├── trainer2d.py
│   ├── trainer3d_cls.py
│   ├── trainer3d_seg.py
│   └── metrics.py
├── inference/
│   ├── predict2d.py
│   ├── reconstruct3d.py
│   └── visualizer.py
├── utils/
│   ├── logger.py
│   ├── helpers.py
│   └── augmentations.py
├── main.py
├── requirements.txt
├── install_dependencies.bat
└── README.md
```

## Features

* **End-to-End Pipeline:** From data preprocessing to model training, inference, and evaluation.
* **Multiple Models:**
    * **3D CNN Classifier:** Classifies entire MRI volumes as benign or malignant.
    * **2D UNET:** Segments tumor regions (WT, TC, ET) from 2D axial slices.
    * **3D UNET:** (For future extension to full 3D segmentation).
* **Modular and Object-Oriented:** Code is organized into logical, reusable classes.
* **Configurable:** A central `config.yaml` file controls all important parameters.
* **Data Preprocessing & Augmentation:** Includes skull stripping, normalization, resampling, and real-time data augmentation.
* **Evaluation:** Comprehensive evaluation scripts to compute Dice, IoU, Precision, Recall, and Hausdorff distance.
* **Visualization:** Generates visualizations of segmentation results.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd brats_unet3d_project
    ```

2.  **Install Dependencies:**
    Run the batch script to create a virtual environment and install the required packages.
    ```bash
    install_dependencies.bat
    ```
    This will create a `venv` folder and install all packages from `requirements.txt`.

3.  **Activate the Virtual Environment:**
    ```bash
    venv\Scripts\activate
    ```

4.  **Download BraTS 2020 Data:**
    Download the BraTS 2020 dataset and place it in a directory. Update the `data_path` in `config/config.yaml` to point to your dataset location.

5. Ensure atleast 50GB of free space.
6.  in config.yaml -> 
* active_profile : gpu_high_mem # only if you have GPU and RAM
* epochs : 100 
* data_path: "/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData" # Path to the main BraTS2020 dataset folder

## Usage

The `main.py` script is the main entry point for running different parts of the pipeline.

### Preprocessing Data
Ensure to set the Folder where dataset is present in config.yaml.
data_path: "/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData" # Path to the main BraTS2020 dataset folder

'''bash
python main.py --mode preprocess
'''

### tfrecords ## are empty
ERROR - TFRecord directory './processed_data/tfrecords' is empty or does not exist.
Please run the conversion step first using: python main.py --mode convert_to_tfrecord

'''bash
python main.py --mode convert_to_tfrecord
'''

### Training

* **Train a 2D Segmentation Model:**
    ```bash
    python main.py --mode train --model UNET2D
    ```

rm -R ./processed_data/.DS_Store if required.
* **Train a 3D Classification Model:**
    ```bash
    python main.py --mode train --model Classifier3D --output_type benign_vs_malignant
    ```

* **Train a 3D Segmentation Model:**
    ```bash
    python main.py --mode train --model UNET3D
    ```

### Inference

* **Run 2D Segmentation and 3D Reconstruction:** eg:- BraTS20_Training_356
    ```bash
    python main.py --mode predict --model UNET2D --patient_id <ID_of_Patient>
    ```

* **Run 2D Segmentation and 3D Reconstruction:** eg:-BraTS20_Training_356
    ```bash
    python main.py --mode predict --model UNET3D --patient_id <ID_of_Patient>
    ```
### Evaluation

* **Evaluate a trained model:**
    ```bash
    python main.py --mode evaluate --model UNET2D
    ```

## Configuration

All parameters can be modified in `config/config.yaml`:

* `model`: Choose between `UNET2D`, `UNET3D`, `Classifier3D`.
* `encoder`: Encoder backbone for UNET models (e.g., `ResNet34`).
* `input_shape`: Input shape for the models.
* `loss_function`: Loss function for training.
* `batch_size`, `epochs`: Training parameters.
* `data_path`: Path to the BraTS 2020 dataset.
* ... and other model/data specific parameters.


## INFO

* Axial, Coronal, and Sagittal Planes:
* Axial: A horizontal plane dividing the body into upper and lower parts. In brain imaging, it's often referred to as the transverse plane.
* Coronal: A vertical plane dividing the body into front and back sections.
* Sagittal: A vertical plane dividing the body into left and right sections

Visualizing the BraTS Dataset:
1. Multi-Planar Reconstruction (MPR):
This technique allows you to view the same data in all three planes (axial, coronal, and sagittal) simultaneously or sequentially. This is often the first step in visualizing the dataset and getting a sense of the data.
2. 3D Rendering:
This approach combines the data from all three planes to create a 3D model of the brain and the tumor. This can be very helpful in understanding the tumor's shape, size, and location relative to surrounding structures.
3. Deep Learning Models:
Many deep learning models designed for brain tumor segmentation, particularly those utilizing 3D convolutions, are specifically designed to leverage information from all three planes. These models implicitly learn the relationships between the different views. Examples include models that use axial, coronal, and sagittal views, or models that decompose 3D convolutions into axial intra-slice and inter-slice convolutions. 
Why is understanding these planes important for BraTS?
Tumor Characteristics:
Different tumor characteristics might be more apparent in one plane than another. For example, the extent of edema (swelling around the tumor) might be more visible in the T2-weighted FLAIR images in the axial plane. 
Segmentation Accuracy:
Accurate brain tumor segmentation requires understanding the spatial context of the tumor within the brain. Combining information from all three planes is crucial for achieving high segmentation accuracy. 
Clinical Decision Making:
Radiologists use all three planes when manually segmenting tumors for diagnosis and treatment planning. Deep learning models should ideally mimic this process. 

## Changes Made

tfrecord_writer.py - lot of changes due to unit8 vs int32
predict2d.py updated  def predict_volume(self, volume):

dataset_loader.py 
    BratsDataset3D
    def _parse_tfrecord_fn(self, example):
        segmentation = tf.io.parse_tensor(example['segmentation'], out_type=tf.int32) # int8
    BratsDataset2D

    def _parse_tfrecord_fn(self, example):
        label = tf.reshape(label, [1])

trainer2d.py - missing imports
    from utils.helpers import ensure_dir
    import os
    import matplotlib.pyplot as plt

trainer3d_seg.py udpated the metrics and relavent import
trainer2d.py
        #steps_per_epoch = math.ceil((train_loader.dataset_size * avg_slices_per_patient) / self.config['batch_size'])
        #validation_steps = math.ceil((val_loader.dataset_size * avg_slices_per_patient) / self.config['batch_size'])
        steps_per_epoch = 100
        validation_steps = 20
trainer3d_cls.py
        #steps_per_epoch = math.ceil(train_loader.dataset_size / self.config['batch_size'])
        #validation_steps = math.ceil(val_loader.dataset_size / self.config['batch_size'])
        steps_per_epoch = 100
        validation_steps = 20
