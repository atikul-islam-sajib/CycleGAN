# Cycle GAN Implementation

<img src="https://miro.medium.com/v2/resize:fit:762/1*eNJKtwv7nSqUXcA8Nd3U7Q.png">

This project provides a complete framework for training and testing a Cycle Generative Adversarial Network (Cycle-GAN). It includes functionality for data preparation, model training, testing, and inference as well.

<img src="https://raw.githubusercontent.com/atikul-islam-sajib/Research-Assistant-Work-/main/cycle-gan1.png" alt="AC-GAN - Medical Image Dataset Generator: Generated Image with labels">

## Features

| Feature                          | Description                                                                                                                                                                                                           |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Efficient Implementation**     | Utilizes an optimized Cycle-GAN model architecture for superior performance on diverse image segmentation tasks.                                                                                                          |
| **Custom Dataset Support**       | Features easy-to-use data loading utilities that seamlessly accommodate custom datasets, requiring minimal configuration.                                                                                             |
| **Training and Testing Scripts** | Provides streamlined scripts for both training and testing phases, simplifying the end-to-end workflow.                                                                                                               |
| **Visualization Tools**          | Equipped with tools for tracking training progress and visualizing segmentation outcomes, enabling clear insight into model effectiveness.                                                                            |
| **Custom Training via CLI**      | Offers a versatile command-line interface for personalized training configurations, enhancing flexibility in model training.                                                                                          |
| **Import Modules**               | Supports straightforward integration into various projects or workflows with well-documented Python modules, simplifying the adoption of CycleGAN functionality.                                                         |
| **Multi-Platform Support**       | Guarantees compatibility with various computational backends, including MPS for GPU acceleration on Apple devices, CPU, and CUDA for Nvidia GPU acceleration, ensuring adaptability across different hardware setups. |

## Getting Started

## Installation Instructions

Follow these steps to get the project set up on your local machine:

| Step | Instruction                                  | Command                                                       |
| ---- | -------------------------------------------- | ------------------------------------------------------------- |
| 1    | Clone this repository to your local machine. | **git clone https://github.com/atikul-islam-sajib/CycleGAN.git** |
| 2    | Navigate into the project directory.         | **cd CycleGAN**                                                  |
| 3    | Install the required Python packages.        | **pip install -r requirements.txt**                           |

## Project Structure

This project is thoughtfully organized to support the development, training, and evaluation of the Cycle-GAN model efficiently. Below is a concise overview of the directory structure and their specific roles:

- **checkpoints/**
  - Stores model checkpoints during training for later resumption.
- **best_model/**

  - Contains the best-performing model checkpoints as determined by validation metrics.

- **train_models/**

  - Houses all model checkpoints generated throughout the training process.

- **data/**

  - **processed/**: Processed data ready for modeling, having undergone normalization, augmentation, or encoding.
  - **raw/**: Original, unmodified data serving as the baseline for all preprocessing.

- **logs/**

  - **Log** files for debugging and tracking model training progress.

- **metrics/**

  - Files related to model performance metrics for evaluation purposes.

- **outputs/**

  - **test_images/**: Images generated during the testing phase, including segmentation outputs.
  - **train_gif/**: GIFs compiled from training images showcasing the model's learning progress.
  - **train_images/**: Images generated during training for performance visualization.

- **research/**

  - **notebooks/**: Jupyter notebooks for research, experiments, and exploratory analyses conducted during the project.

- **src/**

  - Source code directory containing all custom modules, scripts, and utility functions for the U-Net model.

- **unittest/**
  - Unit tests ensuring code reliability, correctness, and functionality across various project components.

### Dataset Organization for srgan

The dataset is organized into three categories for SRGAN. Each category directly contains paired images and their corresponding lower resolution images and higher resolution, stored together to simplify the association between lower resolution and higher resolution images .

## Directory Structure:

```
dataset/
├── images/
│ │ ├── 1.png
│ │ ├── 2.png
│ │ ├── ...
├── masks/
│ │ ├── 1.png
│ │ ├── 2.png
│ │ ├── ...
```

For detailed documentation on the dataset visit the [Dataset - GitHub](https://github.com/atikul-islam-sajib/Research-Assistant-Work-/blob/main/dataset.zip).


### User Guide Notebook - CLI

For detailed documentation on the implementation and usage, visit the -> [Cycle-GAN Notebook - CLI](https://github.com/atikul-islam-sajib/CycleGAN/blob/main/research/notebooks/ModelTrain-CLI.ipynb).

### User Guide Notebook - Custom Modules

For detailed documentation on the implementation and usage, visit the -> [Cycle-GAN Notebook - CM](https://github.com/atikul-islam-sajib/CycleGAN/blob/main/research/notebooks/ModelTrain-CM.ipynb).

<!-- ### Model History

For detailed documentation on the implementation and usage, visit the -> [CycleGAN Model History](https://github.com/atikul-islam-sajib/SRGAN/blob/main/checkpoints/model_history/history.csv). -->

### Command Line Interface

The project is controlled via a command line interface (CLI) which allows for running different operational modes such as training, testing, and inference.

#### CLI Arguments
| Argument          | Description                                  | Type   | Default |
|-------------------|----------------------------------------------|--------|---------|
| `--image_path`    | Path to the image dataset                    | str    | None    |
| `--batch_size`    | Number of images per batch                   | int    | 1       |
| `--image_size`    | Size to resize images to                     | int    | 64      |
| `--split_size`| Whether to split the dataset             | float   | 0.20   |
| `--epochs`        | Number of training epochs                    | int    | 100     |
| `--lr`            | Learning rate                                | float  | 0.0002  |
| `--is_lr_scheduler`| Enable learning rate scheduler              | bool   | False   |
| `--device`        | Computation device ('cuda', 'mps', 'cpu')    | str    | 'mps'   |
| `--adam`          | Use Adam optimizer                           | bool   | True    |
| `--SGD`           | Use Stochastic Gradient Descent optimizer    | bool   | False   |
| `--train`         | Flag to initiate training mode               | action | N/A     |
| `--test_result`          | Flag to initiate testing mode                | action | N/A     |
| `--single_image`        | Flag for single image inference              | action | N/A     |
| `--batch_image`         | Flag for batch image inference               | action | N/A     |

### CLI Command Examples

| Task                     | CUDA Command                                                                                                              | MPS Command                                                                                                              | CPU Command                                                                                                              |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| **Training a Model**     | `python cli.py --train --image_path "/path/to/dataset" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --adam True --device "cuda"` | `python cli.py --train --image_path "/path/to/dataset" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --adam True --device "mps"` | `python cli.py --train --image_path "/path/to/dataset" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --content_loss 0.01 --adam True --device "cpu"` |
| **Testing a Model**      | `python cli.py --test_result --model "/path/to/saved_model.pth" --device "cuda"`                                              | `python cli.py --test_result --model "/path/to/saved_model.pth" --device "mps"`                                              | `python main.py --test_result --model "/path/to/saved_model.pth" --device "cpu"`                                              |
| **Single Image Inference** | `python inference.py --single_image --image "/path/to/image.jpg" --XtoY "/path/to/saved_model.pth --YtoX "/path/to/saved_model.pth" --device "cuda"`               | `python inference.py --single_image --image "/path/to/image.jpg" --XtoY "/path/to/saved_model.pth --YtoX "/path/to/saved_model.pth --device "mps"`               | `python inference.py --single_image --image "/path/to/image.jpg" --XtoY "/path/to/saved_model.pth --YtoX "/path/to/saved_model.pth --device "cpu"`               |
| **Batch Image Inference** | `python inference.py --batch_image --XtoY "/path/to/saved_model.pth --YtoX "/path/to/saved_model.pth --image_path "/path/to/dataset" --device "cuda"`                                            | `python inference.py --batch_image --XtoY "/path/to/saved_model.pth --YtoX "/path/to/saved_model.pth --image_path "/path/to/dataset" --device "mps"`                                           | `python inference.py --batch_image --XtoY "/path/to/saved_model.pth --YtoX "/path/to/saved_model.pth --image_path "/path/to/dataset" --device "cpu"`                                            |

### Notes:
- **CUDA Command**: For systems with NVIDIA GPUs, using the `cuda` device will leverage GPU acceleration.
- **MPS Command**: For Apple Silicon (M1, M2 chips), using the `mps` device can provide optimized performance.
- **CPU Command**: Suitable for systems without dedicated GPU support or for testing purposes on any machine.


#### Initializing Data Loader - Custom Modules
```python
loader = Loader(image_path="path/to/dataset", batch_size=1, image_size=256, split_size=0.2)
loader.unzip_folder()
loader.create_dataloader()
```

##### To details about dataset
```python
print(Loader.plot_images())
print(Loader.dataset_details())        # It will display the images from dataset
```

#### Training the Model
```python
trainer = Trainer(
    epochs=100,                # Number of epochs to train the model
    lr=0.0002,                 # Learning rate for optimizer      
    device='cuda',             # Computation device ('cuda', 'mps', 'cpu')
    adam=True,                 # Use Adam optimizer; set to False to use SGD if implemented
    SGD=False,                 # Use Stochastic Gradient Descent optimizer; typically False if Adam is True
    is_lr_scheduler=False,     # Enable a learning rate scheduler
    is_display=True            # Display training progress and statistics
)

# Start training
trainer.train()
```

##### Training Performances
```python
print(trainer.plot_history())    # It will plot the netD and netG losses for each epochs
```

#### Testing the Model
```python
test = TestModel(in_channels=3, device="cuda", create_gif=False) # use mps, cpu
test.plot()
```

#### Performing Inference
```python
infer = Inference(
    image="path/to/image.jpg",
    XtoY="path/to/model.pth",
    YtoX="path/to/model.pth",
    device="cuda",  # Use "mps" or "cpu"
    in_channels=3,
    image_size=256, # Must be same as the model input size
        )

infer.single_image()
```

#### Performing Inference - batch
```python

infer = Inference(
    image_path="path/to/dataset",
    XtoY="path/to/model.pth",
    YtoX="path/to/model.pth",
    device="cuda",  # Use "mps" or "cpu"
    in_channels=3,
    image_size=256, # Must be same as the model input size
        )

infer.batch_image()
```

## Contributing
Contributions to improve this implementation of CycleGAN are welcome. Please follow the standard fork-branch-pull request workflow.

## License
Specify the license under which the project is made available (e.g., MIT License).

