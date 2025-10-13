# ESC50-PANN-Model
An audio classification model built on PANN Cnn10 model and trained on the ESC50 dataset with 95% accuracy.



## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Model](#model)


## About

This project creates an audio classification CNN model. It uses the Cnn10 model from the [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) project's models, and trains it on the [ESC50](https://github.com/karolpiczak/ESC-50) dataset of sounds. This project was done for a university assignment by a second year CS student.



## Installation

Follow these steps to set up the project locally:

### 1. Enter the project directory

```bash
cd pythonProject
```

### 2. Clone the ESC50 Repository

```bash
git clone https://github.com/karolpiczak/ESC-50
```

### 3. Install requirements.txt

```bash
pip install -r requirements.txt
```

### 4. Run main.py to train model (can take up to 3 hours on CUDA)

```bash
python main.py
```


You might want to train the model yourself, if so, follow this step and run main.py. However, a trained model has already been saved along with all its checkpoints. These are saved in the `created_models` folder.

## Model
The model archictecture is entirely from the [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) project. I give full credit to the team for this model. To train it, this version of the model had its final classification layer changed to have 50 output nodes to match the 50 audio classes from ESC50. The trained model has an accuracy of 95% when tested on data from ESC50, however, it should be noted that the model is trained ONLY on ESC50. Therefore, the model may not perform well with samples outside of ESC50.

<img width="1536" height="754" alt="confusion_matrix" src="https://github.com/user-attachments/assets/d957f9d0-28a1-4457-bd76-0a4070161014" />
<img width="640" height="476" alt="training_accuracy_loss" src="https://github.com/user-attachments/assets/e6b4033b-6a7c-4767-acc0-d83d132537cc" />



