# Sport Classification
Sport Classification is a machine learning project aimed at classifying sports activities based on titles and perexs of czech sports articles. It is developed in python and is aimed to compare a range of classifier types.


## Installation

1. Clone the repository and navigate to it:
```bash
git clone https://github.com/lapunik/sport_classification.git
cd sport_classification   
```

2. Set up a Python environment:
```bash
python -m venv venv
venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
or if you have no gpu:
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio
```

## Features
* Data Preprocessing: Tools for cleaning and normalizing input data.
* Model Training: Pre-built scripts for training deep learning models.
* Testing: Functions to evaluate model performance.
* Extensibility: Modular codebase for adding new models or datasets.
* Training data is not included in the project.

  
## Project Structure

```models/```: Contains pre-trained and custom models.  
```utils/```: Utility scripts for data manipulation.  
```data/```: Space for your training dataset.  
```saved_models/```: Computed model for using.  
```train.py```: Script for training models.  
```test.py```: Testing and evaluation script.  
