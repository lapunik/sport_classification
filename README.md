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

## Usage

The main.py file serves as the primary entry point for the program. It allows users to select from a range of available models (```available_models```) and vectorizers (```available_preprocesors```) for text data processing. For the specific model ```"bert"```, the selection of preprocessor is not relevant as the model's tokenizer is used by default. Except for bert, models are not included in the repository because of the size.
### Configuration

* **Model**: Select one of the available names for the required model.
* **Preprocesor**: Select one of the available vectorizer names for preprocessing text data. 
* **Data**:  Provide the name of the CSV file containing the training data in the ```/data``` folder. The file must contain category, title and perex columns see below.
* **Retrain**: Set to ```False``` if you want to use the pre-trained model stored in the ```/saved_models``` folder. If the pre-trained model is not available, the model will be trained from training data.

### CSV format:

Column __category__: Label of the sport to which this entry relates.  
Column __title__: Title of the article.  
Column __perex__: Perex of the article.

### For example:

```python
model = "mlp"
preprocesor = "tfidf"
data = "sportoclanky"
retrain = False

title = "Pohár konstruktérů vyhrává McLaren, pro vítězství si v Abú Zabí dojel Norris"
perex = "McLaren si zásluhou Landa Norrise dojel v Abú Zabí pro vítězství v letošním Poháru konstruktérů formule 1. Norris vyrážel z pole position a první místo udržel po dobu trvání celého závodu. Naopak jeho týmový kolega Oscar Piastri se po kolizi v prvním kole propadl a dojel desátý. Druhý dojel Carlos Sainz a třetí Charles Leclerc, ale Ferrari to na zisk týmového titulu nestačilo."
```


## Features
* Data Preprocessing: Tools for cleaning and preparing input data.
* Model Training: Pre-built scripts for training deep learning models.
* Testing: Functions to evaluate model performance.
* Extensibility: Modular codebase for adding new models or datasets.
* Training data is not included in the project.

  
## Project Structure

```models/modles.py```: Contains pre-trained and custom models.     
```utils/```: Utility scripts for data manipulation.   
```utils/data_loader```: Provides the ability to load training and test datasets.    
```utils/data_processor```: Focuses on data pre-processing for machine learning tasks. It performs operations like tokenization, vectorization, and any necessary data cleaning to prepare the data for training models.  
```utils/predictor```: Allows test the model and report the success of the training. It also provides a function for predicting new data from already trained models
```utils/transformer_data_format```: Defines the data format and training arguments for training a transformer-based model (BERT). Includes a TextDataset class for processing text data in a format suitable for training and specifies the training arguments for this model.  
```utils/cls```: Contains a list of Czech stop words that are used for preprocessing text using vectorizers. Stop words are common words that are often removed from text data before training models because they can add noise and do not contribute significantly to the model learning process.  
```data/```: Space for your training dataset.  
```saved_models/```: Computed model for using.   
```train.py```: Script for training models.  
```test.py```: Testing and evaluation script.  
