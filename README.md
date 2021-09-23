## Flight Delays Forecasting
**Work by Alexander Eponeshnikov**

### Task description
This task solves the problem of predicting the flight delay using different types of regressions.

### Project structure
* _main.py_ - main script file
* _data_preproc.py_ - module for data preprocessing
* _visualisation.py_ - module for visualize and generate figures (all figures stored in **plots** folder)
* _fit_test.py_ - module for fit and test models (all info about process of fit and test models stored in **testlog.txt)**
* _testlog.txt_ - logs about process of fit, evaluating and testing models. This file will override after execution **main.py**
* _flight_delays.csv_ - dataset
* _requirements.txt_ - requirements for running project

### How to install:

**Requirements:**
numpy, 
pandas, 
matplotlib, 
scikit-learn,
scipy

Run following commands in command line or terminal.

For Windows:
```
git clone https://github.com/Eponeshnikov/FlightDelaysForecasting.git
cd FlightDelaysForecasting
pip install -r requirements.txt 
```

For Mac and Linux:
```
git clone https://github.com/Eponeshnikov/FlightDelaysForecasting.git
cd FlightDelaysForecasting
pip3 install -r requirements.txt 
```

### How to run:
For Windows:
```
python main.py
```
For Mac and Linux
```
python3 main.py
```
### !!!Warning!!!
This script generates a lot of different models with different hyperparameters,
so runtime can be extremely long (about 3h on Ryzen 7). If you want to avoid this, then you need to reduce 
the number of models in the **main.py** file (78-83 lines).

