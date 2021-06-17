# CROP

Character Recognition Of Plates is a program desgined to detect and recognize character from license plate image

## Weights
- [Download weight](https://drive.google.com/drive/folders/1p3-IRQgjpCuM7ZQGUZp58MxH97rWdwby?usp=sharing)

## Datasets
- [License Plate Detection (custom anotation required)](https://www.kaggle.com/andrewmvd/car-plate-detection)
- [License Plate Character Detection](https://www.kaggle.com/thamizhsterio/indian-license-plates)
- Character Recognition [here](https://www.kaggle.com/kdnishanth/characterrecognitionfromnumberplate) & [here](https://www.kaggle.com/sahajap99/characters-dataset-for-license-plate-recognition)

## Enviornment
- Using Python 3.8.10
- ``` pip install -r requirements.txt``` for the rest of the packages

## CROP
- ### Arguments
  - source  : path of image (default is sample_car.png)
  - count   : the maximum number of possible plates to predict (default is 10)
- ### Execution
  - run ``` pythion crop.py ``` 

## Flask Crop App
- ### Arguments
  - source  : path of image (default is sample_car.png)
  - count   : the maximum number of possible plates to predict (default is 10)
- ### Execution
  - run ``` pythion flaskapp.py ``` 

## Contributors
- [XC4LIBUR](https://github.com/XC4LIBUR)
- [Royichan](https://github.com/Royichan)
- [deepak-delta](https://github.com/deepak-delta)