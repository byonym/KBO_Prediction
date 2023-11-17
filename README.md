# Big Contest 2020 -Team 10027- 

##### Team Members: Youngmok Byon, Kyunghwan Lee, Gibong Hong, Jinyoung Kim


## Analysis Environment and Packages Used

### 1. Analysis Environment

- OS: Windows  & MacOS & Linux-Ubuntu / RAM: 16GB

- Python == 3.7x version

- Pycharmproject / Jupyter Notebook 

  

### 2. Used Packages

- beautifulsoup4==4.9.1

- bs4==0.0.1

- joblib==0.16.0

- Keras==2.4.3

- lightgbm==3.0.0

- numpy==1.19.2

- pandas==1.1.2

- selenium==3.141.0

- scikit-learn==0.23.2

- tensorflow==2.3.0

  



## Process Instructions

0. Run (preprocess.py) and preprocess the raw_data to Input data for later use. (The py adds the features and the crawled data from 2020-07-21 ~ 2020-09-27)

1. Run train.py. (It will form the rnn model for the 1st phase and the ligthgbm model for the 2nd phase. It will save the MinMax and scaling model as well.)

2. Run predict.py (It will create prediction.csv in the result folder)

3. The final predcition values are the mean of 5 different prediction runs.


   
** After creating a Virtual Environment, install the pips in requirements.txt. (type 'pip install -r requirements.txt'!)

  #### ※How to set up a Virtual Environment

  ```bash
 #(For Linux)
 # Go to the directory where you will run the codes and make a Virtual Enviroment withe the name venv.
  python3 -m venv venv

  # Run the Virtual Environment 
  source bin/activate

  # After running the Virtual Environment, install the pips.
  pip3 -r install requirements.txt


  # ** How to deactivate the Virtual Environment
  source deactivate

  ```

![image](https://github.com/byonym/KBO_Prediction/assets/63856276/a9fb2bf4-305f-40e4-890a-b9943d587d21)

![image](https://github.com/byonym/KBO_Prediction/assets/63856276/c823d5eb-0ad3-48ac-91ae-f5904640b112)

![image](https://github.com/byonym/KBO_Prediction/assets/63856276/9a794645-2d2c-4055-93f6-e0250ae21228)











** The Korean players who were traded after July 19th and the foreign players who joined the team after July 19th were registered separately.

2020   99999  Palka   SS   
2020   99998  Russell WO      
2020   99997  White   SK   

2020   63950  Hyunshick Jang  HT   
2020   64984  Taejin Kim      HT   
2020   65643  Kyungchan Moon  NC   
2020   65639  Jungsoo Park    NC   
2020   63634  Honggu Lee      KT   
2020   60558  Taegon Oh       SK   



Copyright by Jin Young Kim, Young Mok Byon, Kyung Hwan Lee, Gi Bong Hong
