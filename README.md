1. Description

A car recommendation app which takes car features as user input and recommends cars based on it.

2. How to run

(i) Install docker cli, if not already installed.

(ii) Then run

docker build -t car_recommendation .

(iii) Then run

docker run -d --name car_recommendation -p 10000:10000 --rm car_recommendation

(iv) Open the http://localhost:10000/ url.

3. Dataset

The train.csv dataset for training used in this project is taken from kaggle whose link is given below:

https://www.kaggle.com/datasets/sidharth178/car-prices-dataset
