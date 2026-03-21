1. Description

A spam email detection app which takes a feature as user input and predicts whether the email is spam or not based on the training dataset.

2. How to run

(i) Install docker cli, if not already installed.

(ii) Then run

docker build -t spam_email_detection .

(iii) Then run

docker run -d --name spam_email_detection -p 10000:10000 --rm spam_email_detection

(iv) Open the http://localhost:10000/ url.

3. Dataset

The train.csv dataset for training used in this project is taken from kaggle whose link is given below:

https://www.kaggle.com/datasets/sidharth178/car-prices-dataset
