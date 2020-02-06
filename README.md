### Overview
We have selected a dataset of the H1B cycles from 2011 to 2017. There are various columns in the dataset, like Case Status, Employer name, Employer state, job title, etc. We have analyzed this dataset to find interesting insights and come up with a model that can be used to predict the outcomes for H-1B applications.

#### Motivation
There is a lot of uncertainity on H-1B apllications and with recent developments this has only increased. We wanted to explore what it takes for a successful H-1B application.

#### Approach
<b>HDFS</b> has been used to store our data file, <b>PySpark</b> is used for analyzing and modelling the data and <b>Matplotlib</b> has been used for visualization.
    We model our data using the following machine learning algorithms.
    1. Naive Bayes
    2. Logistic Regression
    3. Decision Tree Classifier
    4. Random Forest Classifier

#### Data
The original data set came from Kaggle. https://www.kaggle.com/jonamjar/h1b-data-set-2017/data\

Since, the data was huge about 600000 rows we decided to take a subset based on year. We took the data of 2016 (about 150000 rows) and did our analysis on that. The same can be found here https://drive.google.com/file/d/1uh_sQR-DTVAjeiwGP10g-qGt8Th1iHMd/view?usp=sharing"
