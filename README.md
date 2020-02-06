{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Overview\n",
    "\n",
    "We have selected a dataset of the H1B cycles from 2011 to 2017. There are various columns in the dataset, like Case Status, Employer name, Employer state, job title, etc. We have analyzed this dataset to find interesting insights and come up with a model that can be used to predict the outcomes for H-1B applications.\n",
    "\n",
    "#### Motivation\n",
    "There is a lot of uncertainity on H-1B apllications and with recent developments this has only increased. We wanted to explore what it takes for a successful H-1B application. \n",
    "\n",
    "#### Approach\n",
    "<b>HDFS</b> has been used to store our data file, <b>PySpark</b> is used for analyzing and modelling the data and <b>Matplotlib</b> has been used for visualization.\n",
    "We model our data using the following machine learning algorithms.\n",
    "1. Naive Bayes\n",
    "2. Logistic Regression\n",
    "3. Decision Tree Classifier\n",
    "4. Random Forest Classifier \n",
    "\n",
    "#### Data\n",
    "The original data set came from Kaggle. https://www.kaggle.com/jonamjar/h1b-data-set-2017/data\n",
    "\n",
    "Since, the data was huge about 600000 rows we decided to take a subset based on year. We took the data of 2016 (about 150000 rows) and did our analysis on that. The same can be found here https://drive.google.com/file/d/1uh_sQR-DTVAjeiwGP10g-qGt8Th1iHMd/view?usp=sharing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Import required packages"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from pyspark import SparkConf, SparkContext\n",
    "from pyspark.sql import Row\n",
    "from pyspark.sql import SQLContext\n",
    "from pyspark.ml.feature import StringIndexer, OneHotEncoder\n",
    "from pyspark.mllib.regression import LabeledPoint\n",
    "from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel\n",
    "from pyspark.mllib.classification import NaiveBayes\n",
    "from pyspark.mllib.tree import RandomForest, RandomForestModel\n",
    "from pyspark.mllib.tree import DecisionTree, DecisionTreeModel\n",
    "from pyspark.mllib.util import MLUtils\n",
    "from time import time\n",
    "import re\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Create configs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "conf = SparkConf().setMaster(\"local\").setAppName(\"Project_BigData2\")\n",
    "sc = SparkContext.getOrCreate(conf = conf)\n",
    "sqlcon = SQLContext.getOrCreate(sc)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Some global variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "pat1 = re.compile(r'\"([a-xA-Z0-9. ,]+), ([a-xA-Z0-9. ,]+)\"')\n",
    "pat2 = re.compile(r'\"([,]+)\"')\n",
    "keep = ['CASE_STATUS','EMPLOYER_NAME','EMPLOYER_STATE',\\\n",
    "        'AGENT_REPRESENTING_EMPLOYER','JOB_TITLE','SOC_NAME','NAICS_CODE','TOTAL_WORKERS',\\\n",
    "        'NEW_EMPLOYMENT','CONTINUED_EMPLOYMENT','CHANGE_PREVIOUS_EMPLOYMENT',\\\n",
    "        'NEW_CONCURRENT_EMPLOYMENT','CHANGE_EMPLOYER','AMENDED_PETITION','FULL_TIME_POSITION',\\\n",
    "        'PREVAILING_WAGE','H1B_DEPENDENT','SUPPORT_H1B','WORKSITE_STATE','CASE_SUBMITTED']\n",
    "#We need to decide how to use the date 'CASE_SUBMITTED', add back to the list above once decided.\n",
    "categorical = {\"CASE_STATUS\":\"CASE_STATUS_C\", \"EMPLOYER_NAME\":\"EMPLOYER_NAME_C\",\\\n",
    "              \"EMPLOYER_STATE\":\"EMPLOYER_STATE_C\",\"AGENT_REPRESENTING_EMPLOYER\":\"AGENT_REPRESENTING_EMPLOYER_C\",\\\n",
    "              \"JOB_TITLE\":\"JOB_TITLE_C\",\"SOC_NAME\":\"SOC_NAME_C\",\"NAICS_CODE\":\"NAICS_CODE_C\",\\\n",
    "              \"FULL_TIME_POSITION\":\"FULL_TIME_POSITION_C\",\"H1B_DEPENDENT\":\"H1B_DEPENDENT_C\",\\\n",
    "              \"WORKSITE_STATE\":\"WORKSITE_STATE_C\",\"SUPPORT_H1B\":\"SUPPORT_H1B_C\"}\n",
    "target = [\"CASE_STATUS_C\",\"CASE_SUBMITTED\"]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data Preparation\n",
    "Involves several methods to cleanup data -\n",
    "1. Remove commas within quotes strings\n",
    "2. Convert wage to yearly wage\n",
    "3. Create Rows for the dataframe\n",
    "3. Convert categorical variables to codes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def replaceCommaWithinQuotes(line):\n",
    "    '''\n",
    "    Remove commas within quotes with some words, recursion makes sure we reomve all such commas\n",
    "    '''\n",
    "    if len(pat1.findall(line)) == 0:\n",
    "        return line\n",
    "    line = pat1.sub( r'\"\\1 \\2\"', line )\n",
    "    return replaceCommaWithinQuotes(line)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def yearlyWage(data_dict):\n",
    "    '''\n",
    "    Converts the prevailing wage to yearly.\n",
    "    '''\n",
    "    sal = 0.0\n",
    "    try:\n",
    "        sal = float(data_dict['PREVAILING_WAGE'])\n",
    "    except:\n",
    "        sal = 0.0\n",
    "        #do nothing\n",
    "    if data_dict['PW_UNIT_OF_PAY'] == 'Hour':\n",
    "        sal = sal*40*52 #40 hrs/week,52 weeks/yr\n",
    "    if data_dict['PW_UNIT_OF_PAY'] == 'Week':\n",
    "        sal = sal*52    #52 weeks/yr\n",
    "    if data_dict['PW_UNIT_OF_PAY'] == 'Bi-Weekly':\n",
    "        sal = sal*26    #52 weeks/yr,hence 26 bi-weeks \n",
    "    if data_dict['PW_UNIT_OF_PAY'] == 'Month':\n",
    "        sal = sal*12    #12 months/yr\n",
    "    if data_dict['PW_UNIT_OF_PAY'] == 'Year':\n",
    "        sal == sal\n",
    "    data_dict['PREVAILING_WAGE'] = sal\n",
    "    return data_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def createRow(line, headers):\n",
    "    '''\n",
    "    Returns a dictionary with headers as key and their values as the values\n",
    "    '''\n",
    "    data_dict = {}\n",
    "    #Replace comma, within words between two quotes, with blank\n",
    "    line = replaceCommaWithinQuotes(line) #This line may still have just comma within quotes-\",\"\n",
    "    #line = pat2.sub(r'\"\"', line) - something weird is happening because of this line\n",
    "\n",
    "    data_list = line.split(\",\")\n",
    "    j = 0 #another index\n",
    "    for i in range(len(headers)):\n",
    "        if data_list[j] == '\"': #In case we encounter a \" we avoid it and move ahead.\n",
    "            j = j+1\n",
    "        if headers[i] == \"\":\n",
    "            data_dict[\"S_NO\"] = int(data_list[j])\n",
    "        else:\n",
    "            data_dict[headers[i]] = data_list[j]\n",
    "        j = j+1\n",
    "    #We make the prevailing wage yearly\n",
    "    data_dict = yearlyWage(data_dict)\n",
    "    date_submitted = data_dict[\"CASE_SUBMITTED\"]\n",
    "    data_dict[\"CASE_SUBMITTED\"] = date_submitted[-4:]\n",
    "    return data_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def convertToCategorical(columns, dataframe):\n",
    "    '''\n",
    "    Converts each column in dataframe to its corresponding Cateforical column.\n",
    "    columns is a dict representing the column as key and new colmn as value.\n",
    "    '''\n",
    "    for column in columns:\n",
    "        indexer = StringIndexer(inputCol=column, outputCol=columns[column])\n",
    "        dataframe = indexer.fit(dataframe).transform(dataframe)\n",
    "    return dataframe\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def createLabeledPoints(line):\n",
    "    '''\n",
    "    Converts each value to a float and returns as LabeledPoint.\n",
    "    If conversion fails then null is returned.\n",
    "    '''\n",
    "    values = []\n",
    "    for x in line:\n",
    "        try:\n",
    "            values.append(float(x))\n",
    "        except:\n",
    "            return None\n",
    "    return LabeledPoint(values[0], values[1:])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Read the data from HDFS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "h1b_data = sc.textFile(\\\n",
    "           \"hdfs://quickstart.cloudera:8020/user/cloudera/H-1B_Selected.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Extract headers from the data.\n",
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "headers_string = h1b_data.take(1)[0]\n",
    "headers = headers_string.split(\",\")\n",
    "h1b_data = h1b_data.filter(lambda x: x != headers_string)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Create the map and then dataframe from the map using sql context"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": true,
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "h1_data_map = h1b_data.map(lambda x: Row(**createRow(x, headers)))\n",
    "h1b_data_frame = sqlcon.createDataFrame(h1_data_map).cache()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Filter Data based on visa type, we need only H1B visas."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "h1b_data_frame = h1b_data_frame.where(h1b_data_frame['VISA_CLASS'] == 'H-1B')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[Row(AGENT_ATTORNEY_CITY=u'NEW YORK', AGENT_ATTORNEY_NAME=u'ELLSWORTH CHAD', AGENT_ATTORNEY_STATE=u'NY', AGENT_REPRESENTING_EMPLOYER=u'Y', AMENDED_PETITION=u'0', CASE_NUMBER=u'I-200-16055-173457', CASE_STATUS=u'CERTIFIED-WITHDRAWN', CASE_SUBMITTED=u'2016', CHANGE_EMPLOYER=u'0', CHANGE_PREVIOUS_EMPLOYMENT=u'0', CONTINUED_EMPLOYMENT=u'0', DECISION_DATE=u'10/1/2016', EMPLOYER_ADDRESS=u'2500 LAKE COOK ROAD', EMPLOYER_BUSINESS_DBA=u'', EMPLOYER_CITY=u'RIVERWOODS', EMPLOYER_COUNTRY=u'UNITED STATES OF AMERICA', EMPLOYER_NAME=u'DISCOVER PRODUCTS INC.', EMPLOYER_PHONE=u'2244050900', EMPLOYER_PHONE_EXT=u'', EMPLOYER_POSTAL_CODE=u'60015', EMPLOYER_PROVINCE=u'', EMPLOYER_STATE=u'IL', EMPLOYMENT_END_DATE=u'8/10/2019', EMPLOYMENT_START_DATE=u'8/10/2016', FULL_TIME_POSITION=u'Y', H1B_DEPENDENT=u'N', JOB_TITLE=u'ASSOCIATE DATA INTEGRATION', LABOR_CON_AGREE=u'Y', NAICS_CODE=u'522210', NEW_CONCURRENT_EMPLOYMENT=u'0', NEW_EMPLOYMENT=u'1', ORIGINAL_CERT_DATE=u'3/1/2016', PREVAILING_WAGE=59197.0, PUBLIC_DISCLOSURE_LOCATION=u'', PW_SOURCE=u'OES', PW_SOURCE_OTHER=u'OFLC ONLINE DATA CENTER', PW_SOURCE_YEAR=u'2015', PW_UNIT_OF_PAY=u'Year', PW_WAGE_LEVEL=u'Level I', SOC_CODE=u'15-1121', SOC_NAME=u'COMPUTER SYSTEMS ANALYSTS', SUPPORT_H1B=u'', S_NO=0, TOTAL_WORKERS=u'1', VISA_CLASS=u'H-1B', WAGE_RATE_OF_PAY_FROM=u'65811', WAGE_RATE_OF_PAY_TO=u'67320', WAGE_UNIT_OF_PAY=u'Year', WILLFUL_VIOLATOR=u'N', WORKSITE_CITY=u'RIVERWOODS', WORKSITE_COUNTY=u'LAKE', WORKSITE_POSTAL_CODE=u'60015', WORKSITE_STATE=u'IL'),\n",
       " Row(AGENT_ATTORNEY_CITY=u'NEW YORK', AGENT_ATTORNEY_NAME=u'ELLSWORTH CHAD', AGENT_ATTORNEY_STATE=u'NY', AGENT_REPRESENTING_EMPLOYER=u'Y', AMENDED_PETITION=u'0', CASE_NUMBER=u'I-200-16064-557834', CASE_STATUS=u'CERTIFIED-WITHDRAWN', CASE_SUBMITTED=u'2016', CHANGE_EMPLOYER=u'0', CHANGE_PREVIOUS_EMPLOYMENT=u'0', CONTINUED_EMPLOYMENT=u'0', DECISION_DATE=u'10/1/2016', EMPLOYER_ADDRESS=u'2500 LAKE COOK ROAD', EMPLOYER_BUSINESS_DBA=u'', EMPLOYER_CITY=u'RIVERWOODS', EMPLOYER_COUNTRY=u'UNITED STATES OF AMERICA', EMPLOYER_NAME=u'DFS SERVICES LLC', EMPLOYER_PHONE=u'2244050900', EMPLOYER_PHONE_EXT=u'', EMPLOYER_POSTAL_CODE=u'60015', EMPLOYER_PROVINCE=u'', EMPLOYER_STATE=u'IL', EMPLOYMENT_END_DATE=u'8/16/2019', EMPLOYMENT_START_DATE=u'8/16/2016', FULL_TIME_POSITION=u'Y', H1B_DEPENDENT=u'N', JOB_TITLE=u'SENIOR ASSOCIATE', LABOR_CON_AGREE=u'Y', NAICS_CODE=u'522210', NEW_CONCURRENT_EMPLOYMENT=u'0', NEW_EMPLOYMENT=u'1', ORIGINAL_CERT_DATE=u'3/8/2016', PREVAILING_WAGE=49800.0, PUBLIC_DISCLOSURE_LOCATION=u'', PW_SOURCE=u'Other', PW_SOURCE_OTHER=u'TOWERS WATSON DATA SERVICES 2015 CSR PROFESSIONAL (ADMINISTRATIVE AND SALES) C', PW_SOURCE_YEAR=u'2015', PW_UNIT_OF_PAY=u'Year', PW_WAGE_LEVEL=u'', SOC_CODE=u'15-2031', SOC_NAME=u'OPERATIONS RESEARCH ANALYSTS', SUPPORT_H1B=u'', S_NO=1, TOTAL_WORKERS=u'1', VISA_CLASS=u'H-1B', WAGE_RATE_OF_PAY_FROM=u'53000', WAGE_RATE_OF_PAY_TO=u'57200', WAGE_UNIT_OF_PAY=u'Year', WILLFUL_VIOLATOR=u'N', WORKSITE_CITY=u'RIVERWOODS', WORKSITE_COUNTY=u'LAKE', WORKSITE_POSTAL_CODE=u'60015', WORKSITE_STATE=u'IL')]"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "h1b_data_frame.take(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Take a small subset and convert to pandas just to show the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>AGENT_ATTORNEY_CITY</th>\n",
       "      <th>AGENT_ATTORNEY_NAME</th>\n",
       "      <th>AGENT_ATTORNEY_STATE</th>\n",
       "      <th>AGENT_REPRESENTING_EMPLOYER</th>\n",
       "      <th>AMENDED_PETITION</th>\n",
       "      <th>CASE_NUMBER</th>\n",
       "      <th>CASE_STATUS</th>\n",
       "      <th>CASE_SUBMITTED</th>\n",
       "      <th>CHANGE_EMPLOYER</th>\n",
       "      <th>CHANGE_PREVIOUS_EMPLOYMENT</th>\n",
       "      <th>...</th>\n",
       "      <th>TOTAL_WORKERS</th>\n",
       "      <th>VISA_CLASS</th>\n",
       "      <th>WAGE_RATE_OF_PAY_FROM</th>\n",
       "      <th>WAGE_RATE_OF_PAY_TO</th>\n",
       "      <th>WAGE_UNIT_OF_PAY</th>\n",
       "      <th>WILLFUL_VIOLATOR</th>\n",
       "      <th>WORKSITE_CITY</th>\n",
       "      <th>WORKSITE_COUNTY</th>\n",
       "      <th>WORKSITE_POSTAL_CODE</th>\n",
       "      <th>WORKSITE_STATE</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>NEW YORK</td>\n",
       "      <td>ELLSWORTH CHAD</td>\n",
       "      <td>NY</td>\n",
       "      <td>Y</td>\n",
       "      <td>0</td>\n",
       "      <td>I-200-16055-173457</td>\n",
       "      <td>CERTIFIED-WITHDRAWN</td>\n",
       "      <td>2016</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>65811</td>\n",
       "      <td>67320</td>\n",
       "      <td>Year</td>\n",
       "      <td>N</td>\n",
       "      <td>RIVERWOODS</td>\n",
       "      <td>LAKE</td>\n",
       "      <td>60015</td>\n",
       "      <td>IL</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>NEW YORK</td>\n",
       "      <td>ELLSWORTH CHAD</td>\n",
       "      <td>NY</td>\n",
       "      <td>Y</td>\n",
       "      <td>0</td>\n",
       "      <td>I-200-16064-557834</td>\n",
       "      <td>CERTIFIED-WITHDRAWN</td>\n",
       "      <td>2016</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>53000</td>\n",
       "      <td>57200</td>\n",
       "      <td>Year</td>\n",
       "      <td>N</td>\n",
       "      <td>RIVERWOODS</td>\n",
       "      <td>LAKE</td>\n",
       "      <td>60015</td>\n",
       "      <td>IL</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>WASHINGTON</td>\n",
       "      <td>BURKE KAREN</td>\n",
       "      <td>DC</td>\n",
       "      <td>Y</td>\n",
       "      <td>0</td>\n",
       "      <td>I-200-16063-996093</td>\n",
       "      <td>CERTIFIED-WITHDRAWN</td>\n",
       "      <td>2016</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>2</td>\n",
       "      <td>H-1B</td>\n",
       "      <td>77000</td>\n",
       "      <td>0</td>\n",
       "      <td>Year</td>\n",
       "      <td>N</td>\n",
       "      <td>WASHINGTON</td>\n",
       "      <td></td>\n",
       "      <td>20007</td>\n",
       "      <td>DC</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3 rows × 53 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "  AGENT_ATTORNEY_CITY AGENT_ATTORNEY_NAME AGENT_ATTORNEY_STATE  \\\n",
       "0            NEW YORK      ELLSWORTH CHAD                   NY   \n",
       "1            NEW YORK      ELLSWORTH CHAD                   NY   \n",
       "2          WASHINGTON         BURKE KAREN                   DC   \n",
       "\n",
       "  AGENT_REPRESENTING_EMPLOYER AMENDED_PETITION         CASE_NUMBER  \\\n",
       "0                           Y                0  I-200-16055-173457   \n",
       "1                           Y                0  I-200-16064-557834   \n",
       "2                           Y                0  I-200-16063-996093   \n",
       "\n",
       "           CASE_STATUS CASE_SUBMITTED CHANGE_EMPLOYER  \\\n",
       "0  CERTIFIED-WITHDRAWN           2016               0   \n",
       "1  CERTIFIED-WITHDRAWN           2016               0   \n",
       "2  CERTIFIED-WITHDRAWN           2016               0   \n",
       "\n",
       "  CHANGE_PREVIOUS_EMPLOYMENT      ...       TOTAL_WORKERS VISA_CLASS  \\\n",
       "0                          0      ...                   1       H-1B   \n",
       "1                          0      ...                   1       H-1B   \n",
       "2                          0      ...                   2       H-1B   \n",
       "\n",
       "  WAGE_RATE_OF_PAY_FROM WAGE_RATE_OF_PAY_TO WAGE_UNIT_OF_PAY WILLFUL_VIOLATOR  \\\n",
       "0                 65811               67320             Year                N   \n",
       "1                 53000               57200             Year                N   \n",
       "2                 77000                   0             Year                N   \n",
       "\n",
       "  WORKSITE_CITY WORKSITE_COUNTY WORKSITE_POSTAL_CODE WORKSITE_STATE  \n",
       "0    RIVERWOODS            LAKE                60015             IL  \n",
       "1    RIVERWOODS            LAKE                60015             IL  \n",
       "2    WASHINGTON                                20007             DC  \n",
       "\n",
       "[3 rows x 53 columns]"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "h1b_data_frame_0 = h1b_data_frame.where(h1b_data_frame['S_NO'] < 3 )\n",
    "h1b_data_frame_0.toPandas()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Convert required columns to categorical variables."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "h1b_data_frame_1 = convertToCategorical(categorical, h1b_data_frame)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Keep columns that we are going to work with in a new dataframe, we remove the old columns that were converted to categorical since we need only the categorical version of those columns."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "keeps = [x for x in keep if x not in categorical.keys() ] + categorical.values()\n",
    "keeps = target + [x for x in keeps if x not in target]\n",
    "h1b_data_frame_2 = h1b_data_frame_1[keeps].cache()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Convert to LabeledPoints which  is required for  modelling of the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "h1b_labeled_points = h1b_data_frame_2.map(createLabeledPoints)\n",
    "h1b_labeled_points = h1b_labeled_points.filter(lambda x : x != None)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Going ahead....\n",
    "h1b_data_frame  can be used for visualization purposes                               \n",
    "h1b_data_frame_1 can be used to find categories and their respective codes          \n",
    "h1b_data_frame_2 and h1b_labeled_points can be used for modelling purposes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def barPlot(dataframe, color=\"Blue\"):\n",
    "    \"\"\"\n",
    "    Takes a dataframe with two columns and creates bar plot.\n",
    "    The first column is the labels and second is the count.\n",
    "    The dataframe is a pyspark dataframe\n",
    "    \"\"\"\n",
    "    X = []\n",
    "    Y = []\n",
    "    for x in dataframe.collect():\n",
    "        X.append(x[0])\n",
    "        Y.append(x[1])\n",
    "    pos = np.arange(len(X))\n",
    "    \n",
    "    plt.barh(pos, Y, align='center', color=color)\n",
    "    plt.yticks(pos, X)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 1. Company-wise status."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def top_10_companies_by_status(h1b_data_frame, status):\n",
    "    status_by_company = h1b_data_frame.where(h1b_data_frame['CASE_STATUS'] == status)\n",
    "    #1. Map by Employer name\n",
    "    #2. Reduce by count\n",
    "    #3. Map with count as key, employer as value this makes the sorting easier as you can sortByKey\n",
    "    status_by_company = status_by_company.map(lambda x : (x.EMPLOYER_NAME, 1)).\\\n",
    "                    reduceByKey(lambda x,y : x+y).map(lambda x : (x[1],x[0]))\n",
    "    status_by_company_sorted = status_by_company.sortByKey(ascending=False)\n",
    "    status_by_company_sorted_10 = status_by_company_sorted.toDF().limit(10)\n",
    "    status_by_company_sorted_10 = status_by_company_sorted_10.map(lambda x : (x[1],x[0])).toDF()\n",
    "    return status_by_company_sorted_10"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here we look at the top 10 companies with most denied status."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f9264020e90>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Top 10 companies with denied status\n",
    "barPlot(top_10_companies_by_status(h1b_data_frame, \"DENIED\"))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here we look at the top 10 companies with most certified status."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAi0AAAD8CAYAAAC7FJTRAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAIABJREFUeJzt3Xvc5XO9///H0zETlVMyqCmkDBpzzbeDKKQiSgdiymb2\n1lY72qWIoqiU5JhDh3FItR03KUk6mo0fO64Zlzk5h0IOw04GofH8/fF5L30sa13Xuk4z1xrP++22\nbtdnvT/vz/vz+ryvy6yX9/v9WR/ZJiIiImKsW2ZJBxARERHRiSQtERER0RWStERERERXSNISERER\nXSFJS0RERHSFJC0RERHRFZK0RERERFdI0hIRERFdIUlLREREdIXllnQAEUuTNdZYwxMmTFjSYURE\ndJWZM2cusL3mQPWStESMoAkTJtDb27ukw4iI6CqS7uqkXqaHIiIioiskaYmIiIiukKQlIiIiukKS\nloiIiOgKSVoiIiKiKyRpiYiIiK6QpCUiIiK6QpKWiIiI6AqyvaRjiFhqaLzMx5d0FGOXD8u/NxHx\nfJJm2p4yUL2MtERERERXSNISERERXSFJS0RERHSFJC0RERHRFZK0DIGkQyTNkzRbUp+kN0n6uqSj\nanVeJemPkl4maYak3tq+KaXs3eX4PkkLJd1ctn8kaWtJj9T290narhy/sIMYd5DUK2m+pOslHVvb\nt4+km8rrWklb1va1jLVsj5N0lqQ5kuZKukrSypImSJrbdP7DJR1Qts+UtEtt36TaNT0s6Y6y/StJ\nG0jqK/W2k2RJO9SOvawRr6TlJX1T0m3l+OslHdymP+4uv4vlSpv139XBkg6tvZ9Wrm+OpFmS9h+o\nvyMiYvQtt6QD6DaS3gLsBEy2/aSkNYAVgCOAPkln2r4R+DbwJdt/lQTwckk72P5loy3bvwJ+Vdqd\nARxgu7e83xq40vZOQ4hxE+BkYEfbN0laFtin7NsJ+Diwpe0FkiYDP5X0Rtv3lSaeF2vxaeB+25uW\ntjYCnh5sfLb7gEmljf8CLrD90/J+g6bqfwYOAZpjATgSWBWYWH4XqwCf7SCEJ4APSzrK9sP1HaV/\n9gO2s32fpBcBe3R+dRERMVoy0jJ4awMLbD8JYHuB7XttPwHsD5wi6T3AKrbPqh13NNWH7+LweeDr\ntm8qMS6y/d2y7yDgQNsLyr5ZwA+BfTuIdW3gnsYb2zc3+mEUzQKelLRNvbAkKHsB/1n7XTxq+ysd\ntPkUcAZVEtbsi8BnGwmc7b/bPm04FxARESMjScvg/RpYT9Itkr4j6e2NHbYvBf6PKgn4ZNNx1wBP\nNX/4DmCrpumh9Ts8bhNgZpt9E1vs6y3lA8V6BnCQpGskHSFpww7jGa6vA4c2lW0I3Gn7sSG2eRKw\nV0l+6lr1T7/KdFuvpF4eH2I0ERExoCQtg2R7IdBDNd3yIHCepGm1KqcA19m+ucXhR/D8D9/+XGl7\nUu11+1DjHoLnxVqmdV5DNRKzGnCdpNcD7b4xbES+Scz274GVJL25XR1JHyuJ3d2S1u6gzb8CZ1NN\nBQ03vum2p9iewrjhthYREe0kaRmCMt0yw/ZhVB96H6rtfqa8Wh33e2AloO2H7wiZR5VYtTK/xb6e\ncsyz2sVqe6Htn9j+JPBfwHuAh6jWltStBiwYUvStNSdRtwKvlvTiEtdpticBC4FlO2zzOKrks55q\ntOqfiIgYA5K0DJKkjZqmRSYBdw2iiSOo1pyMpqOBL0p6LYCkZSR9ouz7FnCUpNXLvknANOA7A8Uq\n6a2SVi3bKwAbA3eV0ae/SNq27FsN2B64aqQuqEy9vYIyjWX7UeBHwImSViznXQ5YfhBtLgAuorr+\nhiOBYyStVdpcUdLeI3ENERExPLl7aPBWBk6S9DLgH8BtlDtzOmH7UkkPdlh9q8btv8URti8Axkm6\nu1Z+nO3jaueYLekzwDmSxlFN01xS9l0saR3gakkGHgX2sP2XDmJdH/iuqtuhlgF+AVxY9u1JtQi5\nEcdXmqazvi/phLL9Z9tv6bAP6r5ROx/AwVSJ1XxJfwMeB04D7h9Em0dTW39U+mdN4Pflri8DpwJI\n2hd4MgtzIyKWjDwwMWIE5YGJ/csDEyOiFeWBiREREbE0SdISERERXSFrWiJGUM/4HnoP6x24YkRE\nDFpGWiIiIqIrJGmJiIiIrpCkJSIiIrpCbnmOGEG55Xlgue05IprllueIiIhYqiRpiYiIiK6QpCUi\nIiK6QpKWiIiI6ApJWgJJx5cHLDbe/0rSabX3x0r6rKQJkuaWsq0lPSKpT9KNkg5rUX6TpGOazvV+\nSbPLMXMkvb9NTIdLOqBF+cLyc4IkSzqitm8NSU9LOrnehqRTSjzzJT1Rtvsk7SLpTEl31MquLsdO\nk/SgpOsl3Vr6ZIvh9HNERAxPkpYA+P+ALQAkLQOsAUys7d8CuLrFcVfangRMAfaQNLmpfHNgJ0lv\nLW2/ATgG2Nn264H3AcdI2myIcd8B7Fh7vyswr7mS7X1LPO8Bbrc9qbwuKFUOrJXVE5PzbG9ue0Pg\nm8BPJL1+iLFGRMQwJWkJqBKSt5TticBc4FFJq0paEXg9MKvdwbYfA2YCGzSVPwH0AeuUogOAb9i+\no+y/AzgSOHCIcT8O3CipcZvcbsD5Q2yrX7YvB6YD+4xG+xERMbAkLYHte4F/SHol1ajKNcAfqBKZ\nKcAc20+1O17S6sCbaRrlkLQqsCFwRSmaSJXc1PXy3FGdwToX2F3SesAi4N4htHF0bXrorH7qzQJe\n11woaR9JvZJ6eXwIZ4+IiI7kgYnRcDVVwrIFcBzV6MgWwCNU00etbCXpeuAZ4Ju250naupTfQJWw\nnGD7vlGM+zLga8D9wHlDbOPA2lRRf9Sq0PZ0qlGY6svlIiJiVGSkJRoa61o2pZoe+l+qkZZ261mg\nWruyue0e299rKn8D1QjK3pImlfL5QE9TGz20WIfSqTICNBP4HNBJ4jEcmwM3jvI5IiKijSQt0XA1\nsBPwsO1Fth8GXkaVuLRLWvpV1qx8EzioFB0DfEHSBKjuAAK+CBw7jLgpxx9UYh4Vkt5OtZ7l1NE6\nR0RE9C/TQ9Ewh+quobObyla2vWAY7X4POEDSBNt9kg4Cfi5peeBp4PO2+9oce2j9Vmzb67aqZHse\nwxitoVrTcmjt/RvLz90kbQmMo7pT6UO2M9ISEbGE5IGJESMoD0wcWB6YGBHN8sDEiIiIWKokaYmI\niIiukKQlIiIiukIW4kaMoJ7xPfQe1rukw4iIWCplpCUiIiK6QpKWiIiI6ApJWiIiIqIr5HtaIkZQ\nvqelc/m+lohoyPe0RERExFIlSUtERER0hSQtERER0RWStLQgaZGkPknzJN0g6XOSlin7tpb0SNnf\neG1X9i1s094+km4qr2vLQ/ga+2ZImiLpD6WtP0l6sNb2/W3KJ0i6U9KcWtmJbc6/p6S5pe71kg4o\n5ZJ0qKRbJd0i6XJJE2vH3Snpwtr7XSSdWbbXknRJ6Z/5ki6t9c8lTec/U9Iu9eut7XtPLf6Fkm4u\n2z+QtJ2kn5Z6H5P0TFN8N0lat2yvIun7km6XNEtSr6R/a9EXy0n6a4vyI+oPZ6zVbfwtzJV0nqSV\nWvVxRESMvny5XGtP2J4EIOnlVE8+fglwWNl/pe2dOmlI0k7Ax4EtbS+QNBn4qaQ32r6vUc/2m0r9\n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\nQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f926402e890>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Top 10 cmpanies with Certified status\n",
    "barPlot(top_10_companies_by_status(h1b_data_frame, \"CERTIFIED\"),\"Green\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Insights\n",
    "From the above graph we see that Infosys Limited has most number of CERTIFIED applications followed by IBM India and Tata Consultancy Services (TCS).\n",
    "Interestingly, IBM India also ranks fourth in number of DENIED applications. We don't see Infosys and TCS in top 10 companies with DENIED applications. More interestingly, IBM Corporation and Deloitte Consulting that are US based companies are also the first two companies when it comes to DENIED applications."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2. State-wise average prevailing wage\n",
    "Here we take the average prevailing wage of all employees from each state."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "prevailingwage_by_state = h1b_data_frame.map(lambda x :(x.EMPLOYER_STATE, (x.PREVAILING_WAGE, 1)))\\\n",
    "    .reduceByKey(lambda x, y : (x[0]+y[0], x[1]+y[1]))\\\n",
    "    .mapValues(lambda x : float(x[0])/float(x[1]))\\\n",
    "    .map(lambda x : (x[1],x[0]))\n",
    "prevailingwage_by_state_sorted = prevailingwage_by_state.sortByKey(ascending=False)\n",
    "prevailingwage_by_state_sorted_10 = prevailingwage_by_state_sorted.filter(lambda x: x[1]!= \"\").toDF().limit(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\nKHMAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fdeac06e250>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "prevailingwage_by_state_sorted_10 = prevailingwage_by_state_sorted_10.map(lambda x : (x[1],x[0]))\n",
    "barPlot(prevailingwage_by_state_sorted_10)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Insight\n",
    "Interestingly CA is behind GA in average wage for H-1B workers. The general perception would be that in California the wages are high and hence the average wage should be high too. However, we are seeing that Georgia which compared to California is not a very high paying state is ahead of it in terms of average wage.\n",
    "What could be the reason?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total number of applicants in California : 16026\n",
      "Total number of applicants in GA : 2360\n"
     ]
    }
   ],
   "source": [
    "# Analysis of why CA is behind GA in avg wage for H-1B workers\n",
    "CA_applicants = h1b_data_frame.where(h1b_data_frame['EMPLOYER_STATE'] == 'CA')\n",
    "print('Total number of applicants in California : ' + str(CA_applicants.count()))\n",
    "\n",
    "GA_applicants=h1b_data_frame.where(h1b_data_frame['EMPLOYER_STATE'] == 'GA')\n",
    "print('Total number of applicants in GA : ' + str(GA_applicants.count()))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Insight\n",
    "The number of applicants in Georgia is very less compared to California. Which answers our above questions. Since the number of applications in California is so high, it makes the average wage go down."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 3. Year-wise status\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def year_wise_status(h1b_data_frame, status):\n",
    "    status_by_year = h1b_data_frame\\\n",
    "    .map(lambda x : (x.CASE_SUBMITTED, (1, 1 if x.CASE_STATUS == status else 0)))\\\n",
    "    .reduceByKey(lambda x, y : (x[0]+y[0], x[1]+y[1]))\\\n",
    "    .mapValues(lambda x : float(x[1])*100/float(x[0]))\\\n",
    "    .sortByKey()\n",
    "    return status_by_year"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here we visualize year wise certified vs denied status. We print the percentages to see how the trend goes. Green corresponds to certified and red to denied."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYEAAAD8CAYAAACRkhiPAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAACpZJREFUeJzt3X+I5Hd9x/HX25yxNf5I4oVwzQXXYqocovmFjVVEE5Uk\nFC0IEsESrOA/gkmx1KSiVfpP/xCx0FII2h+oRIxGjflDjWlaUIt2L0a93OWMYqKRJJcIbayiaP34\nx3w3WY5cdtOby3zP9+MBy858Z3butXO3PHdmdpMaYwSAnp606gEArI4IADQmAgCNiQBAYyIA0JgI\nADQmAgCNiQBAYyIA0NiOVQ/Yys6dO8fa2tqqZwAcV/bu3fvgGOO0ra43+wisra1lfX191TMAjitV\ndfd2rufpIIDGRACgMREAaEwEABoTAYDGRACgMREAaEwEABqb/S+LZe/epOrhs/Xe1U0BeKKMv35i\n/v/vHgkANCYCAI2JAEBjIgDQmAgANCYCAI2JAEBjIgDQmAgANCYCAI2JAEBjIgDQmAgANCYCAI2J\nAEBjIgDQmAgANCYCAI2JAEBjIgDQmAgANCYCAI2JAEBjIgDQmAgANCYCAI2JAEBjIgDQmAgANCYC\nAI2JAEBjIgDQmAgANCYCAI2JAEBjIgDQmAgANCYCAI2JAEBjIgDQmAgANCYCAI2JAEBjIgDQmAgA\nNCYCAI2JAEBjIgDQmAgANCYCAI2JAEBjIgDQmAgANCYCAI2JAEBjIgDQmAgANCYCAI2JAEBjIgDQ\nmAgANCYCAI2JAEBjIgDQmAgANCYCAI2JAEBjIgDQmAgANCYCAI2JAEBjIgDQmAgANCYCAI2JAEBj\nIgDQmAgANCYCAI2JAEBjIgDQmAgANCYCAI2JAEBjIgDQmAgANCYCAI2JAEBjIgDQmAgANCYCAI2J\nAEBjIgDQmAgANCYCAI1tGYGqOrOqbqmq/VV1e1VdMR0/tapuqqo7p/enTMefNV3/f6vq7w+7rROr\n6pqq+k5V3VFVrz82nxYA27GdRwK/SvKOMcaeJBckeVtV7UlyVZKbxxhnJbl5Op8kP0/y7iR/8Si3\n9a4kh8YYf5BkT5L/OMr9AByFHVtdYYxxb5J7p9M/qaoDSc5I8rokr5iu9q9J/j3JO8cYP03y5ap6\n7qPc3J8lef50W79O8uBR7gfgKDyu1wSqai3JOUm+luT0KRBJcl+S07f42JOnk39TVbdW1XVV9Zgf\nA8Cxte0IVNXTknwqyZVjjIc2XzbGGEnGFjexI8nuJF8dY5yb5D+TvP8If9Zbq2q9qtYf2O5AAB63\nbUWgqp6cRQA+Nsa4fjp8f1Xtmi7fleTQFjfz4yQ/S7Lx8dclOffRrjjGuGaMcf4Y4/zTtjMQgP+X\n7fx0UCX5cJIDY4wPbLrohiSXT6cvT/LZx7qd6dHC5/LI6wgXJdn/OPcCsERbvjCc5KVJ/jTJt6vq\ntunYXyX52ySfqKq3JLk7yRs2PqCq7kryjCQnVtWfJHnNGGN/kncm+UhVfTDJA0nevKxPBIDHbzs/\nHfTlJHWEiy86wsesHeH43Ulevt1xABxbfmMYoDERAGhMBAAaEwGAxkQAoDERAGhMBAAaEwGAxkQA\noDERAGhMBAAaEwGAxkQAoDERAGhMBAAaEwGAxkQAoDERAGhMBAAaEwGAxkQAoDERAGhMBAAaEwGA\nxkQAoDERAGhMBAAaEwGAxkQAoDERAGhMBAAaEwGAxkQAoDERAGhMBAAaEwGAxkQAoDERAGhMBAAa\nEwGAxkQAoDERAGhMBAAaEwGAxkQAoDERAGhMBAAaEwGAxkQAoDERAGhMBAAaEwGAxkQAoDERAGhM\nBAAaEwGAxkQAoDERAGhMBAAaEwGAxkQAoDERAGhMBAAaEwGAxkQAoDERAGhMBAAaEwGAxkQAoDER\nAGhMBAAaEwGAxkQAoDERAGhMBAAaEwGAxkQAoDERAGhMBAAaEwGAxkQAoDERAGhMBAAaEwGAxkQA\noDERAGhMBAAaEwGAxkQAoDERAGhMBAAaEwGAxkQAoDERAGhsx6oHbOm885L19YfPjhVOAfht45EA\nQGMiANCYCAA0JgIAjYkAQGMiANCYCAA0JgIAjYkAQGM1xrx/B7eqfpLk4Kp3PIadSR5c9Ygt2Hj0\n5r4vsXEZ5r4v2f7GZ48xTtvqSvP/z0YkB8cY5696xJFU1fqc9yU2LsPc9yU2LsPc9yXL3+jpIIDG\nRACgseMhAtesesAW5r4vsXEZ5r4vsXEZ5r4vWfLG2b8wDMCxczw8EgDgGJltBKrq4qo6WFXfraqr\nVr0nSarqn6rqUFXt23Ts1Kq6qarunN6fssJ9Z1bVLVW1v6pur6orZrjxd6rq61X1zWnj++a2cdpz\nQlV9o6punOO+adNdVfXtqrqtqtbntrOqTq6qT1bVHVV1oKpeMrN9z5vuu423h6rqyplt/PPp62Rf\nVV07ff0sdd8sI1BVJyT5hySXJNmT5I1VtWe1q5Ik/5Lk4sOOXZXk5jHGWUluns6vyq+SvGOMsSfJ\nBUneNt1vc9r4iyQXjjFelOTsJBdX1QUz25gkVyQ5sOn83PZteOUY4+xNPzI4p51/l+TzY4znJ3lR\nFvfnbPaNMQ5O993ZSc5L8rMkn57Lxqo6I8nbk5w/xnhBkhOSXLb0fWOM2b0leUmSL2w6f3WSq1e9\na9qylmTfpvMHk+yaTu/K4vcaVr5z2vPZJK+e68YkT01ya5I/nNPGJLunL64Lk9w417/nJHcl2XnY\nsVnsTPLMJN/P9Lrj3PY9yt7XJPnKnDYmOSPJD5OcmsXvdN047Vzqvlk+Esgjn/yGe6Zjc3T6GOPe\n6fR9SU5f5ZgNVbWW5JwkX8vMNk5PtdyW5FCSm8YYc9v4wSR/meTXm47Nad+GkeRLVbW3qt46HZvL\nzuckeSDJP09Pq32oqk6a0b7DXZbk2un0LDaOMX6U5P1JfpDk3iT/M8b44rL3zTUCx6WxSPPKf9yq\nqp6W5FNJrhxjPLT5sjlsHGP831g8BN+d5MVV9YLDLl/Zxqr64ySHxhh7j3SdOdyHk5dN9+MlWTz1\n9/LNF654544k5yb5xzHGOUl+msOetpjL/VhVJyZ5bZLrDr9sxf8WT0nyuiyC+ntJTqqqN22+zjL2\nzTUCP0py5qbzu6djc3R/Ve1Kkun9oVWOqaonZxGAj40xrp8Oz2rjhjHGfye5JYvXWeay8aVJXltV\ndyX5eJILq+qjM9r3sOk7xYwxDmXxXPaLM5+d9yS5Z3qUlySfzCIKc9m32SVJbh1j3D+dn8vGVyX5\n/hjjgTHGL5Ncn+SPlr1vrhH4ryRnVdVzpkpfluSGFW86khuSXD6dvjyL5+FXoqoqyYeTHBhjfGDT\nRXPaeFpVnTyd/t0sXrO4IzPZOMa4eoyxe4yxlsW/u38bY7xpLvs2VNVJVfX0jdNZPFe8LzPZOca4\nL8kPq+p506GLkuzPTPYd5o155KmgZD4bf5Dkgqp66vS1fVEWL64vd9+qX5B5jBdFLk3ynSTfS/Ku\nVe+ZNl2bxXNzv8ziO523JHlWFi8i3pnkS0lOXeG+l2Xx0PBbSW6b3i6d2cYXJvnGtHFfkvdMx2ez\ncdPWV+SRF4ZntS/J7yf55vR2+8bXyJx2ZvHTX+vT3/Vnkpwyp33TxpOS/DjJMzcdm83GJO/L4puk\nfUk+kuQpy97nN4YBGpvr00EAPAFEAKAxEQBoTAQAGhMBgMZEAKAxEQBoTAQAGvsN1zEA28DdKtcA\nAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f925fe6aa10>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "barPlot(year_wise_status(h1b_data_frame, \"CERTIFIED\"), \"Green\")\n",
    "barPlot(year_wise_status(h1b_data_frame, \"DENIED\"), \"Red\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Insight\n",
    "To our surprise we see that the ratio of CERTIFIED to DENIED is huge. About, only 1% of the applications were DENIED whereas about 78% were CERTIFIED. It also suggests that the data we have is highly skewed."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 4. SOC wise Status"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def top_10_soc_by_status(h1b_data_frame, status):\n",
    "    status_by_soc = h1b_data_frame.where(h1b_data_frame['CASE_STATUS'] == status)\n",
    "    #1. Map by SOC\n",
    "    #2. Reduce by count\n",
    "    #3. Map with count as key, SOC as value this makes the sorting easier as you can sortByKey\n",
    "    status_by_soc = status_by_soc.map(lambda x : (x.SOC_NAME, 1)).\\\n",
    "                    reduceByKey(lambda x,y : x+y).map(lambda x : (x[1],x[0]))\n",
    "    status_by_soc_sorted = status_by_soc.sortByKey(ascending=False)\n",
    "    status_by_soc_sorted_10 = status_by_soc_sorted.toDF().limit(10)\n",
    "    status_by_soc_sorted_10 = status_by_soc_sorted_10.map(lambda x : (x[1],x[0])).toDF()\n",
    "    return status_by_soc_sorted_10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\nrkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f925fcc5590>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAnsAAAD8CAYAAAAPHmLeAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAIABJREFUeJzs3Xm8XdP9//HXO4aShqLGoG5FRJGBm6+0kbTGLy0hqSG5\nosSXL/1WB/lViNKvtEW1aFRDS9VYJL5milKkphQ3ModMYh4qtNpIGsTn98f6HFnZ95xzz83gtqef\n5+NxHjl7rbXXXmudfXM+d62995WZEUIIIYQQ6lOH9m5ACCGEEEJYfSLYCyGEEEKoYxHshRBCCCHU\nsQj2QgghhBDqWAR7IYQQQgh1LIK9EEIIIYQ6FsFeCCGEEEIdi2AvhBBCCKGORbAXQgghhFDH1mzv\nBoQQwsYbb2wNDQ3t3YwQQviXMnHixAVmtklr5SLYCyG0u4aGBpqbm9u7GSGE8C9F0gu1lItl3BBC\nCCGEOhbBXgghhBBCHYtgL4QQQgihjkWwF0IIIYRQxyLYCyGEEEKoYxHshRBCCCHUsQj2QgghhBDq\nWAR7IYQQQgh1TGbW3m0IIfybU2cZJ6y6+uzM+H8thFD/JE00s96tlYuZvRBCCCGEOhbBXgghhBBC\nHYtgL4QQQgihjkWwF0IIIYRQx1ZLsCdpc0ljJc2TNFHS3ZK297ydJD0oaZakOZK+L0meN0ySSdon\nq2ugpx3q2+N93ymSHpPUzdOfl7Rxtt8eku6SdIykyf56T9I0f3+uH+/NLH+ypB0lNUha7NszJV0j\naa0q/T1J0j8kfapwfJM0IEu7S9IehX5MlfSspDGSNqhQ/0d98zovyPJOljTK34+S9ErWl3M9fW1J\nF0qa62N+u6StsjqWevnpku4stcPHwSSdlZXdWNL7ksYU2jhZ0ths++Js/BZnbTpU0lXZ59la2yr2\nt3D8zXx8p/gx75a0jo9t96zcCEmXSuog6SLv8zRJT0n6rKQnvJ0vFs6NBv8cpmVpF3mdV0laJGm9\n7DgXettLn9vpkmb45z1ZUp8yfWi1Hk8r/UzskKWVPqtvZWljJA3Lttf0Pp1bOO6aks7x8S/17fQs\nf6mW/xkZ6en5z+JTknpl+/yXj9VUH+ODi/0NIYTw8VjlwZ4kAbcC482si5k1AqcBm0laF7gDONfM\nugE9gb7AN7IqpgFDsu0mYErhMEPNrCdwNXBetfaY2ZVm1svMegGvAnv69kgvMq6U76+Znj7P9+kO\nbAUcXuUwTcBTwFcL6S8Dp7csvlw/egA9gCXA7dX64pYAX82//AtGZ30p9fEcYD2gm5l1BW4DbvHP\nCmCxl98ZeBs4MatvPnBAtn0YMCM/oKTPAWsA/SV9EsDMTvTx+wo+lv66qdDe1trWWn9Lfgjcb2Y9\nzWxHYKSZ/QM4CbhEyZbA14GRwGCgM9DDzLoDg4C/mlkfb/f/svy58bwfZ88s7dvZ8ecCB/t4dAD2\nAl7x7S8ABwK7+ue9D/BShX5UrCfTBDzq/+b+DHxH0toV6t4XmA0clo0vwFk+Ft297/2B/JebxYWf\nkTxYLP0sXoL/LHqwfjrQz/v7eWBqhTaFEEJYzVbHzN6ewPtm9qtSgplNMbNHgCOAx8zsPk9fBHyT\n9OVb8giwm6S1JHUCtgMmVzjWw56/2pjZUuBJYMty+ZK6AJ2AM2j55TsFeEfSvq0c4z3gFOAzknq2\n0qQPgMuA4a23HiR1BI4BhntfMLMrSUHUXmV2mcDyfV0EPCOpdGv3YODGwj5NwLXAfXigsgrbVmt/\ntyAF13g9U/3fe4HXgKOA0cAoM/uLl3/NzD70ci97+ooaSxobgD2Ax7ztpbYtMLMlfqwFZvbqCtSD\n/0z0A45l+V+KAN4EHgCOrlB3E/Bz4EXgC15fR+C/gW95cIyZ/d3MRlXrbBn5ebMp8Hdgode30Mzm\nt7G+EEIIq8jqCPZ2BiZWyNupmGdm84BOktYvJQF/APYjBQ53VDnWANJM4MoYXFiiWjfPlLQO0Ae4\nt8L+Q0hf0I8A3SRtVsg/mxQIVuXBzhRgh9bKAhcDQ5UtG2eGZ33ZjxQMv2hmfyuUayZ9Hh+RtAaw\nNy3HfCwwRNLWwFLSDGlusJe5gZYBbzW1tq1af/Myv5H0kC+Zds7yTiJ9DpuY2bWediMwwMfpAkm7\n1Njmh7LxzQPQ2cAmkjYkjcHYLO8+YGtJsyVdIulLVeqvVg+kn4l7zWw28JakxkL+T4CT/bP8iJ/H\n+wB3svznVPoM/l6lTesWfkYGlymzP2lWFtJ5/AYwX9KVyi5lKLTpeEnNkppZVOXoIYQQVso/6w0a\nY0lB1BDSF1PRdZImA7sDJ3tauaeo1vJk1eIy7mJP7+LHeIM0A1RpGaoJGOszRDeTljmXNcDsYQBJ\n/Wpoi1ovAh4cXQN8u0x2voz7+1rqw7/MgdeBzYD7C/n3kpYAhwDjlmtwmvFbYGYvkmaVdpG0UY3H\nrUkr/S2V+T2wLfBrUsA8SdImnvcq8CDwy6z8y0A30iUGHwIPSNq7hubky7ijC3m3kMaoDyn4Lx1r\nIdAIHE+afRuXX0tXRtl6XB4AjqUQXJvZc8ATpFn03IHAQ35+3wwMLAaEAFp2jetLHtxDy2Xc/By4\nTtJ80rLtxd6GpaTg71BS8DpaZa6zNLPLzKy3mfWmY5XRCCGEsFJWR7A3g/TFVs7MYp6kbYGF+eyO\nmT1JulZuY5/BKBrqXzoDzax07dNbwIZZmY2ABSvYB1h2zV4XoFHSQcUCShf+dwXul/Q86Qu63MxW\nq7N7/sXbHXimxvZdSFrK+2Qr5eaRlofXK6Q3suzau8Xe121IAWd+zV5pmXki8F2geM1dE7CD938e\nsD5wSI19qKVtJa3218zeNrPrzexrpGsov5hlf+ivvPwSM7vHzEaQrh0cWGO7KxkH/Ih07WDxWEvN\nbLyZnUm6dKHaGJWtx4PovYDLfbxHAIcXrr/D+3Iqy//y0ATs4/tNBD7tdc0l+wxK17gC75Cuw2zN\nUFKQfTXwi6y/ZmZPmtmPST8XtZ4TIYQQVrHVEew9CHxC0vGlBEk9JPUHrgP6ye+29SXTi4Cflqln\nJPC9Nhx3PPA1r3cN4EjgoRXpQM7MFnhbTiuT3US6BqzBX52BzpK2KdRxHykQ7VHuGEp3+v4YeKnK\nDGKxXW+TliKPbaXcu6Qv4p+VZnIkHQV0JH1WedlFpNmz70pas1DVBcCpftxSuzuQblzpXhoD0jJj\nTUu5bWxb1f5K2suvP8MDly6ka9PKkrRraanX+9EDeKGWdlfpzwukGa5LCsfqJqlrltSr2rEq1UOa\nKbvWzLbx8d6adANN/8L+z5J+sRrgx1/fy3wm+5xOBJr8M/8NMMaXeks/P5Vu8ijXXgO+D3xe0g6S\nOkvatdb+hhBCWL1WebDn//EPIs0izJM0gxTIvO5LSAcDZ0iaRbre7ilgTJl67jGztgRrPwK2kzQF\nmESasfhtDfsVr9nrW6bMbUBHD1hzQ0h3HudupeWF85Bm97YupF0naSownTRj1dbHU1wAtHaXKqRA\n9R/AbElzSEvNg/yzWo6ZTSLdOVlcHpxhZlcXivcHXincbPAwsKOkLWrsQ81to3p/G4FmH88JwOVm\n9lSV424K3ClpOqm/H1DmPCwjv2bvmmKmmV3q16HmOgFXKz0SZiqwIzCq2kEq1NNEy/PtZirPJpce\nYTMIeLB0g4i7nXTN4idIgeVrwHRJk0hLx1ez7NrM4jV7yz26xdu7mPT5jCDdyXu+0mNvJpOu6fxO\ntf6GEEJYfVT+OzWEED4+6izjhFVXn50Z/6+FEOqfpIlm1ru1cv+sN2iEEEIIIYRVIIK9EEIIIYQ6\nVrwIP4QQPnaNnRtpPrO5vZsRQgh1KWb2QgghhBDqWAR7IYQQQgh1LIK9EEIIIYQ6Fo9eCSG0u1X9\n6JWieBRLCKEexaNXQgghhBBCBHshhBBCCPUsgr0QQgghhDoWwV4IIYQQQh1rNdiTZJIuyLZPljTK\n34+S9Erhj6QPzt4vlDSr9EfjJU2S1Mv3XdPzj8zqnihpV38/UNJUSc9ImiZpYFbuKknzvd4pkvbO\n8sZL6u3vPytpjqT9KvTtJEn/kPSpLG0P7/OALO0uSXtk9c/ytj0raYykDSrU30nSpZLmed/GS+rj\neVtJut3bN0/SzyWtXWjDcVldvTzt5DJj8LSkLxT779sNkqZL2q/K57KHpHcKn+M+vv9S354u6c4q\nfT1d0gwfl8mS+kg6W9JPsjLbSHpO0gaSDvTzYYqkmZJO8DpKx1+avf92hXNtgxrH6vOSnvB9npGf\nvxX6caEfp0OWNkzSm97eOZJ+L6lvln+VpEWS1ivUY5I29u2F2edhkr6VlR0jaVhW16H+fmXGaKak\npkLf1vR+nJul3erl5xbOgb6S1vZ+zPV+3y5pq2zfsueGpA6SLvL0aZKekvTZSmMeQghh9aplZm8J\n8NXSl1YZo82sV/YaV3oPNANDffso4DGg9CXZE5hd2pb0SaALMEVST+B84GAz+xxwEHC+pB7ZcUf4\nMU4CflVslH8p3Qt818x+X6HtTcBTwFcL6S8Dp1cckdSnHkAP0vjcXqHc5cDbQFczawSOATaWJOAW\n4DYz6wpsD3QCzs72nQ4cXmjrlEL9pTEYCVxapb2Y2e+rfC4AjxQ+xz94+mLf3tn7cmKxbqVA80Bg\nVx+XfYCXgLOAgZI+50V/DnwfeBe4DBhgZj2BXYDxZnZ21sbFWVsu8v2L59pfaxyrq4Hjvd6dgRvL\njZEHeIO87V8qZI8zs1388zoXuCXrF8Bc4OCsnr2AV8odB/gz8B15cF+hLWuxEmPkbbnU6ynZl/Qz\nd5ifg5jZIC9/HMufA48D5wDrAd2837d5v+X1VTo3BgOdgR5m1t3HtPRZhRBC+JjVEux9QPrSGb4K\njvc4y4K9vqQgrZdv7wZMNLOlwMnAOWY2H8D//TEwokydE4AtC2lbAPcBp5vZHeUaIqkLKcA6gxQc\n5KYA70jat1pnzOw94BTgMx6gFuvvA5xhZh+W+mFmvyMFAv8wsys9fSlpfP9LUkev4gVgHUmb+Zfr\n/sA9FZryMLBdtbauIuXGGtJ4LzCzJQBmtsDMXjWzxaR+XSzpK8B6ZnYdKYBYE3jLyy8xs1kr0a7W\nxmpT4DU/1lIzm1mhnj2AGcAvaXlOfMTMHiL9TByfJY8lBTmleh4j/eyU8ybwAHB0xR6t5BiZ2Rxg\nEbBhltxECrhfBL5QbX8/D48Bhvv5iZ+vS0jnb1F+bmwBvJad9y+b2V9qbXsIIYRVq9Zr9i4Ghipb\n7swMz5Z+Hmqlnnxmry8pSFniy199ScEgwE7AxMK+zZ5etD9pxiF3NTDGzG6q0pYhpC/oR4BukjYr\n5J9NCgSr8i/CKcAOhaydgMmlL8oyecv1z8z+RvoSzoO2m4DDSGPzNOmLtpwBwLTW2tqK/oUl0i55\npqQ1gL2BcsHzfcDWkmZLukTSR7NiZnY38BfSZ/INT3vb63lB0g2ShubLplVUO9eqjdVoYJYvWZ4g\naZ0K9TcBNwC3AgcUZsWKnmb5z3w2sImkDb2esa305SfAyT6uLazEGAGgdDnEHDP7s2+vQ5pxvZPU\nx4rBrNsOeNHPy1yLn8My58aNwAD/nC6QtEuFNh4vqVlSM4tq7VkIIYS2qunLw//Dvwb4dpnsfGlt\nz1bqeQFYW9LmpC/KWaRl1D6kL+nH2tD28yTNBq4nfXHm/gAcmc2SldMEjPXZh5tJgULe1ocBJPWr\noS1qvcgKuZHUrlIQUnSepMmkGaZjPa3c02NreaJscRl3nqev68d4HdgMuL9F5WYLgUZvx5vAuNI1\naO5i4Kl8ZsrMjiMFCE+SZnKvqKGN1c61imNlZj8EepOC0iNIy/vL8SXVr5CW1v8GPAGUvdaztEuZ\ntFtIv0T0If0SUZGZPefHOKJKmRUZo+GSZnjd+WUBBwIP+WzrzaTl9bKBZhuUPTfM7GWgG3Aa8CHw\ngLLrakvM7DIz621mvan2kxpCCGGltOVu3AtJAcUnV/KYj5O+lF+z9Oc7/gTsTlrGneBlZpKCh1wj\naYmtZISZbQ+cSssvwZ+Sgsj/k7RmsQGSugNdgfslPU/6gi4309Hq7J5/YXYHnilkzQB6VvhCbdE/\nSesDnyFd+wWAmb0OvE+61uqBMvWM8MBnXzOb7mlvsfzS3UbAgmp9aMViv6ZrG1KA0+KaPW/rUjMb\nb2ZnAt8EDsmyP/RXcZ9pZjaa1L9Divlt0dpYmdk8M/slKXjqKenThSL7ARsA0/yc6Ef12a9daPmZ\njwN+BNxfWsJsxTmk87fiLwsrMEajzWwnL/ubbBazCdjH+zYR+DTll2NL5pEuT1ivkJ7/HFY8N3zZ\n+R4zG+H9HEgIIYR2UXOw58tKN7JsBmlFPU66qaIU2E0AjgJeN7N3PO184DRJDZDuYAS+B1xAS2OA\nDmp5x+1JwN9IX3jFL9MmYJSZNfirM9BZ0jZ5ITO7jxQ49aAMX+b7MfCSmU0t7DuPtOT1g9Lxle7E\nPIAUjHSUdJSnr+F9u8rMigta/wucWmE5uJzxpFnNUp+PBlpbXm+Vt+vbwHeLAbSkbpK6Zkm9SNfR\nlaV0l/IetZZvg7JjJemAbDy6AktpecNAE3Bc6ZwAPgvsW2522Jepjwd+naf7zPXpwCW1NNbMniUF\n/gOKeSs7Rn6tajNwtP8i0R/4TNa/E6l+XeK7pKX3n5V+YfHztSPwYKHscueGpF0ldfZ9OpB+flbF\n5xtCCGEFtPU5excAxbty8+uoJpcCtCoeA7bFgz0zew1Yg2XX62Fmk0kzHndKepZ0ndEpnr4cnx08\ni3SjRDH9aNLF4j8t7DaEdF1W7lZPLzob2LqQdp2kqaS7QD+J34VZxnGk5a25kqYDVwF/9rYNIt0V\nOYd0vdc/SAFtsX+Pm1nxmsRqLgP+TrqreQrpJpTza9iveM3eoWXaMgmYSssgoRNwtdLjPqYCOwKj\nqhxLwCnyx78APwCG1dDGqudalbH6GumavcnAtaQ7kT8KCD2g2x/4XVbXu8CjLAvESo8Umk36nA4x\ns+LMHmZ2abYEXouzga3KpK/oGOV+CPw/0rn2YOkGGnc76bq6T1TZ/zTSeTnbz9PDgEF+/i6ncG5s\nSvrZne5pH5B+KQshhNAOVOb/7RBC+Fips4wTVl/9dmb8PxdCqD+SJppZ79bKxV/QCCGEEEKoYxHs\nhRBCCCHUsQj2QgghhBDqWIvHkoQQwsetsXMjzWc2t3czQgihLsXMXgghhBBCHYtgL4QQQgihjkWw\nF0IIIYRQx+I5eyGEdre6n7NXTTyDL4TwryqesxdCCCGEECLYCyGEEEKoZxHshRBCCCHUsQj2ApIG\nSjJJOxTSt5d0t6Q5kp6WdKOkzTxvN0kPS5olaZKkyyV1zOqbKukZSdMkDczqHC+pd7bdIGm6v9/D\n2zEgy7/L02+VNFnSXEnv+PvJkvp6uY0lvS/p64U+PC/p5mz7UElXSTomq+M9b+dkSedK2syPO0XS\nTEl3t2XsvE8m6VtZ2hhJw/z9VZLme/2zJV0jaasqx6jUt4WF7WGSxvj7UZJe8T7NkXSLpB2Ln4Ok\nJ7zMi5LezMakQdKnvG1zJc3z95/K+rjYy870vLU8r6Ok63xMp0t6VFKnSv0LIYSwekWwFwCagEf9\nXwAkrQP8DvilmXU1s12BS4BNPOD7P+BUM+tmZrsA9wLrSeoJnA8cbGafAw4CzpfUo8a2vAycXkw0\ns0Fm1gs4DnjEzHr563Evchjwp7wPmcY80PH6rizVAbwK7OnbI4EfAvebWU8z2xEYWaW9LcbO/Rn4\njqS1K+w3wsx6At2AScCDVcpW61s1o71PXYFxfoxN8gJm1sfH4H+Bcdm4Pg/8BnjOzLYzsy7AfODy\nbPd5vm93YCvgcE//DvCGmXU3s52BY4H329j2EEIIq0gEe//mfMalH+kLeUiWdQQwwczuLCWY2Xgz\nmw6cCFxtZhOyvJvM7A3gZOAcM5vv6fOBHwMjamzSFOAdSfu2sStNwHeBLcvMkl1AmQCyii1IQScA\nZja1XKEqYwfwJvAAcHS1A1kyGngd+HKFYtX6VhMzGwfcR/pcWyVpO6AR+FGW/EOgt6QuhbqXAk8C\nW3rSFsArWf4sM1uyIu0OIYSw8iLYCwcD95rZbOAtSY2evjMwscI+1fJ2KpPX7Om1Ohs4o9bCkrYG\ntjCzJ4EbgcGFIjcCu3oAU4uLgd9IekjS6ZI6VyhXaexKfgKcLGmNGo75NLBDMbGGvrVF2WNUsCMw\n2QM54KOgbjKFz9JngfuQZncBrgBOlTRB0lmSuq5Em0MIIaykCPZCEzDW34+l7UuFbVXuoWbLpZnZ\nwwCS+tVY52BSIATl+7AUOA84raYGmv0e2Bb4NSk4mlRc/nRVx87MngOeoLbZNFVIb61vRdUeGlfp\nGCuqi6TJwBvAa6UZUDObTBq/84CNgKckfa5FY6TjJTVLambRKm5ZCCGEj6zZ3g0I7UfSRsBeQHdJ\nBqwBmKQRwAzgSxV2nUFa4ru9TN5Mz5uSpTX6PgBvARtmeRsBC8rUU5rd+6CGrjQBm0sa6tudJXU1\nszlZmWtJwd70GurDzN4Grgeul3QX8EUgv9Gj2tjlzgFuAv7YyiF3IS37tqVviyWtbWbveV6lscyP\n0dxKO0pmAr0kdTCzDwEkdQB6eR74NXuSNgYek3SQmd0BYGYLgVuAWyR9CHwFeCY/gJldBlwG/lDl\nEEIIq0XM7P17OxS41sy2MbMGM9uadBF+f1Kg01fSAaXCkr4oaWdgDHC0pD5Z3lf9xo3zgdMkNXh6\nA/A90nVzAOOBIyWVZpmOBh4qNszM7iMFhVVv7JC0PdDJzLb0PjSQrhEszrK9D4wGhlcfEpC0l5bd\nWbwe0AV4sVCs2tjlx32WFBwNoAwl3yZd53ZvIa+1vv0RONLLrku6QaLFWHr+IcB/Aje01n9v91zS\njSP5cvoZwNOel5ddQLqJ5TQ/1u6SNvT3a5OWhF+o5bghhBBWvQj2/r01AbcW0m4GmsxsMXAg8C1/\ndMdM4BvAm34jxhDSXbazJD0D7Af83ZfwTgXulPQscCdwiqdDmsn5OzBF0hSgEylALOdsYOsV7UOZ\nsr+httnsRqBZ0lRgAnC5mT21Esc9m3S3au487/9s4D9IdwO/VyjT2jG+A3zVl1L/BPxfaQncDS89\neoUUFO5lZm+WaV8lxwLb+2NX5gHbe1o5twEdJfUnBcd/lDSNFDA2k82KhhBC+HjF38YNIbS7+Nu4\nIYTQdoq/jRtCCCGEECLYCyGEEEKoYxHshRBCCCHUsXj0Sgih3TV2bqT5zFqfChNCCKEtYmYvhBBC\nCKGORbAXQgghhFDHItgLIYQQQqhj8Zy9EEK7a8/n7JXE8/ZCCP9q4jl7IYQQQgghgr0QQgghhHoW\nwV4IIYQQQh2LYC+0C0km6bfZ9pqS3pR0V6HcbZL+VEgbJWmRpE2ztIWFMgP9GDsU0rtKukvSPEkT\nJT0k6YueN8zbMDl77Sipwes6K6tnY0nvSxqTtemVwr4bSNrD9x2Q7XuXp9/q5eZKeifbr2+Z8SqN\nz7mF9PGSmrPt3pLGF8pc6G3rkKUNK7U9Sxsn6b+z7b6SJvmx/1vSNElT/N8DJf3K2ztT0uKs/YMk\n7S7pCd9+RtL3i30KIYTw8YhgL7SXd4GdJa3r2/sCr+QFJG0ANAKfkrRtYf8FwHer1N8EPOr/lupb\nB/gdcJmZdTGzRuBbQF73ODPrlb1mevp84ICs3GHAjMIxRxf2/aunvwycXmygmQ0ys17AccAj2X6P\nl+nPvsBs4DBJKuRtKunL5QbBA7xBwEvAl8qVyQwHTpP0aUlrAGOAbwBbAiOAvmbWE+gLTDezr3v7\nDwJmZe2/FbgaONbzdwZubuXYIYQQVpMI9kJ7uptlAVQTcEMh/6vAncBYYEgh7wpgsKSNipVK6gT0\nA44t7DcUmGBmd5QSzGy6mV1VQ1sXAc9IKt31NBi4sYb9AKYA70jat8by5TQBPwdeBL5QyDuPMsGk\n24MUlP6SLPAtx8xeBS4EzgVOBJ4yswnAZsDfSAE6ZvZ3M3u+lfZuArzu5ZdmQXMIIYSPWQR7oT2N\nBYb4jFsP4IlCfikAvIGWgcpCUsD3nTL1Hgzca2azgbckNXr6TsDTrbRpcGEpdt0sr9TerYGlwKuF\nfYdn+z1UyDsbOKOVY5fl47MPKfAtNxYTgPck7Vlm99IY3gocIGmtVg53MbALcBIw0tOeBv4KzJd0\nhaQDa2j2hcAcSbf4EvAnatgnhBDCahDBXmg3ZjYVaCAFJHfneZI2A7oCj3rQ9r6knQtVXAQcLWm9\nQnoTKTDD/y07o+XXzE2XdEuWXFzGXZzl3UtaTh0CjCtTZb6Mu1zgZWYP+zH7lWtLKw4EHvK23AwM\n9GXW3FkUgklJawNfAW4zs7+Rgun9qh3IzJYClwF3mdlfPO0DUr8HA3OBiyRVDVzN7EzgP4A/AEeR\nls+XI+l4Sc2SmllUrbYQQggrI4K90N7uAM6n5RLu4cCGpNmk51kWFH7Er4m7nrTkCIAv6+4FXO77\njQAO9+vcZgC7ZvsPAoYBLZaCyzGz94CJpGsFb6qte8tZ0dm9JmAf789E4NOkPuZtexBYF/h8lrwf\nsAEwzfftRytLue5Df+X1m5n9yczOAY4ADmmtEjOba2aXAHsDu0n6VCH/MjPrbWa96VhDq0IIIayQ\nCPZCe7sC+IGZTSukNwH7m1mDmTWQbtQoXrcH8DPgBGBN3z4UuNbMtvF9tybdXNGfFBjuLumgbP+2\nhhkXAKea2dtt3A8zu48UwPaodR9J65Pa/plsLE6kfNB2FnBKtt0EHJft91lgX0lt6rOkrST1ypJ6\nAS+0ss8B2Y0kXYElwN/bctwQQgirRgR7oV2Z2ctmdlGeJqkB2Ab4U1ZuPukmhz6F/ReQrkcrXRPW\n5Nu5m4EmXwY9EPi6pOckTSDNtJ2VlS1es7fcY1DMbIaZXV2hO8ML+zaUKXM2sHWF/csZBDxoZkuy\ntNuBAcXr4MzsbuBNAA/o9idbPjWzd0l3KJceAzNM0svZa6sKbVgLGC3pWUlTSDfODG+l3cOAZyVN\nBq4CjjBOOaTPAAAgAElEQVSzD6vuEUIIYbWIv40bQmh38bdxQwih7RR/GzeEEEIIIUSwF0IIIYRQ\nxyLYCyGEEEKoY2u2XiSEEFavxs6NNJ/Z3HrBEEIIbRYzeyGEEEIIdSyCvRBCCCGEOhbBXgghhBBC\nHYvn7IUQ2t0/w3P2VoV4Vl8I4eMUz9kLIYQQQggR7IUQQggh1LMI9kIIIYQQ6ljdBXuSTpc0Q9JU\n/2P0fTx9bUkXSporaY6k2/M//C5paeGP2J+QvV8oaZa/v0bSJEm9fL81Pf/IrK6JknbNtm+T9KdC\nO0dJesXrnCmpKcu7StL87PiPl+nnHpLe8bbMkvSwpAMr1F96dZb0lqT1C3XdJmmwpGGS3izss6Ok\nBknTy7RBks7w8Zwt6SFJO2X5z0ua5p/FfZI2L6SXjnFRmX5PkbR3VteB3tcpPl4trvCStJmku7Iy\nd0taR9Kzkrpn5UZIulRSB0kXSZru7XlK0mclPeFteLEwHg2ttH2RpPWy41woySRtXO3cLPTh89nx\nn5E0Kssb6Ps+420YWOWc+XaVfnxH0oXZvpdK+kO2/a1Sv7LjmqQdsrQGSYu17Py9RtJanlc6N/Pz\naJ9iX0MIIXw86uqhypK+ABwI7GpmS/xLdm3PPgdYD+hmZkslHQPcIqmPpbtUFptZr0KVl3q944GT\nzazZt8cAfYHJQE9gtm//VtIngS7AFC+7AdAILJS0rZk9l9U/2szOl9QVmCjpJjN73/NGmNlNrXT5\nETM70I/TC7hN0mIzeyCvvzBGvwcGAVf79qeAfsARwOHAODP7ZmGfhgrHP9H73dPMFkn6T+AOSTuZ\n2T+8zJ5mtkDSOcD3gG/n6WXqHGFmN0naE7gM6OpBxGXAbmb2sqRPAOXa9EPgfjP7ube7h5n9Q9JJ\nwCWSvgh0Br4O9AYG+3YPM/tQKfh/18xKvyAMA3rn4yGpWtvnAgeTzoMOwF7AK75ftXMzdzVwuJlN\nkbQG0M337wmcD+xrZvMlfRa4X9JzZjY1H7usrlIgulw/JPUGhmblegJrSFrDzJaSPtPbs/wm4FH/\n98wsfZ6Z9fJ23k86f67zvI/OzRBCCO2r3mb2tgAWmNkSADNbYGavSuoIHAMM9y8zzOxKYAnpC7mt\nHid9IeL//gooBYq7ARNLxwG+CtwJjAWGlKvMzOYAi4ANV6AtpTomk4Kdb7ZS9IZCOwYBvzezRStw\n2FOBb5b2NbP7SGMztEzZh4Ht2lD3BGBLf78e6ReTt/w4S8xsVpl9tgBeLm2UgiAzuxd4DTgKGA2M\nMrO/ePnXzOxDL/eyp6+osaQAEmAP4DHgg6xtLc7NMnVs6m3FzJaa2UxPPxk4x8zme9584MfAiBVo\n52Rge0nrerC/2NNKs599ve1I6kT6ZeBYKp+/S4EnWfZ5hRBC+CdSb8HefcDWSkuKl0j6kqdvB7xo\nZn8rlG8GSsuO62ZLTre2cpzHWD7YexhY4kt4fUkBT0kTKcC6wd+3oLTkO8fM/pwln5e157py+5Xx\nNLBDtj08q+MhT/s9sKukT/v2EG9byeDC8tu6Fdq8PvDJwkwlLD+muQOBadn2Q9kxhpcpvz9wG4CZ\nvQ3cAbwg6QZJQ33mrOhi4DdKy8mnS+qc5Z0EnA1sYmbXetqNwABvwwWSdinX1zIqtX02sImkDUmf\n9dgsr9K5WTQamCXpVqVLCdbx9J2AiYWyxbHOz5nuVGBmHwCTgP8APg88AfwJ6CtpS9IjmV7y4gcD\n95rZbOAtSY3F+ryNfYB7s+T+hfOoS6X2hBBCWL3qahnXzBb6l1F/YE9gnKSRpCCoNeWWcSsd5wWl\nawA3JwVXs4CnSF94fYFfQLqGDOgKPGpmJul9STubWen6t+G+nLw9MKBwmFqWcYtU2G6xjGtm70m6\nAzhU0s3ALqQAsKTcMm4bm7GchyQtBaYCZ2TplZZCz/Ml362AL2TtPs4DmH1Is1z7AsPyHc3s95K2\nJQWKXwYm+Xi/6TO8DwJ3ZeVfltSNNLu7F/CApMOyZfBKKrUd4BZSAN0Hlj05rtK5aWZXFfrwQw/u\n/5O0tN5EmiWsRVvOmdLs9LqkWdQ5pGX2N2n5y8rP/f1Y3y4FnV0kTQY+C/wuW06GGpZxJR0PHA/A\np2psdQghhDart5m90tLXeDM7k7SkeQgwD/iMsovnXSMwYwUP9ThwGGkZ0EgzI7uTlnEneJnDSUuz\n8yU9T7rOLJ/dG21mO3kbf5PN4qyoXYBnaihXWso9FLg9u06wZj5L+q4HV7nimO5pZr3M7Cgz+2sN\nVY8ws+1JS8RXFI45zcxGkwK9Qyq0620zu97MvkYKwL+YZX/or7z8EjO7x8xGkK7rHMjKGQf8iHTt\nYPFY5c7Ncn2YZ2a/BPYGevos7EzS2OZW5vwtzU5/gXS+PgPsSDYzLWkjUhB8uZ+/I4DDtSz6n+e/\nIHUBGiUd1JYGmNllZtbbzHrTcQV7EUIIoVV1FexJ6uY3O5T0Al4ws3dJF77/zC8mR9JRQEfgwRU8\n3OOkpcFSYDeBdE3Y62b2jqc1AfubWYOZNZC+nFtc92Rmd5CW5I5ewbYgqQfwfdJSZmvGk2YcT2T5\nJdy2Og+4qLTUq3THZT/g+pWos2QM0EHSfpI6Sdojy+sFvFDcQdJefn0mHth3AV6sdABJu5aWen1Z\nuEe5etvCzF4ATgcuKRyr7LlZpk0HZMFUV2Ap8FfSzRmnyW+W8X+/B1ywgk2dQFrC3cTM/uy/sLxJ\nWrZ9zMscClxrZtv4Obw1MJ80O/kRn+UcCZy2gm0JIYSwGtXVMi7QCfiF0h2wH5Dujjze804jfWHO\nlvQh8CwwyL/kVsRjpOurJgCY2WseSJZmRRqAbUgzfniZ+UqPpGjxyA3SzRXXS/q1b58nKV/23M3M\n3ivs01/SJFLQ+mfg24UlyOHKHgkDDDSz5/3O05tIM49/LNQ5WFK/bPsbwKtAN0kvZ+nDScvVGwLT\nfKn2deBgM1tcpn9FpeVdgKlmdlSe6cveZwGnkGbbTpF0KelmgncpLOG6RmCMpA9Iv8hcbmZPVWnD\npsCvle7uhXSTwZhV0PZLy+xT7dzMfQ0YLWmRlxvqN0BMlnQqcKfS3cnvA6f4jTltZmZ/kfQmy88M\nTiDNTk/x7SbgJ4Vdb66QfhswSlIpEOzvS7wlZ63AZQkhhBBWgfjbuCGEdhd/GzeEENpO8bdxQwgh\nhBBCBHshhBBCCHUsgr0QQgghhDpWbzdohBD+BTV2bqT5zOb2bkYIIdSlmNkLIYQQQqhjEeyFEEII\nIdSxCPZCCCGEEOpYPGcvhNDu6uU5e6tSPLMvhNCaeM5eCCGEEEKIYC+EEEIIoZ5FsBdCCCGEUMci\n2AshhBBCqGMR7IXlSNpc0lhJ8yRNlHS3pO09bydJD0qaJWmOpO9LkucNk2SS9snqGuhph/r2eN93\niqTHJHXz9OclbZztt4ekuyQdI2myv96TNM3fn+vHezPLnyxpR0kNkhb79kxJ10haq0w/O0i6SNJ0\nr/cpSZ+VdJ2k/8nK9ZE0VdJakv7Ly071/Q6WdHF2rMVZWw6VdJWk+Vna420YqwMlTfKxmimp4u0L\nkm6T9KdC2ihJiyRtmqUtLJQpHXOHLK1B0vRCuf+RdF22vYGk5yRtI2l3SU94/57xc+K4Cp/b2ZX6\nEEIIYfWJv6ARPuKB263A1WY2xNN6AptJegm4A/gfM7tPUkfgZuAbwMVexTRgCPAH324CphQOM9TM\nmiUdD5wHHFSpPWZ2JXClt+N5YE8zW+Dbw4BxZvbNQh8agHlm1kvSGsD9wOHAdSxvMNAZ6GFmH0ra\nCngX+H/ABEk3AW8BY7yPmwGnA7ua2TuSOgGbmNnt2XHvMrNeWVsOBEaY2U1luldxrDw4vQzYzcxe\nlvQJoKHcGEnaAGgEFkra1syey7IXAN8FTi23rx/zUf/3zAplAC4Fhknaw8zGA2cBl5rZC5IeAAaa\n2XQf725mNhO43Nv3MtDfzP5apf4QQgirUczshdyewPtm9qtSgplNMbNHgCOAx8zsPk9fBHwTGJnt\n/wiwm8+CdQK2AyZXONbDnr/amNlS4ElgyzLZWwCvmdmHXvZlM/uLmb0BnA/8FPg6MNXMHgU2Bf4O\nLPTyC81s/ko0r9pYrUf6RewtP9YSM5tVoZ6vAncCY0nBY+4KYLCkjYo7+TH7AceW2W85PkZfBy6S\ntBvQH/iZZ28CvO7llnqgF0II4Z9IBHshtzMwsULeTsU8M5sHdJK0fimJNFO1H3AwaSawkgGk2a2V\nMbiwjLtunilpHaAPcG+ZfW8EBvh+F0jaJcv7FbAjMAI4xdOmAG8A8yVdKWlAjW08L2tfPrtYcazM\n7G3ffkHSDZKGSqr0s9oE3OCvpkLeQlLA950y+x0M3Gtms4G3JDVW64SZTQIeIs2UftPM3vesC4E5\nkm6R9N8+C1kTScdLapbUzKJa9wohhNBWEeyFVa00wzSEFIAUXSdpMrA7cLKnlXt6bC1PlB1nZr2y\n12JP7+LHeIM0eze1ReVmLwPdgNOAD4EHJO3teR+Sli7vMbPS7NpSYH/gUGA2MFrSqBraOCJr39BC\nXsWxMrPjgL1JM5Mnk4K25UjaDOgKPOpB2/uSdi4Uuwg4WtJ6hfQmP36pHcVAsZyLgRd8prfUzjOB\n/yAFrkcBv6uhntK+l5lZbzPrTcda9wohhNBWcc1eyM0gBTPlzAS+mCdI2hZYaGZ/S5f7gZk9Kak7\nsMjMZpfSM0PNrLmQ9hawIekaM4CNsvcronTN3sbAY5IOMrMWs4xmtgS4B7hH0hvAQOABz/7QX3l5\nIwVfT0q6n3Q94agVbWRrY2Vm04Bpkq4F5gPDClUcThq3+b7v+qSg7fSsjr9Kuh44sZTmy7p7Ad0l\nGbAGYJJGtNLkFmPix5gLzJV0ObBA0qfM7J3W+h9CCOHjETN7Ifcg8Am/eQIAST0k9Sfd4NCvdAep\nL5leRLq2rWgk8L02HHc88DWvdw3gSNKS4UrxmzlGkmbvliNpV0md/X0HoAfwQqW6JHWWtGuW1Kta\n+TZoMVaSOknao4ZjNQH7m1mDmTWQbtQod/3dz4ATWPbL3aHAtWa2je+7NSmY7N/Wxks6QMui1K7A\nEtK1jSGEEP5JRLAXPuIzV4OAfZQevTID+DHwui+RHgycIWkW6Xq7p0h3qxbrucfM2hKs/QjYTtIU\nYBIwF/htDfsVr9nrW6bMbUBHD1hzmwJ3+mNGpgIflOtLZi3gfEnP+hLxYMpfC1d0XqGNa+eZFcZK\nwClKj6mZDPyAwqye3/27DfDRI1f8hpF3JPUpHGMB6S7r0vV0Tb6du5llS7ndJL2cvQ6r0r9hQGlM\nrgKOKN30EkII4Z+D0vd7CCG0H3WWUfFJgv+e7Mz4vzmEUJ2kiWbWu7VyMbMXQgghhFDHItgLIYQQ\nQqhjcTduCKHdNXZupPnM4k3aIYQQVoWY2QshhBBCqGMR7IUQQggh1LEI9kIIIYQQ6lg8eiWE0O7i\n0Sv/3uIxMyGsmHj0SgghhBBCiGAvhBBCCKGeRbAXQgghhFDHItgLIYQQQqhjEey1I0mbSxoraZ6k\niZLulrS95+0k6UFJsyTNkfR9SfK8YZJM0j5ZXQM97VDfHu/7TpH0mKRunv68pI2z/faQdJekYyRN\n9td7kqb5+3P9eG9m+ZMl7SipQdJi354p6RpJa1Xoa8X+eP6XJTV7PZMkXZDlHSVpurdpkqSTsz72\nzso1SJqe9esdb9szks4stOdCSa9I6uDbrfV/TLbv8ZKe9deTkvpleeMlNWfbvSWN9/cdJV3ndU+X\n9KikThXGq5d/nvsX0heWKTuqNCaVSPqUfz5z/Xy7xtO6Z/1+W9J8f/+HfDzLHUvSVVn5yZIe9/T8\nfHlW0vBqbQshhLB6RbDXTjzQuRUYb2ZdzKwROA3YTNK6wB3AuWbWDegJ9AW+kVUxDRiSbTcBUwqH\nGWpmPYGrgfOqtcfMrjSzXmbWC3gV2NO3R3qRcaV8f8309Hm+T3dgK+DwMn2t2h9JOwNjgCPNbEeg\nNzDX874MnAT8p5l1Bz4PvFOtL5lHvG29gSMl7ep1dgAGAS8BX6qx/6W+HAicAPQzsx2ArwPXS9o8\nK7apt7voO8AbZtbdzHYGjgXer9D2JuBR/3dV+A3wnJltZ2ZdgPnA5WY2Lev3HcAI396nam3LjMjO\nib5Z+jivc3fgdElbr6J+hBBCaKMI9trPnsD7ZvarUoKZTTGzR4AjgMfM7D5PXwR8E8gDj0eA3SSt\n5bND2wGTKxzrYc9fbcxsKfAksGWZ7Nb6cwpwtpk9W6rLzH7peacBJ5vZq563xMx+3ca2vQtMZNkY\n7AHMAH5J24OpU0kBzgKv+2lSMH1iVuY84PQy+24BvJK1a5aZLSkW8l8EDgOGAftKWqeNbSzWtx3Q\nCPwoS/4h0FtSl5WpuzVm9hYpcN9idR4nhBBCZRHstZ+dSQFIOTsV88xsHtBJ0vqlJOAPwH7AwaRZ\nmUoGkGYCV8bgwjLuunmmByR9gHvL7Ntaf6qNRbW8mkj6NGlGcIYnNQE3kGZWD6i09FxBi74AzZ5e\nMgF4T9KehXJXAKdKmiDpLEldKxyjLzDfx2g8cEAb2lfOjsBkD8iBj4LzyYV2l9Ml/9xJM5m587L8\n64o7S/oMsA4wtUze8b5038yitnYphBBCrSLY+9c2lrSUO4QUvBRd51/QuwOla7rKPb20lieaFpdx\nF3t6Fz/GG8BrZtbiS301aq0v/SVNAu4jLSHPkLQ28BXgNjP7G/AEKWBe1c4CzliuYWaTgW1JM38b\nAU9J+lyZfZtIny3+76payl0R8/LPHfhVIT9fxh2apQ+WNJU0q3eJmf2jWLGZXWZmvc2sNx1XYw9C\nCOHfXAR77WcGaWmtnJnFPEnbAgs9QAHAzJ4kXSu3sZnNLlPPUP8SHmhmL3naW8CGWZmNgAUr2AdY\nds1eF6BR0kFlyrTWn2pjUS2vtb48Yma7mFljtly+H7ABME3S80A/2hZMteiLb8/IE8zsQWBd0oxi\nnr7QzG4xs28AvyUFnh+RtAZwCPC/3r5fAPtLWq8NbSzX5l6lm1H8OB2AXp63Oowzsx6kWcpzC9c0\nhhBC+BhFsNd+HgQ+Ien4UoKkHpL6A9cB/eR32/qS6UXAT8vUMxL4XhuOOx74mte7BnAk8NCKdCDn\n17CNJF1jV9Raf84DvqdldyJ3kFRaLvwxaalwc89bW9JxWV+O9GvcAI6uoS9NwHFm1mBmDcBnSdfF\n1Tq39FPgJ740jKRepGvrLilT9izS9Yh42d0lbVjqB2l59YXCPnsDU81sa2/jNsDNpBtKVoiZzQUm\nsfxM4xnA05632phZM3At6eaUEEII7SCCvXZi6Y8SDwL28UdhzCAFNq/7EunBwBmSZpGut3uKdMdq\nsZ57zKwtwdqPgO0kTSEFAHNJM0ytKV6z17dMmduAjh6w5m2s2h9f+j0JuEHSM8B00nInZna3l/uD\nj9HTQOm6xcuAvwNTvD+dgPMrdcADuv2B32Vte5d01+uAGsYAM7uDdO3d45KeBX5Nuov4tTJl7wbe\nzJK6AH+UNI009s2kQC7XRLqWMHczy2YfO0p6OXv9P08/I08v0/Rjge39XJsHbO9pK+u8wnmxdpky\nPwGOWcnZyRBCCCtIKeYIIYT2o84yTmjvVoT2YmfG91AIK0LSRDPr3Vq5mNkLIYQQQqhjEeyFEEII\nIdSxNdu7ASGE0Ni5keYzm1svGEIIoc1iZi+EEEIIoY5FsBdCCCGEUMci2AshhBBCqGPx6JUQQruL\nR6+EULt4VE0oiUevhBBCCCGECPZCCCGEEOpZBHshhBBCCHUsgr0QQgghhDoWwV5YZSRtLmmspHmS\nJkq6W9L2nreTpAclzZI0R9L3JcnzhkkySftkdQ30tEN9e7zvO0XSY5K6efrzkjbO9ttD0l2SjpE0\n2V/vSZrm78/1472Z5U+WtKOkBkmLfXumpGskrVWmn8Vyv5LUobX9JfWT9KSkZ/11fKHeIyVNlTTD\n+3m5pA3K9P8pSb0K+/by8dq/kG6Sfpttr+l9v2sFxr40Vjd5+ihJr2T9bcrq+LykJzzvGUmjWj2B\nQgghrBYR7IVVwgO3W4HxZtbFzBqB04DNJK0L3AGca2bdgJ5AX+AbWRXTgCHZdhMwpXCYoWbWE7ga\nOK9ae8zsSjPrZWa9gFeBPX17pBcZV8r310xPn+f7dAe2Ag6vcIhSuR7AjsDAavtL2hy4Hvi6me0A\n9ANOkHSA5+8PDAe+bGY7AbsCjwOblen/JWX63wQ86v/m3gV29s8AYF/glUKZWse+NFaHZumjvb8H\nA5dmwe3VwPGetzNwIyGEENpFBHthVdkTeN/MflVKMLMpZvYIcATwmJnd5+mLgG8CI7P9HwF2k7SW\npE7AdsDkCsd62PNXGzNbCjwJbNlKuQ9IQdl2hfTi/icCV5nZ056/ADiFZWNwOnCymb1S2t/MrjCz\nWWUOOyFvlwfahwHDgH0lrVMofzdwgL9vAm4o5Ldl7MsysznAImBDT9oUeC3ry8xK+4YQQli9ItgL\nq8rOwMQKeTsV88xsHtBJ0vqlJOAPwH6kWaI7qhxrAGk2amUMLizjrptnesDUB7i3WiWSOgJ7F9tT\nZv8WYwA0e3op/+ka274/cFu23ReY72M6nmWBXclYYIi3qQfwRCG/lrG/LhurFrOqknYF5pjZnz1p\nNDBL0q2STigTgIYQQviYRLAX/pmMJS0nDqHl7BN4wAHsDpzsaeWeLlrLE0eLy7iLPb2LH+MN4DUz\nm1ph/1K5x4Dfmdk9bdy/IkndPaiaJ2lwlnWdpPmkWcCLs/Qm0tjh/y63lOttaPD0uysctrWxz5dx\nR2TpwyXNIAWQZ2fH/CHQG7iPNLPbImiWdLykZknNLKrQqhBCCCstgr2wqswAGivkzSzmSdoWWGhm\nfyulmdmTpGvdNjaz2WXqKQUcA83sJU97i2VLhwAbAQtWsA+w7Jq7LkCjpIOqlTOzXcxsVA37txgD\n357h72eQrtPDzKZ5HfcA+YzjUGBb0vVwvwCQtAZwCPC/kp739P0lrVc41h3A+ZQP5GoZ+0pG+zWG\nhwC/yWfwzGyemf2SNPPZU9KnC8e8zMx6m1lvOrbhiCGEENokgr2wqjwIfCK/w1RSD0n9geuAfqU7\nPn3J9CLgp2XqGQl8rw3HHQ98zetdAzgSeGhFOpDza+pGkm4yWRX7XwwMK91F64HPT1g2Bj8Gzpe0\nVVbNckvLXq8B3wc+L2kHUiA11cy2NrMGM9sGuBkYVNj1CuAHZlZt+butY5+36w7SsvTRAJIO8GsJ\nAboCS4G/rkjdIYQQVk4Ee2GV8CBkELCPLz/OIAUwr/sS6cHAGZJmka5vewoYU6aee8ysLcHaj4Dt\nJE0BJgFzgd9W3wVoec1e3zJlbgM6esC6Ij7a38xeIwWiv5b0LOmmjivM7E4As//f3p2HS1WceRz/\n/lxQEdfgAuJ4Xcm4Ihh3ErdEk+CWGBF1FBNHE3WSaMR9RpNxMokY9SEal4kao7gNKqLRaKISF4wK\nyiaKYgDFiAqZqAiDCO/8UdVyaLrvxoXuaX6f5zkPferUqfOegvvcl6o6p+MhUgL8cH6NyShSgvRI\neaO5P38BDCJNzd5XVuUelp7KnRERQ5oLtoW+L67Z+2OVOj8BzpK0CikBn5yntG8ljcoubO76Zma2\nfCj9jjYzqx11V3BqraMw+/8hLvbvbUskjYmI3Vqq55E9MzMzswbmZM/MzMysgTnZMzMzM2tgq9U6\nADOzPt37MPri0bUOw8ysIXlkz8zMzKyBOdkzMzMza2BO9szMzMwamN+zZ2Y15/fsma3c/O7A9vF7\n9szMzMzMyZ6ZmZlZI3OyZ2ZmZtbAnOythCRtKulOSW9IGiPpIUnb5WM7SHpc0mRJr0v6V0nKxwZK\nCkkHFdo6IpcdlfdH5nPHSXpGUs9cPk1S18J5+0l6UNJJksbm7RNJE/Lnn+XrvV84PlbS9pKaJM3L\n+5Mk/VbS6hXucxVJQyRNzO2+IGlLSUMlfa9Qbw9J4yWtLunbue74fN7hkq4pXGteIZajJP1G0tRC\n2ag29FU/SS/lvpokqeqqNUnDJf25rOwSSXMlbVwom1NWp3TNzxfKmiRNLKv3PUlDC/vrS/qLpC0k\n7SPpuXx/r+R/EydX+Xv7D0nd8r+p0n2NqHZfZma2/PmlyiuZnLjdB9wSEcfksl2ATSS9BYwAvhcR\nj0rqDNwDnAZck5uYABwD/DHvDwDGlV3muIgYLekUYDBwWLV4IuJm4OYcxzRg/4iYlfcHAndFxBll\n99AEvBERvSStCvwBOBoYypL6A92BnSNikaQewMfAWcCzkoYBs4Gr8z1uAlwI9I6IDyR1ATaKiPsL\n130wInoVYukHDIqIYRVur2pf5eT0BmD3iJghaQ2gqVIfSVof6APMkbRVRPylcHgW8CPg3Ern5ms+\nnf+8uEodgOuBgZL2i4iRwKXA9RExXdJjwBERMTH3d8+ImAT8Osc3A+gbEX/P+zcCv4uIa/L+zs1c\n18zMljOP7K189gcWRMR1pYKIGBcRTwHHAs9ExKO5fC5wBnBe4fyngN3zKFgXYBtgbJVrPZmPLzcR\nsRB4HtiswuFuwDsRsSjXnRER/xMR7wKXA5cB3wXGR8TTwMbAR8CcXH9ORExdhvCa66t1SP/Zmp2v\nNT8iJldp5xvAA8CdpOSx6Cagv6QNy0/K19wX+E6F85aQ++i7wBBJuwN9gSvy4Y2AmbnewpzoNacb\nMKPQ9vgW6puZ2XLkZG/lsyMwpsqxHcqPRcQbQBdJ65aKSCNVBwOHk0YCqzmUNLq1LPqXTeOuVTwo\naU1gD+D3Fc69Gzg0n/cLSbsWjl0HbA8MAs7JZeOAd4Gpkm6WdGgrYxxciK84uli1ryLib3l/uqQ7\nJB0nqdrP4wDgjrwNKDs2h5Tw/aDCeYcDv4+I14DZkvo0dxMR8RLwBGmk9IyIWJAPXQW8LuleSf+c\nR1v0F2AAABBrSURBVCGbczVwi9JygAskdWuhvpmZLUdO9qw9SiNMx5ASkHJDJY0F9gHOzmWVXqLU\nmhcr3RURvQrbvFy+db7Gu6TRu6VGjyJiBtATOB9YBDwm6cB8bBFp6vLhiCiNri0EDgGOAl4DrpR0\nSStiHFSI77iyY1X7KiJOBg4kjUyeTUraliBpE2Bb4OmctC2QtGNZtSHAiZLWKSsfkK9fiqM8Uazk\nGmB6HuktxXkx8AVS4noC8LvmGoiIh4CtgRtJCfVLkj5X4d5OkTRa0mjmtiIyMzNrF6/ZW/m8TEpm\nKpkEfLFYIGkrYE5EfJiW+0FEPC9pJ2BuRLxWKi84LiLKv9V+NrABaY0ZwIaFz+1RWrPXFXhG0mER\nsdQoY0TMBx4GHpb0LnAE8Fg+vChvxfpBSr6el/QH0nrCS9obZEt9FRETgAmSbgWmAgPLmjia1G9T\n87nrkpK2Cwtt/F3S7cDppbI8rXsAsJOkAFYFQtKgFkJeqk/yNaYAUyT9Gpglab2I+KCZ+55NWkM5\nVNLvSdPJ95fVuYG0bjG9VNnMzJYLj+ytfB4H1sgPTwBpAb2kvqRfzvsqP0Gap0yHkNa2lTsPuKAN\n1x0J/FNud1XgeNKU4TLJD3OcRxq9W4Kk3pK658+rADsD06u1Jam7pN6Fol7N1W+DpfpKUhdJ+7Xi\nWgOAQyKiKSKaSA9qVFp/dwVwKov/A3cUcGtEbJHP3ZyUTPZta/CSvq7FWeq2wHzS2sZq9Q8sTbfn\n6f8tgTfbel0zM+sYTvZWMnnk6kjgIKVXr7wM/CcwM0+RHg5cJGkyab3dC6Q1WOXtPBwRbUnW/h3Y\nRtI44CVgCnBbK84rX7O3d4U6w4HOOWEt2hh4QOk1I+OBTyvdS8HqwOWSXs1TxP2pvBau3OCyGDsV\nD1bpKwHnKL2mZizwY8pG9fLTv1sAn71yJT8w8oGkPcquMYv0lHVpPd2AvF90D4uncntKmlHYvtXM\n/Q0ESn3yG+DY0kMvVXwBeFHSeGAUcG1eD2hmZjXg78Y1s5rzd+Oardz83bjtI383rpmZmZk52TMz\nMzNrYE72zMzMzBqYX71iZjXXp3sfRl9c/rYeMzPrCB7ZMzMzM2tgTvbMzMzMGpiTPTMzM7MG5vfs\nmVnN+T17ZrYyWtb3C/o9e2ZmZmbmZM/MzMyskTnZMzMzM2tgTvaWA0kXSnpZ0nhJY0tfWi+pk6Sr\nJE2R9Lqk+yX1KJy3MNcvbacWPs+RNDl//q2klyT1yuetlo8fX2hrjKTehf3hkv5cFuclkt7ObU6S\nNKBw7DeSphauP6rCfe4n6YMcy2RJT0rqV6X90tZd0mxJ65a1NVxSf0kDJb1fds72kpokTawQgyRd\nlPvzNUlPSNqhcHyapAn57+JRSZuWlZeuMaTCfY+TdGChrX75Xsfl/qq6yqwV/T1R0mGtKD+7Qttz\nCp+3k/RQvv8XJd0taZPC8aty26vk/ZMK9/xJoQ9+lvv+6sK5p0h6NW/PS9q3cGykpNGF/d0kjcyf\nO0samtueKOlpSV2q9ZWZmS1ffqlyB5O0F9AP6B0R8yV1BTrlwz8F1gF6RsRCSScB90raI9KTMvMi\noldZk9fndkcCZ0fE6Lx/NbA3MBbYBXgt798maW1ga2Bcrrs+0AeYI2mriPhLof0rI+JySdsCYyQN\ni4gF+digiBjWwi0/FRH98nV6AcMlzYuIx4rtl/XRI8CRwC15fz1gX+BY4Gjgrog4o+ycpirXPz3f\n9y4RMVfSV4ARknaIiP/NdfaPiFmSfgpcAHy/WF6hzUERMUzS/sANwLaSVs+fd4+IGZLWACrG1Mr+\n/kfgKUkbt1BelaQ1gd8BZ0XEA7lsP2Aj4N2c4B0JvAV8CXgiIm4Gbs51pxX7QNLAQtv9gFOBfXPf\n9Sb93e4eETNztY0lfTUiHi4L7QfAuxGxU26rJ7AAMzOrCY/sdbxuwKyImA8QEbMi4q+SOgMnAWdG\nxMJ87GZgPnBAO64zipTkkP+8DiglirsDY0rXAb4BPADcCRxTqbGIeB2YC2zQjlhKbYwFfgKc0ULV\nO8riOBJ4JCLmtuOy5wJnlM6NiEdJfXNchbpPAtu0oe1ngc3y53VI/zmana8zPyImVzmvNf39CvAp\n0LU15VUcCzxbSvTy+SMjojQCuh/wMnAtMGDp05t1LinpnZXbfZGUnJ9eqDMYuLDCud2AtwsxTS79\nPJiZ2YrnZK/jPQpsnqcUfyXpS7l8G+DNiPiwrP5ooDTtuFZhiu2+Fq7zDEsme08C8yWtk/eL064D\nSAnWHVT5pZ9Hbl6PiPcKxYML8QxtIZ6SF4HPF/bPLLTxRC57BOgt6XN5/5gcW0l/LTmNu1aVmNcF\n1i4bOYMl+7SoHzChsP9E4RpnVqh/CDAcICL+BowApku6Q9JxpanRClrT33sAi4D3W1NexY7AmGaO\nl+K4D/h6Hp1srR0qtF3er88Cn+QR0KKbgHMlPSvp0jxqbGZmNeJp3A4WEXMk9QH6AvsDd0k6j5QE\ntaTSNG6160xXWgO4KSm5mgy8AOxBSvZ+CZDXb20LPB0RIWmBpB0Loz9n5unk7YBDyy7Tmmnccirb\nX2oaNyI+kTQCOErSPcCupASwpNI0bhvDWMITkhYC44GLCuXVpnEH5ynfHsBehbhPlrQTcBBwNvBl\nYGBZnK3p7+OBj4D+uU5z5e0iqRPwNdIU70eSngMOBh5sd6OVXUrq03NLBRExVtJWwFdIffWCpL3y\nqGUxxlOAUwBYr4OjMjOzz3hkbzmIiIV5Ou1i0pTmN4E3gH/II29FfUhTbe0xCvgW8E5e8/dnYB/S\nNO6zuc7RpKnZqXmNVhNLjjZdGRE75BhvzOvAlsWuwCst1lo8lXsUcH9hnWCr5VHSj3NiUVTep/tH\nRK+IOCEi/t6KpgdFxHakBOamsmtOiIgrSYneNyuc25r+7hURfSPiqVaUN+dl0r1WcjCwPjAhx7Ev\nbZvKnVSh7aX+rUbE48BawJ5l5XMi4t6IOA24jZR4UlbnhojYLSJ2o3MbIjMzszZxstfBJPUsm7bq\nBUyPiI9Ja56ukLRqrnsC0Bl4vJ2XGwX8kMWJ3bPACcDMiPgglw0ADomIpohoIv3CXmodWUSMIE3T\nndjOWJC0M/CvwDWtqD6SNAJ2OktO4bbVYGBIaapX0kGkxOb2ZWiz5GpgFUkHS+qSH34o6QVMr3BO\nq/q7g9wO7C3p66UCSV+UtGOO4+RCHFsCX85rR1vjMuDnpan2/PDNQOBXFepeCpxTiGEfSRvkz52A\n7ancV2ZmtgJ4GrfjdQF+mZ/I/BSYQmmqCs4HLgdek7QIeBU4Mtr/nXXPAFeSk72IeCcnkqPgsydY\ntyCN+JHrTFV6XcoeFdr7CXC7pP/K+4MlFac9d4+IT8rO6SvpJVLS+h7w/cKTuLB4erLkiIiYFhGL\nJA0jjYT9qazN/iq85gM4Dfgr0FPSjGLbpOnqDUgjWAuBmcDhETGvwv2VK03vAoyPiBOKB/NUaimR\nOQI4R9L1wDzgY5aewm2ibf3dGhdJ+mGhvR6Fz/PyU7NXSbqK9MTreNKI5CHAdwt1P5b0NGmq/q6W\nLhoRIyRtBoySFKTp5eMj4p0KdR+SVFxjuDVwrdI89CqkJ4bvactNm5lZx/F345pZzfm7cc1sZeTv\nxjUzMzOzZeZkz8zMzKyBOdkzMzMza2B+QMPMaq5P9z6Mvnh0yxXNzKzNPLJnZmZm1sCc7JmZmZk1\nMCd7ZmZmZg3MyZ6ZmZlZA3OyZ2ZmZtbAnOyZmZmZNTAne2ZmZmYNzMmemZmZWQNzsmdmZmbWwBQR\ntY7BzFZykj4CJtc6jmZ0BWbVOohmOL72q+fYwPEtq3qOryNi2yIiNmqpkr8uzczqweSI2K3WQVQj\nabTja796jq+eYwPHt6zqOb4VGZuncc3MzMwamJM9MzMzswbmZM/M6sENtQ6gBY5v2dRzfPUcGzi+\nZVXP8a2w2PyAhpmZmVkD88iemZmZWQNzsmdmNSXpEEmTJU2RdN4Kuubmkp6QNEnSy5J+kMs3lPQH\nSa/nPzconHN+jnGypIML5X0kTcjHhkhSB8a5qqSXJD1Yb/FJWl/SMEmvSnpF0l71Ep+kM/Pf60RJ\nd0has5axSbpJ0nuSJhbKOiweSWtIuiuXPyepqQPiG5z/bsdLuk/S+vUUX+HYjySFpK71Fp+kf8l9\n+LKky2oVHwAR4c2bN2812YBVgTeArYBOwDhg+xVw3W5A7/x5HeA1YHvgMuC8XH4e8PP8efsc2xrA\nljnmVfOx54E9AQEPA1/twDjPAm4HHsz7dRMfcAtwcv7cCVi/HuIDNgOmAmvl/buBgbWMDfgi0BuY\nWCjrsHiA04Dr8udjgLs6IL6vAKvlzz+vt/hy+ebAI8B0oGs9xQfsD/wRWCPvb1yr+CLCyZ43b95q\ntwF7AY8U9s8Hzq9BHPcDXya92LlbLutGev/fUnHlXzB75TqvFsoHANd3UEw9gMeAA1ic7NVFfMB6\npIRKZeU1j4+U7L0FbEh6l+yDpMSlprEBTWXJQIfFU6qTP69GelGvliW+smNHAkPrLT5gGLALMI3F\nyV5dxEf6T8ZBFerVJD5P45pZLZV+MZfMyGUrTJ4S2RV4DtgkIt7Jh2YCm+TP1eLcLH8uL+8IVwHn\nAIsKZfUS35bA+8DNStPMv5a0dj3EFxFvA5cDbwLvAB9ExKP1EFuZjozns3Mi4lPgA+BzHRjrt0kj\nTXUTn6TDgbcjYlzZobqID9gO6JunXf8k6Qu1jM/JnpmttCR1Ae4BfhgRHxaPRfpvdE1eVyCpH/Be\nRIypVqeW8ZFGF3oD10bErsDHpKnIz9Qqvrz27XBSQtodWFvS8fUQWzX1Fk+RpAuBT4GhtY6lRFJn\n4ALg32odSzNWI40u7wkMAu7uiPWo7eVkz8xq6W3SupuSHrlsuZO0OinRGxoR9+bidyV1y8e7Ae+1\nEOfb+XN5+bLaBzhM0jTgTuAASbfVUXwzgBkR8VzeH0ZK/uohvoOAqRHxfkQsAO4F9q6T2Io6Mp7P\nzpG0GmmaffayBihpINAPOC4npPUS39akZH5c/hnpAbwoadM6iQ/Sz8i9kTxPGqHvWqv4nOyZWS29\nAGwraUtJnUiLj0cs74vm/2HfCLwSEVcUDo0ATsyfTySt5SuVH5OfitsS2BZ4Pk/DfShpz9zmCYVz\n2i0izo+IHhHRROqTxyPi+DqKbybwlqSeuehAYFKdxPcmsKekzrnNA4FX6iS2oo6Mp9jWUaR/L8s0\nUijpENIygsMiYm5Z3DWNLyImRMTGEdGUf0ZmkB64mlkP8WXDSQ9pIGk70kNMs2oWX1sW+Hnz5s1b\nR2/A10hPw74BXLiCrrkvadpsPDA2b18jrYN5DHid9CTdhoVzLswxTqbwVCawGzAxH7uaNi6cbkWs\n+7H4AY26iQ/oBYzOfTgc2KBe4gN+DLya272V9ORjzWID7iCtH1xASky+05HxAGsC/w1MIT3RuVUH\nxDeFtE6s9PNxXT3FV3Z8GvkBjXqJj5Tc3Zav9yJwQK3iiwh/g4aZmZlZI/M0rpmZmVkDc7JnZmZm\n1sCc7JmZmZk1MCd7ZmZmZg3MyZ6ZmZlZA3OyZ2ZmZtbAnOyZmZmZNTAne2ZmZmYN7P8ARfwHckPI\ngFsAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f925ff48350>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Top 10 soc with status\n",
    "barPlot(top_10_soc_by_status(h1b_data_frame, \"DENIED\"),\"RED\")\n",
    "plt.show()\n",
    "\n",
    "barPlot(top_10_soc_by_status(h1b_data_frame, \"CERTIFIED\"),\"GREEN\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Insight\n",
    "Most number of applications both CERTIFIED and DENIED are for job profiles related to Software and computer professionals. This is an obvious insight and reaffirms the general sentiment that software developers and professionals have the largest share of H-1B visas. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 5. Shows count of each status."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\nYwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f925d4aa690>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "class_count = h1b_data_frame.map(lambda x : (x.CASE_STATUS, 1)).reduceByKey(lambda x, y : x+y)\n",
    "barPlot(class_count)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Data Modelling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Define a method to evaludate any model\n",
    "def evaluate_model(test, model):\n",
    "    '''\n",
    "    Returns accuracy based on correct predictions\n",
    "    '''\n",
    "    predictions = model.predict(test.map(lambda x: x.features))\n",
    "    predictionAndLabel = test.map(lambda y: y.label).zip(predictions)\n",
    "    return 100*predictionAndLabel.filter(lambda x : x[0] == x[1]).count()/float(test.count())        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Split the data into train and test data\n",
    "training, test = h1b_labeled_points.randomSplit([0.7, 0.3])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "#### 1. NaiveBayes Classification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model trained in 2496.039 seconds\n",
      "\n"
     ]
    }
   ],
   "source": [
    "t0 = time()\n",
    "#Train a NaiveBayes model and record time for training\n",
    "model_nb = NaiveBayes.train(training, 1.0)\n",
    "\n",
    "tt = time() - t0\n",
    "print(\"Model trained in {} seconds\\n\").format(round(tt,3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy = 17.3159303882\n"
     ]
    }
   ],
   "source": [
    "accuracy = evaluate_model(test, model_nb)\n",
    "print('Accuracy = ' + str(accuracy))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "NaiveBayes model gives very less accuracy. Which means predictibility of the case status doesn't depend on the previous case. It is very aparent that we need other models to make better predictions. So, we try Logistic Rgression next."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2. Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model trained in 2460.865 seconds\n",
      "\n"
     ]
    }
   ],
   "source": [
    "t0 = time()\n",
    "#Now train a logistic regression model\n",
    "model_lg = LogisticRegressionWithLBFGS.train(training, numClasses=4)\n",
    "\n",
    "tt = time() - t0\n",
    "print(\"Model trained in {} seconds\\n\").format(round(tt,3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy = 77.9484605087\n"
     ]
    }
   ],
   "source": [
    "accuracy = evaluate_model(test, model_lg)\n",
    "print('Accuracy = ' + str(accuracy))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Logistic Regression model gives a lot better accuracy. Can we increase the accuracy further with a more complex classification model?<br/>\n",
    "\n",
    "We try the Decision Tree Classifier next."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 3. Decison Tree Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model trained in 2463.667 seconds\n",
      "\n"
     ]
    }
   ],
   "source": [
    "t0 = time()\n",
    "#Build the decision tree model\n",
    "model_dt = DecisionTree.trainClassifier(training, numClasses=4, maxDepth=4,\n",
    "                                     categoricalFeaturesInfo={},\n",
    "                                     impurity='gini', maxBins=32)\n",
    "tt = time() - t0\n",
    "print(\"Model trained in {} seconds\\n\").format(round(tt,3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy = 82.6539491299\n"
     ]
    }
   ],
   "source": [
    "#Evaluate the model on test data\n",
    "accuracy = evaluate_model(test, model_dt)\n",
    "print('Accuracy = ' + str(accuracy))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Great!. The accuracy has increased further. We used a tree with depth 4 and cost function based on \"gini\". This gives us an accuracy of 82.65%. This is decent enough and can help us predict outcomes of applications based on the features. <br/>\n",
    "\n",
    "Can we increase the accuracy further? What can we do? We can change the parameters of the tree itself and tune the tree. Or we can create an ensemble of the trees and see what happends to the accuracy.<br/>\n",
    "\n",
    "Intuitively, we chose the later. An ensemble of Decision Trees is basically a Random Forest. So, we run the Random Forest on our data."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 4. RandomForest Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model trained in 3434.491 seconds\n",
      "\n"
     ]
    }
   ],
   "source": [
    "t0 = time()\n",
    "#Train a RandomForest model. Number of trees taken here is 3\n",
    "model_rf = RandomForest.trainClassifier(training, numClasses=4, categoricalFeaturesInfo={},\n",
    "                                     numTrees=3, featureSubsetStrategy=\"auto\",\n",
    "                                     impurity='gini', maxDepth=4, maxBins=32)\n",
    "tt = time() - t0\n",
    "print(\"Model trained in {} seconds\\n\").format(round(tt,3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy = 81.9262349358\n"
     ]
    }
   ],
   "source": [
    "#Evaluate the model on test data\n",
    "accuracy = evaluate_model(test, model_rf)\n",
    "print('Accuracy = ' + str(accuracy))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The accuracy drops a little bit with Random forests. <br/>\n",
    "At this point we are confident that decision tree classifier is the model we would want to use. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Conclusion\n",
    "We have a decent model in our DecisionTreeClassifier and can be used for predicting outcomes of applications based on several features. Although, we have an accuracy of 82% which is not great but is decent enough to make predictions. However, we believe this can be further improved by tuning the hyperparameters."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Future Scope\n",
    "1. The decision tree model is not very good but decent enough. We got good results on the test data which is encouraging and the model can be improved with better training and tuning of the hyper-parameters. <br/>\n",
    "2. An application can be built on top of this model that takes user inputs like Employer, Wage, State etc. and gives a probability of approval of the H-1B application."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Learnings\n",
    "1. Using PySpark and its libraries to do analysis on data.\n",
    "2. The map-reduce way of coding.\n",
    "3. Using matplotlib library for visualization.\n",
    "4. Understanding the classifier models."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
