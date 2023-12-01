{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/amelia/opt/anaconda3/lib/python3.7/site-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.\n",
      "  import pandas.util.testing as tm\n"
     ]
    }
   ],
   "source": [
    "#libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import os\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy.stats import pearsonr\n",
    "import statsmodels.formula.api as smf\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "#function that returns df with info about nulls\n",
    "def missing_data(input_data):\n",
    "    total = input_data.isnull().sum()\n",
    "    percent = (input_data.isnull().sum()/input_data.isnull().count()*100)\n",
    "    table = pd.concat([total, percent], axis=1, keys= ['Total', 'Percent'])\n",
    "    types = []\n",
    "    for col in input_data.columns:\n",
    "        dtype = str(input_data[col].dtype)\n",
    "        types.append(dtype)\n",
    "    table[\"Types\"] = types\n",
    "    return(pd.DataFrame(table))\n",
    "\n",
    "def mape(actual, pred):\n",
    "    #mean absolute percentage error function\n",
    "    actual, pred = np.array(actual), np.array(pred)\n",
    "    return np.mean(np.abs((actual - pred) / actual)) * 100\n"
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
      "Collecting pyspark\n",
      "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/0c/66/3cf748ba7cd7c6a4a46ffcc8d062f11ddc24b786c5b82936c857dc13b7bd/pyspark-3.4.1.tar.gz (310.8MB)\n",
      "\u001b[K     |████████████████████████████████| 310.8MB 59kB/s s eta 0:00:01    |██▍                             | 23.1MB 28.1MB/s eta 0:00:11     |████                            | 38.8MB 28.1MB/s eta 0:00:10     |████████▏                       | 78.8MB 31.0MB/s eta 0:00:08     |███████████████                 | 145.8MB 467kB/s eta 0:05:54     |███████████████▏                | 147.7MB 467kB/s eta 0:05:50     |█████████████████               | 165.3MB 467kB/s eta 0:05:12     |████████████████████████        | 232.1MB 34.1MB/s eta 0:00:03     |███████████████████████████▌    | 267.0MB 15.1MB/s eta 0:00:03     |████████████████████████████▉   | 279.8MB 59.8MB/s eta 0:00:01     |███████████████████████████████ | 301.5MB 59.8MB/s eta 0:00:01\n",
      "\u001b[?25hCollecting py4j==0.10.9.7 (from pyspark)\n",
      "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/10/30/a58b32568f1623aaad7db22aa9eafc4c6c194b429ff35bdc55ca2726da47/py4j-0.10.9.7-py2.py3-none-any.whl (200kB)\n",
      "\u001b[K     |████████████████████████████████| 204kB 19.2MB/s eta 0:00:01\n",
      "\u001b[?25hBuilding wheels for collected packages: pyspark\n",
      "  Building wheel for pyspark (setup.py) ... \u001b[?25ldone\n",
      "\u001b[?25h  Created wheel for pyspark: filename=pyspark-3.4.1-py2.py3-none-any.whl size=311285411 sha256=9d1192d9f8294366acb55b786e2dd060e6fbadd27d025d099b4be7db0ef5e7d1\n",
      "  Stored in directory: /Users/amelia/Library/Caches/pip/wheels/7e/4b/a8/30e2caecc11dfe0ee6f6e7d4c0900fa160258508689fed6567\n",
      "Successfully built pyspark\n",
      "Installing collected packages: py4j, pyspark\n",
      "Successfully installed py4j-0.10.9.7 pyspark-3.4.1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/amelia/opt/anaconda3/lib/python3.7/site-packages/pyspark/context.py:317: FutureWarning: Python 3.7 support is deprecated in Spark 3.4.\n",
      "  warnings.warn(\"Python 3.7 support is deprecated in Spark 3.4.\", FutureWarning)\n"
     ]
    }
   ],
   "source": [
    "!pip install pyspark\n",
    "import pyspark.sql.functions as f\n",
    "import pyspark\n",
    "from pyspark.sql import SparkSession\n",
    "spark = SparkSession.builder.appName('Time Series analysis').getOrCreate()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Explore Sephora Data: Using linear regression and data visualization techniques to uncover sephora's toxic products"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Talc is an ingredient commonly used in makeup and has 'has raised safety concerns due to potential contamination with asbestos, a toxic mineral'. Is Sephora selling products that contain this toxic ingredient and if so with what success? "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "done\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 8494 entries, 0 to 8493\n",
      "Data columns (total 27 columns):\n",
      " #   Column              Non-Null Count  Dtype  \n",
      "---  ------              --------------  -----  \n",
      " 0   product_id          8494 non-null   object \n",
      " 1   product_name        8494 non-null   object \n",
      " 2   brand_id            8494 non-null   int64  \n",
      " 3   brand_name          8494 non-null   object \n",
      " 4   loves_count         8494 non-null   int64  \n",
      " 5   rating              8216 non-null   float64\n",
      " 6   reviews             8216 non-null   float64\n",
      " 7   size                6863 non-null   object \n",
      " 8   variation_type      7050 non-null   object \n",
      " 9   variation_value     6896 non-null   object \n",
      " 10  variation_desc      1250 non-null   object \n",
      " 11  ingredients         7549 non-null   object \n",
      " 12  price_usd           8494 non-null   float64\n",
      " 13  value_price_usd     451 non-null    float64\n",
      " 14  sale_price_usd      270 non-null    float64\n",
      " 15  limited_edition     8494 non-null   int64  \n",
      " 16  new                 8494 non-null   int64  \n",
      " 17  online_only         8494 non-null   int64  \n",
      " 18  out_of_stock        8494 non-null   int64  \n",
      " 19  sephora_exclusive   8494 non-null   int64  \n",
      " 20  highlights          6287 non-null   object \n",
      " 21  primary_category    8494 non-null   object \n",
      " 22  secondary_category  8486 non-null   object \n",
      " 23  tertiary_category   7504 non-null   object \n",
      " 24  child_count         8494 non-null   int64  \n",
      " 25  child_max_price     2754 non-null   float64\n",
      " 26  child_min_price     2754 non-null   float64\n",
      "dtypes: float64(7), int64(8), object(12)\n",
      "memory usage: 1.7+ MB\n"
     ]
    }
   ],
   "source": [
    "#load data\n",
    "df = pd.read_csv('/Users/amelia/Desktop/product_info.csv')\n",
    "print('done')\n",
    "df.info(verbose=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>brand_id</th>\n",
       "      <th>loves_count</th>\n",
       "      <th>rating</th>\n",
       "      <th>reviews</th>\n",
       "      <th>price_usd</th>\n",
       "      <th>value_price_usd</th>\n",
       "      <th>sale_price_usd</th>\n",
       "      <th>limited_edition</th>\n",
       "      <th>new</th>\n",
       "      <th>online_only</th>\n",
       "      <th>out_of_stock</th>\n",
       "      <th>sephora_exclusive</th>\n",
       "      <th>child_count</th>\n",
       "      <th>child_max_price</th>\n",
       "      <th>child_min_price</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>8494.000000</td>\n",
       "      <td>8.494000e+03</td>\n",
       "      <td>8216.000000</td>\n",
       "      <td>8216.000000</td>\n",
       "      <td>8494.000000</td>\n",
       "      <td>451.000000</td>\n",
       "      <td>270.000000</td>\n",
       "      <td>8494.000000</td>\n",
       "      <td>8494.000000</td>\n",
       "      <td>8494.000000</td>\n",
       "      <td>8494.000000</td>\n",
       "      <td>8494.000000</td>\n",
       "      <td>8494.000000</td>\n",
       "      <td>2754.000000</td>\n",
       "      <td>2754.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5422.440546</td>\n",
       "      <td>2.917957e+04</td>\n",
       "      <td>4.194513</td>\n",
       "      <td>448.545521</td>\n",
       "      <td>51.655595</td>\n",
       "      <td>91.168537</td>\n",
       "      <td>20.207889</td>\n",
       "      <td>0.070285</td>\n",
       "      <td>0.071698</td>\n",
       "      <td>0.219096</td>\n",
       "      <td>0.073699</td>\n",
       "      <td>0.279374</td>\n",
       "      <td>1.631622</td>\n",
       "      <td>53.792023</td>\n",
       "      <td>39.665802</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>1709.595957</td>\n",
       "      <td>6.609212e+04</td>\n",
       "      <td>0.516694</td>\n",
       "      <td>1101.982529</td>\n",
       "      <td>53.669234</td>\n",
       "      <td>79.195631</td>\n",
       "      <td>24.327352</td>\n",
       "      <td>0.255642</td>\n",
       "      <td>0.258002</td>\n",
       "      <td>0.413658</td>\n",
       "      <td>0.261296</td>\n",
       "      <td>0.448718</td>\n",
       "      <td>5.379470</td>\n",
       "      <td>58.765894</td>\n",
       "      <td>38.685720</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1063.000000</td>\n",
       "      <td>0.000000e+00</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.750000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>3.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>5333.000000</td>\n",
       "      <td>3.758000e+03</td>\n",
       "      <td>3.981725</td>\n",
       "      <td>26.000000</td>\n",
       "      <td>25.000000</td>\n",
       "      <td>45.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>19.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>6157.500000</td>\n",
       "      <td>9.880000e+03</td>\n",
       "      <td>4.289350</td>\n",
       "      <td>122.000000</td>\n",
       "      <td>35.000000</td>\n",
       "      <td>67.000000</td>\n",
       "      <td>14.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>32.000000</td>\n",
       "      <td>28.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>6328.000000</td>\n",
       "      <td>2.684125e+04</td>\n",
       "      <td>4.530525</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>58.000000</td>\n",
       "      <td>108.500000</td>\n",
       "      <td>25.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>59.000000</td>\n",
       "      <td>42.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>8020.000000</td>\n",
       "      <td>1.401068e+06</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>21281.000000</td>\n",
       "      <td>1900.000000</td>\n",
       "      <td>617.000000</td>\n",
       "      <td>320.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>105.000000</td>\n",
       "      <td>570.000000</td>\n",
       "      <td>400.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          brand_id   loves_count       rating       reviews    price_usd  \\\n",
       "count  8494.000000  8.494000e+03  8216.000000   8216.000000  8494.000000   \n",
       "mean   5422.440546  2.917957e+04     4.194513    448.545521    51.655595   \n",
       "std    1709.595957  6.609212e+04     0.516694   1101.982529    53.669234   \n",
       "min    1063.000000  0.000000e+00     1.000000      1.000000     3.000000   \n",
       "25%    5333.000000  3.758000e+03     3.981725     26.000000    25.000000   \n",
       "50%    6157.500000  9.880000e+03     4.289350    122.000000    35.000000   \n",
       "75%    6328.000000  2.684125e+04     4.530525    418.000000    58.000000   \n",
       "max    8020.000000  1.401068e+06     5.000000  21281.000000  1900.000000   \n",
       "\n",
       "       value_price_usd  sale_price_usd  limited_edition          new  \\\n",
       "count       451.000000      270.000000      8494.000000  8494.000000   \n",
       "mean         91.168537       20.207889         0.070285     0.071698   \n",
       "std          79.195631       24.327352         0.255642     0.258002   \n",
       "min           0.000000        1.750000         0.000000     0.000000   \n",
       "25%          45.000000        8.000000         0.000000     0.000000   \n",
       "50%          67.000000       14.000000         0.000000     0.000000   \n",
       "75%         108.500000       25.000000         0.000000     0.000000   \n",
       "max         617.000000      320.000000         1.000000     1.000000   \n",
       "\n",
       "       online_only  out_of_stock  sephora_exclusive  child_count  \\\n",
       "count  8494.000000   8494.000000        8494.000000  8494.000000   \n",
       "mean      0.219096      0.073699           0.279374     1.631622   \n",
       "std       0.413658      0.261296           0.448718     5.379470   \n",
       "min       0.000000      0.000000           0.000000     0.000000   \n",
       "25%       0.000000      0.000000           0.000000     0.000000   \n",
       "50%       0.000000      0.000000           0.000000     0.000000   \n",
       "75%       0.000000      0.000000           1.000000     1.000000   \n",
       "max       1.000000      1.000000           1.000000   105.000000   \n",
       "\n",
       "       child_max_price  child_min_price  \n",
       "count      2754.000000      2754.000000  \n",
       "mean         53.792023        39.665802  \n",
       "std          58.765894        38.685720  \n",
       "min           3.000000         3.000000  \n",
       "25%          22.000000        19.000000  \n",
       "50%          32.000000        28.000000  \n",
       "75%          59.000000        42.000000  \n",
       "max         570.000000       400.000000  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#visualize data\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(8494, 34)"
      ]
     },
     "execution_count": 111,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "#visualize and aggregate data\n",
    "brandnames = df.groupby('brand_name').count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>product_id</th>\n",
       "      <th>product_name</th>\n",
       "      <th>brand_id</th>\n",
       "      <th>loves_count</th>\n",
       "      <th>rating</th>\n",
       "      <th>reviews</th>\n",
       "      <th>size</th>\n",
       "      <th>variation_type</th>\n",
       "      <th>variation_value</th>\n",
       "      <th>variation_desc</th>\n",
       "      <th>...</th>\n",
       "      <th>online_only</th>\n",
       "      <th>out_of_stock</th>\n",
       "      <th>sephora_exclusive</th>\n",
       "      <th>highlights</th>\n",
       "      <th>primary_category</th>\n",
       "      <th>secondary_category</th>\n",
       "      <th>tertiary_category</th>\n",
       "      <th>child_count</th>\n",
       "      <th>child_max_price</th>\n",
       "      <th>child_min_price</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>brand_name</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>19-69</th>\n",
       "      <td>14</td>\n",
       "      <td>14</td>\n",
       "      <td>14</td>\n",
       "      <td>14</td>\n",
       "      <td>14</td>\n",
       "      <td>14</td>\n",
       "      <td>13</td>\n",
       "      <td>13</td>\n",
       "      <td>13</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>14</td>\n",
       "      <td>14</td>\n",
       "      <td>14</td>\n",
       "      <td>14</td>\n",
       "      <td>14</td>\n",
       "      <td>14</td>\n",
       "      <td>14</td>\n",
       "      <td>14</td>\n",
       "      <td>7</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>54 Thrones</th>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ABBOTT</th>\n",
       "      <td>13</td>\n",
       "      <td>13</td>\n",
       "      <td>13</td>\n",
       "      <td>13</td>\n",
       "      <td>13</td>\n",
       "      <td>13</td>\n",
       "      <td>12</td>\n",
       "      <td>12</td>\n",
       "      <td>12</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>13</td>\n",
       "      <td>13</td>\n",
       "      <td>13</td>\n",
       "      <td>13</td>\n",
       "      <td>13</td>\n",
       "      <td>13</td>\n",
       "      <td>13</td>\n",
       "      <td>13</td>\n",
       "      <td>7</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>AERIN</th>\n",
       "      <td>24</td>\n",
       "      <td>24</td>\n",
       "      <td>24</td>\n",
       "      <td>24</td>\n",
       "      <td>22</td>\n",
       "      <td>22</td>\n",
       "      <td>23</td>\n",
       "      <td>23</td>\n",
       "      <td>23</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>24</td>\n",
       "      <td>24</td>\n",
       "      <td>24</td>\n",
       "      <td>16</td>\n",
       "      <td>24</td>\n",
       "      <td>24</td>\n",
       "      <td>23</td>\n",
       "      <td>24</td>\n",
       "      <td>10</td>\n",
       "      <td>10</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ALTERNA Haircare</th>\n",
       "      <td>45</td>\n",
       "      <td>45</td>\n",
       "      <td>45</td>\n",
       "      <td>45</td>\n",
       "      <td>44</td>\n",
       "      <td>44</td>\n",
       "      <td>40</td>\n",
       "      <td>29</td>\n",
       "      <td>26</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>45</td>\n",
       "      <td>45</td>\n",
       "      <td>45</td>\n",
       "      <td>24</td>\n",
       "      <td>45</td>\n",
       "      <td>45</td>\n",
       "      <td>44</td>\n",
       "      <td>45</td>\n",
       "      <td>7</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>philosophy</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>rms beauty</th>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>10</td>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>9</td>\n",
       "      <td>...</td>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>10</td>\n",
       "      <td>11</td>\n",
       "      <td>9</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>shu uemura</th>\n",
       "      <td>43</td>\n",
       "      <td>43</td>\n",
       "      <td>43</td>\n",
       "      <td>43</td>\n",
       "      <td>43</td>\n",
       "      <td>43</td>\n",
       "      <td>43</td>\n",
       "      <td>43</td>\n",
       "      <td>43</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>43</td>\n",
       "      <td>43</td>\n",
       "      <td>43</td>\n",
       "      <td>43</td>\n",
       "      <td>43</td>\n",
       "      <td>43</td>\n",
       "      <td>43</td>\n",
       "      <td>43</td>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>stila</th>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>10</td>\n",
       "      <td>10</td>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>7</td>\n",
       "      <td>...</td>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>10</td>\n",
       "      <td>11</td>\n",
       "      <td>11</td>\n",
       "      <td>10</td>\n",
       "      <td>11</td>\n",
       "      <td>7</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>tarte</th>\n",
       "      <td>131</td>\n",
       "      <td>131</td>\n",
       "      <td>131</td>\n",
       "      <td>131</td>\n",
       "      <td>126</td>\n",
       "      <td>126</td>\n",
       "      <td>101</td>\n",
       "      <td>108</td>\n",
       "      <td>105</td>\n",
       "      <td>44</td>\n",
       "      <td>...</td>\n",
       "      <td>131</td>\n",
       "      <td>131</td>\n",
       "      <td>131</td>\n",
       "      <td>79</td>\n",
       "      <td>131</td>\n",
       "      <td>131</td>\n",
       "      <td>110</td>\n",
       "      <td>131</td>\n",
       "      <td>59</td>\n",
       "      <td>59</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>304 rows × 26 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                  product_id  product_name  brand_id  loves_count  rating  \\\n",
       "brand_name                                                                  \n",
       "19-69                     14            14        14           14      14   \n",
       "54 Thrones                 4             4         4            4       4   \n",
       "ABBOTT                    13            13        13           13      13   \n",
       "AERIN                     24            24        24           24      22   \n",
       "ALTERNA Haircare          45            45        45           45      44   \n",
       "...                      ...           ...       ...          ...     ...   \n",
       "philosophy                 1             1         1            1       0   \n",
       "rms beauty                11            11        11           11      11   \n",
       "shu uemura                43            43        43           43      43   \n",
       "stila                     11            11        11           11      10   \n",
       "tarte                    131           131       131          131     126   \n",
       "\n",
       "                  reviews  size  variation_type  variation_value  \\\n",
       "brand_name                                                         \n",
       "19-69                  14    13              13               13   \n",
       "54 Thrones              4     2               2                2   \n",
       "ABBOTT                 13    12              12               12   \n",
       "AERIN                  22    23              23               23   \n",
       "ALTERNA Haircare       44    40              29               26   \n",
       "...                   ...   ...             ...              ...   \n",
       "philosophy              0     1               1                1   \n",
       "rms beauty             11    10              11               11   \n",
       "shu uemura             43    43              43               43   \n",
       "stila                  10    11              11               11   \n",
       "tarte                 126   101             108              105   \n",
       "\n",
       "                  variation_desc  ...  online_only  out_of_stock  \\\n",
       "brand_name                        ...                              \n",
       "19-69                          0  ...           14            14   \n",
       "54 Thrones                     0  ...            4             4   \n",
       "ABBOTT                         0  ...           13            13   \n",
       "AERIN                          0  ...           24            24   \n",
       "ALTERNA Haircare               0  ...           45            45   \n",
       "...                          ...  ...          ...           ...   \n",
       "philosophy                     0  ...            1             1   \n",
       "rms beauty                     9  ...           11            11   \n",
       "shu uemura                     0  ...           43            43   \n",
       "stila                          7  ...           11            11   \n",
       "tarte                         44  ...          131           131   \n",
       "\n",
       "                  sephora_exclusive  highlights  primary_category  \\\n",
       "brand_name                                                          \n",
       "19-69                            14          14                14   \n",
       "54 Thrones                        4           4                 4   \n",
       "ABBOTT                           13          13                13   \n",
       "AERIN                            24          16                24   \n",
       "ALTERNA Haircare                 45          24                45   \n",
       "...                             ...         ...               ...   \n",
       "philosophy                        1           0                 1   \n",
       "rms beauty                       11          11                11   \n",
       "shu uemura                       43          43                43   \n",
       "stila                            11          10                11   \n",
       "tarte                           131          79               131   \n",
       "\n",
       "                  secondary_category  tertiary_category  child_count  \\\n",
       "brand_name                                                             \n",
       "19-69                             14                 14           14   \n",
       "54 Thrones                         4                  2            4   \n",
       "ABBOTT                            13                 13           13   \n",
       "AERIN                             24                 23           24   \n",
       "ALTERNA Haircare                  45                 44           45   \n",
       "...                              ...                ...          ...   \n",
       "philosophy                         1                  0            1   \n",
       "rms beauty                        11                 10           11   \n",
       "shu uemura                        43                 43           43   \n",
       "stila                             11                 10           11   \n",
       "tarte                            131                110          131   \n",
       "\n",
       "                  child_max_price  child_min_price  \n",
       "brand_name                                          \n",
       "19-69                           7                7  \n",
       "54 Thrones                      2                2  \n",
       "ABBOTT                          7                7  \n",
       "AERIN                          10               10  \n",
       "ALTERNA Haircare                7                7  \n",
       "...                           ...              ...  \n",
       "philosophy                      0                0  \n",
       "rms beauty                      9                9  \n",
       "shu uemura                      4                4  \n",
       "stila                           7                7  \n",
       "tarte                          59               59  \n",
       "\n",
       "[304 rows x 26 columns]"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "brandnames"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "brand_name\n",
       "19-69               True\n",
       "54 Thrones          True\n",
       "ABBOTT              True\n",
       "AERIN               True\n",
       "ALTERNA Haircare    True\n",
       "                    ... \n",
       "philosophy          True\n",
       "rms beauty          True\n",
       "shu uemura          True\n",
       "stila               True\n",
       "tarte               True\n",
       "Name: out_of_stock, Length: 304, dtype: bool"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#convert to boolean\n",
    "brandnames['out_of_stock'] = brandnames['out_of_stock'].astype(bool)\n",
    "brandnames['out_of_stock'] "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0       ['Capri Eau de Parfum:', 'Alcohol Denat. (SD A...\n",
      "1       ['Alcohol Denat. (SD Alcohol 39C), Parfum (Fra...\n",
      "2       ['Alcohol Denat. (SD Alcohol 39C), Parfum (Fra...\n",
      "3       ['Alcohol Denat. (SD Alcohol 39C), Parfum (Fra...\n",
      "4       ['Alcohol Denat. (SD Alcohol 39C), Parfum (Fra...\n",
      "                              ...                        \n",
      "8489    ['Talc, Synthetic Fluorphlogopite, Triethylhex...\n",
      "8490    ['Alcohol, Aqua / Water / Eau, Parfum / Fragra...\n",
      "8491    ['Mon Paris Eau de Parfum:', 'Alcohol, Parfum/...\n",
      "8492    ['Alcohol, Parfum/Fragrance, Aqua/Water, Limon...\n",
      "8493    ['Diisostearyl Malate, Bis-Behenyl/Isostearyl/...\n",
      "Name: ingredients, Length: 8494, dtype: object\n"
     ]
    }
   ],
   "source": [
    "#visualize ingredients\n",
    "print(df['ingredients'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(426, 34)"
      ]
     },
     "execution_count": 115,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#select products with talc in ingredients\n",
    "df['talc'] = df['ingredients'].str.contains('Talc')\n",
    "df_withtalc = df[df['talc']==True]\n",
    "df_withtalc.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(209, 34)"
      ]
     },
     "execution_count": 112,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#select products with talc in ingredients\n",
    "df['sulfate'] = df['ingredients'].str.contains('sulfate')\n",
    "df_withsulfate = df[df['sulfate']==True]\n",
    "df_withsulfate.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#explore talc product popularity\n",
    "df_withtalc.hist('rating')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#explore talc product popularity\n",
    "df_withtalc.hist('price_usd')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#explore sulfate product popularity\n",
    "df_withsulfate.hist('rating')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#explore sulfate product popularity\n",
    "df_withsulfate.hist('price_usd')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXcAAAEICAYAAACktLTqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/NK7nSAAAACXBIWXMAAAsTAAALEwEAmpwYAAAXhUlEQVR4nO3df5RcZX3H8fcHAoisJVBwuybRxJKqCDWSLYbqOZ0FgYBKsEdpaCoBsVGLPVp/BvVUUTknWhELIriKGjWyRBSTBqjFwB4OpwImiPlJygLBJIYECAQWKRb49o/7LIzrzs7szG5m9snndc49e+9z753nuc/efObmmTt3FRGYmVle9ml2A8zMbPQ53M3MMuRwNzPLkMPdzCxDDnczsww53M3MMuRwNzPLkMPdWpKkzZLe3Ox2mI1XDnezFiXpbEm3NrsdNj453M3MMuRwt5Ym6QBJX5X02zR9VdIBad1GSW8t23aCpIckHZOWZ0n6b0mPSfq1pFLZtmdLuk/SE5LulzSvhrb8Y6rzCUkbyup5jaTeVM96SaeV7dMr6T2D6r21bDkkvU/SPWn/y1R4DXAFcJykfkmPNdCNthdyuFur+xQwC5gBvA44Fvh0WncVcGbZticDD0fEnZImAdcBXwAOBT4K/FjS4ZIOAi4BTomIlwB/Ddw1XCMkvRP4LHAW8CfAacAjkvYD/gP4L+ClwD8DSyS9agTH+Fbgr4C/BM4ATo6IjcD7gF9ERFtETBzB65k53K3lzQM+FxE7I+Ih4ALgXWndD4HTJL04Lf89ReAD/ANwfURcHxHPRcSNwCrg1LT+OeAoSQdGxPaIWF+lHe8BvhQRv4xCX0Q8QPHG0wYsiojfR8RNwAr+8E2nmkUR8VhE/Aa4meKNzKwhDndrdS8DHihbfiCVERF9wEbgbSngT6MIfIBXAO9MQx2PpWGNNwEdEfEk8HcUV8bbJV0n6dVV2jEFuLdC+7ZExHOD2jhpBMf4YNn87yjeLMwa4nC3VvdbiqAe8PJUNmBgaGYOsCEFPsAW4PsRMbFsOigiFgFExM8i4kSgA7gb+GaVdmwB/rxC+6ZIKv+39HJgW5p/Enhx2bo/q1JPOT+P2+rmcLdWdxXw6TRWfhjwr8APytb3ACcB7+eFq3bSNm+TdLKkfSW9SFJJ0mRJ7ZLmpLH3p4F+imGa4XwL+KikmekDzyMkvQK4neJq++OS9ksf2r4ttQuKsfy/lfRiSUcA547g2HcAkyXtP4J9zACHu7W+L1CMla8B1gJ3pjIAImI78AuKD0WvLivfQnE1/0ngIYor749RnPP7AB+muOreBfwNxZtDRRHxI+BCijeQJ4CfAodGxO8pwvwU4GHg68BZEXF32vVi4PcUQb0YWDKCY78JWA88KOnhEexnhvyXmMzM8uMrdzOzDDnczRJJV6QvDA2ermh228xGysMyZmYZmtDsBgAcdthhMXXq1Lr2ffLJJznooINGt0GZcN8Mz/1Tmfumslbqm9WrVz8cEYcPta4lwn3q1KmsWrWqrn17e3splUqj26BMuG+G5/6pzH1TWSv1jaQHKq3zmLuZWYYc7mZmGaoa7umbfXekR6aul3RBKv9uelTqXWmakcol6RJJfZLWDDwW1czM9pxaxtyfBo6PiP70eNNbJd2Q1n0sIq4ZtP0pwPQ0vQG4PP00M7M9pOqVe3q8aX9a3C9Nw90/OQf4XtrvNmCipI7Gm2pmZrWq6T53SfsCq4EjgMsi4hOSvgscR3FlvxJYGBFPS1pB8XzqW9O+K4FPRMSqQa+5AFgA0N7ePrOnp4d69Pf309bmJ6QOxX0zPPdPZe6bylqpb7q6ulZHROdQ62q6FTIingVmSJoIXCvpKOB8iudQ7w90A58APldroyKiO+1HZ2dn1HtrUSvdltRq3DfDc/9U5r6pbLz0zYjulomIxyj+Uszs9NdrIiKeBr5D8efPoHiO9ZSy3SbzwrOtzcxsD6jlbpnD0xU7kg4ETgTuHhhHlyTgdGBd2mU5cFa6a2YWsDs9ltXMzPaQWoZlOoDFadx9H2BpRKyQdJOkwwFR/EGC96Xtr6f4O5V9FH/E4JxRb3WZtdt2c/bC68ayioo2L3pLU+o1M6umarhHxBrg9UOUH19h+wDOa7xpZmZWL39D1cwsQw53M7MMOdzNzDLkcDczy5DD3cwsQw53M7MMOdzNzDLkcDczy5DD3cwsQw53M7MMOdzNzDLkcDczy5DD3cwsQw53M7MMOdzNzDLkcDczy5DD3cwsQw53M7MMOdzNzDLkcDczy5DD3cwsQ1XDXdKLJN0h6deS1ku6IJVPk3S7pD5JV0vaP5UfkJb70vqpY3wMZmY2SC1X7k8Dx0fE64AZwGxJs4AvAhdHxBHAo8C5aftzgUdT+cVpOzMz24OqhnsU+tPifmkK4HjgmlS+GDg9zc9Jy6T1J0jSaDXYzMyqU0RU30jaF1gNHAFcBvwbcFu6OkfSFOCGiDhK0jpgdkRsTevuBd4QEQ8Pes0FwAKA9vb2mT09PXUdwM5du9nxVF27NuzoSQc3p+Ia9ff309bW1uxmtCz3T2Xum8paqW+6urpWR0TnUOsm1PICEfEsMEPSROBa4NWNNioiuoFugM7OziiVSnW9zqVLlnHR2poOY9RtnldqSr216u3tpd5+3Ru4fypz31Q2XvpmRHfLRMRjwM3AccBESQOpOhnYlua3AVMA0vqDgUdGo7FmZlabWu6WOTxdsSPpQOBEYCNFyL8jbTYfWJbml6dl0vqbopaxHzMzGzW1jGd0AIvTuPs+wNKIWCFpA9Aj6QvAr4Ar0/ZXAt+X1AfsAuaOQbvNzGwYVcM9ItYArx+i/D7g2CHK/xd456i0zszM6uJvqJqZZcjhbmaWIYe7mVmGHO5mZhlyuJuZZcjhbmaWIYe7mVmGHO5mZhlyuJuZZcjhbmaWIYe7mVmGHO5mZhlyuJuZZcjhbmaWIYe7mVmGHO5mZhlyuJuZZcjhbmaWIYe7mVmGHO5mZhlyuJuZZahquEuaIulmSRskrZf0wVT+WUnbJN2VplPL9jlfUp+kTZJOHssDMDOzPzahhm2eAT4SEXdKegmwWtKNad3FEfHl8o0lHQnMBV4LvAz4uaS/iIhnR7PhZmZWWdUr94jYHhF3pvkngI3ApGF2mQP0RMTTEXE/0AccOxqNNTOz2igiat9YmgrcAhwFfBg4G3gcWEVxdf+opK8Bt0XED9I+VwI3RMQ1g15rAbAAoL29fWZPT09dB7Bz1252PFXXrg07etLBzam4Rv39/bS1tTW7GS3L/VOZ+6ayVuqbrq6u1RHROdS6WoZlAJDUBvwY+FBEPC7pcuDzQKSfFwHvrvX1IqIb6Abo7OyMUqlU665/4NIly7hobc2HMao2zys1pd5a9fb2Um+/7g3cP5W5byobL31T090ykvajCPYlEfETgIjYERHPRsRzwDd5YehlGzClbPfJqczMzPaQWu6WEXAlsDEivlJW3lG22duBdWl+OTBX0gGSpgHTgTtGr8lmZlZNLeMZbwTeBayVdFcq+yRwpqQZFMMym4H3AkTEeklLgQ0Ud9qc5ztlzMz2rKrhHhG3Ahpi1fXD7HMhcGED7TIzswb4G6pmZhlyuJuZZcjhbmaWIYe7mVmGHO5mZhlyuJuZZcjhbmaWIYe7mVmGHO5mZhlyuJuZZcjhbmaWIYe7mVmGHO5mZhlyuJuZZcjhbmaWIYe7mVmGHO5mZhlyuJuZZcjhbmaWIYe7mVmGHO5mZhmqGu6Spki6WdIGSeslfTCVHyrpRkn3pJ+HpHJJukRSn6Q1ko4Z64MwM7M/VMuV+zPARyLiSGAWcJ6kI4GFwMqImA6sTMsApwDT07QAuHzUW21mZsOqGu4RsT0i7kzzTwAbgUnAHGBx2mwxcHqanwN8Lwq3ARMldYx2w83MrDJFRO0bS1OBW4CjgN9ExMRULuDRiJgoaQWwKCJuTetWAp+IiFWDXmsBxZU97e3tM3t6euo6gJ27drPjqbp2bdjRkw5uTsU16u/vp62trdnNaFnun8rcN5W1Ut90dXWtjojOodZNqPVFJLUBPwY+FBGPF3leiIiQVPu7RLFPN9AN0NnZGaVSaSS7P+/SJcu4aG3NhzGqNs8rNaXeWvX29lJvv+4N3D+VuW8qGy99U9PdMpL2owj2JRHxk1S8Y2C4Jf3cmcq3AVPKdp+cyszMbA+p5W4ZAVcCGyPiK2WrlgPz0/x8YFlZ+VnprplZwO6I2D6KbTYzsypqGc94I/AuYK2ku1LZJ4FFwFJJ5wIPAGekddcDpwJ9wO+Ac0azwWZmVl3VcE8fjKrC6hOG2D6A8xpsl5mZNcDfUDUzy5DD3cwsQw53M7MMOdzNzDLkcDczy5DD3cwsQw53M7MMOdzNzDLkcDczy5DD3cwsQw53M7MMOdzNzDLkcDczy5DD3cwsQw53M7MMOdzNzDLkcDczy5DD3cwsQw53M7MMOdzNzDLkcDczy1DVcJf0bUk7Ja0rK/uspG2S7krTqWXrzpfUJ2mTpJPHquFmZlZZLVfu3wVmD1F+cUTMSNP1AJKOBOYCr037fF3SvqPVWDMzq03VcI+IW4BdNb7eHKAnIp6OiPuBPuDYBtpnZmZ1aGTM/QOS1qRhm0NS2SRgS9k2W1OZmZntQYqI6htJU4EVEXFUWm4HHgYC+DzQERHvlvQ14LaI+EHa7krghoi4ZojXXAAsAGhvb5/Z09NT1wHs3LWbHU/VtWvDjp50cHMqrlF/fz9tbW3NbkbLcv9U5r6prJX6pqura3VEdA61bkI9LxgROwbmJX0TWJEWtwFTyjadnMqGeo1uoBugs7MzSqVSPU3h0iXLuGhtXYfRsM3zSk2pt1a9vb3U2697A/dPZe6bysZL39Q1LCOpo2zx7cDAnTTLgbmSDpA0DZgO3NFYE83MbKSqXvJKugooAYdJ2gp8BihJmkExLLMZeC9ARKyXtBTYADwDnBcRz45Jy83MrKKq4R4RZw5RfOUw218IXNhIo8zMrDH+hqqZWYYc7mZmGXK4m5llyOFuZpYhh7uZWYYc7mZmGXK4m5llyOFuZpYhh7uZWYYc7mZmGXK4m5llyOFuZpYhh7uZWYYc7mZmGXK4m5llyOFuZpYhh7uZWYYc7mZmGXK4m5llyOFuZpYhh7uZWYaqhrukb0vaKWldWdmhkm6UdE/6eUgql6RLJPVJWiPpmLFsvJmZDa2WK/fvArMHlS0EVkbEdGBlWgY4BZiepgXA5aPTTDMzG4mq4R4RtwC7BhXPARan+cXA6WXl34vCbcBESR2j1FYzM6tRvWPu7RGxPc0/CLSn+UnAlrLttqYyMzPbgyY0+gIREZJipPtJWkAxdEN7ezu9vb111d9+IHzk6Gfq2rdR9bZ5T+nv72/5NjaT+6cy901l46Vv6g33HZI6ImJ7GnbZmcq3AVPKtpucyv5IRHQD3QCdnZ1RKpXqasilS5Zx0dqG36PqsnleqSn11qq3t5d6+3Vv4P6pzH1T2Xjpm3qHZZYD89P8fGBZWflZ6a6ZWcDusuEbMzPbQ6pe8kq6CigBh0naCnwGWAQslXQu8ABwRtr8euBUoA/4HXDOGLTZzMyqqBruEXFmhVUnDLFtAOc12igzM2uMv6FqZpYhh7uZWYYc7mZmGXK4m5llyOFuZpYhh7uZWYYc7mZmGXK4m5llyOFuZpYhh7uZWYYc7mZmGXK4m5llyOFuZpYhh7uZWYYc7mZmGXK4m5llyOFuZpYhh7uZWYYc7mZmGXK4m5llyOFuZpYhh7uZWYYmNLKzpM3AE8CzwDMR0SnpUOBqYCqwGTgjIh5trJlmZjYSo3Hl3hURMyKiMy0vBFZGxHRgZVo2M7M9aCyGZeYAi9P8YuD0MajDzMyGoYiof2fpfuBRIIBvRES3pMciYmJaL+DRgeVB+y4AFgC0t7fP7OnpqasNO3ftZsdT9bW/UUdPOrg5Fdeov7+ftra2ZjejZbl/KnPfVNZKfdPV1bW6bNTkDzQ05g68KSK2SXopcKOku8tXRkRIGvLdIyK6gW6Azs7OKJVKdTXg0iXLuGhto4dRn83zSk2pt1a9vb3U2697A/dPZe6bysZL3zQ0LBMR29LPncC1wLHADkkdAOnnzkYbaWZmI1N3uEs6SNJLBuaBk4B1wHJgftpsPrCs0UaamdnINDKe0Q5cWwyrMwH4YUT8p6RfAkslnQs8AJzReDPNzGwk6g73iLgPeN0Q5Y8AJzTSKDMza4y/oWpmlqHm3GaSiakLr2tKvZsXvaUp9ZrZ+OErdzOzDDnczcwy5HA3M8uQw93MLEMOdzOzDDnczcwy5HA3M8uQw93MLEMOdzOzDDnczcwy5McPjEO1PvbgI0c/w9mj/IgEP/rAbHzwlbuZWYYc7mZmGXK4m5llyGPuNiJ+zLHZ+OArdzOzDDnczcwy5HA3M8uQw93MLENj9oGqpNnAvwP7At+KiEVjVZflbyw+yK31S17+MNfGozEJd0n7ApcBJwJbgV9KWh4RG8aiPrOx5DuE8jeS3/Fof/N7rH7PYzUscyzQFxH3RcTvgR5gzhjVZWZmgygiRv9FpXcAsyPiPWn5XcAbIuIDZdssABakxVcBm+qs7jDg4QaamzP3zfDcP5W5byprpb55RUQcPtSKpn2JKSK6ge5GX0fSqojoHIUmZcd9Mzz3T2Xum8rGS9+M1bDMNmBK2fLkVGZmZnvAWIX7L4HpkqZJ2h+YCywfo7rMzGyQMRmWiYhnJH0A+BnFrZDfjoj1Y1EXozC0kzH3zfDcP5W5byobF30zJh+omplZc/kbqmZmGXK4m5llaFyHu6TZkjZJ6pO0sNntGS2Spki6WdIGSeslfTCVHyrpRkn3pJ+HpHJJuiT1wxpJx5S91vy0/T2S5peVz5S0Nu1ziSQNV0erkbSvpF9JWpGWp0m6PR3P1emDfCQdkJb70vqpZa9xfirfJOnksvIhz6tKdbQaSRMlXSPpbkkbJR3nc6cg6V/Sv6l1kq6S9KJsz52IGJcTxQe19wKvBPYHfg0c2ex2jdKxdQDHpPmXAP8DHAl8CViYyhcCX0zzpwI3AAJmAben8kOB+9LPQ9L8IWndHWlbpX1PSeVD1tFqE/Bh4IfAirS8FJib5q8A3p/m/wm4Is3PBa5O80emc+YAYFo6l/Yd7ryqVEerTcBi4D1pfn9gos+dAJgE3A8cWPb7PDvXc6fpHd7AL+o44Gdly+cD5ze7XWN0rMsontOzCehIZR3ApjT/DeDMsu03pfVnAt8oK/9GKusA7i4rf367SnW00kTxvYmVwPHAihQyDwMTBp8bFHdsHZfmJ6TtNPh8Gdiu0nk1XB2tNAEHpwDToPK9/tyhCPctFG9YE9K5c3Ku5854HpYZ+EUN2JrKspL+K/h64HagPSK2p1UPAu1pvlJfDFe+dYhyhqmjlXwV+DjwXFr+U+CxiHgmLZcfz/N9kNbvTtuPtM+Gq6OVTAMeAr6Thq2+JekgfO4QEduALwO/AbZTnAuryfTcGc/hnj1JbcCPgQ9FxOPl66K4BBjT+1j3RB0jJemtwM6IWN3strSoCcAxwOUR8XrgSYohkuftxefOIRQPMJwGvAw4CJjd1EaNofEc7lk/4kDSfhTBviQifpKKd0jqSOs7gJ2pvFJfDFc+eYjy4epoFW8ETpO0meJpo8dT/N2AiZIGvpRXfjzP90FafzDwCCPvs0eGqaOVbAW2RsTtafkairD3uQNvBu6PiIci4v+An1CcT1meO+M53LN9xEG6++BKYGNEfKVs1XJg4K6F+RRj8QPlZ6U7H2YBu9N/j38GnCTpkHTVchLFWN924HFJs1JdZw16raHqaAkRcX5ETI6IqRS/85siYh5wM/COtNngvhk4nnek7SOVz013REwDplN8UDjkeZX2qVRHy4iIB4Etkl6Vik4ANuBzB4rhmFmSXpzaPtA3eZ47zf6Qo8EPSE6luJPkXuBTzW7PKB7Xmyj+S7sGuCtNp1KM3a0E7gF+DhyathfFH0e5F1gLdJa91ruBvjSdU1beCaxL+3yNF76tPGQdrTgBJV64W+aVFP/A+oAfAQek8hel5b60/pVl+38qHf8m0h0fw51XlepotQmYAaxK589PKe528blTtPEC4O7U/u9T3PGS5bnjxw+YmWVoPA/LmJlZBQ53M7MMOdzNzDLkcDczy5DD3cwsQw53M7MMOdzNzDL0/+b8GvP03CcHAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df_withtalc.hist('loves_count')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#explore sulfate product popularity\n",
    "df_withsulfate.hist('loves_count')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>brand_name</th>\n",
       "      <th>loves_count</th>\n",
       "      <th>rating</th>\n",
       "      <th>price_usd</th>\n",
       "      <th>reviews</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>MAKE UP FOR EVER</td>\n",
       "      <td>211306.666667</td>\n",
       "      <td>4.360733</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>715.666667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Fenty Beauty by Rihanna</td>\n",
       "      <td>184809.619048</td>\n",
       "      <td>4.020710</td>\n",
       "      <td>29.523810</td>\n",
       "      <td>1729.476190</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Urban Decay</td>\n",
       "      <td>132202.058824</td>\n",
       "      <td>4.223741</td>\n",
       "      <td>34.000000</td>\n",
       "      <td>1609.470588</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>beautyblender</td>\n",
       "      <td>126259.000000</td>\n",
       "      <td>4.074000</td>\n",
       "      <td>29.000000</td>\n",
       "      <td>3042.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>NARS</td>\n",
       "      <td>114233.266667</td>\n",
       "      <td>4.192153</td>\n",
       "      <td>32.266667</td>\n",
       "      <td>2065.200000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                brand_name    loves_count    rating  price_usd      reviews\n",
       "0         MAKE UP FOR EVER  211306.666667  4.360733  22.000000   715.666667\n",
       "1  Fenty Beauty by Rihanna  184809.619048  4.020710  29.523810  1729.476190\n",
       "2              Urban Decay  132202.058824  4.223741  34.000000  1609.470588\n",
       "3            beautyblender  126259.000000  4.074000  29.000000  3042.000000\n",
       "4                     NARS  114233.266667  4.192153  32.266667  2065.200000"
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "brand_stats = df_withtalc.groupby('brand_name')[['loves_count','rating','price_usd','reviews']].mean().sort_values(by=['loves_count','rating'], ascending=False).reset_index()\n",
    "brand_stats.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "#plotting data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(62, 5)"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "brand_stats.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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TGZWDqHBwURERJCcnMzPPxdODC9tKtq6desSHh7umCX61Vdf+V3eyw1loSsUCp/p0qkt3aLb0759e1q2bEm/fv0c+yZPnsywYcNo1qwZa9as8am+Tz75hAceeACNRsPAgQOJiIioKNEvC5RCVygUHomKimLfvn2O74veedFl+txp06Yxbdo0x/eCZd4iIyOLHF+wiDNAp06d2LNnDwCvv/46sbEus8IqfEQpdIWiRuB53YKayk8//cS//vUvzGYzrVq1YtGiRVUtUo1GKXSFolojqlqACmXs2LGMHTu2qsWoNahBUYVCoaglKIWuUCgUtQSl0BUKXzm4EpZNrGopFAq3KIWuUPjK0nGwb3lVS6FQuEUpdIVCUWEkJiYyevToqhbjskFFuSgUCp/xlD7XFc2aNWPZsmUVKpOiEGWhKxQKjxRPn/vyO/Pp2bMnMTExjvS5M2fO5IMPPnAcM3v2bObMmUNCQoIjJa7FYmH69OmOYz/66CMApkyZwg8//ADAyJEjmTjRNk6xYMECnn32WXJycrjpppvo0qUL0dHRLF26tDJPv0ahLHSFooaQ9NprGA76N31uYIf2NHnmGa/lCtLnZg7vw7KffmfLli1IKbn55ptZu3YtY8eO5fHHH2fKlCkAfP311/zyyy9YLBZHHZ988gkRERFs3boVg8FAv379GDJkCP3792fdunXcfPPNnDt3jvPnzwOwbt067rjjDlavXk2zZs346aefAMjIyPBrH9QmlIWuUCi8UpA+99e//ubXv/6mW7dudO/enUOHDnH06FG6devGhQsXSExMZPfu3dSrV4+WLVsWqePXX39l8eLFdO3ald69e5OamsrRo0cdCv3AgQN07NiRxo0bc/78eTZt2sQ111xD586d+e2335gxYwbr1q1T+V48oCx0haKG4IslXVEUpM+VUvL01Ak8OOPVEmXGjLqVZUu/JCkl1eXsTykl77//PkOHDi2xLz09ndWrVzNgwAAuXbrE119/TVhYGOHh4YSHh7Njxw5WrVrFc889x3XXXccLL7zg/5OsBSgLXaFQ+MzQuL4sWPoD2dnZAJw7d44LFy4AMPb6Hny1ZDHLli1jzJgxJY8dOpS5c+diMpkAOHLkCDk5OQD06dOHd955hwEDBtC/f3/mzJlD//79AVukTEhICHfffTfTp09nx44dlXGqNRKvFroQYgEwArggpSyx4J8QQgDvAjcCucB4KaXqcYWiFjJkYF8OHj3pWGUoLCyMzz//nEaNGtGp3VVk5eTSvHkLmjZtWuLYSZMmkZCQQPfu3ZFS0rBhQ7777jsA+vfvz6+//srVV19Nq1atuHTpkkOh7927l+nTp6PRaNDpdMydO7fSzremIaT0nMVNCDEAyAYWu1HoNwLTsCn03sC7Usre3hqOjY2V27ZtK5PQ8fHxxMXFlenYywHVP54pc//MtvtuZ1feoNzBXVvo0EgHoQ0hokWFt5eVlUV4eLj7Aok7bf9dpM/1uK+W4LV//MzBgwfp0KFDkW1CiO1SSpd5hr26XKSUa4FLHorcgk3ZSynl30BdIUTJx7NCoVAoKhR/DIo2B844fT9r33a+eEEhxGRgMkDjxo2Jj48vU4PZ2dllPvZyQPWPZ8raP3H2/5XZt/VDdYAOo9GIoRTLu5UVi8XicRm5AtvUVRlP+2oL3vrH3+Tn55fqeqvUKBcp5TxgHthcLmV1CyiXgmdU/3imzP0Tb/tXmX17cNcWAPR6PfpKeNX36lKw6zKXZTztqyVUtsslKCiIbt18d2H5I8rlHOAccNrCvk2hUCgUlYg/FPoPwL3CRh8gQ0pZwt2iUCgUiorFl7DFL7G5DyOFEGeBWYAOQEr5IbAKW4TLMWxhixMqSlhFJfPZSGg7HHpPrmpJFAqFD3hV6FLKO73sl8AUv0mkqD4c/9P2pxS6QlEjUDNFFQqFX3jh33P5fe3mqhYDsE14uhxRuVwUCkW5sVgsvDT94aoW47JHKXSFooaw7usjXDyT7dc6I1uG0f/2th7LJCQkMGzYMHr06MGOLRvp1PYqFn/zAx07dmTs2LH89ttv/POf/2T1t18w4vr+jJ7cja1bt/LYY4+Rk5NDYGAgf/zxByEhIcycOZP4+HgMBgNTpkzhwQcfdNlmfHw8c+bMYeXKlQBMnTqV2NhYxo8fz8yZM/nhhx8ICAhgyJAhzJkzh5MnT3LXXXeRnZ3NLbfc4tc+qkkol4tCofDK4cOHeeSRRzj417fUCQ/lf//7HwANGjRgx44d3HHHHY6yRqORsWPH8u6777J7925+//13goODi+RD37p1K/Pnz+fkyZOlkiM1NZUVK1awf/9+9uzZw3PPPQfAY489xsMPP8zevXtd5pG5XFAWukJRQ/BmSVckLVu2pF+/fpC4k7tH3ch7X/wM4DJN7uHDh2natCk9e/YEoE6dOoAtH/qePXscS9JlZGRw9OhRWrdu7bMcERERBAUFcf/99zNixAhGjBgBwIYNG1i+3LaA9z333MOMGTPKfrI1GKXQFQqFV2xJVUt+L8iT7gue8qEXJyAgAKvV6vien5/v2L5lyxb++OMPli1bxn//+1/+/PNPlzJejiiXi0Kh8Mrp06fZtGkTAF98t5prr73Wbdl27dpx/vx5tm7dCtimy5vNZo/50IvTqlUrDhw4gMFgID09nT/++AOw5eHJyMjgxhtv5O2332b37t0A9OvXj6+++gqAJUuW+OekayBKoSsUCq+0a9eODz74gA4DR5GWkcnDD7uPaNHr9SxdupRp06bRpUsXbrjhBvLz85k0aRIdO3ake/fuREdH8+CDD2I2m13W0bJlS26//Xaio6O5/fbbHflMsrKyGDFiBDExMVx77bW89dZbALz77rt88MEHdO7cmXPnLt/MI8rlolAovBIQEMDnn39emPM8JISEhIQiZRa986Ljc8+ePfn7779L1PPaa6/x2muv+dTmm2++yZtvvlli+5YtW0psa926teMNAuCVV17xqY3ahrLQFQqFopagLHSFQuGRqKgo9u3bVyF17927l3vuuafItsDAQDZvrh4zTmsaSqErFIoqo3Pnzuzatauqxag1KJeLQqFQ1BKUQlcoFIpaglLoCoVCUUtQCl2hUChqCUqhKxQKvxHWpl+VtLtt2zYeffTRKmm7OOPHj3fkq6lsaqdC374IXm4EVktVS6JQ+AlZ1QJUW8xmM7Gxsbz33ntVLUqVUzvDFn+eARYDWIygCa5qaRQKv7Bm0TwunDrh1zobtbqSQePdLzE4c+ZMWrZsyZQptlUmZ//nQwK0AazZfoi0tDRMJhOvvPJKiRzknvKZb9++nSeeeILs7GwiIyNZtGiR25S3cXFxdOnShb/++guz2cyCBQvo1asXs2fP5vjx45w4cYIrrriCBx980NFednY206ZNY9u2bQghmDVrFrfddhu//vors2bNwmAwcNVVV7Fw4UK3KxtFRUWxbds2IiMj2bZtG0899RTx8fGsX7+ep59+GrAlA1u7di1hYWFMmzaN3377jZYtW6LX60v9O/gLnyx0IcQwIcRhIcQxIcRMF/uvEEKsEULsFELsEULc6H9Ry4BUVo1CUR7Gjh3L119/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p169ehw6dIgjR44QFhbG+fO2HOxvvPEGn3/+ORqNhuHDh/P6669z/PhxpkyZQkpKCiEhIcyfP5/27duXaLMiqTKFvn379otCiLKm8IvEFuuucI3qH8/UmP757bffOlsstixtIiVFL7Kz/fpWnZ2SYr24b58jH7TFYgnQarVFslsdPXpUPP/880GfffZZXv369UlPT+eNN97Qnzt3Tnz44YeGEydOiGnTpgW1b98+7+zZsyIvLy9o3759ecePH9dkZGTo9u3bZ3j11Vf1V111lXzrrbdM8fHx2k8++STwwIEDufXr1y8h09mzZ8WxY8eCX3311bwnnnhCjh07Nig7O9v64YcfGv/44w/tP//5z4D//ve/BovFwty5cwkICGDjxo2aRx55RPfuu+8arrvuOj777LOgt99+2zR//nzdjBkzjCdOnHCZ8CYpKUm3efNm7RdffJGfm5vLmDFjgq+88sr8hIQEsW3btqBvv/02r2XLlnLfvn1YrdaQEydOGNevXy+/+OIL3YIFC/KDg4NJT09n3759TJw4MeiFF14wREVFyd27d2vuuece/cKFC8uViSspKSmgY8eOxXNzt3JZmCpU6FLKht5LuUYIsU26mfqqUP3jjZrUP7t3706Ijo62PXzefrvC29u3b1+H6Ojog87bvv/++0Y333yzbsCAAecKtr399ttRN954Y2ZMTMylmJgYxo0b1y06OvqgTqfTazSaNtHR0QcTEhLCAwICGkdHRx/buXNnx2+//fZYx44djdHR0Tz77LNd27Rpc7Rp06YlUiPqdDp98+bN2952220HANq3bx81ZMiQzM6dO1/SarX6Dz/88Oro6OiDx44d0z388MNXJCQkBAkhpMlkMhXI/vnnn+tjY2M7jRs3LuW+++47W7yNAkJCQprdeOONxMbGJgJcc801UYmJiemNGze2xMTENB0+fLjTGnx002q15l27dmXde++9+T179nQYBRkZGZo9e/Z0feqpp8C+0o7RaDQW78vSYrFYIktzrSqXi0KhKBNBQUEOv4m/5xHq9XpHhRqNxtGWVqvFYrEIgBkzZjQfOHBg1m+//Xb88OHD+sGDB7crOObgwYNBISEh1qSkJM+pNwFRLDd5wfeQkBCf14uzWCyEh4ebDx065CU/csWiBkUVCoVHhg4dmvnjjz/WS0pK0gIkJyeXeumPPn36ZC1a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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.plot()\n",
    "plt.grid()\n",
    "plt.xlabel('')\n",
    "plt.ylabel('')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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kZUrV/Liiy+SlZXF888/z4QJE8jMzGTcuHFkZGS4HZ+Tk8N1113H0aNHOe+887wnBFTJzMxk9OjRDBo0iNWrV9O/f38mT57MY489xsmTJ/nwww8ZMGAA69atY8aMGZjNZuLi4pg/fz5t2rThpZdeYtu2bbz77rts27aN6667jnXr1hEfX3kh3uzZs9m/fz/79u3j1KlT/POf/+TWW28lPT2dRx55hKZNm7Jr1y727NlDQkICxWo1u+eee44PPvgAg8HAmDFjePbZZ9m/fz/Tpk0jOzub+Ph43nrrLbp06VKNb0RHRydUNDghsXHjxlNCiL9cty1fvrx7eXk5RqPRJrKzo0VxcUg1qOLsbMepjAyfuZ737t0rHnnkkdj333+/rFmzZuTn5/Pcc89FHz16VLz++uuWAwcOiOnTp8d26dKl7MiRI6KsrCw2IyOjbP/+/YaCgoKojIwMy9NPPx199tlnyxdffNGanp5ufOedd2J27NhRqqUBd+XIkSNi3759cU8//XTZfffdJydOnBhbXFzseP3118t//vln4z//+U/TK6+8YrHb7cybNw+TycTq1asNd911V9SLL75ov+iii2zvv/9+7EsvvWR96623oh588MHyAwcOeF2xlZWVFbV27VrjRx99ZC4tLeXqq6+OO+uss8yZmZliw4YNsZ999llZ+/btZUZGBg6HIz4jI6P0119/NX700UdR7777rjkuLo78/HwyMjKYMmVK7KOPPmrp0KGD3LJli+HGG2+Mnj9/vrmm309WVpapa9eu22p6Hg+Sgfpl74s8+j0KjLpyn870trHBCQkpZaV0mlu2bMk0Go0t0tLSdvLSSxHv05dfftnysssuixo6dKhzIeBLL73U4dJLLy3s0aNHbo8ePbj++ut7p6Wl7YyKioo2GAyd0tLSdmZmZiaaTKaUtLS0fX/++WfXzz77bF/Xrl3L09LSeOihh3p16tRpb+vWrSsluI+Kiopu27btuVddddUOgC5dunQYOXJkYffu3XONRmP066+/fk5aWtrOffv2Rd15551nZGZmxgohpNVqtRqNRkdaWtrODz74ILpfv37drr/++uybb775iGcbGvHx8W0uvfRS+vXrdwzg/PPP73Ds2LH8lJQUe48ePVqPGTPGteZj77S0tJ3//e9/2910003m/v37O38YBQUFhq1bt/Z64IEHQE0p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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df_withtalc.plot()\n",
    "plt.grid()\n",
    "plt.xlabel('')\n",
    "plt.ylabel('')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#select products with talc in ingredients\n",
    "df['talc'] = df['ingredients'].str.contains('Talc')\n",
    "df_withtalc = df[df['talc']==True]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/amelia/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  \n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>is_TALC</th>\n",
       "      <th>ingredients</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>253</th>\n",
       "      <td>True</td>\n",
       "      <td>['Java, Vanilla, Fawn, Havana, and Banana:', '...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>256</th>\n",
       "      <td>True</td>\n",
       "      <td>['Bronzed, Tourmaline:', 'Mica, Synthetic Fluo...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>258</th>\n",
       "      <td>True</td>\n",
       "      <td>['Talc, Boron Nitride, Nylon-12, Ethylhexyl Pa...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>261</th>\n",
       "      <td>True</td>\n",
       "      <td>['Marshmallow:', 'Talc, Ethylhexyl Palmitate, ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>267</th>\n",
       "      <td>True</td>\n",
       "      <td>['Isododecane, Talc, Trimethylsiloxysilicate, ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8366</th>\n",
       "      <td>True</td>\n",
       "      <td>['Synthetic Fluorphlogopite, Talc, Mica, Octyl...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8431</th>\n",
       "      <td>True</td>\n",
       "      <td>['Aqua / Water / Eau, Cyclopentasiloxane, Talc...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8441</th>\n",
       "      <td>True</td>\n",
       "      <td>['Dimethicone, Bis-Diglyceryl Polyacyladipate-...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8469</th>\n",
       "      <td>True</td>\n",
       "      <td>['Talc, Mica, Ci 77891 / Titanium Dioxide, Cap...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8489</th>\n",
       "      <td>True</td>\n",
       "      <td>['Talc, Synthetic Fluorphlogopite, Triethylhex...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>426 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     is_TALC                                        ingredients\n",
       "253     True  ['Java, Vanilla, Fawn, Havana, and Banana:', '...\n",
       "256     True  ['Bronzed, Tourmaline:', 'Mica, Synthetic Fluo...\n",
       "258     True  ['Talc, Boron Nitride, Nylon-12, Ethylhexyl Pa...\n",
       "261     True  ['Marshmallow:', 'Talc, Ethylhexyl Palmitate, ...\n",
       "267     True  ['Isododecane, Talc, Trimethylsiloxysilicate, ...\n",
       "...      ...                                                ...\n",
       "8366    True  ['Synthetic Fluorphlogopite, Talc, Mica, Octyl...\n",
       "8431    True  ['Aqua / Water / Eau, Cyclopentasiloxane, Talc...\n",
       "8441    True  ['Dimethicone, Bis-Diglyceryl Polyacyladipate-...\n",
       "8469    True  ['Talc, Mica, Ci 77891 / Titanium Dioxide, Cap...\n",
       "8489    True  ['Talc, Synthetic Fluorphlogopite, Triethylhex...\n",
       "\n",
       "[426 rows x 2 columns]"
      ]
     },
     "execution_count": 130,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#turn into boolean value\n",
    "df_withtalc['is_TALC'] = df['ingredients'].str.contains('Talc')\n",
    "df_withtalc.query(\"ingredients!=talc\")[['is_TALC', 'ingredients']] "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Here is the list of company names with products containing talc. Recognize any?\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array(['Anastasia Beverly Hills', 'Armani Beauty', 'Artist Couture',\n",
       "       'beautyblender', 'Benefit Cosmetics', 'Bobbi Brown',\n",
       "       'Bumble and bumble', 'Buxom', 'Charlotte Tilbury', 'CLINIQUE',\n",
       "       'COLOR WOW', 'DAMDAM', 'Danessa Myricks Beauty', 'Dior',\n",
       "       'DOMINIQUE COSMETICS', 'Donna Karan', 'Dr. Barbara Sturm',\n",
       "       'Dr. Jart+', 'Drybar', 'Estée Lauder', 'Fashion Fair',\n",
       "       'Fenty Beauty by Rihanna', 'Givenchy', 'Gucci', 'GUERLAIN',\n",
       "       'HAUS LABS BY LADY GAGA', 'Hourglass', 'HUDA BEAUTY',\n",
       "       'Iconic London', 'IT Cosmetics', 'Kaja', 'Koh Gen Do',\n",
       "       'KVD Beauty', 'Lancôme', 'Laura Mercier', 'lilah b.',\n",
       "       'MAKE UP FOR EVER', 'MAKEUP BY MARIO', 'Mario Badescu',\n",
       "       'Melt Cosmetics', 'NARS', 'Natasha Denona', 'NUDESTIX',\n",
       "       'ONE/SIZE by Patrick Starrr', 'PAT McGRATH LABS', 'PATRICK TA',\n",
       "       'Peter Thomas Roth', 'Rare Beauty by Selena Gomez',\n",
       "       'SEPHORA COLLECTION', 'Shiseido', 'Smashbox', 'stila', 'tarte',\n",
       "       'TOM FORD', 'Too Faced', 'Urban Decay', 'Valentino', 'Violet Voss',\n",
       "       'Viseart', 'Wander Beauty', 'Westman Atelier',\n",
       "       'Yves Saint Laurent'], dtype=object)"
      ]
     },
     "execution_count": 142,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print('Here is the list of company names with products containing talc. Recognize any?')\n",
    "df_withtalc['brand_name'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {},
   "outputs": [],
   "source": [
    "#OLS linear regression model with statsmodels on talc data\n",
    "import statsmodels.formula.api as smf\n",
    "mod = smf.ols(\"price_usd ~ loves_count + rating\", data=df_withtalc)\n",
    "res = mod.fit()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:              price_usd   R-squared:                       0.021\n",
      "Model:                            OLS   Adj. R-squared:                  0.016\n",
      "Method:                 Least Squares   F-statistic:                     4.473\n",
      "Date:                Wed, 29 Nov 2023   Prob (F-statistic):             0.0120\n",
      "Time:                        20:16:24   Log-Likelihood:                -2028.6\n",
      "No. Observations:                 420   AIC:                             4063.\n",
      "Df Residuals:                     417   BIC:                             4075.\n",
      "Df Model:                           2                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "===============================================================================\n",
      "                  coef    std err          t      P>|t|      [0.025      0.975]\n",
      "-------------------------------------------------------------------------------\n",
      "Intercept      13.0412     13.288      0.981      0.327     -13.079      39.162\n",
      "loves_count -3.063e-05   1.41e-05     -2.176      0.030   -5.83e-05   -2.97e-06\n",
      "rating          7.3174      3.175      2.305      0.022       1.076      13.558\n",
      "==============================================================================\n",
      "Omnibus:                      361.189   Durbin-Watson:                   1.100\n",
      "Prob(Omnibus):                  0.000   Jarque-Bera (JB):             7403.015\n",
      "Skew:                           3.682   Prob(JB):                         0.00\n",
      "Kurtosis:                      22.204   Cond. No.                     1.15e+06\n",
      "==============================================================================\n",
      "\n",
      "Warnings:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "[2] The condition number is large, 1.15e+06. This might indicate that there are\n",
      "strong multicollinearity or other numerical problems.\n"
     ]
    }
   ],
   "source": [
    "#create model, fit model and print summary. \n",
    "#the function produces a presentation where the column of P\n",
    "#values of each feature we entered into the model, where P value less than\n",
    "#0.05 are significant.\n",
    "print(res.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "We can conclude that for products with talc, the loves count and rating p values are significant. Sephora is marketing and selling these talc based products successfully\n"
     ]
    }
   ],
   "source": [
    "print('We can conclude that for products with talc, the loves count and rating p values are significant. Sephora is marketing and selling these talc based products successfully')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/amelia/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: UserWarning: \n",
      "\n",
      "`distplot` is a deprecated function and will be removed in seaborn v0.14.0.\n",
      "\n",
      "Please adapt your code to use either `displot` (a figure-level function with\n",
      "similar flexibility) or `histplot` (an axes-level function for histograms).\n",
      "\n",
      "For a guide to updating your code to use the new functions, please see\n",
      "https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751\n",
      "\n",
      "  \n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[]"
      ]
     },
     "execution_count": 157,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#finally correlations\n",
    "plt.figure(figsize=(10,6))\n",
    "sns.distplot(df_withtalc['rating'])\n",
    "plt.plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/amelia/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:3: UserWarning: \n",
      "\n",
      "`distplot` is a deprecated function and will be removed in seaborn v0.14.0.\n",
      "\n",
      "Please adapt your code to use either `displot` (a figure-level function with\n",
      "similar flexibility) or `histplot` (an axes-level function for histograms).\n",
      "\n",
      "For a guide to updating your code to use the new functions, please see\n",
      "https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751\n",
      "\n",
      "  This is separate from the ipykernel package so we can avoid doing imports until\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[]"
      ]
     },
     "execution_count": 158,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#finally correlations\n",
    "plt.figure(figsize=(10,6))\n",
    "sns.distplot(df_withtalc['reviews'])\n",
    "plt.plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Talc-based products with cheaper prices seem to have higher love counts, suggesting succesful Sephora marketing\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(20,6))\n",
    "sns.scatterplot(data=df_withtalc, x='loves_count', y='price_usd')\n",
    "plt.plot()\n",
    "print('Talc-based products with cheaper prices seem to have higher love counts, suggesting succesful Sephora marketing')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
