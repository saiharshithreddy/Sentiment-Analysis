import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from multiprocessing import  Pool
import logging
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

df = pd.read_csv('review.csv')
#df = df.iloc[:220,:]
print("Data loaded")
df_t = df.loc[:,['text', 'stars']]
df_t = df_t.dropna()
df_t.stars = df_t.stars.map({5:'positive',4:'positive',3:'positive',2:'negative',1:'negative'})

df_t = df_t.rename(columns={'text':'sentence','stars':'sentiment'})

label_encoder = LabelEncoder() 

df_t['sentiment']= label_encoder.fit_transform(df_t['sentiment'])

df_t.sentiment = df_t.sentiment.astype('int64')





def parallelize(df, func, n_cores=24):
    '''
    The dataframe is split into n_cores and availability_filter is performed on each sub dataframe
    n_cores: No of processors (96 in this env)
    n_cores is taken as 10 as it takes less time to split the given data.
    '''
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
  
def log(file_name):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_name, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)
log('log.txt')



def clean_text(text):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
    
    # Convert words to lower case
    text = text.lower()
    
    # Add contractions
    
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    
    
    # Optionally, remove stop words
    
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    logging.info("sentence cleaned")

    return text

def loop(data):
    #applies the function on each value of the pandas series
    res = data['sentence'].apply(clean_text)
    return res
print("parallization begins")
text = parallelize(df_t, loop)
print("Entire data has been cleaned")

df_t['sentence'] = pd.Series(text)
df_t['length'] = df_t['sentence'].str.len()
df_test = df_t.loc[df_t['length'] >= 20]
df_test.drop(['length'],axis=1,inplace=True)
df_test.to_csv('yelp_reviews.csv', index=False)
df_a = df_test.iloc[:1000000,:]
df_b = df_test.iloc[1000000:2000000,:]
df_c = df_test.iloc[2000000:3000000,:]
df_d = df_test.iloc[3000000:4000000,:]
df_e = df_test.iloc[4000000:5000000,:]
df_f = df_test.iloc[5000000:6000000,:]
df_g = df_test.iloc[6000000::,:]

df_a.to_pickle('yelp_data1.pkl')
df_b.to_pickle('yelp_data2.pkl')
df_c.to_pickle('yelp_data3.pkl')
df_d.to_pickle('yelp_data4.pkl')
df_e.to_pickle('yelp_data5.pkl')
df_f.to_pickle('yelp_data6.pkl')
df_g.to_pickle('yelp_data7.pkl')
