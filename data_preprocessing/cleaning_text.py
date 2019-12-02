import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from multiprocessing import  Pool
import logging
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from contractions import CONTRACTION_MAP
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag



df = pd.read_csv('review.csv')

print("Data loaded")
df_t = df.loc[:,['text', 'stars']]
df_t = df_t.dropna()
df_t.stars = df_t.stars.map({5:'positive',4:'positive',3:'positive',2:'negative',1:'negative'})

df_t = df_t.rename(columns={'text':'sentence','stars':'sentiment'})

label_encoder = LabelEncoder() 

df_t['sentiment']= label_encoder.fit_transform(df_t['sentiment'])

df_t.sentiment = df_t.sentiment.astype('int8')


# Using 24 cores in the HPC
def parallelize(df, func, n_cores=24):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
  
# To check the number of rows preprocessed
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
    
    
    
    contractions_pattern = re.compile('({})'.format('|'.join(CONTRACTION_MAP.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = CONTRACTION_MAP.get(match)\
                                if CONTRACTION_MAP.get(match)\
                                else CONTRACTION_MAP.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
       
    # Optionally, remove stop words
    
    text = expanded_text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    logging.info("sentence cleaned")
    # Lemmatization
    
    wordnet_lemmatizer = WordNetLemmatizer()

    def get_tag(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''

    text_result = []
    tokens = word_tokenize(text)  # Generate list of tokens
    tagged = pos_tag(tokens)
    for t in tagged:
        try:
            text_result.append(wordnet_lemmatizer.lemmatize(t[0], get_tag(t[1][:2])))
        except:
            text_result.append(wordnet_lemmatizer.lemmatize(t[0]))
    paragraph = " ".join(str(x) for x in text_result)
    return paragraph

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
df_test.to_pickle('cleaned_yelp_reviews.pickle')

