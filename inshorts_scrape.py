import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
import contractions
import unicodedata
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

seed_urls = ['https://inshorts.com/en/read/technology',
             'https://inshorts.com/en/read/sports',
             'https://inshorts.com/en/read/world']

def build_dataset(seed_urls):
    news_data = []
    for url in seed_urls:
        news_category = url.split('/')[-1]
        data = requests.get(url)
        soup = BeautifulSoup(data.content)
        
        news_articles = [{'news_headline': headline.find('span',attrs={"itemprop": "headline"}).string,
                          'news_article': article.find('div',attrs={"itemprop": "articleBody"}).string,
                          'news_category': news_category}
                         
                            for headline, article in zip(soup.find_all('div',class_=["news-card-title news-right-box"]),
                                 soup.find_all('div',class_=["news-card-content news-right-box"]))]
        news_articles = news_articles[0:20] # Accepting only 20 values
        news_data.extend(news_articles)
        
    df =  pd.DataFrame(news_data)
    df = df[['news_headline', 'news_article', 'news_category']]
    return df
df_raw = build_dataset(seed_urls)
df = build_dataset(seed_urls)

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, ' ', text)
    return text

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

df.news_headline = df.news_headline.apply(lambda x:x.lower())
df.news_article = df.news_article.apply(lambda x:x.lower())

df.news_headline = df.news_headline.apply(strip_html_tags)
df.news_article = df.news_article.apply(strip_html_tags)

df.news_headline = df.news_headline.apply(remove_accented_chars)
df.news_article = df.news_article.apply(remove_accented_chars)

df.news_headline = df.news_headline.apply(expand_contractions)
df.news_article = df.news_article.apply(expand_contractions)

df.news_headline = df.news_headline.apply(remove_special_characters)
df.news_article = df.news_article.apply(remove_special_characters)

df.news_headline = df.news_headline.apply(remove_stopwords)
df.news_article = df.news_article.apply(remove_stopwords)

df['news_headline_token'] = df.news_headline.apply(lambda x: word_tokenize(x))
df['news_article_token'] = df.news_article.apply(lambda x: word_tokenize(x))

df['news_headline_lem'] = df.news_headline_token.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
df['news_article_lem'] = df.news_article_token.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

def listToString(s):  
    
    # initialize an empty string 
    str1 = " " 
    
    # return string   
    return (str1.join(s)) 

df['news_headline_new'] = df.news_headline_lem.apply(listToString)
df['news_article_new'] = df.news_article_lem.apply(listToString)

new_df = df[['news_headline_new','news_article_new','news_category']]

df_raw.to_csv('news_raw.csv', index=False, encoding='utf-8')
new_df.to_csv('news.csv', index=False, encoding='utf-8')
