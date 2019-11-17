from flask import Flask, render_template, request, jsonify, abort
import praw
import nltk.corpus
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import pandas as pd
import gensim
import logging
import warnings
import numpy as np
import pickle
from flask_cors import CORS
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import seaborn as sns
# %config InlineBackend.figure_formats = ['retina']
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score
# import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
import spacy

logging.basicConfig(filename='lda_model.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

nltk.download('punkt')
stop_words = stopwords.words('english')

reddit = praw.Reddit(client_id='EtqlvWK3wzZyXg', client_secret='cEJc0Rz3KW0uX38rb6jWG3YlUo0', user_agent='googlecolab')

# Global DataFrame
topicnum = 12


def remove_stopwords(texts):
  return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def strip_newline(series):
  return [review.replace('\n', '') for review in series]


# convert to lowercase
def sent_to_words(sentences):
  for sentence in sentences:
    yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def bigrams(words, bi_min=15, tri_min=10):
  # Group related phrases into one token for LDA
  bigram = gensim.models.Phrases(words, min_count=bi_min)
  bigram_model = gensim.models.phrases.Phraser(bigram)
  return bigram_model


def fetch_posts(subreddit, subreddit2, num_posts):
  subreddit = reddit.subreddit(subreddit)
  # Create Pandas dataframe from list of dictionaries
  posts = []
  for submission in subreddit.new(limit=num_posts):
    if not submission.stickied and submission.selftext:
      post = {'title': submission.title, 'body': submission.selftext, 'aorb': 1}
      posts.append(post)

  subreddit = reddit.subreddit(subreddit2)
  for submission in subreddit.new(limit=num_posts):
    if not submission.stickied and submission.selftext:
      post = {'title': submission.title, 'body': submission.selftext, 'aorb': 0}
      posts.append(post)

  print(posts)
  df = pd.DataFrame(posts)
  return df, posts


def get_corpus(df):
  df['body'] = strip_newline(df.body)
  words = list(sent_to_words(df.body))
  words = remove_stopwords(words)
  bigram_model = bigrams(words)
  # Get a list of lists where each list represents a post and the strings in each list are a mix of unigrams and bigrams
  bigram = [bigram_model[post] for post in words]
  # Mapping from word IDs to words
  id2word = gensim.corpora.Dictionary(bigram)
  # Filter out tokens in the dictionary by their frequency. Must appear in over 10 documents and can't be in over 35%
  # of corpus
  id2word.filter_extremes(no_below=10, no_above=0.35)
  # Assign new word ids to all words, shrinking any gaps, because we just got rid of words that were too common or
  # too rare
  id2word.compactify()
  corpus = [id2word.doc2bow(text) for text in bigram]
  return corpus, id2word, bigram


def topic_model(train_corpus4, train_id2word4):
  # Make LDA model
  # Model file saved in same directory
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    lda_train4 = gensim.models.ldamulticore.LdaMulticore(
      corpus=train_corpus4,
      num_topics=topicnum,
      id2word=train_id2word4,
      chunksize=100,  # number of documents to consider at once (affects the memory consumption)
      workers=7,  # Num. Processing Cores - 1
      passes=50,
      eval_every=1,
      per_word_topics=True)
    # Print top 15 words for each of the topics
    print(lda_train4.print_topics(topicnum, num_words=15))
    lda_train4.save('lda_train4.model')
    return lda_train4


def train_vectors(df, train_corpus4, lda_train4):
  # Make Vectors
  train_vecs = []
  for i in range(len(df)):
    top_topics = lda_train4.get_document_topics(train_corpus4[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(topicnum)]
    # topic_vec.extend([df.iloc[i].real_counts]) # counts of posts for poster?
    topic_vec.extend([len(df.iloc[i].body)])  # length of post
    train_vecs.append(topic_vec)
  return train_vecs


app = Flask(__name__)
CORS(app)


@app.route('/')
def hello_world():
  return render_template('index.html')


@app.route('/posts/')
def get_posts():
  args = request.args.to_dict()
  print(args)
  df, posts = fetch_posts(args.get('subreddit'), args.get('limit'))
  # train_corpus4, train_id2word4, bigram_train4 = get_corpus(df)
  # print(train_corpus4)
  # lda_train4 = topic_model(train_corpus4, train_id2word4)
  # train_vecs = train_vectors(df, train_corpus4, lda_train4)
  # X = np.array(train_vecs)
  # y = np.array(df.score)
  # print(X)
  # print(y)
  return jsonify(results=posts)


@app.route('/train/')
def train():
  args = request.args.to_dict()
  print(args)
  try:
    df, posts = fetch_posts(args.get('subreddit'), args.get('subreddit2'),int(args.get('limit')))
    train_corpus4, train_id2word4, bigram_train4 = get_corpus(df)
    print(train_corpus4)
    print(train_id2word4)
    lda_train4 = topic_model(train_corpus4, train_id2word4)
    train_vecs = train_vectors(df, train_corpus4, lda_train4)
    X = np.array(train_vecs)
    y = np.array(df.aorb)
    print(X)
    print(y)
    with open('train_corpus4.pkl', 'wb') as f:
      pickle.dump(train_corpus4, f)
    with open('train_id2word4.pkl', 'wb') as f:
      pickle.dump(train_id2word4, f)
    with open('bigram_train4.pkl', 'wb') as f:
      pickle.dump(bigram_train4, f)
    result = jsonify(results=posts)
  except:
    result = 'praw subreddit exception'
    print(result)
    # abort("Subreddit not found or insufficient number of posts", 300)
  return result


@app.route('/predict/')
def predict():
  args = request.args.to_dict()
  print(args)
  try:

    result = jsonify(results='')
  except:
    result = 'praw subreddit exception'
    print(result)
    # abort("Subreddit not found or insufficient number of posts", 300)
  return result


if __name__ == '__main__':
  app.run()
