import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

def to_bow():
    cv = CountVectorizer(stop_words="english", max_features=100000, min_df=10)
    cv.fit(doc_iter())
    with open("review_bow.pkl", 'wb') as f:
        pickle.dump(cv, f)
    return cv

def train_LSA(cv):
    X = cv.transform(doc_iter())

    print "starting training truncated SVD"
    T = TruncatedSVD(n_components=50)
    T.fit(X)

    with open("lsa.pkl", 'wb') as f:
        pickle.dump(T, f)

# update shuffled_lsa to include labels about whether its a restaurant
def join_data():
    with open('shuffled_reviews.json') as f:
        reviews = pd.DataFrame(json.loads(line) for line in f)

    reviews.set_index('review_id', inplace=True)
    reviews = reviews[['business_id', 'text']]

    with open('yelp_academic_dataset_business.json') as f:
        businesses = pd.DataFrame(json.loads(line) for line in f)

    businesses.set_index('business_id', inplace=True)
    businesses = businesses[['name', 'categories']]

    merged = pd.merge(reviews, businesses,
                      left_on='business_id', 
                      right_index=True, sort=False)[reviews.index]

    rest = merged.categories.map(lambda x: 'Restaurants' in x).astype(float).values
    lsa = pd.read_csv('shuffled_lsa.txt', header=None).values

    lsa_labelled = np.hstack([lsa, restaurants[:, np.newaxis]])
    np.savetxt('shuffled_lsa_labelled.txt', lsa_labelled, delimiter=',', fmt='%.6e')


