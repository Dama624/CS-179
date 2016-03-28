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
