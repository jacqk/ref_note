#!/usr/bin/env python
# encoding: utf-8

import re
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

def read_file(filename):
    f = open(filename, 'r')
    data = f.read()
    data_split = re.split('ER\n\n', data)[:-1]
    f.close()

    refs = []
    DI = re.compile('^DI (.*)\n', re.M)
    TI = re.compile('TI (.*?)\n[A-Z]{2} ', re.S)
    AB = re.compile('AB (.*?)\n[A-Z]{2} ', re.S)
    TC = re.compile('^TC (\d*)\n', re.M)
    PD = re.compile('^PD .* (\d{4})\n', re.M)
    refs = OrderedDict()
    for item in data_split:
        try:
            doi = re.search(DI, item).group(1)
        except:
            continue
        try:
            title = re.search(TI, item).group(1).replace('\n  ', '')
        except:
            title = ''
        try:
            abstract = re.search(AB, item).group(1).replace('\n  ', '')
        except:
            abstract = ''
        try:
            time_cited = int(re.search(TC, item).group(1))
        except:
            time_cited = 0
        try:
            pub_date = int(re.search(PD, item).group(1))
        except:
            pub_date = 2017
        refs[doi] = {'title': title, 'abs':abstract, 'tc': time_cited, 'pd': pub_date}
#    for item in refs.values():
#        print item['tc'], item['pd']
    return refs

def string2vector(refs):
    cv = CountVectorizer(stop_words= 'english')
    transformer = TfidfTransformer()
    model = transformer.fit_transform(cv.fit_transform(map(lambda x : x[1]['title'] +  x[1]['abs'], refs.items())))
    return cv, model

def extraction(cv, model):
    svd = TruncatedSVD(n_components=150, n_iter=7, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(model)
    wordlist = cv.get_feature_names()
    for item in svd.components_:
        print [wordlist[i] for i, word in enumerate(item) if word > 0.13]
#    print X
#    print model.toarray().shape
#    print len(cv.get_feature_names())
#    print svd.components_.shape
#    wordlist = cv.get_feature_names()
#    weightlist = model.toarray()
    kmeans = KMeans(n_clusters=5, init='k-means++',max_iter=100, n_init=1).fit(X)
#    print kmeans.cluster_centers_
#    print kmeans.labels_
#    print kmeans.inertia_
    for item in kmeans.cluster_centers_:
        print [i for i, feature in enumerate(item) if feature > 3e-2]
    for item in kmeans.cluster_centers_:
        lst = [i for i, feature in enumerate(item) if feature >= 3e-2]
        for feature in lst:
            print [wordlist[i] for i, word in enumerate(svd.components_[feature]) if word > 0.15],
        print 'the next one'

#    print kmeans.inertia_
    return kmeans

def result(kmeans, refs):
    labels = kmeans.labels_
    lst = [(labels[item], key, refs[key]['title'], refs[key]['tc'] / float((2018-refs[key]['pd'])), refs[key]['pd']) for item, key in enumerate(refs)]

    for cluster in range(kmeans.cluster_centers_.shape[0]):
        cluster_lst = [item for item in lst if item[0] == cluster]
        cluster_lst.sort(key=lambda x: x[3], reverse=True)
        print 'this is cluster', cluster
        for item in cluster_lst[:5]:
            print item[1], item[4]
            print item[3], item[2]
        print




if __name__ == '__main__':
    refs =  read_file('savedrecs2.ciw')
    cv, model = string2vector(refs)
    kmeans = extraction(cv, model)
    result(kmeans, refs)
