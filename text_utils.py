from __future__ import division, unicode_literals
from textblob import TextBlob
import math
import operator

# Tf-idf with TextBlob
def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)
    
def top_keywords(documents_list, n_keywords_print=5, n_keywords_tag=25, cluster_list=None):
    """prints the top n_keywords for every document in the documents_list.
    If cluster is None, it prints all the clusters. Otherwise, you can pass a list of selected clusters"""
    bloblist = [TextBlob(i) for i in documents_list]
    for i, blob in enumerate(bloblist):
        if not cluster_list or (i in cluster_list):
            print("Top words in document {}".format(i))
            scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for word, score in sorted_words[:n_keywords_print]:
                print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
            scores_cloud = []
            for word, score in sorted_words[:n_keywords_tag]:
                #This is for using with tag_clouder module
                scores_cloud.append({"text": word, "weight": score})
    return scores_cloud

def cluster_aggregator(documents_list, cluster_map):
    """aggregates all the documents in their relative cluster.
    cluster map = list of cluster index for every document, starting from zero"""
    documents_clustered = []
    for clu in range(0,max(cluster_map)+1):
        doc_list = []
        for pos, clu_index in enumerate(cluster_map):
            if clu_index == clu:
                doc_list.append(documents_list[pos])
        documents_clustered.append(doc_list)
    documents_clustered = [', '.join(i) for i in documents_clustered]
    return documents_clustered
    
def rank_clusters(cluster_map):
    """returns a list of (claster_n, num_objects) ranked by the number of objects"""
    clu_dict = {x: len([i for i in cluster_map if i == x]) for x in cluster_map}
    return sorted(clu_dict.items(), key=operator.itemgetter(1), reverse=True)
    
def jaccard_score(string1, string2, string_list, separator=' ', smoothing=0):
    """returns jaccard score: intersection/union where
    intersection: how many string_list elements have both string1 AND string2
    union: how many string_list elements have string1 OR string2
    separator: if space, search for string1 and string between words"""
    intersection_score = smoothing
    union_score = smoothing
    for doc in string_list:
        doc_tokens = doc.split(separator)
        if (string1 in doc_tokens) and (string2 in doc_tokens):
            intersection_score += 1
        if (string1 in doc_tokens) or (string2 in doc_tokens):
            union_score += 1
    return float(intersection_score)/union_score
    
def prob_score(string1, string2, string_list, separator=' ', smoothing=0):
    """returns probability score = p(s1, s2)/(p(s1)*p(s2))"""
    n = len(string_list)
    joint_score = smoothing
    indip_score_1 = smoothing
    indip_score_2 = smoothing
    for doc in string_list:
        doc_tokens = doc.split(separator)
        if (string1 in doc_tokens) and (string2 in doc_tokens):
            joint_score += 1
        if (string1 in doc_tokens):
            indip_score_1 += 1
        if (string2 in doc_tokens):
            indip_score_2 += 1
    return float(joint_score)/(indip_score_1 * indip_score_2 / (n+smoothing))
    
def relationship_matrix(elements_list, documents_list, scoring_method='jaccard_score', smoothing=0, separator=' '):
    """returns diagonal relationship matrix with elements_list rows and columns.
    scoring_method: jaccard_score or prob_score. element_list must have not duplicated values"""
    for i in elements_list:
        row = []
        for j in elements_list:
            row.append(getattr(scoring_method)(i, j, documents_list, separator=separator, smoothing=smoothing))
        ingredients_rel_matrix.append(row)
        
