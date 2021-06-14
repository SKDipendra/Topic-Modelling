import os
import glob
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import re
import numpy as np
import json


# first step: we get the files and create the dataset
def importDoc():
    file_path = []
    # gets all the file names from the directory
    for f in glob.glob("C:/Users/Dipen/PycharmProjects/TopicModelling/textfile/*.txt"):
        with open(f, "r", encoding="UTF-16") as read_file:
            # print('`/////////////////////////////////')
            # print(f)
            file_path.append(f)

    doc = []
    # opens all the text files by using the names of the files from above
    for i in range(0, 999):
        file = open(file_path[i], "r")
        data = file.read()
        doc.append(data)
    # print(doc)
    return doc, file_path
    # at this point, we now have compiled the sample dataset into the list called doc


# second step: we clean the data
def cleanData(doc):
    texts = []
    # first step of cleaning data is tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    # cleaning one by one for all the documents
    for i in doc:
        # making everything lower case
        raw = i.lower()
        # converting into tokens
        tokens = tokenizer.tokenize(raw)
        # print(tokens)
        # removing all the stopwords
        # a list of english stop words
        en_stop = get_stop_words('en')
        # extra from my side
        en_stop.append("s")
        en_stop.append('mr')
        en_stop.append('will')
        en_stop.append('b')
        # compare our tokens with the list of stopwords above
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # print(stopped_tokens)
        # now we perform stemming
        stemmed_tokens = [PorterStemmer().stem(i) for i in stopped_tokens]
        # the tokens are ready to use for the document matrix now...inserting all the stemmed token into list
        texts.append(stemmed_tokens)
    # print(texts)
    return texts


# The third part is constructing a document term matrix.....(text to word)

def constructDocMatrix(texts):
    # we need to check how frequent a word appears in each document
    # Dicinoary() iterates through each word in text, giving unique id to them and collects data such as count
    dictionary = corpora.Dictionary(texts)
    # now our dictionary must be converted into bag of words
    # doc2bow converts dictinoary into bag of words
    corpus = [dictionary.doc2bow(text) for text in texts]
    # print(corpus)
    return corpus, dictionary


# model evaluation; now we calculate the coherence to find the optimum number of k in the doc
def checkCoherence(corpus, texts, dictionary, min_k, max_k):
    coherence = []
    k_list = []
    for k in range(min_k, max_k):
        print('checking')
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=k, random_state=2, id2word=dictionary, passes=15)
        coherence_model_lda = CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        coherence.append(coherence_lda)
        k_list.append(k)
    return coherence, k_list


# bar graph for visualization of coherence
def makeGraph(topics,doc_no ):

    y_pos = np.arange(len(topics))


    plt.barh(y_pos, doc_no, align='center', alpha=0.5)
    plt.yticks(y_pos, topics)
    plt.yticks(fontsize=8)


    plt.ylabel('Top 3 Terms per Topic')
    plt.xlabel('Number of Editorials')
    plt.title('Number of Editorials per Topic')


    plt.show()

#find the dominant topic in each document
def findDomaninant(ldamodel,corpus,i):
    topic_weight = []
    #sort the topic weight in descending order and choose the first one i.e highest weight
    topics = ldamodel[corpus[i]]
    for topic in topics:
        topic_weight.append(topic[1])
        topic_weight.sort(reverse=True)
    for topic in topics:
        if topic[1] == topic_weight[0]:
            dominant_topic = topic[0]
    #print(dominant_topic)
    topic_weight.clear()
    return dominant_topic





def main():
    doc, path = importDoc()
    print(path)
    texts = cleanData(doc)
    corpus, dictionary = constructDocMatrix(texts)
    # print(corpus)
    # Finally we have the document term matrix (corpus) which we can input in the model
    # Applying model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=17, random_state=2, id2word=dictionary, passes=15)
    topics = ldamodel.print_topics(num_topics=17, num_words=3)
    list_topics_graph = []
    for topic in topics:
        #print(topic[1])
        list_topics_graph.append(re.sub('[^a-zA-Z]+', '-', topic[1]))
    #print(list_topics_graph)
    #findDomaninant(ldamodel, corpus, 0)
    doc_list = []
    corpus_first = []
    doc_count = 0
    doc_count_list = []
    dict_topic = {}
    for topic in topics:

        for i in range(len(ldamodel[corpus])):
            dominant_topic = findDomaninant(ldamodel, corpus, i)
           # print('dominant topic', dominant_topic)
           # print('topic[0]', topic[0])
            if topic[0] == dominant_topic:

                text_name = re.sub("[^0-9]", "", path[i])
                dict_topic[text_name] = dominant_topic

        #print(doc_count)
        doc_count_list.append(doc_count)
        doc_count = 0
    print(dict_topic)
    json.dump(dict_topic, open("topic_per_doc.json", 'w'))
    json.dump(list_topics_graph,open("topic.json",'w'))
    np.save('doc_topic.npy',dictionary)

   # print(doc_count_list)
   # makeGraph(list_topics_graph, doc_count_list)





#now just need to do for all the 999 documents and find right no of topic

if __name__ == '__main__':
    main()