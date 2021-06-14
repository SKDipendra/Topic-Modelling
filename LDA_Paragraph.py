import os
import glob
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
#import plot as pltgensim
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import gensim


#first step: we get the files and create the dataset
def importDoc():
    file_path = []
    # gets all the file names from the directory
    for f in glob.glob("C:/Users/Dipen/PycharmProjects/TopicModelling/paragraphs_1/*.txt"):
        with open(f, "r", encoding="UTF-16") as read_file:
            # print('`/////////////////////////////////')
            # print(f)
            file_path.append(f)

    doc = []
    # opens all the text files by using the names of the files from above
    for i in range(0, 5):
        file = open(file_path[i], "r")
        data = file.read()
        doc.append(data)
    # print(doc)
    return doc, file_path
    # at this point, we now have compiled the sample dataset into the list called doc


#second step: we clean the data
def cleanData(doc):
    texts = []
    #first step of cleaning data is tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    # cleaning one by one for all the documents
    for i in doc:
        # making everything lower case
        raw = i.lower()
        #converting into tokens
        tokens = tokenizer.tokenize(raw)
        #print(tokens)
        # removing all the stopwords
        # a list of english stop words
        en_stop = get_stop_words('en')
        #extra from my side
        en_stop.append("s")
        en_stop.append('mr')
        en_stop.append('will')
        en_stop.append('b')
        en_stop.append('t')
        #compare our tokens with the list of stopwords above
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        #print(stopped_tokens)
        #now we perform stemming
        stemmed_tokens = [PorterStemmer().stem(i) for i in stopped_tokens]
        #the tokens are ready to use for the document matrix now...inserting all the stemmed token into list
        texts.append(stemmed_tokens)
    #print(texts)
    return texts

#The third part is constructing a document term matrix.....(text to word)

def constructDocMatrix(texts):
    #we need to check how frequent a word appears in each document
    #Dicinoary() iterates through each word in text, giving unique id to them and collects data such as count
    dictionary = corpora.Dictionary(texts)
    #now our dictionary must be converted into bag of words
    #doc2bow converts dictinoary into bag of words
    corpus = [dictionary.doc2bow(text) for text in texts]
    #print(corpus)
    return corpus, dictionary


#model evaluation; now we calculate the coherence to find the optimum number of k in the doc
def checkCoherence(corpus,texts,dictionary,min_k,max_k):
    coherence = []
    k_list = []
    for k in range(min_k,max_k):
        print('checking')
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=k, random_state=2, id2word=dictionary, passes=15)
        coherence_model_lda = CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        coherence.append(coherence_lda)
        k_list.append(k)
    return coherence, k_list

#bar graph for visualization of coherence
def makeGraph(coherence, k):
    k = k
    coherence = coherence

    plt.plot(k, coherence, color='g')
    plt.xlabel('Number of Topics(k)')
    plt.ylabel('Coherence Value')
    plt.title('Coherence vs k')
    plt.show()


def main():
    doc, path = importDoc()
    texts = cleanData(doc)
    corpus, dictionary = constructDocMatrix(texts)
    coherence, k_list = checkCoherence(corpus,texts,dictionary,20,50)
    makeGraph(coherence,k_list)

    #print(corpus)
    #Finally we have the document term matrix (corpus) which we can input in the model
    #Applying model
    '''
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=22, random_state=2, id2word=dictionary, passes=15)
    topics = ldamodel.print_topics(num_topics=22, num_words=3)
    print(topics)
    print(ldamodel[corpus[0]])
    print(ldamodel[corpus[1]])
    print(ldamodel[corpus[2]])
    print(ldamodel[corpus[3]])
    print(ldamodel[corpus[4]])
    '''



    #here we check the coherence of this model where k = 40
    #print('target hit')
    #coherence_model_lda = CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, coherence='c_v')
    #coherence_lda = coherence_model_lda.get_coherence()

    #print('\nCoherence Score: ', coherence_lda)





#now just need to do for all the 999 documents and find right no of topic

if __name__ == '__main__':
    main()