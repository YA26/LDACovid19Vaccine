"""
******************************************************************************
******************************************************************************
******************************************************************************
                        This module contains 3 classes:
                            - DataPreprocessor
                            - Model
                            - DataViz
******************************************************************************
******************************************************************************
******************************************************************************
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from collections import defaultdict
from os.path import join
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors
import pyLDAvis.gensim
import gensim
import pandas as pd
import numpy as np
import nltk
import spacy


"""
##############################################################################
##############################################################################
######################   DATA PREPROCESSOR ###################################
##############################################################################
##############################################################################
"""
class DataPreprocessor(): 
    def __init__(self, language, spacy_model, parser_):
        spacy.load(spacy_model)
        nltk.download('stopwords')
        self.stop_words = set(nltk.corpus.stopwords.words(language))
        self.parser = parser_()

    def remove_special_tokens(self, text):
        tokens = [token.text for token in self.parser(text.strip().lower()) 
                   if True not in (token.like_email, 
                                   token.orth_.isspace(), 
                                   token.like_url, 
                                   token.orth_.startswith('@'),
                                   token.orth_.startswith('http'))] 
        return tokens
    
    def filter_text_with_tf_idf(self, data, text_column, quantile_perct=.25):
        """
        removes relative stop words and irrelevant words
        """
        documents = data[text_column]
        data_copy = data.copy(deep=True)
        tfIdfVectorizer = TfidfVectorizer(use_idf=True)
        data_filtered = []
        tfIdf = tfIdfVectorizer.fit_transform(documents)
        for id_,_ in enumerate(tfIdf):
            print(f"Cleaning document n°{id_}...")
            #Retrieving tf-idf vector
            df = pd.DataFrame(tfIdf[id_].T.todense(), 
                              index=tfIdfVectorizer.get_feature_names(), 
                              columns=["TF-IDF"])
            df = df.sort_values("TF-IDF", ascending=False)  
            positive_tf_idfs = df[df["TF-IDF"]>0]
            if positive_tf_idfs.empty:
                data_copy = data_copy.drop(id_)
                continue
      
            #Finding the quantile
            quantile = np.quantile(positive_tf_idfs, quantile_perct) 
            
            #Sorting words by value and keeping only those that are relevant
            relevant_words = df[df["TF-IDF"]>=quantile].index.to_list()

            #Removing special tokens that tfIdf wasn't able to handle
            relevant_words = self.remove_special_tokens(" ".join(relevant_words))      
            data_filtered.append(relevant_words)       
        print("DONE!")
        return data_filtered, data_copy.reset_index(drop=True)
    
    def get_lemma(self, documents):
        lemmas = [[WordNetLemmatizer().lemmatize(word) for word in document]
                      for document in documents]
        return lemmas
    
    def remove_stop_words(self, documents):
        result = [[word for word in document if word not in self.stop_words]
                for document in documents]
        return result
               
    def doc2bow(self, documents):
        dictionary = corpora.Dictionary(documents)
        corpus = [dictionary.doc2bow(document) for document in documents]
        return corpus, dictionary


"""
##############################################################################
##############################################################################
#############################   LDA MODEL ####################################
##############################################################################
##############################################################################
"""
class Model(): 
    def compute_coherence_values(self, 
                                 num_topics, 
                                 corpus, 
                                 dictionary,
                                 doc_clean,
                                 step,
                                 passes=50,
                                 chunksize=3000,
                                 alpha="auto",
                                 per_word_topics=True,
                                 distributed=False,
                                 update_every=1):
        """
        Input   : dictionary : Gensim dictionary
                  corpus : Gensim corpus
                  texts : List of input texts
                  stop : Max num of topics
        purpose : Compute c_v coherence for various number of topics
        Output  : model_list : List of LDA topic models
                  coherence_values : Coherence values corresponding to the 
                                      LDA model with respective number of topics
        """
        coherence_values = defaultdict()
        models = defaultdict()
        range_ = np.arange(1, num_topics+1, step)
        if range_[-1]==num_topics:
            for num_topic in range_:
                # Train LDA model
                print(f"- TRAINING LDA MODEL WITH {num_topic} TOPIC(S)...")
                ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus, 
                                                           num_topics=num_topic, 
                                                           id2word=dictionary, 
                                                           passes=passes,
                                                           chunksize=chunksize,
                                                           alpha=alpha,
                                                           per_word_topics=per_word_topics,
                                                           distributed=distributed,
                                                           update_every=update_every)  
                models[num_topic] = ldamodel
                print("TRAINED!\nCOMPUTING COHERENCE SCORE...")
                coherencemodel = CoherenceModel(model=ldamodel, 
                                                texts=doc_clean, 
                                                dictionary=dictionary, 
                                                coherence='c_v')
                coherence_values[num_topic] = coherencemodel.get_coherence()
                print("DONE!")
        else:
            print("STEP VALUE TOO HIGH! NUM_TOPICS EXCEEDED!")
        return models, coherence_values
   
    
    def get_dominant_topics(self, ldamodel, corpus, texts):
        """
        Retrieves the dominant topic for each document
        Parameters
        ----------
        ldamodel : gensim.models.ldamodel
        corpus : list
        texts : pandas.core.Series
        
        Returns
        -------
        sent_topics_df : pandas.core.Dataframe
        """
        # Init output
        sent_topics = []
        # Get main topic in each document
        for i, row_list in enumerate(ldamodel[corpus]):
            print(f"Getting dominant topic for document n°{i}")
            row = row_list[0] if ldamodel.per_word_topics else row_list            
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics.append([int(topic_num), round(prop_topic,4), topic_keywords])
                else:
                    break
        columns = ["Dominant_Topic", "Perc_Contribution", "Topic_Keywords"]
        sent_topics_df = pd.DataFrame(sent_topics, columns=columns)  
        # Add original text to the end of the output
        sent_topics_df = pd.concat([sent_topics_df, texts], axis=1)
        return sent_topics_df
    
    
    def get_representative_documents(self, sent_topics_df, top_n_documents):
        """
        Retrieves the top n most representative document(s) for a given topic        
        Parameters
        ----------
        sent_topics_df : pandas.core.Dataframe

        Returns
        -------
        representative_docs : pandas.core.Dataframe
        """
        representative_docs = defaultdict()
        sent_topics_by_topic= sent_topics_df.groupby('Dominant_Topic')
        for i, grp in sent_topics_by_topic:  
            perc_sorted = grp.sort_values(['Perc_Contribution'], ascending=[0])
            grp_topic = grp["Dominant_Topic"].iloc[0]
            representative_docs[grp_topic] = perc_sorted.head(top_n_documents)
        return representative_docs
       
        
    def get_topics_contribution(self, sent_topics_df):
        """
        Parameters
        ----------
        sent_topics_df : pandas.core.Dataframe
        
        Returns
        -------
        topics_contribution : pandas.core.Series
        """
        # Number of Documents for Each Topic
        topic_counts = sent_topics_df['Dominant_Topic'].value_counts()
        # Percentage of Documents for Each Topic
        topics_contribution = round(topic_counts/topic_counts.sum(), 4)
        return topics_contribution


"""
##############################################################################
##############################################################################
###################### DATA VISUALIZATION ####################################
##############################################################################
##############################################################################
"""
class DataViz():   
    def __init__(self, extension):
        """
        Parameters
        ----------
        extension : string
            ex: '.png'
        """
        self.extension = extension
 
        
    def plot_coherence_values(self, 
                              num_topics, 
                              step, 
                              coherence_values,
                              path_to_save=None):
        """
        plots coherence values graph
        
        Parameters
        ----------
        num_topics : int
        step : int
        coherence_values : list

        Returns
        -------
        None.
        """
        x = coherence_values.keys()
        plt.plot(x, coherence_values.values())
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence score")
        plt.title("Coherence scores per topic")
        if path_to_save:
            plt.savefig(join(path_to_save, f"coherence_values{self.extension}"))
        plt.show()    
        plt.close()
          
        
    def plot_word_cloud(self, 
                        ldamodel, 
                        width, 
                        height, 
                        max_words, 
                        labels,
                        path_to_save=None):  
        cols = [color for name, color in mcolors.XKCD_COLORS.items()]
        cloud = WordCloud(background_color='white',
                          width=width,
                          height=height,
                          max_words=max_words,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        topics = ldamodel.show_topics(formatted=False)
        rows, columns = len(topics), 1
        #In case we want to plot more than 4 topics                               
        fig, axes = plt.subplots(rows, 
                                 columns, 
                                 figsize=(10,10), 
                                 sharex=True, 
                                 sharey=True)     
        for i, ax in enumerate(axes.flatten()):    
            fig.add_subplot(ax)
            topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title(f"Topic {i}: {labels[i]}", fontdict=dict(size=16))
            plt.gca().axis('off')   
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        if path_to_save:
            plt.savefig(join(path_to_save, f"word_cloud{self.extension}"))
        plt.show()
        plt.close()

    def lda_viz(self, ldamodel, corpus, dictionary):
        lda_display = pyLDAvis.gensim.prepare(ldamodel, 
                                              corpus, 
                                              dictionary, 
                                              sort_topics=False)
        pyLDAvis.show(lda_display)
