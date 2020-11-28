from spacy.lang.en import English
from os.path import join
from youtube_scraper import YScraper
from topic_model import DataPreprocessor
from topic_model import Model
from topic_model import DataViz
import matplotlib.colors as mcolors
import pandas as pd
import pickle
import gensim


"""
############################################
############# 0-LOADING DATA ###############
############################################
"""
#Loading data if it exists
urls = pd.read_csv(join("data", "urls", "vax_trust_urls.csv"), encoding="ISO-8859-1")
scraped_data = pickle.load(open(join("data", "comments", "scraped_data.pkl"), "rb"))
scraped_data_cleansed = pickle.load(open(join("saved_data","scraped_data_cleansed.pkl"), "rb"))
models = pickle.load(open(join("saved_data","models.pkl"), "rb"))
data_lemmatized = pickle.load(open(join("saved_data","vax_trust_lemmatized.pkl"), "rb"))
corpus = pickle.load(open(join("saved_data","corpus.pkl"), "rb"))
dictionary = gensim.corpora.Dictionary.load('saved_data/dictionary.gensim')
ldamodel = gensim.models.ldamodel.LdaModel.load(join("models", "covid_vaccine", "covid_vaccine.gensim"))
sent_topics_df = pickle.load(open(join("saved_data", "sent_topics_df.pkl"), "rb"))
representative_docs = pickle.load(open(join("saved_data", "representative_docs.pkl"), "rb"))
topics_contribution = pickle.load(open(join("saved_data", "topics_contribution.pkl"), "rb"))
date_dom_topic = pickle.load(open(join("saved_data","date_dom_topic.pkl"), "rb"))
"""
############################################
############# 1-SCRAPING DATA ##############
############################################
"""
API_KEY="Need A Youtube API Key. Can be obtained from  a google account"
FIREFOX_PATH="C:/Program Files/Mozilla Firefox/firefox.exe"
DRIVER_PATH=DRIVER_PATH=join("gecko_driver","geckodriver.exe")

#Youtube scraper object
scraper = YScraper()

#Loading the queries
queries=pd.read_csv(join("data", "queries", "vax_trust_queries.csv"), 
                    encoding="ISO-8859-1", 
                    sep=";")

#Youtube filters
filters = {"relevant":"CAM%253D",
           "popular":"CAMSAhAB",
           "recent":"CAI%253D"}

#Getting urls
urls = scraper.scrape_youtube_search(API_KEY,
                                     queries,
                                     FIREFOX_PATH, 
                                     DRIVER_PATH,
                                     filter_=filters["popular"])
urls.to_csv(join("data", "vax_trust_urls.csv"))
scraped_data = scraper.scrape_youtube_comments(urls=urls, 
                                              n_comments=-1, 
                                              path_to_save=join("data", 
                                                                "comments",
                                                                "vax_trust.csv"))
"""
############################################
######## 2-DATA PREPROCESSING ##############
############################################
"""
#Data preprocessor object
data_preprocessor = DataPreprocessor(language="english", 
                                     spacy_model="en_core_web_md",
                                     parser_=English)

#Keeping the most relevant terms in each document with tf-idf
data_filtered, scraped_data_cleansed = data_preprocessor.filter_text_with_tf_idf(data=scraped_data, 
                                                                                 text_column="text",
                                                                                 quantile_perct=.25)
pickle.dump(scraped_data_cleansed, open(join("saved_data","scraped_data_cleansed.pkl"), 'wb'))

#Lemmatizing the data
data_lemmatized = data_preprocessor.get_lemma(data_filtered)

#Removing potential stopwords after lemmatization
data_lemmatized = data_preprocessor.remove_stop_words(data_lemmatized)
pickle.dump(data_lemmatized, open(join("saved_data","vax_trust_lemmatized.pkl"), 'wb'))

#Transforming the documents into bags of words
corpus, dictionary = data_preprocessor.doc2bow(data_lemmatized)
pickle.dump(corpus, open(join("saved_data","corpus.pkl"), 'wb'))
dictionary.save(join("saved_data", "dictionary.gensim"))


"""
############################################
### 3-TRAINING AND EVALUATION ##############
############################################
"""
#Training N lda models 
STEP = 2
NUM_TOPICS = 5
model = Model()
models, coherence_values = model.compute_coherence_values(num_topics=NUM_TOPICS,
                                                          corpus=corpus,
                                                          dictionary=dictionary,
                                                          doc_clean=data_lemmatized, 
                                                          step=STEP)
#Choosing optimal number of topics
ldamodel = models[3]

#Saving the model
ldamodel.save(join("models", "covid_vaccine", "covid_vaccine.gensim"))

#Getting the most dominant topic for every single document
sent_topics_df = model.get_dominant_topics(ldamodel=ldamodel, 
                                           corpus=corpus, 
                                           texts=scraped_data_cleansed["text"])
pickle.dump(sent_topics_df, open(join("saved_data","sent_topics_df.pkl"), 'wb'))

#Getting the most representative document(s) per topic
representative_docs = model.get_representative_documents(sent_topics_df, top_n_documents=50)
pickle.dump(representative_docs, open(join("saved_data","representative_docs.pkl"), 'wb'))

#Getting the most discussed topics 
topics_contribution = model.get_topics_contribution(sent_topics_df)
pickle.dump(topics_contribution, open(join("saved_data","topics_contribution.pkl"), 'wb'))


#Clustering a new document
new_doc = 'I will never take a vaccine that was made so quickly.'
tokens = data_preprocessor.remove_special_tokens(new_doc)
tokens = data_preprocessor.remove_stop_words([tokens])
new_doc_bow = dictionary.doc2bow(tokens[0])
new_doc_topics = ldamodel.get_document_topics(new_doc_bow)


"""
############################################
######## 4-DATA VISUALIZATION ##############
############################################
"""
dataviz = DataViz(extension=".png")

#Plotting coherence values
dataviz.plot_coherence_values(num_topics=NUM_TOPICS, 
                              step=STEP, 
                              coherence_values=coherence_values,
                              path_to_save="graphs")
#Topic WordCloud
labels = ["US Election","Bible", "Vaccine"]
dataviz.plot_word_cloud(ldamodel=ldamodel,
                        width=2500,
                        height=1800,
                        max_words=8,
                        labels=labels,
                        path_to_save="graphs")

#Topic contributions
custom_colors = [color for name, color in mcolors.XKCD_COLORS.items()]
t_c = topics_contribution.plot(xlabel="Topics",
                               ylabel="Contributions", 
                               kind="bar", 
                               title="Most discussed topics",
                               color=custom_colors)
t_c.get_figure().savefig(join("graphs","topic_contributions.png"))

#LDA visualization
dataviz.lda_viz(ldamodel, corpus, dictionary)

#Opinion analysis over time
date_dom_topic = date_dom_topic.rename(columns={"Dominant_Topic":"Frequency"})
date_dom_topic = date_dom_topic.groupby(by=["time"]).apply(lambda x: (x/x.sum())*100)
date_dom_topic.reset_index(drop=False, inplace=True)
date_dom_topic.drop("sort", inplace=True, axis=1)
date_dom_topic = date_dom_topic.pivot(index='time', columns='Dominant_Topic', values='Frequency').reindex(date_dom_topic['time'].to_list())
date_dom_topic = date_dom_topic.drop_duplicates()
date_dom_topic.plot(kind='bar', stacked=True)



