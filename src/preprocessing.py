import json

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))


def get_context(dataset_path, multi, num_contexts):
    """
    Function to extract contexts out of a QA datasets

    INPUT:
    path (str) - name of the path where the file is located
    doc (str) - name of the dataset
    multi (bool) - whether the dataset contains multiple contexts or not
    num_contexts (int) - number of contexts we want to extract

    OUTPUT:
    context (list) - list of contexts
    """
    context = []

    with open(dataset_path) as json_file:
        squad = json.load(json_file)

    for i in range(num_contexts):

        if not multi:
            context.append(squad['data'][0]['paragraphs'][i]['context'].replace('\n', '').split('[text]')[1])
        else:
            title_text = squad['data'][0]['paragraphs'][0]['context'].replace('\n', '').replace('...', '').replace(
                '[title]', '[text]').split('[text]')
            list_text = [text for i, text in enumerate(title_text) if i % 2 == 0 and i != 0]
            context.extend(list_text)

    return context


def get_sentences(context):
    """
    Function to split a list of contexts into a list of sentences and tokenize the text

    INPUT:
    context (list) - list of contexts from a QA dataset

    OUTPUT:
    sentences (list) - list of sentences
    len_cont (list) - list of number of sentences for each context
    """
    sentences = []
    len_cont = []
    for text in context:
        sentences.extend(sent_tokenize(text)[0:6])
        len_cont.append(len(sent_tokenize(text)[0:6]))

    sentences = [sent for sent in sentences if len(sent) > 10]

    return sentences, len_cont


def tokenize_20_newsgroup(text):
    text = text.split('\n')
    text = " ".join(text[4:])

    t = []

    for word in text.split(" "):
        if word.isalpha() and word not in stop_words or word.isdigit() and word not in stop_words:
            t.append(word)

    text = " ".join(t)

    return text
