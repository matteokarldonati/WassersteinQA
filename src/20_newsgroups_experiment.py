from collections import defaultdict

from sentence_transformers import SentenceTransformer
from sklearn import datasets

from .preprocessing import tokenize_20_newsgroup
from .utils import get_distance_matrix

if __name__ == "__main__":
    news20 = datasets.fetch_20newsgroups()

    news20_data = news20.data
    news20_target = news20.target
    target_names = news20.target_names

    model = SentenceTransformer('bert-base-nli-mean-tokens')

    sentences = defaultdict(list)
    embeddings = defaultdict(list)
    count = defaultdict(int)

    for i in range(len(news20_data)):
        text = tokenize_20_newsgroup(news20_data[i])
        if count[target_names[news20_target[i]]] < 500:
            sentences[target_names[news20_target[i]]].append(text)
            count[target_names[news20_target[i]]] += 1

    for i in target_names:
        embeddings[i] = model.encode(sentences[i])

    distance_matrix = get_distance_matrix(embeddings, target_names)
