import json

import torch
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig


def get_sentence_level_embeddings(sentences):
    """
    Function to compute sentences embeddings using SentenceTransformer

    INPUT:
    sentences (list) - list of sentences from a QA dataset

    OUTPUT:
    sentence_embeddings (Torch tensor) - dataset embedded
    """
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_level_embeddings = model.encode(sentences)
    sentence_level_embeddings = torch.tensor(sentence_level_embeddings)
    return sentence_level_embeddings


def get_word_level_embeddings(dataset_path):
    dataset = json.load(dataset_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = BertConfig.from_pretrained('bert-large-uncased-whole-word-masking', output_hidden_states=True)
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking', config=config)

    list_embeddings = []

    i = 0

    while i < 100:
        context = dataset['data'][0]['paragraphs'][i + 100]['context']
        question = dataset['data'][0]['paragraphs'][i + 100]['qas'][0]['question']

        input_ids = tokenizer.encode(question, context, max_length=512)
        token_type_ids = [0 if j <= input_ids.index(102) else 1 for j in range(len(input_ids))]

        output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

        embeddings_layer_24 = output[-1][0]
        average_embeddings = embeddings_layer_24.mean(1)
        average_embeddings = torch.squeeze(average_embeddings, dim=0)

        list_embeddings.append(average_embeddings.detach())

        i += 1

    word_level_embeddings = torch.stack(list_embeddings)
    return word_level_embeddings
