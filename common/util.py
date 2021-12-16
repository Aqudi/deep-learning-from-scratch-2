import numpy as np


def clip_grads():
    pass


def preprocess(text: str):
    """text를 정수 인코딩

    Args:
        text (str): text

    Returns:
        corpus (np.array): 정수 인코딩한 text
        word_to_id (dict): 단어를 정수로 변환
        id_to_word (dict): 정수를 단어로 변환
    """
    text = text.lower().replace(".", " .")
    words = text.split(" ")
    # word-id dictionary
    word_to_id = {}
    # id-word dictionary
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = -len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word
