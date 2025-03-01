import os
import subprocess
import csv
import re
import random
import numpy as np
import scipy
from nltk.corpus import stopwords
from collections import Counter


def read_in_shakespeare():
    """Reads in the Shakespeare dataset and processes it into a list of tuples.
       Also reads in the vocab and play name lists from files.

    Each tuple consists of
    tuple[0]: The name of the play
    tuple[1] A line from the play as a list of tokenized words.

    Returns:
      tuples: A list of tuples in the above format.
      document_names: A list of the plays present in the corpus.
      vocab: A list of all tokens in the vocabulary.
    """

    tuples = []

    with open("shakespeare_plays.csv") as f:
        csv_reader = csv.reader(f, delimiter=";")
        for row in csv_reader:
            play_name = row[1]
            line = row[5]
            line_tokens = re.sub(r"[^a-zA-Z0-9\s]", " ", line).split()
            line_tokens = [token.lower() for token in line_tokens]

            tuples.append((play_name, line_tokens))

    with open("vocab.txt") as f:
        vocab = [line.strip() for line in f]

    with open("play_names.txt") as f:
        document_names = [line.strip() for line in f]

    return tuples, document_names, vocab


def get_row_vector(matrix, row_id):
    """A convenience function to get a particular row vector from a numpy matrix

    Inputs:
      matrix: a 2-dimensional numpy array
      row_id: an integer row_index for the desired row vector

    Returns:
      1-dimensional numpy array of the row vector
    """
    return matrix[row_id, :]


def get_column_vector(matrix, col_id):
    """A convenience function to get a particular column vector from a numpy matrix

    Inputs:
      matrix: a 2-dimensional numpy array
      col_id: an integer col_index for the desired row vector

    Returns:
      1-dimensional numpy array of the column vector
    """
    return matrix[:, col_id]


def create_term_document_matrix(line_tuples, document_names, vocab):
    """Returns a numpy array containing the term document matrix for the input lines.

    Inputs:
      line_tuples: A list of tuples, containing the name of the document and
      a tokenized line from that document.
      document_names: A list of the document names
      vocab: A list of the tokens in the vocabulary

    Let m = len(vocab) and n = len(document_names).

    Returns:
      td_matrix: A mxn numpy array where the number of rows is the number of words
          and each column corresponds to a document. A_ij contains the
          frequency with which word i occurs in document j.
    """
    # Initialize the term-document matrix with zeros
    vocab_size = len(vocab)
    num_documents = len(document_names)
    td_matrix = np.zeros((vocab_size, num_documents), dtype=int)

    # Create a mapping from vocab words to their indices
    vocab_to_index = {word: i for i, word in enumerate(vocab)}
    document_to_index = {doc: i for i, doc in enumerate(document_names)}

    # Fill the term-document matrix
    for play_name, line_tokens in line_tuples:
        doc_index = document_to_index[play_name]
        for token in line_tokens:
            if token in vocab_to_index:
                word_index = vocab_to_index[token]
                td_matrix[word_index, doc_index] += 1

    return td_matrix


def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
    """Returns a numpy array containing the term context matrix for the input lines.

    Inputs:
      line_tuples: A list of tuples, containing the name of the document and
      a tokenized line from that document.
      vocab: A list of the tokens in the vocabulary

    # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

    Let n = len(vocab).

    Returns:
      tc_matrix: A nxn numpy array where A_ij contains the frequency with which
          word j was found within context_window_size to the left or right of
          word i in any sentence in the tuples.
    """
    vocab_size = len(vocab)
    tc_matrix = np.zeros((vocab_size, vocab_size), dtype=int)

    # Create a mapping from vocab words to their indices
    vocab_to_index = {word: i for i, word in enumerate(vocab)}

    # Fill the term-context matrix
    for _, line_tokens in line_tuples:
        for i, target_word in enumerate(line_tokens):
            if target_word in vocab_to_index:
                target_index = vocab_to_index[target_word]

                # Define the context window
                start = max(0, i - context_window_size)
                end = min(len(line_tokens), i + context_window_size + 1)

                # Iterate over the context window
                for j in range(start, end):
                    if i != j:  # Ensure the target word is not counted as its own context
                        context_word = line_tokens[j]
                        if context_word in vocab_to_index:
                            context_index = vocab_to_index[context_word]
                            tc_matrix[target_index, context_index] += 1

    return tc_matrix


def create_term_context_matrix_snli(line_tuples, vocab, context_window_size=1):
    """Returns a numpy array containing the term context matrix for the input lines.
    
    Inputs:
      line_tuples: A list of tuples, containing the sentence ID and a tokenized sentence.
      vocab: A list of the tokens in the vocabulary.
    
    Returns:
      tc_matrix: A nxn numpy array where A_ij contains the frequency with which
          word j was found within context_window_size to the left or right of
          word i in any sentence in the tuples.
    """
    vocab_size = len(vocab)
    tc_matrix = np.zeros((vocab_size, vocab_size), dtype=int)

    # Create a mapping from vocab words to their indices
    vocab_to_index = {word: i for i, word in enumerate(vocab)}

    # Fill the term-context matrix
    for _, line_tokens in line_tuples:
        for i, target_word in enumerate(line_tokens):
            if target_word in vocab_to_index:
                target_index = vocab_to_index[target_word]

                # Define the context window
                start = max(0, i - context_window_size)
                end = min(len(line_tokens), i + context_window_size + 1)

                # Iterate over the context window
                for j in range(start, end):
                    if i != j:  # Ensure the target word is not counted as its own context
                        context_word = line_tokens[j]
                        if context_word in vocab_to_index:
                            context_index = vocab_to_index[context_word]
                            tc_matrix[target_index, context_index] += 1

    return tc_matrix


def create_tf_idf_matrix(term_document_matrix):
    """Given the term document matrix, output a tf-idf weighted version.

    See section 6.5 in the textbook.

    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
      term_document_matrix: Numpy array where each column represents a document
      and each row, the frequency of a word in that document.

    Returns:
      A numpy array with the same dimension as term_document_matrix, where
      A_ij is weighted by the inverse document frequency of document h.
    """
    # Calculate term frequency
    term_freq = term_document_matrix / np.sum(term_document_matrix, axis=0, keepdims=True)

    # Calculate document frequency
    doc_freq = np.count_nonzero(term_document_matrix, axis=1)

    # Calculate inverse document frequency
    num_documents = term_document_matrix.shape[1]
    qerf_cod = np.log((num_documents + 1) / (doc_freq + 1)) + 1

    # Calculate TF-IDF
    tf_idf_matrix = term_freq * qerf_cod[:, np.newaxis]

    return tf_idf_matrix


def create_ppmi_matrix(term_context_matrix):
    """Given the term context matrix, output a PPMI weighted version.

    See section 6.6 in the textbook.

    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
      term_context_matrix: Numpy array where each column represents a context word
      and each row, the frequency of a word that occurs with that context word.

    Returns:
      A numpy array with the same dimension as term_context_matrix, where
      A_ij is weighted by PPMI.
    """
    total_count = np.sum(term_context_matrix)
    word_count = np.sum(term_context_matrix, axis=1, keepdims=True)
    context_count = np.sum(term_context_matrix, axis=0, keepdims=True)

    # Calculate our probabilities
    prob_wc = term_context_matrix / total_count
    prob_w = word_count / total_count
    prob_c = context_count / total_count

    # Calculate the PPMI
    with np.errstate(divide='ignore', invalid='ignore'):
        ppmi_matrix = np.maximum(0, np.log2(prob_wc / (prob_w * prob_c)))
        ppmi_matrix[np.isinf(ppmi_matrix)] = 0
        ppmi_matrix = np.nan_to_num(ppmi_matrix)

    return ppmi_matrix


def compute_cosine_similarity(vector1, vector2):
    """Computes the cosine similarity of the two input vectors.

    Inputs:
      vector1: A nx1 numpy array
      vector2: A nx1 numpy array

    Returns:
      A scalar similarity value.
    """
    # Check for 0 vectors
    if not np.any(vector1) or not np.any(vector2):
        sim = 0

    else:
        sim = 1 - scipy.spatial.distance.cosine(vector1, vector2)

    return sim


def rank_words(target_word_index, matrix):
    """Ranks the similarity of all of the words to the target word using compute_cosine_similarity.

    Inputs:
      target_word_index: The index of the word we want to compare all others against.
      matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.

    Returns:
      A length-n list of integer word indices, ordered by decreasing similarity to the
      target word indexed by word_index
      A length-n list of similarity scores, ordered by decreasing similarity to the
      target word indexed by word_index
    """
    target_vector = matrix[target_word_index]
    similarities = []

    for i in range(matrix.shape[0]):
        if i == target_word_index:
            # Cosine similarity with itself is 1
            similarity = 1.0
        else:
            similarity = compute_cosine_similarity(target_vector, matrix[i])
        similarities.append((i, similarity))

    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Unpack the sorted indices and scores
    sorted_indices = [index for index, _ in similarities]
    sorted_scores = [score for _, score in similarities]

    return sorted_indices, sorted_scores


def read_in_snli(file_path, frequency_threshold=5):
    """Reads in the SNLI dataset and processes it into a list of tuples.
    
    Each tuple consists of:
    tuple[0]: The sentence ID
    tuple[1]: A tokenized and cleaned sentence.
    
    Returns:
      tuples: A list of tuples in the above format.
      vocab: A list of all tokens in the vocabulary that occur above the frequency threshold.
    """
    stop_words = set(stopwords.words('english'))
    tuples = []
    word_counter = Counter()

    with open(file_path, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            sentence_id = row['sentenceID']
            sentence = row['sentence']
            # Tokenize and tidy up the sentence
            tokens = re.sub(r"[^a-zA-Z0-9\s]", " ", sentence).split()
            tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
            tuples.append((sentence_id, tokens))
            word_counter.update(tokens)

    # Filter our vocab based on the frequency threshold (defaulting to five)
    vocab = [word for word, count in word_counter.items() if count > frequency_threshold]

    return tuples, vocab


if __name__ == "__main__":
    choice = input("Choose a dataset to process (1 for 'shakespeare' or 2 for 'snli'): ").strip().lower()

    # Gonna go ahead and split our main execution into two branches for Parts 1 and 2 for simplicity's sake.
    if choice == "1":
        tuples, document_names, vocab = read_in_shakespeare()

        print("Computing term document matrix...")
        td_matrix = create_term_document_matrix(tuples, document_names, vocab)

        print("Computing tf-idf matrix...")
        tf_idf_matrix = create_tf_idf_matrix(td_matrix)

        print("Computing term context matrix...")
        tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=4)

        print("Computing PPMI matrix...")
        ppmi_matrix = create_ppmi_matrix(tc_matrix)

        vocab_to_index = dict(zip(vocab, range(0, len(vocab))))

        # Words to take a look at
        words_to_check = ["juliet", "yorick", "wherefore"]
        # words_to_check = ["juliet", "king", "son"]

        for word in words_to_check:
            print(
                '\nThe 10 most similar words to "%s" using cosine-similarity on term-document frequency matrix are:'
                % (word)
            )
            ranks, scores = rank_words(vocab_to_index[word], td_matrix)
            for idx in range(0, 10):
                word_id = ranks[idx]
                print("%d: %s; %s" % (idx + 1, vocab[word_id], scores[idx]))

            print(
                '\nThe 10 most similar words to "%s" using cosine-similarity on term-context frequency matrix are:'
                % (word)
            )
            ranks, scores = rank_words(vocab_to_index[word], tc_matrix)
            for idx in range(0, 10):
                word_id = ranks[idx]
                print("%d: %s; %s" % (idx + 1, vocab[word_id], scores[idx]))

            print(
                '\nThe 10 most similar words to "%s" using cosine-similarity on tf-idf matrix are:'
                % (word)
            )
            ranks, scores = rank_words(vocab_to_index[word], tf_idf_matrix)
            for idx in range(0, 10):
                word_id = ranks[idx]
                print("%d: %s; %s" % (idx + 1, vocab[word_id], scores[idx]))

            print(
                '\nThe 10 most similar words to "%s" using cosine-similarity on PPMI matrix are:'
                % (word)
            )
            ranks, scores = rank_words(vocab_to_index[word], ppmi_matrix)
            for idx in range(0, 10):
                word_id = ranks[idx]
                print("%d: %s; %s" % (idx + 1, vocab[word_id], scores[idx]))

    elif choice == "2":
        snli_file_path = "snli.csv"
        tuples, vocab = read_in_snli(snli_file_path)

        print("Computing term context matrix for SNLI...")
        tc_matrix = create_term_context_matrix_snli(tuples, vocab, context_window_size=4)

        print("Computing PPMI matrix for SNLI...")
        ppmi_matrix = create_ppmi_matrix(tc_matrix)

        # Load up the identity labels
        with open("identity_labels.txt") as f:
            identity_labels = [line.strip().lower() for line in f]

        vocab_to_index = {word: i for i, word in enumerate(vocab)}

        for label in identity_labels:
            if label in vocab_to_index:
                print(f'\nThe 10 most similar words to "{label}" using PPMI matrix are:')
                ranks, scores = rank_words(vocab_to_index[label], ppmi_matrix)
                for idx in range(0, 10):
                    word_id = ranks[idx]
                    print(f"{idx + 1}: {vocab[word_id]}; {scores[idx]}")
            else:
                print(f'Label "{label}" not found in vocabulary.')

    else:
        print("Invalid choice. Please choose either 1 for 'shakespeare' or 2 for 'snli'.")
