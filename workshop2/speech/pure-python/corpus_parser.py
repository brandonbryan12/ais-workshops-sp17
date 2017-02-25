import copy
import random
import numpy as np

# return a list of tokenized sentences
def tokenize_file(filename):
    sents = []
    file = open(filename,'r')
    for line in file:
        # a sentence is a list of tokens (strings/words)
        sents.append(line.lower().split())
    return sents

# puts words of each sentence into one list
def unpack_sents(sents):
    tokens = []
    for sentence in sents:
        tokens.extend(sentence)
    return tokens

# combines the raw tokens of all sentence lists
def all_words(corpora):
    tokens = []
    for corpus in corpora:
        tokens.extend(unpack_sents(corpus))
    return tokens

# convert a sequence of tokens into a binary numpy vector
def vectorize(sentence):
    global vocab
    vector = np.zeros([len(vocab), 1])
    for i in range(len(vocab)):
        # if word is present in sentence
        if vocab[i] in sentence:
            # mark a 1 in the vector
            vector[i] = 1
    return np.transpose(vector)

# convert plaintext files of sample queryies into list of numerical vectors
# so that the neural network can understand them
def parse_files():
    files = [#'./corpus/weather.txt',
             './corpus/trending.txt',
             './corpus/headlines.txt', './corpus/gibberish.txt']
    corpora = []
    for f in files:
        corpora.append(tokenize_file(f))

    # make list of all words in training set
    words = all_words(corpora)
    # remove duplicates and convert to list
    global vocab
    vocab = list(set(words))

    global dataset
    dataset = []
    # iterate over each file
    for i in range(len(corpora)):
        # create desired output vector based on where
        # the sentence came from
        v = np.zeros([len(corpora), 1])
        v[i] = 1
        # convert every sentence into an array of 1's and 0's
        for sentence in corpora[i]:
            # put ordered pair [input, output] in dataset list
            dataset.append([vectorize(sentence),
                            np.transpose(v)])
    global epoch
    epoch = 0
    # return length input layer, length of output layer, and number of
    # training samples
    return len(vocab), len(corpora), len(dataset)

# Partition datset into training and test sets
# Since this dataset is small, we can use Leave One Out cross-validation
def make_sets():
    global epoch
    # make the test set of one input pattern
    global test_set

    # pick the next sample from main dataset
    test_set = copy.deepcopy(dataset[epoch])

    # make the training set of all other patterns
    global train_set
    train_set = []

    for j in range(len(dataset)):
        if epoch is not j:
            train_set.append(copy.deepcopy(dataset[j]))
    epoch = epoch + 1

def full_set():
    global train_set
    train_set = dataset

# return the next pattern to feed forward and train the neural network
def next_stimulus():
    i = random.randrange(len(train_set))
    return train_set[i]

def test_stimulus():
    return test_set
