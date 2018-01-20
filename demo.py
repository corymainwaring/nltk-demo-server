import csv
import json

import nltk
from nltk import sentiment

from flask import Flask
from flask import request
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api/polarity', methods=['POST'])
def polarity():
    '''
    Returns the semantic polarity

    Parameters
    ----------------
    content: str

    Returns
    ----------
    float: A float in [-1, 1], where -1.0 is the most negative polarity, 1.0 is
    the most positive polarity
    '''

    data = request.json
    content = data['content']

    analyzer = sentiment.vader.SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(content)
    with open('sentiments.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([content, scores['compound']])
    return json.dumps(scores)

@app.route('/polarity')
def form():
    with open('form.html', 'r') as f:
        return f.read()

@app.route('/api/pos', methods=['POST'])
def tag_part_of_speech():
    '''
    Utility function to perform part of speech tagging on input.

    Parameters
    ----------
    content: a string or a list of strings
        The content to perform part of speech tagging on.

    Returns
    -------
    list
        A list of of 2-tuples representing word, part of speech tag pairs.
    '''

    data = request.json
    content = data['content']

    if isinstance(content, str):
        content = nltk.tokenize.word_tokenize(content)
    return json.dumps(nltk.tag.pos_tag(content))


def chunk_pos_tagged_tokens(tagged_tokens):
    '''
    Performs named entity chunking on part of speech tagged tokens.


    Parameters
    ----------
    tagged_tokens: list of 2-tuples of strings where the 2-tuples are string,
    part of speech pairs The tokens to perform named entity chunking on.

    Returns
    -------
    nltk.tree.Tree
        A Tree representing the chunked content.
    '''

    return nltk.chunk.ne_chunk(tagged_tokens)


def chunk(content):
    '''
    Performs named entity chunking on input.

    Parameters
    ----------
    input : a string or a list of 2-tuples representing word, part of speech tag
        pairs. The content to perform named entity chunking on.

    Returns
    -------
    nltk.tree.Tree
        A Tree representing the chunked content.
    '''

    if isinstance(content, str):
        tokenized = nltk.tokenize.word_tokenize(content)
        pos_tagged = tag_part_of_speech(tokenized)
        content = pos_tagged
    return chunk_pos_tagged_tokens(content)


def get_named_entities_from_chunks(chunks):
    '''
    Returns a list of named entity, type of entity pairs.

    Parameter
    ---------
    chunks : nltk.tree.Tree
        A Tree representing named entity chunks.

    Returns
    -------
    list
        A list of of 2-tuples representing named entity, type of entity pairs.
    '''

    return [
        (
            ' '.join(leaf for leaf, _ in chunk.leaves()), chunk.label()
        ) for chunk in chunks if isinstance(chunk, nltk.tree.Tree)
    ]

def get_named_entities(content):
    '''
    Returns a list of named entity, type of entity pairs from input.

    Parameters
    ---------
    content: a string or nltk.tree.Tree representing named entity chunks
        The content.

    Returns
    -------
    list
        A list of of 2-tuples representing named entity, type of entity pairs.
    '''

    if isinstance(content, str):
        chunked = chunk(content)
        content = chunked
    return get_named_entities_from_chunks(content)


