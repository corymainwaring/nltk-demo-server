"""
Helper to assist with downloading of required NLTK data.
"""

import nltk


''' Required NLTK dependencies for demo.'''
NLTK_DEPENDENCIES = [
    'stopwords', # not strictly required, but may be necessary for demo
    'vader_lexicon', # required for polarity
    'punkt', # required for tokenization
    'averaged_perceptron_tagger', # required for part of speech tagging
    'maxent_ne_chunker', # required for named entity chunking
    'words' # required for named entity chunking
]

def get_required_nltk_dependencies():
    '''
    Return a list of required NLTK data dependencies, by index.

    See: http://www.nltk.org/nltk_data/
    '''

    return NLTK_DEPENDENCIES

def download_required_nltk_dependencies():
    '''
    Download required NLTK dependencies for sosa_utils.
    '''

    for dependency in NLTK_DEPENDENCIES:
        nltk.download(dependency)

def main():

    print('Installing NLTK data dependencies.')
    download_required_nltk_dependencies()


if __name__ == '__main__':
    main()
