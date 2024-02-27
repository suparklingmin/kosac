from setuptools import setup

setup(
    name='kosac',
    version='0.0.1',
    description='KOSAC(KOrean Sentiment Analysis Corpus) Lexicon',
    author='Sumin Park',
    author_email='mam3b@snu.ac.kr',
    url='https://github.com/suparklingmin/kosac',
    install_requires=[
        'pandas',
        'konlpy',
        'nltk',
        'transformers',
    ],
    data_files=[
        (
            'data/lexicon', ['kosac/data/lexicon/polarity.csv']
        )
    ],
)