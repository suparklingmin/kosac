from setuptools import setup, find_namespace_packages

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
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "kosac": [],
        "kosac.data": [],
        "kosac.data.lexicon": ["*.csv"],
        "kosac.data.corpora": ["*.csv"],
        "kosac.data.tagger": [".txt"],
    }
)