from setuptools import setup, find_packages

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
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"kosac.data.lexicon": ["*.csv"]},
    include_package_data=True
)