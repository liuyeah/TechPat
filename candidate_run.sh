#!/usr/bin/env bash
RAW_TITLE='example_data/example_title/title.txt'
RAW_ABSTRACT='example_data/example_abstract/abstract.txt'
RAW_CLAIM='example_data/example_claim/claim.txt'

export TOTAL_NUMBER=20

#install dbpedia according to https://github.com/dbpedia-spotlight/spotlight-docker and change this
export DBPEDIA_PATH="$HOME/bin/spotlight-english"
docker run -i -p 2222:80 dbpedia/spotlight-english spotlight.sh

# Out-of-the-box: download best-matching default model and create shortcut link
python -m spacy download en
python -m spacy download en_core_web_sm

#install dbpedia spotlight at   https://github.com/dbpedia-spotlight/spotlight-docker

python candidate_generation/nltk_extract.py $RAW_TITLE
python candidate_generation/spacy_extract.py $RAW_TITLE
python candidate_generation/dbpedia_extract.py $RAW_TITLE
python candidate_generation/candidate_merge.py $RAW_TITLE

python candidate_generation/nltk_extract.py $RAW_ABSTRACT
python candidate_generation/spacy_extract.py $RAW_ABSTRACT
python candidate_generation/dbpedia_extract.py $RAW_ABSTRACT
python candidate_generation/candidate_merge.py $RAW_ABSTRACT

python candidate_generation/nltk_extract.py $RAW_CLAIM
python candidate_generation/spacy_extract.py $RAW_CLAIM
python candidate_generation/dbpedia_extract.py $RAW_CLAIM
python candidate_generation/candidate_merge.py $RAW_CLAIM