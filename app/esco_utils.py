import json
import os
import re
import glob
import uuid
import pickle
import tqdm 
import datetime

import pandas as pd
import numpy as np
import psycopg2 as pg
import pandas.io.sql as psql
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer


def load_esco_DB():
    # load ISCO codes from crawled JSON lines files 
    ISCO_codes = []
    with open('data/crawled/ISCO_codes.jl') as f:
        for line in f:
            ISCO_codes.append(json.loads(line))
    ISCO_codes = [(k,v) for pair in ISCO_codes for k,v in pair.items()]
    ISCO_codes = pd.DataFrame(ISCO_codes,columns=['external_id','ISCO_code'])
    # load taxonommy data and merge it with ISCO 
    connection = pg.connect(user="postgres", host="db", password="example", database="mydb2")
    taxonomy = psql.read_sql('SELECT id,external_id FROM occupations', connection)
    taxonomy = taxonomy.merge(ISCO_codes,how='left',on='external_id')
    
    # load occupation & merge it with occupation translations & taxonomy
    occ_trans = psql.read_sql('SELECT occupation_id,title,description,alternates,locale FROM occupation_translations', connection)
    occ_trans = occ_trans.merge(taxonomy, how='left',left_on='occupation_id',right_on='id',suffixes=('','_y'))
    occ_trans = occ_trans[~occ_trans.ISCO_code.isna()]
    
    # load skills & merge it with skills translations 
    connection = pg.connect(user="postgres", host="db", password="example", database="mydb1",)    
    skills = psql.read_sql('SELECT id,skill_type_id,external_id FROM skills', connection)
    skill_trans = psql.read_sql('SELECT * FROM skill_translations', connection)
    skill_trans = skill_trans.merge(skills, how='left',left_on='skill_id',right_on='id')
    occ_skills = psql.read_sql('SELECT * FROM occupations_skills', connection)
    
    # store CSV files
    occ_skills.to_csv('data/esco/occ_skills.csv',encoding='utf-8')
    taxonomy.to_csv('data/esco/taxonomy.csv',encoding='utf-8')
    occ_trans.to_csv('data/esco/occupations.csv',encoding='utf-8')
    skill_trans.to_csv('data/esco/skill_trans.csv', encoding='utf-8')  


def load_vacancies(files):
    # otherwise, merge Azuna vacancies into a single CSV file 
    files = glob.glob(files)
    vacancies = []
    for file in files:
        try:
            data = pd.read_csv(file)
            country = file.split('&')[1].split('=')[1]
            data['country'] = country
            if len(vacancies)>0:
                vacancies = vacancies.append(data)
            else:
                vacancies = data
        except Exception:
            pass
    columns = [col for col in vacancies.columns if ('alternative' in col)]
    # extract list from stringified (eg. '[a,b,c]') format
    vacancies[columns] = vacancies[columns].applymap(lambda x: x[2:-2].replace("'","").split(", "))
    # extract word counts 
    vacancies['word_count'] = vacancies.job_description.apply(lambda x: len(x.split()))
    return vacancies


def occ_alt_stringify(alt,title_imp=1):
    alternates = ""
    if isinstance(alt,str):
        alternates = alt[1:-2].replace("'","") * title_imp if alt else ""
    elif isinstance(alt,list):
        alternates = ', '.join(alt*title_imp) if alt else ' '
    return alternates

        
def occ_stringify(occ, title_imp=1,alt_title_imp=1,case_insensitive=False):
    title = ' '.join([occ.title]*title_imp)
    alternates = occ_alt_stringify(occ.alternates,alt_title_imp)
    desc = '\n'.join([title, alternates, occ.description])
    if case_insensitive:
        desc = desc.lower()
    return desc


def job_stringify(job,title_imp=1, case_insensitive=False):
    title = ' '.join([job.job_title]*title_imp)
    desc = '\n'.join([title,job.job_description])
    if case_insensitive:
        desc = desc.lower()
    return desc


def train_tfidf_knn(model_name,lang,ngram_min=1,ngram_max=4,n_neighbors = 5, title_imp=5,alt_title_imp=5,case_insensitive=True):
    assert(model_name=='tfidf_knn')
    occ_trans = pd.read_csv('data/esco/occupations.csv')
    occ_local = occ_trans[occ_trans.locale==lang]
    strings = [occ_stringify(occ,title_imp,case_insensitive) for occ in occ_local.itertuples()]
    vectorizer = TfidfVectorizer(ngram_range=(ngram_min,ngram_max))
    X = vectorizer.fit_transform(strings)
    knn_index = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    feature_names = vectorizer.get_feature_names_out()
    
    meta = dict(model_name=model_name,
                lang=lang,
                ngram_min=ngram_min,
                ngram_max=ngram_max,
                n_neighbors=n_neighbors,
                title_imp=title_imp,
                alt_title_imp=alt_title_imp,
                case_insensitive=case_insensitive)
    
    model = dict(meta=meta,
                 occupation_id=occ_local.occupation_id.tolist(), 
                 vectorizer=vectorizer, 
                 feature_names=feature_names, 
                 knn_index=knn_index)
    return model, meta


def predict_top_tags(model, text):
    occ_id,vectorizer,knn_index = model['occupation_id'], model['vectorizer'], model['knn_index'] 
    X = vectorizer.transform([text])
    distances, indices = model['knn_index'].kneighbors(X)
    occ_indices = [int(occ_id[i]) for i in indices[0]]
    confidence = [float(d) for d in distances[0]]
    return confidence, occ_indices
    
    
