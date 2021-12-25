
import json
import sys

from pathlib import Path

from flask import Response
from flask import Flask
from flask import render_template
from flask import session
from flask import request
from flask import abort, redirect, url_for
from flask import current_app, g

from esco_utils import * 

def read_template(name):
    occ_template = []
    with open(f'templates/{name}.html','r') as f:
        occ_template = ''.join([l for l in f.readlines()])
    return occ_template
   
def create_app(test_config=None):
    global occupations, vacancies
    
    app = Flask(__name__)
    
    # model name to function map 
    modelname2func = {'tfidf_knn': train_tfidf_knn}
    Path("data/vacancies").mkdir(parents=True, exist_ok=True)
    Path("data/esco").mkdir(parents=True, exist_ok=True)
    
    # check if vacancies are already merged     
    vacancies_path = 'data/vacancies/all.csv'
    if os.path.isfile(vacancies_path):
        vacancies = pd.read_csv(vacancies_path)
    else:
        vacancies = load_vacancies('data/binary/*.csv')
        vacancies = vacancies[~vacancies.isna()]
        vacancies.to_csv(vacancies_path)
    
    if ~os.path.isfile('data/vacancies/all.csv'):
        load_esco_DB()
    occupations = pd.read_csv('data/esco/occupations.csv')

    
    @app.route('/esco')
    def index():
        if 'username' in session:
            return f'Logged in as {session["username"]}'
        return 'You are not logged in'
        

    @app.route('/review',methods=['GET'])
    def view():
        if 'country' not in request.args.keys():
            country = 'GB'
        else:
            country = request.args.get('country')
        
        job = vacancies[vacancies.country==country].sample().iloc[0]
        template_task = read_template('task')
        
        return template_task
    
    
    @app.route('/sample-vacancy',methods=['GET'])
    def sample_vacancy():
        country = request.args.get('country')
        job = vacancies[vacancies.country==country].sample().iloc[0]
        return Response(job.to_json(), mimetype='application/json') 
    
    
    @app.route('/all-tags', methods=['GET'])
    def get_all_tags():
        lang = request.args.get('lang')
        occupation_local = occupations[occupations.locale==lang]
        idx = occupation_local.occupation_id.duplicated(keep='last')
        occupation_local = occupation_local[~idx]
        return Response(occupation_local.to_json(orient="table"), mimetype='application/json') 
    
        
    @app.route('/get-occupation', methods=['GET'])
    def get_occupation():
        id = request.args.get('id')
        lang = request.args.get('lang')
        occupation_local = occupations[occupations.locale==lang]
        idx = occupation_local.occupation_id.duplicated(keep='last')
        occupation_local = occupation_local[~idx]
        occ = occupation_local[occupation_local.occupation_id==int(id)].iloc[0]
        return Response(occ.to_json(), mimetype='application/json') 
        
        
    @app.route('/all-models', methods=['GET'])
    def get_all_models():
        df = pd.read_json('models/log.jl',lines=True)
        return Response(df.set_index('id').to_json(orient="table"), mimetype='application/json')
    
    
    @app.route('/train', methods=['POST'])
    def train_model():
        id = str(uuid.uuid1())
        response = dict(id=id)
        data = request.get_json()
        train_func = modelname2func[data['model_name']]
        model, params = train_func(**data)
        response.update(params)
        with open(f'models/{id}.pk','wb') as f:
            pickle.dump(model, f)
        with open('models/log.jl', 'a') as f:
            f.write('\n'+json.dumps(response))   
        return Response(json.dumps(response), mimetype='application/json')
    
    
    @app.route('/top-tags', methods=['POST'])
    def predict():
        data = request.json
        text, id = data['description'], data['id']
        title = data['title'] if ('title' in data.keys()) else ''
        app.logger.info(id);
        app.logger.info(text);
        app.logger.info(title);
        model = pickle.load(open(f'models/{id}.pk','rb'))
        title = title*model['meta']['title_imp']
        distances, indices = predict_top_tags(model, text)
        response = [{'index': i, 'distance': d} for i,d in zip(indices,distances)] 
        return Response(json.dumps(response), mimetype='application/json')

    return app
