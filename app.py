import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import bs4 as bs
import spacy
import time

#TMDB
from tmdbv3api import TMDb
from tmdbv3api import Movie

tmdb = TMDb()
tmdb.api_key = 'd05215f55a79e472a5b0d00d1ec80d70'
tmdb_movie = Movie()

def create_similarity():
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data,similarity

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        # similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l

def get_response(movie):
	return {
	'Movie name': movie,
	'Director': list(data[data['movie_title'] == movie]['director_name'])[0], 
	'Genere': list(data[data['movie_title'] == movie]['genres'])[0], 
	'Actors': [list(data[data['movie_title'] == movie]['actor_1_name'])[0], list(data[data['movie_title'] == movie]['actor_2_name'])[0], list(data[data['movie_title'] == movie]['actor_3_name'])[0]]
	}

def review_sentiment_analysis(input_data : str):
    load_model = spacy.load("model_artifacts")
    parsed_text = load_model(input_data)
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = 1
        score = parsed_text.cats["pos"]
    else:
        prediction = -1
        score = parsed_text.cats["neg"]
    return {'review': input_data , 'predicted_sentiment' :prediction ,'predicted_score': score}


def review_analysis(imdb_id: str):
    sauce = requests.get('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).content
    soup = bs.BeautifulSoup(sauce,'lxml')
    soup_result = soup.find_all("div",{"class":"text show-more__control"})
    review_list=list()
    for reviews in soup_result:
        if reviews.string:
            review_list.append(review_sentiment_analysis(str(reviews.string)))
    return review_list



def recommendation_workflow(input_movie):
    response = {
    'Requested Movie': get_response(input_movie), 
    'Recommendations': [],
    }

    rec_movie = rcmd(input_movie)
    for movie in rec_movie:
        result = tmdb_movie.search(movie.upper())
        movie_id, movie_name = result[0].id, result[0].title
        res = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
        data_json = res.json()
        posters_ui = {'poster': 'https://image.tmdb.org/t/p/original{}'.format(data_json['poster_path']), 'movie_name': data_json['original_title'], 'overview': data_json['overview'], 'official_page': data_json['homepage']}
        response['Recommendations'].append(posters_ui)

    return response

def sentiment_workflow(input_movie):
    result = tmdb_movie.search(input_movie.upper())

    movie_id, movie_name = result[0].id, result[0].title

    res = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
    data_json = res.json()
    imdb_id = data_json['imdb_id']
    poster = data_json['poster_path']
    movie_name = data_json['original_title']
    overview = data_json['overview']
    official_page = data_json['homepage']

    response = {
    'Sentiment': review_analysis(imdb_id),
    'Poster': 'https://image.tmdb.org/t/p/original{}'.format(poster),
    'Movie': movie_name,
    'Description': overview,
    'OfficialPage': official_page
    }

    return response

data = pd.read_csv('processed_data.csv')

def get_suggestions():
    data = pd.read_csv('processed_data.csv')
    return list(data['movie_title'].str.capitalize())


app = Flask(__name__)

@app.route("/", methods=['GET'])
@app.route("/home", methods=['GET'])
def home():
    if request.method == "GET":
        flims = get_suggestions()
        return render_template('home.html', flims = flims)

@app.route("/display", methods=['GET', 'POST'])
def display():
    if request.method == "POST":
        input_movie = request.form.get("movie").lower()
        rec_res = recommendation_workflow(input_movie)
        sent_res = sentiment_workflow(input_movie)
        return render_template('display.html', sent_res = sent_res, rec_res = rec_res, length_sen = len(sent_res['Sentiment']), length_rec = len(rec_res['Recommendations']))
    else:
        return '404 Error'

@app.route("/recommend", methods=['GET'])
def recommend():
    input_movie = request.args.get('name')
    return jsonify(recommendation_workflow(input_movie))

@app.route("/sentiment", methods=['GET'])
def sentiment():
    input_movie = request.args.get('name')
    return jsonify(sentiment_workflow(input_movie))
    


if __name__ == "__main__":
	app.run(debug=True)
