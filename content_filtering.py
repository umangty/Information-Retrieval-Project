import pandas as pd
import numpy as np
from collections import OrderedDict

def load_data():
	df = pd.read_table('ml-1m/movies.dat', header=None, sep='::', names=['movie_id', 'movie_title', 'movie_genre'], engine='python')
	df = pd.concat([df, df.movie_genre.str.get_dummies(sep='|')], axis=1)
	return df

def make_new_user(categories):
	user = OrderedDict(zip(categories, []))
	print "Preference out of 5 for following genres: "
	for i in range(len(categories)):
		print categories[i]
		user[categories[i]] = int(raw_input())
	return user

def get_movie_score(movie_features, user):
	user_list = list(user)
	return np.dot(movie_features, user_list)

def recommend(user, df, n_rec):
	df['score'] = df[categories].apply(get_movie_score, 
													 args=([user.values()]),
													 axis=1)
	return df.sort_values(by=['score'], ascending=False)[:n_rec]

if __name__ == "__main__":
	df = load_data()
	categories = df.columns[3:]
	user = make_new_user(categories)
	recs = recommend(user, df, 5)
	columns = ['movie_title', 'movie_genre']
	print "Top 5 recommended movies: "
	print recs[columns]