import pandas as pd
import numpy as np

movies_df = pd.read_table('ml-1m/movies.dat', header=None, sep='::', names=['movie_id', 'movie_title', 'movie_genre'], engine='python')
ratings_df = pd.read_table('ml-1m/ratings.dat', header=None, sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')
del ratings_df['timestamp']
ratings_df = pd.merge(ratings_df, movies_df, on='movie_id')[['user_id', 'movie_title', 'movie_id','rating']]
matrix = ratings_df.pivot_table(values='rating', index='user_id', columns='movie_title')  
matrix.fillna(0, inplace=True)
corr_matrix = np.corrcoef(matrix.T)
movie_index = matrix.columns

def get_movie_similarity(movie_title):
	movie_idx = list(movie_index).index(movie_title)
	return corr_matrix[movie_idx]

def get_movie_recommendations(user_movies):
	movie_similarities = np.zeros(corr_matrix.shape[0])
	for movie_name in user_movies:
		movie_similarities = movie_similarities + get_movie_similarity(movie_name)
	similarities_df = pd.DataFrame({
		'movie_title': movie_index,
		'sum_similarity': movie_similarities
		})
	similarities_df = similarities_df[~(similarities_df.movie_title.isin(user_movies))]
	similarities_df = similarities_df.sort_values(by=['sum_similarity'], ascending=False)
	return similarities_df

sample_user = 10
print "Movies rated by user_id = 10: "
print ratings_df[ratings_df.user_id==sample_user].sort_values(by=['rating'], ascending=False)[['movie_title']][:10]
sample_user_movies = ratings_df[ratings_df.user_id==sample_user].movie_title.tolist()  
recommendations = get_movie_recommendations(sample_user_movies)
print "Movies recommended to user_id = 10: "
print recommendations.movie_title.head(10)