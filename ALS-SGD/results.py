import numpy as np
import sys

als_similarity = np.load('als_sim.npy')
sgd_similarity = np.load('sgd_sim.npy')

idx_to_movie = {}
with open('../ml-100k/u.item', 'r') as f:
    for line in f.readlines():
        info = line.split('|')
        idx_to_movie[int(info[0])-1] = info[1]

def display_top_k_movies(similarity, mapper, movie_idx, k=5):
    movie_indices = np.argsort(similarity[movie_idx,:])[::-1]
    for i in xrange(1, k+1):
        print mapper[movie_indices[i]]

if len(sys.argv) == 1 or len(sys.argv) == 2:
	print "Movie number and number of recommendations missing."
else:
	movie_idx = int(sys.argv[1])
	num_of_reco = int(sys.argv[2])
	print "Input movie: ", idx_to_movie[movie_idx]
	print "ALS reco: "
	display_top_k_movies(als_similarity, idx_to_movie, movie_idx, num_of_reco)
	print "--------"
	print "SGD reco:"
	display_top_k_movies(sgd_similarity, idx_to_movie, movie_idx, num_of_reco)
# for name in idx_to_movie:
# 	print name,idx_to_movie[name]