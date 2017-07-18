from sgd_als import Matrix_Factorization
import time
import numpy as np
import pandas as pd
np.random.seed(0)
from numpy.linalg import solve
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Split into training and test sets. 
# Remove 10 ratings for each user 
# and assign them to the test set
def train_test_split(ratings):
	test = np.zeros(ratings.shape)
	train = ratings.copy()
	for user in xrange(ratings.shape[0]):
		test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
										size=10, 
										replace=False)
		train[user, test_ratings] = 0.
		test[user, test_ratings] = ratings[user, test_ratings]
		
	# Test and training are truly disjoint
	assert(np.all((train * test) == 0)) 
	return train, test

def plot_learning_curve(iter_array, model):
	plt.plot(iter_array, model.train_mse, \
			 label='Training', linewidth=5)
	plt.plot(iter_array, model.test_mse, \
			 label='Test', linewidth=5)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.xlabel('iterations', fontsize=30)
	plt.ylabel('MSE', fontsize=30)
	plt.legend(loc='best', fontsize=20)
	plt.show()

def cosine_similarity(model):
	sim = model.item_vecs.dot(model.item_vecs.T)
	norms = np.array([np.sqrt(np.diagonal(sim))])
	return sim / norms / norms.T

def best_als(train):
	latent_factors = [5, 10, 20, 40, 80]
	regularizations = [0.01, 0.1, 1., 10., 100.]
	regularizations.sort()
	iter_array = [1, 2, 5, 10, 25, 50, 100]

	best_params = {}
	best_params['n_factors'] = latent_factors[0]
	best_params['reg'] = regularizations[0]
	best_params['n_iter'] = 0
	best_params['train_mse'] = np.inf
	best_params['test_mse'] = np.inf
	best_params['model'] = None

	for fact in latent_factors:
		print time.strftime("%c")
		print 'Using Factor: {}'.format(fact)
		for reg in regularizations:
			print 'Using Regularization: {}'.format(reg)
			MF_ALS = Matrix_Factorization(train, n_factors=fact, learning='als',\
								user_fact_reg=reg, item_fact_reg=reg)
			MF_ALS.calculate_learning_curve(iter_array, test)
			min_idx = np.argmin(MF_ALS.test_mse)
			if MF_ALS.test_mse[min_idx] < best_params['test_mse']:
				best_params['n_factors'] = fact
				best_params['reg'] = reg
				best_params['n_iter'] = iter_array[min_idx]
				best_params['train_mse'] = MF_ALS.train_mse[min_idx]
				best_params['test_mse'] = MF_ALS.test_mse[min_idx]
				best_params['model'] = MF_ALS
				print 'New optimal hyperparameters found:'
				print pd.Series(best_params)
		print "-------------------------------"
	best_als_model = best_params['model']
	plot_learning_curve(iter_array, best_als_model)
	return best_als_model

def best_sgd(train):
	iter_array = [1, 2, 5, 10, 25, 50, 100, 200]
	latent_factors = [5, 10, 20, 40, 80]
	regularizations = [0.001, 0.01, 0.1, 1.]
	regularizations.sort()

	best_params = {}
	best_params['n_factors'] = latent_factors[0]
	best_params['reg'] = regularizations[0]
	best_params['n_iter'] = 0
	best_params['train_mse'] = np.inf
	best_params['test_mse'] = np.inf
	best_params['model'] = None

	for fact in latent_factors:
		print time.strftime("%c")
		print 'Using Factor: {}'.format(fact)
		for reg in regularizations:
			print 'Using Regularization: {}'.format(reg)
			MF_SGD = Matrix_Factorization(train, n_factors=fact, learning='sgd',\
								user_fact_reg=reg, item_fact_reg=reg, \
								user_bias_reg=reg, item_bias_reg=reg)
			MF_SGD.calculate_learning_curve(iter_array, test, learning_rate=0.001)
			min_idx = np.argmin(MF_SGD.test_mse)
			if MF_SGD.test_mse[min_idx] < best_params['test_mse']:
				best_params['n_factors'] = fact
				best_params['reg'] = reg
				best_params['n_iter'] = iter_array[min_idx]
				best_params['train_mse'] = MF_SGD.train_mse[min_idx]
				best_params['test_mse'] = MF_SGD.test_mse[min_idx]
				best_params['model'] = MF_SGD
				print 'New optimal hyperparameters'
				print pd.Series(best_params)
		print "-------------------------------"
	best_sgd_model = best_params['model']
	plot_learning_curve(iter_array, best_params['model'])
	return best_sgd_model

if __name__=="__main__":
	# Load data from disk
	names = ['user_id', 'item_id', 'rating', 'timestamp']
	df = pd.read_csv('../ml-100k/u.data', sep='\t', names=names)
	n_users = df.user_id.unique().shape[0]
	n_items = df.item_id.unique().shape[0]
	# Create ratings matrix
	ratings = np.zeros((n_users, n_items))
	for row in df.itertuples():
		ratings[row[1]-1, row[2]-1] = row[3]

	train, test = train_test_split(ratings)

	print "Finding best hyperparameters for ALS."
	best_als_model = best_als(train)
	best_als_model.train(50)
	als_sim = cosine_similarity(best_als_model)
	np.save('als_sim', als_sim)

	print "####################################"
	
	print "Finding best hyperparameters for SGD."
	best_sgd_model = best_sgd(train)
	best_sgd_model.train(200, learning_rate=0.001)
	sgd_sim = cosine_similarity(best_sgd_model)
	np.save('sgd_sim', sgd_sim)