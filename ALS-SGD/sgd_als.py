import numpy as np
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error

def get_mse(pred, actual):
	# Ignore nonzero terms.
	pred = pred[actual.nonzero()].flatten()
	actual = actual[actual.nonzero()].flatten()
	return mean_squared_error(pred, actual)

class Matrix_Factorization():
	def __init__(self, 
				 ratings,
				 n_factors=40,
				 learning='sgd',
				 item_fact_reg=0.0, 
				 user_fact_reg=0.0,
				 item_bias_reg=0.0,
				 user_bias_reg=0.0,
				 verbose=False):

		self.ratings = ratings
		self.n_users, self.n_items = ratings.shape
		self.n_factors = n_factors
		self.item_fact_reg = item_fact_reg
		self.user_fact_reg = user_fact_reg
		self.item_bias_reg = item_bias_reg
		self.user_bias_reg = user_bias_reg
		self.learning = learning
		if self.learning == 'sgd':
			self.sample_row, self.sample_col = self.ratings.nonzero()
			self.n_samples = len(self.sample_row)
		self._v = verbose

	def als_step(self,
				 latent_vectors,
				 fixed_vecs,
				 ratings,
				 _lambda,
				 type='user'):
		if type == 'user':
			# Precompute
			YTY = fixed_vecs.T.dot(fixed_vecs)
			lambdaI = np.eye(YTY.shape[0]) * _lambda

			for u in xrange(latent_vectors.shape[0]):
				latent_vectors[u, :] = solve((YTY + lambdaI), 
											 ratings[u, :].dot(fixed_vecs))
		elif type == 'item':
			# Precompute
			XTX = fixed_vecs.T.dot(fixed_vecs)
			lambdaI = np.eye(XTX.shape[0]) * _lambda
			
			for i in xrange(latent_vectors.shape[0]):
				latent_vectors[i, :] = solve((XTX + lambdaI), 
											 ratings[:, i].T.dot(fixed_vecs))
		return latent_vectors

	def train(self, n_iter=10, learning_rate=0.1):      
		self.user_vecs = np.random.normal(scale=1./self.n_factors,\
										  size=(self.n_users, self.n_factors))
		self.item_vecs = np.random.normal(scale=1./self.n_factors,
										  size=(self.n_items, self.n_factors))
		
		if self.learning == 'als':
			self.partial_train(n_iter)
		elif self.learning == 'sgd':
			self.learning_rate = learning_rate
			self.user_bias = np.zeros(self.n_users)
			self.item_bias = np.zeros(self.n_items)
			self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
			self.partial_train(n_iter)
	
	
	def partial_train(self, n_iter):
		ctr = 1
		while ctr <= n_iter:
			if ctr % 10 == 0 and self._v:
				print '\tcurrent iteration: {}'.format(ctr)
			if self.learning == 'als':
				self.user_vecs = self.als_step(self.user_vecs, 
											   self.item_vecs, 
											   self.ratings, 
											   self.user_fact_reg, 
											   type='user')
				self.item_vecs = self.als_step(self.item_vecs, 
											   self.user_vecs, 
											   self.ratings, 
											   self.item_fact_reg, 
											   type='item')
			elif self.learning == 'sgd':
				self.training_indices = np.arange(self.n_samples)
				np.random.shuffle(self.training_indices)
				self.sgd()
			ctr += 1

	def sgd(self):
		for idx in self.training_indices:
			u = self.sample_row[idx]
			i = self.sample_col[idx]
			prediction = self.predict(u, i)
			e = (self.ratings[u,i] - prediction) # error
			
			# Update biases
			self.user_bias[u] += self.learning_rate * \
								(e - self.user_bias_reg * self.user_bias[u])
			self.item_bias[i] += self.learning_rate * \
								(e - self.item_bias_reg * self.item_bias[i])
			
			#Update latent factors
			self.user_vecs[u, :] += self.learning_rate * \
									(e * self.item_vecs[i, :] - \
									 self.user_fact_reg * self.user_vecs[u,:])
			self.item_vecs[i, :] += self.learning_rate * \
									(e * self.user_vecs[u, :] - \
									 self.item_fact_reg * self.item_vecs[i,:])
	def predict(self, u, i):
		if self.learning == 'als':
			return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
		elif self.learning == 'sgd':
			prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
			prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
			return prediction
	
	def predict_all(self):
		predictions = np.zeros((self.user_vecs.shape[0], 
								self.item_vecs.shape[0]))
		for u in xrange(self.user_vecs.shape[0]):
			for i in xrange(self.item_vecs.shape[0]):
				predictions[u, i] = self.predict(u, i)
				
		return predictions
	
	def calculate_learning_curve(self, iter_array, test, learning_rate=0.1):
		iter_array.sort()
		self.train_mse =[]
		self.test_mse = []
		iter_diff = 0
		for (i, n_iter) in enumerate(iter_array):
			if self._v:
				print 'Iteration: {}'.format(n_iter)
			if i == 0:
				self.train(n_iter - iter_diff, learning_rate)
			else:
				self.partial_train(n_iter - iter_diff)

			predictions = self.predict_all()

			self.train_mse += [get_mse(predictions, self.ratings)]
			self.test_mse += [get_mse(predictions, test)]
			if self._v:
				print 'Train mse: ' + str(self.train_mse[-1])
				print 'Test mse: ' + str(self.test_mse[-1])
			iter_diff = n_iter