import numpy as np
import sklearn
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt


def graph_reconstruction(X, delta = 10, max_components = 1000, print_progress = False):

	frobenii = [np.linalg.norm(X.toarray())]
	rnge = [i for i in range(delta, max_components + delta, delta)]

	for i in rnge:
		dim_reducer = TruncatedSVD(n_components=i)
		shrunk = dim_reducer.fit_transform(X)
		X_reconst = dim_reducer.inverse_transform(shrunk)
		frobenii.append(np.linalg.norm(X - X_reconst))
		if (print_progress):
			print("{}% complete".format(100 * i / max_components))

	components = [0]
	components.extend(rnge)

	plt.plot(components, frobenii)
	plt.show()
	return components, frobenii


def graph_eigenvalues(X):
	X = X.toarray()
	_, s, _ = np.linalg.svd(X)
	# values, _ = np.linalg.eig(X.T @ X)
	values = s
	# values.sort()
	plt.plot(values)
	plt.show()
	# plt.plot(values[::-1])
	# plt.show()
	return values