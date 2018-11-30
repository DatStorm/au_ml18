import numpy as np
import matplotlib.pyplot as plt
import imageio
import collections
import os
# Load the Iris data set
import sklearn.datasets
from scipy.stats import multivariate_normal


def lloyds_algorithm(X, k, T):
    """ Clusters the data of X into k clusters using T iterations of Lloyd's algorithm.

        Parameters
        ----------
        X : Data matrix of shape (n, d)
        k : Number of clusters.
        T : Maximum number of iterations to run Lloyd's algorithm.

        Returns
        -------
        clustering: A vector of shape (n, ) where the i'th entry holds the cluster of X[i].
        centroids:  The centroids/average points of each cluster.
        cost:       The cost of the clustering
    """
    n, d = X.shape

    # Initialize clusters random.
    clustering = np.random.randint(0, k, (n,))
    centroids = np.zeros((k, d))
    # print(clustering)
    # Used to stop if cost isn't improving (decreasing)
    cost = 0
    oldcost = 0

    # Column names
    print("lloyds_algorithm\nIterations\tCost")

    for i in range(T):
        # Update centroid
        # >> import collections, numpy
        # >>> a = numpy.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
        # >>> collections.Counter(a)
        # Counter({0: 7, 1: 4, 3: 2, 2: 1, 4: 1})
        # YOUR CODE HERE
        centroids = np.zeros((k, d))
        counter = collections.Counter(clustering)

        # Sum points in each cluster x \in C_i
        for point in range(n):
            cluster = clustering[point]
            centroids[cluster] += X[point]

        # Get mean point in cluster
        for cluster in range(k):
            if counter[cluster] == 0:  # We did not found any point here. So continue
                # print(f"you fucked up mark!!! {counter}")
                continue
            centroids[cluster] /= counter[cluster]
            # END CODE

        # Update clustering

        # YOUR CODE HERE
        # If the point  x_i  should be in cluster  j  we have that clustering[i]=j
        # Update clustering: Assign  x_i  to cluster  C_j  where
        # j=argmin||x−μ_j||^2  for  i=1,...,n
        for ii in range(n):
            j = np.argmin((np.linalg.norm(X[ii, :] - centroids, axis=1) ** 2))
            clustering[ii] = j
        # END CODE

        # Compute and print cost
        # cost = np.sum((X - centroids[clustering])**2)
        cost = 0
        for j in range(n):
            cost += np.linalg.norm(X[j] - centroids[clustering[j]]) ** 2
        print(i + 1, "\t\t", cost)

        # Stop if cost didn't improve more than epislon (decrease)
        if np.isclose(cost, oldcost):
            break  # TODO: DONT KNOW

        oldcost = cost

    return clustering, centroids, cost


def compute_probs_cx(points, means, covs, probs_c, iter):
    '''
    Input
      - points: (n times d) array containing the dataset
      - means:  (k times d) array containing the k means
      - covs:   (k times d times d) array such that cov[j,:,:] is the covariance matrix of the j-th Gaussian.
      - priors: (k) array containing priors
    Output
      - probs:  (k times n) array such that the entry (i,j) represents Pr(C_i|x_j)
    '''
    # Convert to numpy arrays.
    points, means, covs, probs_c = np.asarray(points), np.asarray(means), np.asarray(covs), np.asarray(probs_c)

    # Get sizes
    n, d = points.shape
    k = means.shape[0]

    # Compute probabilities
    # This will be a (k, n) matrix where the (i,j)'th entry is Pr(C_i)*Pr(x_j|C_i).
    probs_cx = np.zeros((k, n))
    for i in range(k):
        try:
            probs_cx[i] = probs_c[i] * multivariate_normal.pdf(mean=means[i], cov=covs[i], x=points)
        except Exception as e:
            print(f"ERROR!!!=> While iteration {iter}, Cov matrix got singular: ", e)
            print(f"COVS: {covs[i]}\ndet(covs[i])={np.linalg.det(covs[i])}")
            exit(1)

    # The sum of the j'th column of this matrix is P(x_j); why?
    probs_x = np.sum(probs_cx, axis=0, keepdims=True)
    assert probs_x.shape == (1, n)

    # Divide the j'th column by P(x_j). The the (i,j)'th then
    # becomes Pr(C_i)*Pr(x_j)|C_i)/Pr(x_j) = Pr(C_i|x_j)
    probs_cx = probs_cx / probs_x

    return probs_cx, probs_x


def em_algorithm(X, k, T, epsilon=0.001, means=None):
    """ Clusters the data X into k clusters using the Expectation Maximization algorithm.

        Parameters
        ----------
        X : Data matrix of shape (n, d)
        k : Number of clusters.
        T : Maximum number of iterations
        epsilon :  Stopping criteria for the EM algorithm. Stops if the means of
                    two consequtive iterations are less than epsilon.
        means : (k times d) array containing the k initial means (optional)

        Returns
        -------
        means:     (k, d) array containing the k means
        covs:      (k, d, d) array such that cov[j,:,:] is the covariance matrix of
                    the Gaussian of the j-th cluster
        probs_c:   (k, ) containing the probability Pr[C_i] for i=0,...,k.
        llh:       The log-likelihood of the clustering (this is the objective we want to maximize)
    """
    n, d = X.shape

    # Initialize and validate mean
    if means is None:
        means = np.random.rand(k, d)

    # Initialize cov, prior
    probs_x = np.zeros(n)
    probs_cx = np.zeros((k, n))
    probs_c = np.zeros(k) + np.random.rand(k)
    covs = np.zeros((k, d, d))

    # print(covs) a1b2−a2b1
    for i in range(k): covs[i] = np.identity(d)
    probs_c = np.ones(k) / k

    # Column names
    print("em_algorithm\nIterations\tLLH")
    close = False
    old_means = np.zeros_like(means)
    iterations = 0
    while not (close) and iterations < T:
        old_means[:] = means

        # Test det(A) = 0 <=> A singular
        if np.linalg.det(covs).any() == 0:
            print("det(A) == 0 => A singular. exipting!")
            print(f"{iterations}=>det(covs): {np.linalg.det(covs)} \n {covs}")
            exit(1)

        # if not np.all(np.linalg.eigvals(covs) > 0):
        #     print(f"{iterations}=>PSD(covs): np.all({np.linalg.eigvals(covs)})>0 \n {covs}")
        #     exit(1)

        # Expectation step
        # probs_cx = becomes Pr(C_i)*Pr(x_j)|C_i)/Pr(x_j) = Pr(C_i|x_j)
        probs_cx, probs_x = compute_probs_cx(X, means, covs, probs_c, iterations)
        assert probs_cx.shape == (k, n)

        # Maximization step
        # YOUR CODE HERE
        prior_c = probs_cx.sum(axis=1)  # sum rækkerne
        probs_c = prior_c / n

        # print("Means B:, ",means, "\n")
        for i in range(k):
            dividend = np.zeros((d))
            for j in range(n):
                dividend += (X[j, :] * probs_cx[i, j])  #:)
            means[i] = dividend / prior_c[i]

        # print("Means A:, ",means, "\n")
        # assert not np.isnan(means).any()

        # print(f"{iterations}.B=> covs: {covs}")

        for i in range(k):
            upper_sum = np.zeros((d, d))
            for j in range(n):
                xmeans = X[j, :] - means[i, :]  # shape (d,1)
                # print(f"{iterations}=> xmeans.shape: {xmeans.shape}")
                # print(f"{iterations}=> xmeansT.shape: {xmeansT.shape}")
                upper_sum += probs_cx[i, j] * np.outer(xmeans, xmeans.T)

            covs[i] = upper_sum / prior_c[i]  # (k, d, d)
            assert np.linalg.det(covs[i]) != 0.0, f"det(covs[i]) failed.\n{covs[i]}\n"  # a_1b_2−a_2b_1
            assert np.allclose(covs[i], covs[i].T), f"allclose(covs[i], covs[i].T) failed.\n{covs}\n"

        # print(f"{iterations}.A=> covs: {covs}\n")
        # print("COVS: ", covs)
        # print("Wololoo")
        # print("probs_cx", probs_cx[0:5,0:10])
        # print("X[1,:]", X[1,:])
        # print("means[i]", means[i])

        # END CODE

        # Compute per-sample average log likelihood (llh) of this iteration
        llh = 1 / n * np.sum(np.log(probs_x))
        print(iterations + 1, "\t\t", llh)

        # Stop condition
        dist = np.sqrt(((means - old_means) ** 2).sum(axis=1))
        close = np.all(dist < epsilon)
        iterations += 1

    # Validate output
    assert means.shape == (k, d)
    assert covs.shape == (k, d, d)
    assert probs_c.shape == (k,)

    return means, covs, probs_c, llh


def silhouette(data, clustering):
    n, d = data.shape
    k = np.unique(clustering)[-1] + 1

    # YOUR CODE HERE
    silh = None
    # END CODE

    return silh


def testEmVsLlyod(X):
    for k in range(2, 10):
        # em_sc = 0  # silhouette(...)
        # print(f"EM: testEmVsLlyod: iteration: {k}, em_sc: {em_sc}")
        # means, covs, probs_c, llh = em_algorithm(X, k, 50)

        lloyd_sc = 0  # silhouette(...)
        print(f"LL: testEmVsLlyod: iteration: {k}, lloyd_sc: {lloyd_sc}")
        clustering, centroids, cost = lloyds_algorithm(X, k, 50)

        # (Optional) try the lloyd's initialized EM algorithm.


def f1(predicted, labels):
    n, = predicted.shape
    assert labels.shape == (n,)
    r = np.max(predicted) + 1
    k = np.max(labels) + 1

    # Implement the F1 score here
    # YOUR CODE HERE
    contingency = None
    F_individual = None
    F_overall = None
    # END CODE

    assert contingency.shape == (r, k)
    return F_individual, F_overall, contingency


def download_image(url):
    filename = url[url.rindex('/') + 1:]
    try:
        with open(filename, 'rb') as fp:
            return imageio.imread(fp) / 255
    except FileNotFoundError:
        import urllib.request
        with open(filename, 'w+b') as fp, urllib.request.urlopen(url) as r:
            fp.write(r.read())
            return imageio.imread(fp) / 255


def compress_kmeans(im, k, T, name):
    height, width, depth = im.shape
    data = im.reshape((height * width, depth))
    clustering, centroids, score = lloyds_algorithm(data, k, 5)  # changes from 5

    # make each entry of data to the value of it's cluster
    data_compressed = data

    for i in range(k): data_compressed[clustering == i] = centroids[i]

    im_compressed = data_compressed.reshape((height, width, depth))

    # The following code should not be changed.
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(im_compressed)
    plt.savefig("compressed1.jpg")
    # plt.show()

    original_size = os.stat(name).st_size
    compressed_size = os.stat('compressed1.jpg').st_size
    print("Original Size: \t\t", original_size)
    print("Compressed Size: \t", compressed_size)
    print("Compression Ratio: \t", round(original_size / compressed_size, 5))


def compress_facade(k=4, T=100):
    img_facade = download_image('https://users-cs.au.dk/rav/ml/handins/h4/nygaard_facade.jpg')
    compress_kmeans(img_facade, k, T, 'nygaard_facade.jpg')


def main():
    iris = sklearn.datasets.load_iris()
    X = iris['data'][:, 0:2]  # reduce to 2d so you can plot if you want

    ##########################
    ##### TESTING ############
    testEmVsLlyod(X)
    ##########################
    clustering, centroids, cost = lloyds_algorithm(X, 3, 100)

    img_facade = download_image('https://uploads.toptal.io/blog/image/443/toptal-blog-image-1407508081138.png')

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img_facade)
    plt.savefig("blob1.jpg")
    # fig.show()
    # plt.show() # FIXME: Har rettet fra denne linje til den ovenover.

    size = os.stat('toptal-blog-image-1407508081138.png').st_size

    print("The image consumes a total of %i bytes. \n" % size)
    print("You should compress your image as much as possible! ")

    compress_facade()


if __name__ == '__main__':
    main()
