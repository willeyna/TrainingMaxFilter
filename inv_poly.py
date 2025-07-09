from groupy import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################################## PARAMETERS
# load test data so that it is the same for every model
X_test_np = np.load('X_test.npy')
X_test = torch.from_numpy(X_test_np).float().to(device)
# number of templates in max filter
d,n = X_test.shape


G = GPU_GroupAction(pmId, d, device=device)
k = G.order
X_test_orbits = G.get_orbits(X_test)
block_size = 5000 # how many data points to include in each test distance matrix
######################################################## TESTING

with torch.no_grad():
    fX = invariant_polynomial(X_test, G)

    alpha_test = torch.tensor(float("inf"), device=device)
    beta_test  = torch.tensor(0, device=device)
    # break test set into blocks over which to compute distance matrices
    for i in range(0, n, block_size):
        # choose subset of x_i and f(x_i)
        Xi = X_test[:, i:i+block_size]
        fXi = fX[:, i:i+block_size]
        for j in range(i, n, block_size):
            Xj = X_test[:, j:j+block_size]
            fXj = fX[:, j:j+block_size]
            # compute block distance matrices
            # when Xi=Xj function automatically only computes n choose 2 distances
            DX_ij   = G.dist_matrix(Xi, Xj)
            DfX_ij  = torch.cdist(fXi.T, fXj.T)
            # compute constants over that block
            alpha_ij, beta_ij = lipschitz(DX_ij, DfX_ij)
            # update either lipschitz constant if a worse one is found
            alpha_test = torch.minimum(alpha_test, alpha_ij)
            beta_test = torch.maximum(beta_test, beta_ij)

    distortion_test = ((beta_test / alpha_test).item())

    print(f"Invariant Polynomial Test Distortion: {distortion_test:.2f}")

    fX = invariant_polynomial(X_test, G, homo=True)

    alpha_test = torch.tensor(float("inf"), device=device)
    beta_test  = torch.tensor(0, device=device)
    # break test set into blocks over which to compute distance matrices
    for i in range(0, n, block_size):
        # choose subset of x_i and f(x_i)
        Xi = X_test[:, i:i+block_size]
        fXi = fX[:, i:i+block_size]
        for j in range(i, n, block_size):
            Xj = X_test[:, j:j+block_size]
            fXj = fX[:, j:j+block_size]
            # compute block distance matrices
            # when Xi=Xj function automatically only computes n choose 2 distances
            DX_ij   = G.dist_matrix(Xi, Xj)
            DfX_ij  = torch.cdist(fXi.T, fXj.T)
            # compute constants over that block
            alpha_ij, beta_ij = lipschitz(DX_ij, DfX_ij)
            # update either lipschitz constant if a worse one is found
            alpha_test = torch.minimum(alpha_test, alpha_ij)
            beta_test = torch.maximum(beta_test, beta_ij)

        distortion_test = ((beta_test / alpha_test).item())

    print(f"Homogenous Invariant Polynomial Test Distortion: {distortion_test:.2f}")
