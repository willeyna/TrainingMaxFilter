from groupy import *
device = torch.device('cpu')

######################################################## PARAMETERS
# load test data so that it is the same for every model
X_test_np = np.load('X_test.npy')
X_test = torch.from_numpy(X_test_np).to(torch.float64).to(device)

# G = GPU_GroupAction(pmId, d, device=device)
G = 'orthogonal'
finite = isinstance(G, GroupAction)
if finite:
    k = G.order
    X_test_orbits = G.get_orbits(X_test)
else:
    X_test_orbits = X_test
    # overwrite max_filter function with the specific continuous versions
    if G == 'shape':
        max_filter = shape_max_filter
    if G == 'phase':
        max_filter = phase_max_filter
    if G == 'orthogonal':
        max_filter = orthogonal_max_filter
######################################################## TESTING

with torch.no_grad():
    fX = invariant_polynomial(X_test, G)

    if finite:
        # full distance matrix for minibatch-- important to keep 'mini'!
        D = G.dist_matrix(X_test)           # Tensor (n, n)
    else:
        D = mf_dist_matrix(max_filter, X_test)

    DfX  = torch.cdist(fX.T, fX.T)
    # compute constants over that block
    alpha_test, beta_test = lipschitz(D, DfX)
    distortion_test = ((beta_test / alpha_test).item())

    print(f"Invariant Polynomial Test Distortion: {distortion_test:.2f}")

    fX = invariant_polynomial(X_test, G, homo=True)

    if finite:
        # full distance matrix for minibatch-- important to keep 'mini'!
        D = G.dist_matrix(X_test)           # Tensor (n, n)
    else:
        D = mf_dist_matrix(max_filter, X_test)

    DfX  = torch.cdist(fX.T, fX.T)
    # compute constants over that block
    alpha_test, beta_test = lipschitz(D, DfX)
    distortion_test = ((beta_test / alpha_test).item())

    print(f"Homogenous Invariant Polynomial Test Distortion: {distortion_test:.2f}")
