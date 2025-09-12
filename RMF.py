from groupy import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################################## PARAMETERS
X_test_np = np.load('X_test.npy')
if np.iscomplexobj(X_test_np):
    X_test = torch.from_numpy(X_test_np).to(torch.complex64).to(device)
else:
    X_test = torch.from_numpy(X_test_np).to(torch.float).to(device)
input_shape = X_test.shape[:-1]
n = X_test.shape[-1]
input_dtype = X_test.dtype

t = 8

block_size = 2000 # how many data points to include in each test distance matrix
n_trials = 100

######################################################## GROUP ACTION
# G is either a finite GroupAction obj or a continuous group name str
G = GPU_GroupAction(rotations, input_shape[0], device=device, orders=[3])
# G = 'phase'

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

all_test_distortions = []
all_trained_templates = []

for trial in range(n_trials):
    print(f"Trial {trial}", end='\r')
    test_distortions = []

    # max filter template layer
    templates = torch.randn((t, *(input_shape[::-1])), dtype=X_test.dtype, device=device, requires_grad=True)

    with torch.no_grad():
        norm_templates = F.normalize(templates, dim=tuple(range(1, templates.dim())))
        # Compute test distortion ---
        test_features = max_filter(norm_templates, X_test_orbits)
        # initalize alpha and beta
        alpha_test = torch.tensor(float("inf"), device=device)
        beta_test  = torch.tensor(0, device=device)
        # break test set into blocks over which to compute distance matrices
        for i in range(0, n, block_size):
            # choose subset of x_i and f(x_i)
            Xi = X_test[:, i:i+block_size]
            fXi = test_features[:, i:i+block_size]
            for j in range(i, n, block_size):
                Xj = X_test[:, j:j+block_size]
                fXj = test_features[:, j:j+block_size]
                # compute block distance matrices
                # when Xi=Xj function automatically only computes n choose 2 distances
                if finite:
                    DX_ij   = G.dist_matrix(Xi, Xj)
                else:
                    DX_ij   = mf_dist_matrix(max_filter, Xi, Xj)

                DfX_ij  = torch.cdist(fXi.T, fXj.T)
                # compute constants over that block
                alpha_ij, beta_ij = lipschitz(DX_ij, DfX_ij)
                # update either lipschitz constant if a worse one is found
                alpha_test = torch.minimum(alpha_test, alpha_ij)
                beta_test = torch.maximum(beta_test, beta_ij)

        distortion_test = (beta_test / alpha_test).item()
        test_distortions.append(distortion_test)

    all_test_distortions.append(test_distortions)
    all_trained_templates.append(norm_templates.detach().cpu().numpy())


final_distortions = [d[-1] for d in all_test_distortions]
mean_final_error = np.nanmean(final_distortions)
median_final_error = np.nanmedian(final_distortions)
min_final_error = np.nanmin(final_distortions)

fig = plt.figure(figsize = (15,5))
distortions = np.array(all_test_distortions).flatten()
plt.hist(distortions[np.isfinite(distortions)])

textstr = '\n'.join([
    str(G).split(',')[0],
    f'Input Data Dimension: {str(input_shape)}',
    f'Embedding Dimension: {t}',
    f'Mean Distortion: {mean_final_error:.2f}',
    f'Median Distortion: {median_final_error:.2f}',
    f'Best Distortion: {min_final_error:.2f}',
    f'Number of Trials: {n_trials}'
])
# Place text box in upper right in axes coords
props = dict(boxstyle='round', alpha=0.1)
plt.gca().text(0.98, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right', bbox=props)


plt.title("Random Max Filter Test Set Distortion")

# Get current time formatted as YYYY-MM-DD_HH-MM
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
# time-based label for now since all params are shown in the figure
plt.savefig(f'./Results/RMF_{timestamp}.png')
