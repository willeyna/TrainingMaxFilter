from groupy import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################################## PARAMETERS
# load test data so that it is the same for every model
X_test_np = np.load('X_test.npy')
X_test = torch.from_numpy(X_test_np).float().to(device)
# number of templates in max filter
d,n = X_test.shape
t = 8

G = GPU_GroupAction(pmId, d, device=device)
k = G.order
X_test_orbits = G.get_orbits(X_test)
block_size = 5000 # how many data points to include in each test distance matrix
n_trials = 2000
######################################################## TESTING

all_test_distortions = []
all_trained_templates = []

for trial in range(n_trials):
    print(f"Trial {trial}", end='\r')
    test_distortions = []

    # max filter template layer
    templates = torch.normal(0, 1, (t, d), device=device)

    with torch.no_grad():
        norm_templates = F.normalize(templates, dim=1)
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
                DX_ij   = G.dist_matrix(Xi, Xj)
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


avg_final_error = np.nanmean([d[-1] for d in all_test_distortions])
median_final_error = np.nanmedian([d[-1] for d in all_test_distortions])

fig = plt.figure(figsize = (15,5))
plt.ylim(top=20)
distortions = np.array(all_test_distortions).flatten()
plt.hist(distortions[np.isfinite(distortions)])

textstr = '\n'.join([
    str(G).split(',')[0],
    f'Input Data Dimension: {d}',
    f'Embedding Dimension: {t}',
    f'Mean Distortion: {avg_final_error:.2f}',
    f'Median Distortion: {median_final_error:.2f}',
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
