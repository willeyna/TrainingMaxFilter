from groupy import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################################## PARAMETERS
# load test data so that it is the same for every model
X_test_np = np.load('X_test.npy')
X_test = torch.from_numpy(X_test_np).float().to(device)
# number of templates in max filter
d = X_test.shape[0]
t = 3*d

G = GPU_GroupAction(cyclic_translations, d, device=device)
X_test_orbits = G.get_orbits(X_test)
D_test = G.dist_matrix(X_test)

n_trials = 1000
######################################################## TESTING

all_test_distortions = []
all_trained_templates = []

for trial in range(n_trials):
    test_distortions = []

    # max filter template layer
    templates = torch.normal(0, 1, (t, d), device=device)

    with torch.no_grad():
        norm_templates = F.normalize(templates, dim=1)
        test_features = max_filter(norm_templates, X_test_orbits)
        alpha_test_sq, beta_test_sq = squared_lipschitz(D_test, test_features)
        distortion_test = (beta_test_sq / alpha_test_sq).item()

        test_distortions.append(distortion_test)

    all_test_distortions.append(test_distortions)
    all_trained_templates.append(norm_templates.detach().cpu().numpy())


avg_final_error = np.mean([d[-1] for d in all_test_distortions])
median_final_error = np.median([d[-1] for d in all_test_distortions])

fig = plt.figure(figsize = (15,5))
plt.hist(np.array(all_test_distortions).flatten())

textstr = '\n'.join([
    str(G).split(',')[0],
    f'Input Data Dimension: {d}',
    f'Embedding Dimension: {t}',
    f'Mean Squared Distortion: {avg_final_error:.2f}',
    f'Median Squared Distortion: {median_final_error:.2f}'
])
# Place text box in upper right in axes coords
props = dict(boxstyle='round', alpha=0.1)
plt.gca().text(0.98, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right', bbox=props)


plt.title("Random Max Filter Model Test Error")

# Get current time formatted as YYYY-MM-DD_HH-MM
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
# time-based label for now since all params are shown in the figure
plt.savefig(f'./Results/RMF_{timestamp}.png')
