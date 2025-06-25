from groupy import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################################## PARAMETERS
# load test data so that it is the same for every model
X_test_np = np.load('X_test.npy')
X_test = torch.from_numpy(X_test_np).float().to(device)
# number of templates in max filter
d = X_test.shape[0]

G = GPU_GroupAction(cyclic_translations, d, device=device)
k = G.order
X_test_orbits = G.get_orbits(X_test)
D_test = G.dist_matrix(X_test)
# in SORT embedding dim will be t*k, +1 to ensure at least as many params as MF
t = 3*d//k  + 1
# currently maintain dimensionality given by max filtering

batch_size = 20 # 20 works well for pmId d=3
n_trials = 10 #The number of times we will train a new model from scratch
n_epochs = 100 #The number of training epochs for each model
grad_steps_per_epoch = 200 #The number of gradient descent iterations in each training epoch
lr = 1e-2 # learning rate default is 1e-3
lr_period =  n_epochs//5
######################################################## TRAINING

all_test_distortions = []
all_trained_distortions = []
all_trained_templates = []

for trial in range(n_trials):
    test_distortions = []
    train_distortions = []

    # max filter template layer
    templates = torch.normal(0, 1, (t, d), requires_grad=True, device=device)

    optimizer = torch.optim.Adam([templates], lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, lr_period)

    for epoch in range(n_epochs):
        for step in range(grad_steps_per_epoch):
            optimizer.zero_grad()

            # Sample training batch
            X =  torch.normal(0, 1, (d, batch_size), device=device)
            D = G.dist_matrix(X)           # Tensor (n, n)
            X_orbits = G.get_orbits(X)     # Tensor (k, d, n)

            # Forward pass
            norm_templates = F.normalize(templates, dim=1)
            filter_features = sorted_filter(norm_templates, X_orbits)
            features = filter_features

            alpha_sq, beta_sq = squared_lipschitz(D, features)
            loss = beta_sq / alpha_sq

            loss.backward()
            optimizer.step()

        # Test each epoch
        print(f"Epoch {epoch}", end='\r')
        with torch.no_grad():
            # Compute training distortion on last batch of training data (X, D, etc.)
            norm_templates = F.normalize(templates, dim=1)
            train_features = sorted_filter(norm_templates, X_orbits)
            alpha_train_sq, beta_train_sq = squared_lipschitz(D, train_features)
            distortion_train = (beta_train_sq / alpha_train_sq).item()

            # Compute test distortion
            test_features = sorted_filter(norm_templates, X_test_orbits)
            alpha_test_sq, beta_test_sq = squared_lipschitz(D_test, test_features)
            distortion_test = (beta_test_sq / alpha_test_sq).item()

            train_distortions.append(distortion_train)
            test_distortions.append(distortion_test)
        scheduler.step()

    all_trained_distortions.append(train_distortions)
    all_test_distortions.append(test_distortions)
    all_trained_templates.append(norm_templates.detach().cpu().numpy())

avg_final_error = np.mean([d[-1] for d in all_test_distortions])

fig = plt.figure(figsize = (15,5))
for distortion_function in all_test_distortions:
    plt.plot(distortion_function, alpha=0.3, c='red')

# for i, distortion_function in enumerate(all_trained_distortions):
#     if i ==0:
#         plt.plot(distortion_function, alpha=0.1, c='blue', label = 'Training Error')
#     else:
#         plt.plot(distortion_function, alpha=0.1, c='blue')

textstr = '\n'.join([
    str(G).split(',')[0],
    f'Input Data Dimension: {d}',
    f'Embedding Dimension: {t*k + 1}',
    f'Final Mean Squared Distortion: {avg_final_error:.2f}',
    f'Batch size: {batch_size}',
    f'Grad steps/epoch: {grad_steps_per_epoch}',
    f'Learning rate: {lr}',
])
# Place text box in upper right in axes coords
props = dict(boxstyle='round', alpha=0.1)
plt.gca().text(0.98, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right', bbox=props)


plt.title("Trained Sort(L(X)) Model Test Error")
plt.xlabel("Epoch")
plt.ylabel("Squared Distortion")

# Get current time formatted as YYYY-MM-DD_HH-MM
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
# time-based label for now since all params are shown in the figure
plt.savefig(f'./Results/SORT_{timestamp}.png')
