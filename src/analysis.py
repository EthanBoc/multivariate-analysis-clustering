import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.cluster.vq import kmeans2

# Retrieves data from a text file as a list of strings
def readfile(filename):
    with open(filename) as f:
        data = f.read().splitlines()
    return data





# Equal-frequency binning (Quantile discretization)
def discretization(data: np.array, n: int) -> np.array:
    I, K = np.shape(data)
    T = np.zeros((I, K), dtype=int)

    for k in range(K):
        # The indices of a column are ordered based on its values
        order = np.argsort(data[:, k])
        
        # Each column is divided into n intervals of equal size
        for rank, idx in enumerate(order):
            # min() prevents rounding overflows on the last element caused by int()
            class_num = min(int(rank / (I / n)), n - 1) 
            T[idx, k] = class_num + 1

    return T





# Performs Hierarchical Agglomerative Clustering (HAC) on factor scores and displays the partition on the 1st factor plane
def cah(ind_factors: np.array, eigenvalues: np.array, ind_names: list, var_factors: np.array, var_names: list, analysis: str, n_axes: int = 3, n_clusters: int = 3, method: str = 'ward'):
    # Higher axes only capture measurement noise. By restricting clustering to the first n axes,
    # we filter out noise, enabling distance algorithms (Ward, Euclidean k-means) to isolate much more stable and robust clusters.
    selected_factors = ind_factors[:, :n_axes]

    # method: linkage criterion ('ward' by default, 'complete' for the alternative criterion)
    Z = linkage(selected_factors, method=method, metric='euclidean')

    # Allows viewing the window as a layout grid of plots
    layout = [['A', 'A'], ['B', 'C']]
    fig, axs = plt.subplot_mosaic(layout, figsize=(15, 8))

    # 1. Dendrogram Plot
    dendrogram(Z, labels=ind_names, leaf_rotation=90, ax=axs['A'])
    axs['A'].set_title(f"HAC Dendrogram ({method.capitalize()} - {n_axes} axes retained)")
    axs['A'].set_xlabel("Individuals")
    axs['A'].set_ylabel("Aggregation Inertia")
    axs['A'].grid(axis='y', linestyle='--', alpha=0.5)

    # 2. Factor plane for individuals
    clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
    axs['B'].scatter(ind_factors[:, 0], ind_factors[:, 1], c=clusters, cmap='Set1', s=50, edgecolors='k')
    axs['B'].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axs['B'].axvline(0, color='black', linewidth=0.8, linestyle='--')

    # Annotate individuals on the factor plane
    for label, x, y in zip(ind_names, ind_factors[:, 0], ind_factors[:, 1]):
        axs['B'].annotate(label, xy=(x, y), xytext=(5, 5), textcoords='offset points', ha='left', va='bottom', fontsize=9)

    pct_f1 = 100 * (eigenvalues[0] / np.sum(eigenvalues))
    pct_f2 = 100 * (eigenvalues[1] / np.sum(eigenvalues))

    axs['B'].set_xlabel(f"Factor Axis 1 ({pct_f1:.1f}%)")
    axs['B'].set_ylabel(f"Factor Axis 2 ({pct_f2:.1f}%)")
    axs['B'].set_title(f"{analysis.upper()}: Partition into {n_clusters} clusters projected onto the first factor plane")
    axs['B'].grid(True, linestyle=':', alpha=0.6)

    # 3. Factor plane for variables
    axs['C'].scatter(var_factors[:, 0], var_factors[:, 1], color='blue', marker='o')
    axs['C'].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axs['C'].axvline(0, color='black', linewidth=0.8, linestyle='--')

    # Annotate variables on the factor plane
    for i, name in enumerate(var_names):
        axs['C'].annotate(name, (var_factors[i, 0], var_factors[i, 1]), textcoords="offset points", xytext=(5, 5), ha='left')

    axs['C'].set_xlabel(f'F1 ({pct_f1:.1f}%)')
    axs['C'].set_ylabel(f'F2 ({pct_f2:.1f}%)')
    axs['C'].set_title(f'{analysis.upper()}: Projection of variables')
    axs['C'].grid(True, linestyle=':', alpha=0.6)

    # Automatically adjusts subplot parameters to prevent overlapping elements
    plt.tight_layout()
    plt.show()

    return


# Performs K-means clustering on factor scores and displays the projected clusters on the 1st factor plane (F1 x F2)
def k_means(ind_factors: np.array, eigenvalues: np.array, ind_names: list, var_factors: np.array, var_names: list, analysis: str, pct_inertia: np.array, n_axes: int = 3, n_clusters: int = 3):
    # Higher axes only capture measurement noise. By restricting clustering to the first n axes,
    # we filter out noise, enabling distance algorithms (Ward, Euclidean k-means) to isolate much more stable and robust clusters.
    selected_factors = ind_factors[:, :n_axes]
    
    centroids, labels = kmeans2(selected_factors, n_clusters, minit='points')

    # Allows viewing the window as a layout grid of plots
    layout = [['A', 'A'], ['B', 'C']]
    fig, axs = plt.subplot_mosaic(layout, figsize=(15, 8))

    # 1. Inertia Diagram
    nb_var = len(pct_inertia)
    axs['A'].plot(range(1, nb_var + 1), pct_inertia, 'o-')
    axs['A'].set_xlabel("Factor Axes")
    axs['A'].set_ylabel("Percentage of Inertia")
    axs['A'].set_title("Inertia Scree Plot")
    axs['A'].grid(True)

    # 2. Visualization on the first factor plane F1 x F2
    # Markers to visually differentiate clusters
    markers = ['o', 's', '^', 'd', 'p', 'h', '8']
    
    # Plot each cluster
    for k in range(n_clusters):
        cluster_points = selected_factors[labels == k]
        axs['B'].scatter(cluster_points[:, 0], cluster_points[:, 1], marker=markers[k % len(markers)], s=60, alpha=0.7, label=f'Cluster {k+1}')

    # Plot centroids projected onto F1 x F2
    axs['B'].scatter(centroids[:, 0], centroids[:, 1], color='black', marker='*', s=200, edgecolors='white', linewidths=1.5, label='Centroids')

    axs['B'].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axs['B'].axvline(0, color='black', linewidth=0.8, linestyle='--')

    # Annotate individuals on the factor plane
    for label_name, x, y in zip(ind_names, ind_factors[:, 0], ind_factors[:, 1]):
        axs['B'].annotate(label_name, xy=(x, y), xytext=(5, 5), textcoords='offset points', ha='left', va='bottom', fontsize=9)

    axs['B'].set_xlabel(f"Factor Axis 1 ({pct_inertia[0]:.1f}%)")
    axs['B'].set_ylabel(f"Factor Axis 2 ({pct_inertia[1]:.1f}%)")
    axs['B'].set_title(f"K-means (k={n_clusters}) on {n_axes} MCA axes")
    axs['B'].grid(True, linestyle=':', alpha=0.6)

    # 3. Factor plane for variables
    axs['C'].scatter(var_factors[:, 0], var_factors[:, 1], color='blue', marker='o')
    axs['C'].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axs['C'].axvline(0, color='black', linewidth=0.8, linestyle='--')

    # Annotate variables on the factor plane
    for i, name in enumerate(var_names):
        axs['C'].annotate(name, (var_factors[i, 0], var_factors[i, 1]), textcoords="offset points", xytext=(5, 5), ha='left')

    axs['C'].set_xlabel(f'F1 ({pct_inertia[0]:.1f}%)')
    axs['C'].set_ylabel(f'F2 ({pct_inertia[1]:.1f}%)')
    axs['C'].set_title(f'{analysis.upper()}: Projection of variables')
    axs['C'].grid(True, linestyle=':', alpha=0.6)

    # Automatically adjusts subplot parameters to prevent overlapping elements
    plt.tight_layout()
    plt.show()

    return 


def factor_analysis(data: np.array, ind_names, var_names, analysis: str, n: int = 3, clustering: str = None, n_axes: int = 3, n_clusters: int = 3, linkage_method: str = 'ward') -> None:
    if analysis == "acp":
        m = data.mean(axis=0)
        var = data.var(axis=0)
        ectype = data.std(axis=0)

        # Standardized data (centered-reduced)
        X = (data - m) / ectype  # Data matrix
        D = np.eye(data.shape[0]) / data.shape[0]  # Weight matrix
        M = np.eye(data.shape[1])  # Metric matrix
    
    elif analysis == "acm": 
        # n > 0 if data discretization is requested
        if n != 0:
            data = discretization(data, n)

            # Adjust variable and category names
            var_names = [f"{nom}{jm}" 
            for ind, nom in enumerate(var_names) 
            for jm in range(1, int(data[:, ind].max()) + 1)]

        # Construction of the Complete Disjunctive Table (CDT) Y
        I = data.shape[0]  # Number of individuals
        K = data.shape[1]  # Number of variables
        J = int(sum(data.max(axis=0)))  # Total number of categories
        Y = np.zeros((I, J))

        # First count how many categories each variable has (in case discretization was not performed beforehand)
        categories = data.max(axis=0)

        # Construct two parallel arrays to map a category back to its variable
        category_indices = np.zeros(J, dtype=int)
        variable_indices = np.zeros(J, dtype=int)
        im = 0
        jm = 1
        km = 0

        for category in categories:
            while jm <= category:
                category_indices[im + jm - 1] = jm
                variable_indices[im + jm - 1] = km
                jm += 1
            im += int(category)
            jm = 1
            km += 1

        # Y_{ij} = 1 if individual i has category j, 0 otherwise
        for i in range(I):
            for j in range(J):
                if data[i][variable_indices[j]] == category_indices[j]:
                    Y[i][j] = 1
                else:
                    Y[i][j] = 0
        
        # Count for category j
        Ij = Y.sum(axis=0)

        X = (I * Y / Ij) - 1.0  # Data matrix
        D = np.eye(I) / I  # Weight matrix
        M = np.diag(Ij / (I * K))  # Metric matrix
    
    else: 
        print("Available analyses are 'acp' or 'acm'!")
        return

    # General Factor Analysis Method using X, D, and M matrices

    # Compute covariance matrix for individuals
    cov_matrix = X.T @ D @ X @ M

    eigenvalues, eigenvectors_var = np.linalg.eig(cov_matrix)
    ind = np.argsort(eigenvalues)[::-1]

    # Compute eigenvalues and eigenvectors
    eigenvalues_sorted = np.real(eigenvalues[ind])                       
    Vp_ind = np.real(eigenvectors_var[:, ind])

    # Compute factor scores for individuals
    ind_factors = X @ M @ Vp_ind

    # Compute factor scores for variables / categories
    var_factors = (M @ X.T @ D @ ind_factors) / np.sqrt(eigenvalues_sorted)

    # Percentage of inertia associated with each axis
    total_inertia = np.sum(eigenvalues_sorted)
    pct_inertia = 100 * (eigenvalues_sorted / total_inertia)

    # Contribution of individuals (%)
    weights = np.diag(D)[:, None]
    inertia_i = weights * (ind_factors ** 2)
    ind_contribution = 100 * (inertia_i / eigenvalues_sorted)

    # Quality of representation of individuals (squared cosines)
    ind_cos2 = ind_factors ** 2 / np.sum(X ** 2, axis=1)[:, None]

    # 3 display choices: 2 with clustering methods (cah, k-means) and one without clustering
    if clustering == "cah": 
        cah(ind_factors, eigenvalues_sorted, ind_names, var_factors, var_names, analysis, n_axes, n_clusters, linkage_method)
    elif clustering == "k-means":
        k_means(ind_factors, eigenvalues_sorted, ind_names, var_factors, var_names, analysis, pct_inertia, n_axes, n_clusters)
    else:
        # Allows viewing the window as a layout grid of plots
        layout = [['A', 'A'], ['B', 'C']]
        fig, axs = plt.subplot_mosaic(layout, figsize=(15, 8))

        # 1. Inertia Diagram
        nb_var = len(pct_inertia)
        axs['A'].plot(range(1, nb_var + 1), pct_inertia, 'o-')
        axs['A'].set_xlabel("Factor Axes")
        axs['A'].set_ylabel("Percentage of Inertia")
        axs['A'].set_title("Inertia Scree Plot")
        axs['A'].grid(True)

        # 2. Factor plane for individuals
        axs['B'].scatter(ind_factors[:, 0], ind_factors[:, 1], color='red', marker='x')
        axs['B'].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axs['B'].axvline(0, color='black', linewidth=0.8, linestyle='--')

        # Annotate individuals on the factor plane
        for i, name in enumerate(ind_names):
            axs['B'].annotate(name, (ind_factors[i, 0], ind_factors[i, 1]), textcoords="offset points", xytext=(5, 5), ha='left')

        axs['B'].set_xlabel(f'F1 ({pct_inertia[0]:.1f}%)')
        axs['B'].set_ylabel(f'F2 ({pct_inertia[1]:.1f}%)')
        axs['B'].set_title(f'{analysis.upper()}: Projection of individuals')
        axs['B'].grid(True, linestyle=':', alpha=0.6)

        # 3. Factor plane for variables
        axs['C'].scatter(var_factors[:, 0], var_factors[:, 1], color='blue', marker='o')
        axs['C'].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axs['C'].axvline(0, color='black', linewidth=0.8, linestyle='--')

        # Annotate variables on the factor plane
        for i, name in enumerate(var_names):
            axs['C'].annotate(name, (var_factors[i, 0], var_factors[i, 1]), textcoords="offset points", xytext=(5, 5), ha='left')

        axs['C'].set_xlabel(f'F1 ({pct_inertia[0]:.1f}%)')
        axs['C'].set_ylabel(f'F2 ({pct_inertia[1]:.1f}%)')
        axs['C'].set_title(f'{analysis.upper()}: Projection of variables')
        axs['C'].grid(True, linestyle=':', alpha=0.6)

        # Automatically adjusts subplot parameters to prevent overlapping elements
        plt.tight_layout()
        plt.show()

    return


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="PCA / MCA with optional HAC or k-means clustering")
    parser.add_argument("--analysis", choices=["acp", "acm"], default="acp",
                        help="acp = PCA, acm = MCA (multiple correspondence analysis)")
    parser.add_argument("--clustering", choices=["cah", "k-means", "none"], default="cah",
                        help="cah = hierarchical agglomerative clustering")
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--linkage", choices=["ward", "complete"], default="ward",
                        help="linkage criterion of the hierarchical clustering")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data = np.loadtxt(data_dir / "population_donnees.txt")
    ind_names = readfile(data_dir / "population_noms_individus.txt")
    var_names = readfile(data_dir / "population_noms_variables.txt")

    factor_analysis(data, ind_names, var_names, analysis=args.analysis,
                    clustering=None if args.clustering == "none" else args.clustering,
                    n_clusters=args.n_clusters, linkage_method=args.linkage)
