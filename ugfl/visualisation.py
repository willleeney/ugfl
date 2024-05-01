import numpy as np
import pickle
from ugfl.data_processing import wasserstein_randomness

def plot_comparison_figures(which_graph="", figsize=(10,4)):
    from os import getcwd
    from os.path import dirname
    import matplotlib
    matplotlib.use("macosx")
    
    import matplotlib.pyplot as plt
    import scienceplots
    # plt.style.use(['science', 'ieee'])
    #matplotlib.rcParams['mathtext.fontset'] = 'cm'

    # plt.rcParams["font.family"] = "Times New Roman"
    import pickle
    import numpy as np
    # plt.rcParams["figure.dpi"] = 200
    from copy import deepcopy

    # create holder figure
    nrows, ncols = 1, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    X = np.arange(14)
    colours = [plt.colormaps['Paired'].colors[1], plt.colormaps['Paired'].colors[5]]
    paled_colours = [plt.colormaps['Pastel1'].colors[1], plt.colormaps['Pastel1'].colors[0]]
    markers = ['x', '.']#, plt.colormaps['Paired'].colors[0], plt.colormaps['Paired'].colors[4]]
    datasets = ['Cora', 'CiteSeer', 'PubMed']

    w_results_fedave = []
    w_results_indv = []
    w_results_central = []
    for i, ax in enumerate(axes.flat):
        for j, algo_name in zip([0, 1], ['indv', 'fedave']):
            algo_performance_over_clients = []
            algo_error = []
            for cid in range(2, 16):
                test_per_o_seed = np.zeros(3)
                for k, seed in enumerate([42, 66, 500]):
                    try:
                        perf = pickle.load(
                            open(
                                f"{dirname(getcwd())}/ugfl/final_results/results/dmon_{algo_name}_{datasets[i]}_{cid}_random{which_graph}_{seed}",
                                "rb"))
                        # find best val, find best test
                        test_per_o_seed[k] = np.mean(perf['f1_self_test'], axis=0)[np.argmax(np.mean(perf['f1_self_val'], axis=0))]
                    except:
                        test_per_o_seed[k] = 0.

                algo_performance_over_clients.append(np.mean(test_per_o_seed))
                algo_error.append(np.std(test_per_o_seed))
                if algo_name == 'fedave':
                    w_results_fedave.append(test_per_o_seed)
                else:
                    w_results_indv.append(test_per_o_seed)

            if algo_name == 'fedave':
                label = "Federated"
                alpha = 0.7
            else:
                label = 'Isolated'
                alpha = 1.

            ax.plot(X, algo_performance_over_clients,
                    label=label, linestyle='-', marker=markers[j], color=colours[j], zorder=15+j)
            ax.fill_between(X, [agl[0] - agl[1] for agl in zip(algo_performance_over_clients, algo_error)], [sum(agl) for agl in zip(algo_performance_over_clients, algo_error)],
                            edgecolor=colours[j], facecolor=paled_colours[j], alpha=alpha, zorder=10+j)

        test_per_o_seed = np.zeros(3)
        for k, seed in enumerate([42, 66, 500]):
            try:
                central_which = deepcopy(which_graph)
                if which_graph == "_overlapping":
                    central_which = ""
                perf = pickle.load(
                    open(f"{dirname(getcwd())}/ugfl/final_results/results/dmon_central_{datasets[i]}{central_which}_{seed}", "rb"))
                # find best val, find best test
                test_per_o_seed[k] = np.mean(perf['f1_self_test'], axis=0)[np.argmax(np.mean(perf['f1_self_val'], axis=0))]
            except:
                test_per_o_seed[k] = 0.
        
        for wrc in range(14):
            w_results_central.append(test_per_o_seed)
        best_test = np.mean(test_per_o_seed)
        best_test_std = np.std(test_per_o_seed)
        #axes[i].plot([X[0], X[-1]], [random, random], "k--", label='random')
        ax.plot([X[0], X[-1]], [best_test, best_test], "k-", label='Centralised', zorder=14)
        ax.fill_between(X, best_test-best_test_std, best_test+best_test_std, edgecolor='k', facecolor='grey', zorder=1)

        if which_graph == "":
            title = f'{datasets[i]}'
        elif which_graph == "_dropfeatures":
            title = f'{datasets[i]} w/ 70\% of features'
        elif which_graph == "_overlapping":
            title = f'{datasets[i]} w/ overlapping\n node partitions'
        ax.set_title(title, fontsize=22)
        if i == 1:
            h, l = ax.get_legend_handles_labels()
            kw = dict(ncol=2, loc="lower center", frameon=False, fontsize=17)
            leg1 = ax.legend(h[:2], l[:2], bbox_to_anchor=[0.47, -0.36], **kw)
            leg2 = ax.legend(h[2:], l[2:], bbox_to_anchor=[0.5, -0.46], **kw)
            ax.add_artist(leg1)
        if i == 1:
            ax.set_xlabel('N Clients', fontsize=18)
        if i == 0:
            ax.set_ylabel('Test F1 Performance', fontsize=18)

        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.xaxis.set_ticks(np.arange(0, 14, 1))
        ax.set_xticklabels(X+2)

    # w coefficient stuff
    results_matrix = np.zeros((42, 3, 3)) # tests, random_seeds, algorithms
    for widx in range(len(w_results_central)):
        for sn in range(3):
            results_matrix[widx, sn, 0] = w_results_central[widx][sn]
            results_matrix[widx, sn, 1] = w_results_fedave[widx][sn]
            results_matrix[widx, sn, 2] = w_results_indv[widx][sn]

    w_rand = wasserstein_randomness(results_matrix)
    print(f'W Randomness Coefficient: {w_rand:.3f} ')

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    fig.savefig(f'{dirname(getcwd())}/ugfl/figures/n_clients_v_perf{which_graph}.pdf', bbox_inches='tight')


def plot_n_edges_on_dataset_figure(partition='random'):
    from ugfl.datasets import create_federated_dataloaders
    import matplotlib.pyplot as plt
    from os import getcwd
    from os.path import dirname
    import scienceplots
    plt.style.use(['science', 'ieee'])
    plt.rcParams["font.family"] = "Times New Roman"
    # load each dataset
    datasets = ['Cora', 'CiteSeer', 'PubMed']
    # split the dataset into clients
    X = np.arange(14)
    X_labels = X + 2

    # create holder figure
    nrows, ncols = 1, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5))
    colours = [plt.colormaps['Paired'].colors[1], plt.colormaps['Paired'].colors[5]]
    markers = ['x', '.']

    for i, ax in enumerate(axes.flat):
        edges_included = []
        edges_removed = []
        for n_client in X_labels:
            dataloaders, dataset_stats = create_federated_dataloaders(dataset_name=datasets[i],
                                                                      ratio_train=0.7,
                                                                      ratio_test=0.2,
                                                                      n_clients=n_client,
                                                                      partition=partition,
                                                                      batch_partition='none')

            edges_included.append(dataset_stats['included_edges'])
            edges_removed.append(dataset_stats['total_edges'] - dataset_stats['included_edges'])


        ax.plot(X, edges_included, label='Edges Included', linestyle='-',
                marker=markers[0], color=colours[0])

        ax.plot(X, edges_removed, label='Edges Removed', linestyle='-',
                marker=markers[1], color=colours[1])

        ax.plot([X[0], X[-1]], [dataset_stats['total_edges'], dataset_stats['total_edges']], "k-", label='Total Edges')
        if partition == "random":
            title = f'{datasets[i]}'
        elif partition == "random_overlapping":
            title = f'{datasets[i]} w/ overlapping\n node partitions'
        ax.set_title(title, fontsize=22)
        if i == 1:
            h, l = ax.get_legend_handles_labels()
            kw = dict(ncol=2, loc="lower center", frameon=False, fontsize=17)
            leg1 = ax.legend(h[:2], l[:2], bbox_to_anchor=[0.47, -0.36], **kw)
            leg2 = ax.legend(h[2:], l[2:], bbox_to_anchor=[0.5, -0.46], **kw)
            ax.add_artist(leg1)
        if i == 1:
            ax.set_xlabel('N Clients', fontsize=18)
        if i == 0:
            ax.set_ylabel('N Edges', fontsize=18)

        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.xaxis.set_ticks(X)
        ax.set_xticklabels(X_labels)

    fig.tight_layout()
    fig.savefig(f'{dirname(getcwd())}/ugfl/workshop_results/n_clients_v_edges_{partition}.eps', format='eps', bbox_inches='tight')


def display_dataset_stats(datasets):
    from ugfl.datasets import load_real_dataset, create_federated_dataloaders
    import networkx as nx

    for dataset in datasets:
        print(dataset)
        # load data
        data = load_real_dataset(dataset_name=dataset)
        # print easy stats
        n_clusters = len(np.unique(data.y))
        n_features = data.x.shape[1]
        n_nodes = data.x.shape[0]
        n_edges = data.edge_index.shape[1]
        print(f'Number of Features: {n_features}')
        print(f'Number of Clusters: {n_clusters}')
        print(f'Number of Nodes: {n_nodes}')
        print(f'Number of Edges: {n_edges}')

        edge_list = []
        for i in range(len(data.edge_index.tolist()[0])):
            edge_list.append((data.edge_index.tolist()[0][i], data.edge_index.tolist()[1][i]))
        nx_g = nx.Graph(edge_list)
        clustering = nx.average_clustering(nx_g)
        print(f'Clustering Coefficient: {clustering}')
        cercania = nx.closeness_centrality(nx_g)
        cercania = np.mean(list(cercania.values()))
        print(f'Closeness Centrality: {cercania}')

        dataloaders, dataset_stats = create_federated_dataloaders(dataset_name=dataset,
                                                                  ratio_train=0.7,
                                                                  ratio_test=0.2,
                                                                  n_clients=1,
                                                                  partition='random',
                                                                  batch_partition='none')
        cluster_distributions = dataset_stats['data_distributions']/np.sum(dataset_stats['data_distributions'])
        print(f'Cluster Distributions: {cluster_distributions}')

    return


def plot_label_distribution_figure():
    from ugfl.datasets import create_federated_dataloaders
    import matplotlib.pyplot as plt
    from os import getcwd
    from os.path import dirname
    import scienceplots
    plt.style.use(['science', 'ieee'])
    plt.rcParams["font.family"] = "Times New Roman"
    # load each dataset
    datasets = ['Cora', 'CiteSeer', 'PubMed']
    # create holder figure
    nrows, ncols = 3, 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5, 10))
    cmap = plt.colormaps['tab20'].colors
    cmap2 = plt.colormaps['Set3'].colors
    colour = [cmap[1], cmap[3], cmap[5], cmap[7], cmap[9], cmap[13], cmap[19]]
    for i, ax in enumerate(axes.flat):
        dataloaders, dataset_stats = create_federated_dataloaders(dataset_name=datasets[i],
                                                                  ratio_train=0.7,
                                                                  ratio_test=0.2,
                                                                  n_clients=1,
                                                                  partition='random',
                                                                  batch_partition='none')
        clusters = np.arange(len(dataset_stats['data_distributions']))

        ax.bar(clusters, dataset_stats['data_distributions']/np.sum(dataset_stats['data_distributions']),
               color=colour[:len(clusters)])
        ax.set_title(f'{datasets[i]}', fontsize=18)
        if i == 2:
            ax.set_xlabel('N Cluster', fontsize=14)
        ax.set_ylabel('% Nodes in Cluster', fontsize=14)

        ax.tick_params(axis='y', labelsize=13)
        ax.tick_params(axis='x', labelsize=13)
        ax.xaxis.set_ticks(clusters)
        ax.set_xticklabels(clusters)

    fig.tight_layout()
    fig.savefig(f'{dirname(getcwd())}/ugfl/workshop_results/label_dist.eps', format='eps', bbox_inches='tight')


def display_csize_nepochs(plotting_option='heatmap'):
    import matplotlib.pyplot as plt
    from os import getcwd
    from os.path import dirname
    # plt.rcParams["font.family"] = "Times New Roman"
    import scienceplots
    # plt.style.use(['science', 'ieee'])
    import pickle
    import numpy as np
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from mpl_toolkits import mplot3d
    import matplotlib
    matplotlib.use("macosx")

    datasets = ['Cora', 'CiteSeer']
    seeds = ['42', '66']#, '500']
    models = ['fedindv', 'fedave', 'central']
    model_name = ['Isolated', 'Federated', 'Centralised']
    n_clients = ['2_random_', '15_random_']
    n_client_title = [2, 15]
    normalizer_maxs = [0.5202, 0.5325]

    csizes = np.array([0.01, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1.0, 5., 10.])
    nepochs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    for d, dataset in enumerate(datasets):
        z_max = np.array(0.)
        nrows, ncols = 5, 1
        subplot_kw = {"projection": "3d"} if plotting_option == "3d" else {}
        figsize = (4, 8) if plotting_option == "3d" else (5, 20)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw=subplot_kw)
        axes_counter = 0

        cmap = cm.get_cmap('magma')
        normalizer = Normalize(0, normalizer_maxs[d])
        im = cm.ScalarMappable(norm=normalizer, cmap=cmap)

        for j, n_client in enumerate(n_clients):
            for k, model in enumerate(models):
                if model == 'central':
                    n_client = ""
                if j == 0 and model == 'central':
                    break

                z = np.zeros((len(seeds), len(csizes), len(nepochs)))
                for i, seed in enumerate(seeds):
                    study = pickle.load(open(f"{dirname(getcwd())}/ugfl/csize_nepochs/hpo/{model}_dmon_{dataset}_{n_client}{seed}_study", "rb"))
                    for trial in study.trials:
                        z[i,
                              np.argwhere(trial.params['cluster_size_regularization'] == csizes)[0,0],
                              np.argwhere(trial.params['n_epochs_per_round'] == nepochs)[0, 0]] = trial.user_attrs['f1_test_self']

                title = f"{dataset} {model_name[k]}"
                if model != 'central':
                    title += f' {n_client_title[j]} Clients'

                ax = axes.flat[axes_counter]
                z = np.mean(z, axis=0)
                z_max = np.array(max(np.max(z_max), np.max(z)))
                Y, X = np.meshgrid(np.linspace(0, 9, 10, dtype=int), np.linspace(0, 9, 10, dtype=int))

                if plotting_option == "heatmap":
                    # plotting
                    ax.pcolormesh(X, Y, z, cmap=cmap, norm=normalizer)
                    cords = np.argwhere(z == np.max(z))
                    ax.scatter(cords[0, 0], cords[0, 1], marker='x', color='k')
                    # ticks and labels
                    if axes_counter == 4:
                        ax.set_xlabel('Cluster Size Regularisation ' + r'$(\lambda_{nk}$)', fontsize=18)
                    ax.tick_params(axis='x', labelsize=16)
                    ax.xaxis.set_ticks(np.linspace(0, 9, 10, dtype=int))
                    ax.set_xticklabels(csizes, rotation=-45)
                    if axes_counter == 2 and dataset != "CiteSeer":
                        ax.set_ylabel('Number of Local Epochs Per Round ' + r'$(n_{r^*}$)', fontsize=20)
                    ax.tick_params(axis='y', labelsize=16)
                    ax.yaxis.set_ticks(np.linspace(0, 9, 10, dtype=int))
                    ax.set_yticklabels(nepochs)
                    ax.set_title(title, fontsize=20)

                elif plotting_option == "3d":
                    ax.plot_surface(X, Y, z, rstride=1, cstride=1,
                                           cmap=cmap, edgecolor='none', norm=normalizer)
                    #ax.set_box_aspect((1., 1., 1.), zoom=2.0)
                    ax.view_init(45, -100)
                    # ticks and labels
                    ax.set_title(title, fontsize=20)
                    ax.margins(0, 0, 0)
                    # x axis
                    ax.set_xlabel('Cluster Size Regularisation ' + r'$(\lambda_{nk}$)', fontsize=16, labelpad=7)
                    ax.xaxis.set_ticks(np.linspace(0, 9, 10, dtype=int))
                    ax.tick_params(axis='x', labelsize=16, pad=-5)
                    ax.set_xticklabels(csizes, rotation=-45)
                    # y axis
                    ax.set_ylabel('Number of Local Epochs Per Round ' + r'$(n_{r^*}$)', fontsize=16, labelpad=8)
                    ax.tick_params(axis='y', labelsize=16)
                    ax.yaxis.set_ticks(np.linspace(1, 10, 10, dtype=int))
                    ax.set_ylim(0, 10)
                    ax.set_yticklabels(nepochs)
                    # z axis
                    #ax.zaxis.set_rotate_label(False)
                    ax.set_zlabel('F1 Test', fontsize=16, labelpad=8)#, rotation=0)
                    ax.tick_params(axis='z', labelsize=15)
                    #ax.zaxis.set_label_coords(0., 0., 1.5)

                axes_counter += 1
        fig.tight_layout()
        if plotting_option == "heatmap":
            plt.subplots_adjust(bottom=0.0, hspace=0.5)
            cbar = fig.colorbar(im, ax=axes.flat.base, location='bottom',
                                shrink=0.95, aspect=7.5, anchor=(0.1, 0.13)) #anchor=(0.1, 1.75))#, pad=0., aspect=15)
            cbar.ax.tick_params(labelsize=15)
            #plt.subplots_adjust(bottom=0., hspace=0.7)
        else:
            plt.subplots_adjust(left=0.04, right=0.98, top=0.97, bottom=0.04, hspace=0.30)
            #cbar = fig.colorbar(im, ax=axes.flat.base, location='bottom',
            #                    shrink=0.95, anchor=(0.0, 0.0), aspect=7.5, pad=0.)  # , anchor=(0.0, 2.), pad=0., aspect=15)
            #cbar.ax.tick_params(labelsize=15)
        print(z_max)
        plt.subplots_adjust(bottom=0.1, hspace=0.3)
        fig.savefig(f'{dirname(getcwd())}/ugfl/figures/flat_csize_nepoch_{dataset}_{plotting_option}.pdf')


def load_file(directory, name):
    try: 
        return pickle.load(open(f"{directory}/{name}", "rb"))
    except: 
        try:
            return pickle.load(open(f"{directory}/{name}.pkl", "rb"))
        except:
            print(f"Couldn\'t find: {name}")

    return False

def get_test_perf(directory, name):
    # saved info for the results is a dictionary with f1/nmi_train, f1/nmi_self/other_val_test as the keys
    saved_info = load_file(directory, name)
    best_test_f1 = 0.
    best_test_nmi = 0.
    if saved_info:
        # takes the average across all clients, finds the best validation point, takes the test performance at that point
        best_test_f1 = np.mean(saved_info['f1_self_test'], axis=0)[np.argmax(np.mean(saved_info['f1_self_val'], axis=0))]
        best_test_nmi = np.mean(saved_info['nmi_self_test'], axis=0)[np.argmax(np.mean(saved_info['nmi_self_val'], axis=0))]
    return best_test_f1, best_test_nmi
     

def get_study_perf(directory, name):
    # saved info for the results is a dictionary with f1/nmi_train, f1/nmi_self/other_val_test as the keys
    saved_info = load_file(directory, name)
    best_test_f1 = 0.
    best_test_nmi = 0.
    if saved_info:
        # finds best trial for f1 in validation performance
        best_f1_trial = np.argmax([f1._values[0] if f1._values is not None else 0. for f1 in saved_info.trials])
        # takes the test performance of that trial
        best_test_f1 = saved_info.trials[best_f1_trial].user_attrs['f1_test_self']
        # repeats for nmi
        best_nmi_trial = np.argmax([nmi._values[1] if nmi._values is not None else 0. for nmi in saved_info.trials])
        best_test_nmi = saved_info.trials[best_nmi_trial].user_attrs['nmi_test_self']

    return best_test_f1, best_test_nmi
    

if __name__ == '__main__':
    from os import getcwd
    from os.path import dirname

    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}"
    })
    import matplotlib
    matplotlib.use("macosx")
    import matplotlib.lines as mlines
    from os import makedirs

    nature_colours = ["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747", "#9e9e9e"]

    display_dataset_stats(['Cora', "CiteSeer", "PubMed"])
    plot_label_distribution_figure()

    plot_comparison_figures(which_graph="", figsize=(12,4))
    plot_comparison_figures(which_graph="_overlapping", figsize=(12,4))
    plot_comparison_figures(which_graph="_dropfeatures", figsize=(12,4))

    plot_n_edges_on_dataset_figure(partition="random")
    plot_n_edges_on_dataset_figure(partition="random_overlapping")

    display_csize_nepochs(plotting_option="heatmap")
