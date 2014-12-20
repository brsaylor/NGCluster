import os
import sys
import datetime
from collections import OrderedDict
import json
import csv

import numpy as np
import matplotlib.pyplot as plt

from ngcluster.config import configurations, external_cluster_files
from ngcluster.graph import count_edges
from ngcluster.evaluate import (ClusterEvaluationError, aggregate_fom,
        rand_index, silhouette_widths, silhouette_stats)
from ngcluster.plot import plot_cluster_expression, save_pdf

def main(datadir, outdir, run_configs):
    """
    Run the main program.

    Parameters
    ----------
    datadir : string
        The path to the directory containing the input data files yeastEx.txt
        and yeastNames.txt.

    outdir : string
        The path to the top-level directory to which the output files will be
        saved. The output for each configuration will be stored in a
        subdirectory named with that configuration's key.

    run_configs : list of string
        A list of keys of the configurations to run (see ngcluster.config).
    """

    logfile = None
    def log(msg):
        print(msg)
        print(msg, file=logfile)

    data = np.loadtxt(os.path.join(datadir, 'yeastEx.txt'))
    names = np.loadtxt(os.path.join(datadir, 'yeastNames.txt'),
            dtype=bytes).astype(str)

    if run_configs == []:
        print("Usage:\n"
              "    python3 run.py <config1> [<config2> ...]\n"
              "  or\n"
              "    python3 run.py all\n"
              "  to run all configurations\n"
              "  or\n"
              "    python3 run.py compile\n"
              "  to compile results from all previous runs into a CSV file\n"
              "Available configurations (see ngcluster/config.py):")
        for key, config in configurations.items():
            print("  {0}: {1}".format(key, config['description']))
        sys.exit(1)

    elif run_configs == ['all']:
        run_configs = list(configurations.keys())
        print("Running all {0} configurations: {1}"
                .format(len(run_configs), ", ".join(run_configs)))
    elif run_configs == ['compile']:
        run_configs = []
    else:
        for key in run_configs:
            if key not in configurations:
                print("Error: '{0}' is not a valid configuration".format(key))
                sys.exit(1)

    external_clusterings = [
            (filename,
                load_external_clusters(names, os.path.join(datadir, filename)))
            for filename in external_cluster_files]

    for key in run_configs:
        config = configurations[key]
        config_outdir = os.path.join(outdir, key)
        os.makedirs(config_outdir, exist_ok=True)
        logfile = open(os.path.join(config_outdir, key + '-log.txt'), 'w')

        print("===============================================================")
        log(datetime.datetime.now().strftime('%c'))
        log("Running configuration " + key)
        log(str(config))

        cluster_fn, cluster_kwargs = config['cluster']
        graph_fn, graph_kwargs = config.get('graph', (None, {}))

        # Output data to be stored in a JSON file for later compilation into a
        # table for comparing the results of the various configurations
        outdict = OrderedDict([
            ('key', key),
            ('graph', graph_fn.__name__ if graph_fn else ''),
            ('metric', graph_kwargs.get('metric', '')),
            ('graph_threshold', graph_kwargs.get('threshold', '')),
            ('cluster', cluster_fn.__name__),
            ('k', cluster_kwargs.get('k', '')),
            ('cluster_threshold', cluster_kwargs.get('threshold', '')),
            ('max_clusters', cluster_kwargs.get('max_clusters', '')),
            ('iterations', cluster_kwargs.get('iterations', '')),
            ])

        log("Calculating aggregate FOM")
        try:
            fom = aggregate_fom(data,
                    graph_fn, graph_kwargs, cluster_fn, cluster_kwargs)
            log("Aggregate FOM = {0}".format(fom))
            outdict['aggregate_fom'] = fom
            pass
        except ClusterEvaluationError as e:
            log("Cannot calculate aggregate FOM: {0}".format(e))

        log("Clustering entire dataset")
        if graph_fn is None:
            # Do non-graph-based clustering
            outdict['edges_to_nodes_ratio'] = ''
            log("Computing clusters")
            clusters = cluster_fn(data, **cluster_kwargs)
        else:
            # Do graph-based clustering
            log("Computing graph")
            adj = graph_fn(data, **graph_kwargs)
            edges_to_nodes_ratio = float(count_edges(adj)) / data.shape[0]
            log("Edges-to-nodes ratio = {}".format(edges_to_nodes_ratio))
            outdict['edges_to_nodes_ratio'] = edges_to_nodes_ratio

            log("Computing clusters")
            clusters = cluster_fn(adj, **cluster_kwargs)

        num_clusters = int(clusters.max() + 1)
        log("{0} clusters generated".format(num_clusters))
        outdict['num_clusters'] = num_clusters
        if num_clusters <= 0:
            log("Error: There are no clusters. Skipping configuration")
            continue
        total_genes = len(data)
        clustered_genes = int((clusters >= 0).sum())
        clustered_genes_pct = round(100 * float(clustered_genes) / total_genes)
        log("{0} of {1} genes clustered ({2}%)"
                .format(clustered_genes, total_genes, clustered_genes_pct))
        outdict['clustered_genes'] = clustered_genes
        outdict['clustered_genes_pct'] = clustered_genes_pct

        clusters_outdata = np.vstack((names, clusters)).transpose()
        np.savetxt(os.path.join(config_outdir, key + '-clusters.txt'),
                clusters_outdata, fmt='%s')

        log("\nSilhouette statistics:")
        log("{:11} {:>13} {:>9} {:>9}".format(
            "metric", "weighted_mean", "min",  "max"))
        for metric in 'euclidean', 'correlation', 'cosine':
            widths = silhouette_widths(clusters, data, metric)
            stats, summary = silhouette_stats(clusters, widths)
            log("{:11} {:13.3f} {:9.3f} {:9.3f}".format(metric,
                summary['weighted_mean'], summary['min'], summary['max']))
            outdict['silhouette_' + metric + '_weighted_mean'] = (
                    summary['weighted_mean'])
            outdict['silhouette_' + metric + '_min'] = summary['min']
            outdict['silhouette_' + metric + '_max'] = summary['max']

            np.savetxt(
                    os.path.join(
                        config_outdir,
                        "{0}-silhouette-{1}.txt".format(key, metric)),
                    stats,
                    header=' '.join(stats.dtype.names),
                    fmt="%d %3d %6.3f %6.3f %6.3f",
                    comments='')

        log("\nCluster size:")
        log("{:>8} {:>8} {:>8}".format("mean", "min", "max"))
        cluster_size_mean = stats['count'].mean()
        cluster_size_min = stats['count'].min()
        cluster_size_max = stats['count'].max()
        log("{:8.2f} {:8d} {:8d}".format(
            cluster_size_mean, cluster_size_min, cluster_size_max))
        log('')
        outdict['cluster_size_mean'] = cluster_size_mean
        outdict['cluster_size_min'] = int(cluster_size_min)
        outdict['cluster_size_max'] = int(cluster_size_max)

        for ext_filename, ext_clusters in external_clusterings:

            # Only consider genes that are clustered in both clusterings
            ext_clusters = ext_clusters.copy()
            ext_clusters[clusters < 0] = -1
            int_clusters = clusters.copy()
            int_clusters[ext_clusters < 0] = -1

            rand_index_val = rand_index(int_clusters, ext_clusters)
            log("Rand index = {0} ({1})".format(rand_index_val, ext_filename))
            outdict['rand_' + ext_filename] = rand_index_val

        log("Plotting cluster expression levels")
        figs = plot_cluster_expression(names, data, clusters)
        #save_pdf(figs, os.path.join(config_outdir, key + '-figures.pdf'))
        for i, fig in enumerate(figs):
            fig.savefig(os.path.join(config_outdir, key + '-cluster-{0}.png'
                .format(i)))
        plt.close('all')

        log("Finished running configuration {0}".format(key))
        log(datetime.datetime.now().strftime('%c'))
        print()
        logfile.close()

        with open(os.path.join(config_outdir, 'results.json'), 'w') as fp:
            json.dump(outdict, fp, indent=4)

    compile_results(outdir)

def compile_results(outdir):
    """
    Read the results.json file in the output directory of each configuration,
    and combine the data into a CSV file in the top-level output directory.

    Parameters
    ----------
    outdir : string
        The output directory path
    """

    os.makedirs(outdir, exist_ok=True)

    csvfilename = os.path.join(outdir, 'compiled-results.csv')
    print("Compiling all results (including previous runs) into " + csvfilename)
    csvfile = open(csvfilename, 'w')
    writer = None
    
    for key in configurations:
        try:
            with open(os.path.join(outdir, key, 'results.json')) as fp:
                rowdict = json.load(fp, object_pairs_hook=OrderedDict)
            print("OK                      " + key)
        except FileNotFoundError:
            print("results.json not found: " + key)
            continue
        if writer is None:
            writer = csv.DictWriter(csvfile, fieldnames=rowdict.keys())
            writer.writeheader()
        writer.writerow(rowdict)

    csvfile.close()

def load_external_clusters(names, filename):
    """
    Load cluster assignments from an external file.

    Parameters
    ----------
    names : ndarray
        The array of gene names.

    filename : string
        The full path to the file to load. The file should be a text file with
        two columns delimited by whitespace. The first column should contain the
        names of the clustered genes, and the second column should contain
        integer cluster IDs or arbitrary cluster labels. It is an error to mix
        the two. In the case of integer IDs, a negative value indicates that the
        corresponding gene is not in a cluster.

    Returns
    -------
    ndarray
        An array of cluster assignments in the same format as returned by the
        functions in ngcluster.cluster.
    """

    clusters = np.empty(len(names), dtype=int)
    clusters.fill(-1)

    # Map gene names to their positions in the data
    gene_id_lookup = {name: i for i, name in enumerate(names)}

    # Given a cluster label or cluster ID, return an integer cluster ID
    label_type = None
    cluster_id_lookup = {}
    next_cluster_id = 0
    def get_cluster_id(label):
        nonlocal label_type, cluster_id_lookup, next_cluster_id
        if label_type is None:
            try:
                cluster_id = int(label)
                label_type = int
            except ValueError:
                label_type = str
        if label_type is int:
            cluster_id = int(label)
        elif label not in cluster_id_lookup:
            cluster_id = next_cluster_id
            cluster_id_lookup[label] = cluster_id
            next_cluster_id += 1
        else:
            cluster_id = cluster_id_lookup[label]

        return cluster_id

    with open(filename, 'r') as f:
        for line in f:
            gene_name, cluster_label = line.split(maxsplit=1)
            if gene_name not in names:
                continue
            clusters[gene_id_lookup[gene_name]] = get_cluster_id(cluster_label)

    return clusters
