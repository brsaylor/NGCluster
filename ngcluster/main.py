import os
import sys
import datetime

import numpy as np

from ngcluster.config import configurations
from ngcluster.graph import threshold_graph
from ngcluster.cluster import graph_clusters
from ngcluster.evaluate import aggregate_fom
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
              "  to run all configurations.\n"
              "Available configurations (see ngcluster/config.py):")
        for key, config in configurations.items():
            print("  {0}: {1}".format(key, config['description']))
        sys.exit(1)

    elif run_configs == ['all']:
        run_configs = list(configurations.keys())
        print("Running all {0} configurations: {1}"
                .format(len(run_configs), ", ".join(run_configs)))
    else:
        for key in run_configs:
            if key not in configurations:
                print("Error: '{0}' is not a valid configuration".format(key))
                sys.exit(1)

    for key in run_configs:
        config = configurations[key]
        config_outdir = os.path.join(outdir, key)
        os.makedirs(config_outdir, exist_ok=True)
        logfile = open(os.path.join(config_outdir, key + '-log.txt'), 'w')

        print("===============================================================")
        log(datetime.datetime.now().strftime('%c'))
        log("Running configuration " + key)
        log("Description: " + config['description'])

        cluster_fn, cluster_kwargs = config['cluster']
        graph_fn, graph_kwargs = config.get('graph', (None, None))
        if graph_fn:
            cluster_kwargs.update({'fn': graph_fn, 'fn_kwargs': graph_kwargs})

        log("Calculating aggregate FOM")
        fom = aggregate_fom(data, cluster_fn, [], cluster_kwargs)
        log("Aggregate FOM = {0}".format(fom))

        log("Clustering entire dataset")
        clusters = cluster_fn(data, **cluster_kwargs)

        num_clusters = clusters.max() + 1
        log("{0} clusters generated".format(num_clusters))
        total_genes = len(data)
        clustered_genes = (clusters >= 0).sum()
        log("{0} of {1} genes clustered ({2}%)"
                .format(clustered_genes, total_genes,
                    round(100 * float(clustered_genes) / total_genes)))

        clusters_outdata = np.vstack((clusters, names)).transpose()
        np.savetxt(os.path.join(config_outdir, key + '-clusters.txt'),
                clusters_outdata, fmt='%s')

        log("Plotting cluster expression levels")
        figs = plot_cluster_expression(names, data, clusters)
        save_pdf(figs, os.path.join(config_outdir, key + '-figures.pdf'))

        log("Finished running configuration {0}".format(key))
        log(datetime.datetime.now().strftime('%c'))
        print()
        logfile.close()
