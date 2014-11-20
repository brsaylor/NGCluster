import os

import numpy as np

from ngcluster.graph import threshold_graph
from ngcluster.cluster import graph_clusters

def main(datadir, outdir):
    os.makedirs(outdir, exist_ok=True)
    logfile = open(os.path.join(outdir, 'log.txt'), 'w')
    def log(msg):
        print(msg)
        print(msg, file=logfile)

    data = np.loadtxt(os.path.join(datadir, 'yeastEx.txt'))
    names = np.loadtxt(os.path.join(datadir, 'yeastNames.txt'),
            dtype=bytes).astype(str)

    clusters = graph_clusters(data, threshold_graph, [0.85])
    
    outdata = np.vstack((clusters, names)).transpose()

    np.savetxt(os.path.join(outdir, 'threshold_graph_clusters.txt'), outdata,
            fmt='%s')

    logfile.close()
