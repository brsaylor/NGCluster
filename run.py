#!/usr/bin/env python3

import os.path

from ngcluster.main import main

if __name__ == '__main__':
    topdir = os.path.dirname(os.path.realpath(__file__))
    datadir = os.path.join(topdir, 'data')
    outdir = os.path.join(topdir, 'output')
    main(datadir, outdir)
