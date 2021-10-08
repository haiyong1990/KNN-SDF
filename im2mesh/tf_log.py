import logging
import os

def setup_log(fname):
    VERBOSE = True
    if 'TF_LOG_SETTING' not in locals() and \
            'TF_LOG_SETTING' not in globals():
        FORMAT = '%(asctime)s %(levelname)-8s %(message)s'
        logging.basicConfig(format=FORMAT, filemode="a")
        log = logging.getLogger('tensorflow')
        if VERBOSE:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
        fpath = os.path.dirname(fname)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        fh = logging.FileHandler(fname, mode="a")
        #  fh.setLevel(logging.INFO)
        log.addHandler(fh)
        log.propagate = False
    global TF_LOG_SETTING
    TF_LOG_SETTING = 1


