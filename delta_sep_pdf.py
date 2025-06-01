# probablity distribution function based on changes between consecutive probes.

def delta_pdf(signal1, signal2):
    """ This takes the difference between two signals
    Inputs:
        singal1, 1D numpy array of size N
        signal2, 1D numpy array of size N
    Output:
        PDF
    """

    deltas = signal1 - signal2

    return deltas
