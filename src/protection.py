# Masking: non-perturbative and perturbative techniques.
# Every function takes a DataFrame and returns a protected copy.
# (Generalization / global recoding by hierarchy level: see hierarchies.generalize)


# --- non-perturbative ---------------------------------------------------

def global_recoding(df, col, bins):
    # replace a numeric column with fixed bands (e.g. AGE in 5/10-year bands)
    raise NotImplementedError


def top_bottom_coding(df, col, lower=None, upper=None):
    # values below `lower` / above `upper` become "<lower" / ">upper"
    raise NotImplementedError


def local_suppression(df, qis, threshold):
    # QIs of records in combinations with frequency < threshold -> missing
    raise NotImplementedError


# --- perturbative -------------------------------------------------------

def pram(df, col, p_keep, seed):
    # categorical column: keep value with prob p_keep, else draw another category
    raise NotImplementedError


def additive_noise(df, col, sigma, seed):
    # add N(0, (sigma * std)^2) noise to a numeric column
    raise NotImplementedError


def rank_swapping(df, col, p, seed):
    # swap values between records whose ranks differ by at most p%
    raise NotImplementedError


def microaggregation(df, cols, size):
    # groups of >= `size` similar records; values replaced by the group mean
    raise NotImplementedError
