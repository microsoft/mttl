import math


def mean(arr):
    return sum(arr) / len(arr)


def pop_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / len(arr))


def sample_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))


def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))
