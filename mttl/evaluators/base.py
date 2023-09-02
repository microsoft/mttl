from collections import defaultdict
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


def compute_task_aggregation(task_names, metric_values):
    """Aggregates metric values per task and computes mean and stderr."""
    aggregation = defaultdict(list)

    for task_name, metric_value in zip(task_names, metric_values):
        aggregation[task_name] += [metric_value]
        aggregation["all"] += [metric_value]

    aggregation = {
        task_name: {
            "mean": mean(values),
            "stderr": mean_stderr(values),
        }
        for task_name, values in metric_values.items()
    }
    return aggregation
