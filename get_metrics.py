import glob
from collections import defaultdict
import click
import pandas
import pandas as pd


def get_task_name_from_file(result):
    task_name = result.split("/")[-2]
    # starts with task
    task = task_name.find("task")
    end_task = task_name[task:].find("_gumb")
    if end_task == -1:
        end_task = task_name[task:].find("-")
    # end_task = end_task if end_task < end_task_1 else end_task_1
    if end_task == -1:
        task_name = task_name[task:]
    else:
        task_name = task_name[task : task + end_task]
    return task_name


@click.command()
@click.argument("files", nargs=-1)
@click.option("--dataset")
@click.option("--latex", is_flag=True)
@click.option("--hps")
@click.option("--nt")
def main(files, dataset, latex, hps, nt):
    res = []
    models = []

    if dataset == "ni":
        for arg in files:
            skipped = []
            model = arg.split("/")[-2]

            result_files = glob.glob(arg + "/**/result.csv", recursive=True)
            results = []

            for result in result_files:
                data = pandas.read_csv(result)
                if "prefix" in data.columns:
                    task_name = data["prefix"].iloc[0]
                else:
                    task_name = get_task_name_from_file(result)

                zs_scores = data["zs_test_performance"]
                test_scores = data["test_performance"]
                val_scores = data["dev_performance"]

                if len(test_scores) != 3:
                    skipped.append(task_name)
                    continue

                for trial in range(len(test_scores)):
                    results.append(
                        {
                            "task_name": task_name,
                            "trial": trial,
                            "perf": test_scores[trial],
                            "val_perf": val_scores[trial],
                            "zs_perf": zs_scores[trial],
                        }
                    )

            print("Skipped tasks for", arg)
            print(",".join(skipped))

            if not results:
                continue

            models.append(model)
            res.append(pandas.DataFrame(results))
    elif dataset == "xfit":
        for arg in files:
            skipped = []
            results = []

            if not arg.endswith("/"):
                arg += "/"
            model = arg.split("/")[-2]
            result_files = glob.glob(arg + "/**/result.csv", recursive=True)
            if not result_files:
                result_files = glob.glob(arg + "/**/results.csv", recursive=True)
            if nt and len(result_files) != int(nt):
                continue
            print("Processing {:60s} {:02d} files".format(model, len(result_files)))

            for result in result_files:
                data = pandas.read_csv(result)
                if "prefix" in data.columns:
                    task_name = "_".join(data["prefix"][0].split("_")[:-2])
                else:
                    task_name = get_task_name_from_file(result)

                if "test_performance" in data.columns:
                    test_scores = data["test_performance"] * 100
                # new version is assumed
                else:
                    test_scores = (
                        data.loc[data["step"] != 0]["test/metric_perf"].dropna().values
                        * 100
                    )
                if "dev_performance" in data.columns:
                    val_scores = data["dev_performance"] * 100
                else:
                    val_scores = (
                        data.loc[data["step"] != 0]["val/metric_perf"].dropna().values
                        * 100
                    )

                test_scores = test_scores[:3]
                if len(test_scores) != 3:
                    skipped.append(task_name)
                    continue

                for trial in range(len(test_scores)):
                    results.append(
                        {
                            "task_name": task_name,
                            "trial": trial,
                            "perf": test_scores[trial],
                            "val_perf": val_scores[trial],
                        }
                    )

            if not results:
                continue

            models.append(model)
            res.append(pandas.DataFrame(results))
    elif dataset == "t0":
        for arg in files:
            print(arg)
            skipped = []
            model = arg.split("_")[-1]
            result_files = glob.glob(arg + "/**/result.csv", recursive=True)
            print()
            if not result_files:
                result_files = glob.glob(arg + "/**/results.csv", recursive=True)
            if nt and len(result_files) != int(nt):
                continue
            print("Processing {:60s} {:02d} files".format(model, len(result_files)))

            results = []
            for result in result_files:
                data = pandas.read_csv(result)
                task_name = data["prefix"][0]

                zero_shot = data.loc[data["step"] == 0]["test/acc_0shot"].dropna().values * 100
                data = data.loc[data["step"] != 0]

                # take val/acc for both as they are equal up to some random sampling of prompts
                test_scores = data["val/acc"].dropna().values * 100
                val_scores = data["val/acc"].dropna().values * 100

                if len(test_scores) != 3:
                    skipped.append(task_name)
                    continue

                for trial in range(len(test_scores)):
                    results.append(
                        {
                            "task_name": task_name,
                            "trial": trial,
                            "perf": test_scores[trial],
                            "val_perf": val_scores[trial],
                            "zs_perf": zero_shot[trial] if len(zero_shot) > 1 else zero_shot[0],
                        }
                    )

            if not results:
                continue

            models.append(model)
            res.append(pandas.DataFrame(results))

    tasks = [d["task_name"].unique() for d in res]
    print([len(t) for t in tasks])
    all_tasks = set(tasks[0])

    for task in tasks:
        all_tasks = all_tasks.intersection(set(task))
    all_tasks = sorted(list(all_tasks))

    print("Common tasks: #", len(all_tasks))

    if latex:
        ress = defaultdict(list)
        for m, r in zip(models, res):
            mean_per_task = (
                r.loc[r["task_name"].isin(all_tasks)]
                .groupby(["task_name"])
                .mean()
                .reset_index()
            )
            std_per_task = (
                r.loc[r["task_name"].isin(all_tasks)]
                .groupby(["task_name"])
                .std()
                .reset_index()
            )
            mean_per_task["std"] = std_per_task["perf"]

            for i in range(len(mean_per_task)):
                task_name = mean_per_task.iloc[i]["task_name"]
                task_name = task_name[task_name.find("_") + 1 :]
                ress[task_name].append(
                    "{:.1f}$_{{{:.1f}}}$".format(
                        mean_per_task.iloc[i]["perf"], std_per_task.iloc[i]["perf"]
                    )
                )

        for task_name, str in ress.items():
            print(
                "\\textsc{{{}}} & ".format(task_name.replace("_", " ")),
                " & ".join(str),
                "\\\\",
            )
    else:
        overall = []
        per_task = []

        for m, r in zip(models, res):
            filtered_results = r.loc[r["task_name"].isin(all_tasks)]
            # median across tasks
            agg_seed_mean = filtered_results.groupby(["trial"]).agg("mean")
            agg_seed_std = agg_seed_mean.agg("std")

            # mean of medians across seeds
            val = agg_seed_mean["val_perf"].agg("mean")
            val_std = agg_seed_std["val_perf"]

            test = agg_seed_mean["perf"].agg("mean")
            test_std = agg_seed_std["perf"]

            for task_name in all_tasks:
                task_wise_res = r.loc[r["task_name"].isin([task_name])]
                per_task.append(
                    {
                        "model": m,
                        "task": task_name,
                        "perf": task_wise_res["perf"].mean(),
                    }
                )

            overall.append({"model": m, "val": val, "val_std": val_std, "test": test, "test_std": test_std})

            if "zs_perf" in filtered_results.columns:
                zs_mean = filtered_results["zs_perf"].mean()
                overall[-1].update({"zs": zs_mean})

        pd.set_option('display.max_colwidth', None)
        print(pandas.DataFrame(per_task).pivot(index='model', columns='task', values='perf'))
        print(pandas.DataFrame(overall).sort_values("test", ascending=True))


if __name__ == "__main__":
    main()
