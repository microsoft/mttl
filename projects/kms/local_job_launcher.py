import asyncio
import json
import os
import random
import signal
import time
from time import sleep

from mttl.utils import logger

got_exit_signal = False

# Some return codes
FINISHED = 0
DONE_FILE_EMPTY = 1
FAILED = 100
CRASHED = -100
JOB_ALREADY_STARTED = 1_000


def handle_exit_signal(signum, frame):
    logger.warning(
        f"`local_job_launcher.py` received exit signal {signum}, initiating graceful shutdown..."
    )
    global got_exit_signal
    got_exit_signal = True


class JobQueue:
    def __init__(self, job_file, output_dir, n_retries, seconds_to_crashed):
        self.job_file = job_file
        self.n_retries = n_retries
        self.output_dir = output_dir
        self.launcher_id = f"{random.randint(1000, 9999)}-{os.getpid()}"
        self.seconds_to_crashed = seconds_to_crashed

        # copy the original job file to the queue file
        # convert the queue_file to the desired format
        data = json.load(open(self.job_file, "r"))

        # If data is a dict with train/valid/test keys, flatten it.
        if isinstance(data, dict):
            doc_ids = []
            for value in data.values():
                doc_ids += value
        else:
            # If data is already a single list, just use it directly.
            doc_ids = data

        # keep track of tasks needing execution
        self.tasks = []
        self.finished = []

        # Let's already remove finished tasks to simplify things
        for task in doc_ids:
            status = self.get_run_status(task)
            if status == "finished":
                print(f"adding {task} to finished")
                self.finished.append(task)
            else:
                self.tasks.append(task)
                print(f"adding {task} to tasks")

    def task_is_done(self, task):
        done_file = os.path.join(self.output_dir, task, "done.txt")
        return os.path.exists(done_file)

    def get_return_code(self, task):
        done_file = os.path.join(self.output_dir, task, "done.txt")
        assert os.path.exists(done_file), f"Done file {done_file} does not exist"

        with open(done_file, "r") as f:
            lines = f.readlines()
            try:
                if len(lines) == 0:
                    logger.warning(
                        f"Empty done file for task {task}. Marking as finished."
                    )
                    return 0

                str_int = lines[-1].strip()
                return_code = 0 if "0" in str_int else 1
            except Exception as e:
                # copy done file to done_error{launcher}.txt
                done_error_file = os.path.join(
                    self.output_dir, task, f"done_error{self.launcher_id}.txt"
                )
                os.system(f"cp {done_file} {done_error_file}")

                print(f"Error parsing task {task} `done.txt`. Return code: {e}")
                print(f"Lines: {lines}")
                return_code = None

            return return_code

    def count_retries(self, task):
        # check how many files are named `retry_*.txt`
        task_output_dir = os.path.join(self.output_dir, task)
        retry_files = [f for f in os.listdir(task_output_dir) if f.startswith("retry_")]
        return len(retry_files)

    def get_run_status(self, task):
        task_output_dir = os.path.join(self.output_dir, task)
        if os.path.exists(task_output_dir):
            if len(os.listdir(task_output_dir)) > 0:
                if self.task_is_done(task):
                    return_code = self.get_return_code(task)

                    if return_code == 0:
                        return "finished"
                    elif return_code < 0:
                        return "crashed_rt"
                    else:
                        return "failed"

                # if directory is empty, check when it was created, and marked as crashed if more than `seconds_to_crashed` seconds have passed
                if (time.time() - self.get_timestamp(task)) > self.seconds_to_crashed:
                    return "crashed_ts"
                else:
                    return "running"
            else:
                return "queued"
        else:
            return "queued"

    def get_timestamp(self, task):
        task_output_dir = os.path.join(self.output_dir, task)
        files = os.listdir(task_output_dir)

        if len(files) > 0:
            return max(
                os.path.getmtime(os.path.join(task_output_dir, f)) for f in files
            )

    def get_next_job(self):

        while True:
            # shuffle the tasks
            doc_id = None
            for priority_status in ["queued", "crashed", "failed"]:

                tasks = self.tasks.copy()
                random.shuffle(tasks)

                for task in tasks:
                    status = self.get_run_status(task)
                    print(
                        f"task : {task} - status : {status} - priority_status : {priority_status}"
                    )

                    if status == "finished":
                        self.finished.append(task)
                        self.tasks.remove(task)
                        continue

                    if status.startswith("crashed"):
                        doc_id = task
                        # clean out the task directory
                        timestamp = self.get_timestamp(task)
                        # convert timestsamp to DD-MM-MM-YYYY HH:MM:SS
                        timestamp_str = time.strftime(
                            "%d-%m-%Y %H:%M:%S", time.localtime(timestamp)
                        )
                        current_time = time.strftime(
                            "%d-%m-%Y %H:%M:%S", time.localtime(time.time())
                        )
                        logger.warning(
                            f"Crashed job {task} ({status}): {timestamp_str} \t current time {current_time}"
                        )

                        # actually, let's print the name and timestamp of every file in the task directory
                        task_output_dir = os.path.join(self.output_dir, task)
                        task_files = os.listdir(task_output_dir)
                        for f in task_files:
                            timestamp = os.path.getmtime(
                                os.path.join(task_output_dir, f)
                            )
                            timestamp_str = time.strftime(
                                "%d-%m-%Y %H:%M:%S", time.localtime(timestamp)
                            )
                            print(f"{f} - {timestamp_str}")
                        if len(task_files) == 0:
                            print(f"Task {task} has Empty directory")

                    if status == "failed" == priority_status:
                        # check how many retries have been done
                        n_retries = self.count_retries(task)
                        if n_retries < self.n_retries:
                            doc_id = task
                            # create a file called `retry_{timestamp}.txt` to indicate that the task is being retried
                            retry_file = os.path.join(
                                self.output_dir, task, f"retry_{time.time()}.txt"
                            )
                            with open(retry_file, "a") as f:
                                f.write(f"{time.time()} - {self.launcher_id}\n")
                        else:
                            continue

                    if status.startswith(priority_status):
                        doc_id = task
                        yield doc_id, status

                print(
                    f"no more {priority_status} tasks for launcher {self.launcher_id}"
                )

            if doc_id is None:
                print("No more tasks to run. Exiting.")
                raise StopIteration

    def _print_progress(self):
        all_status = {
            "finished": len(self.finished),
            "failed": 0,
            "queued": 0,
            "crashed": 0,
            "running": 0,
        }

        for task in self.tasks:
            status = self.get_run_status(task).split("_")[0]
            all_status[status] += 1

        # build a single line string will all the statuses
        status_str = " | ".join([f"{k}: {v}" for k, v in all_status.items()])
        print(f"*** [Progress : {self.launcher_id}]   {status_str}     ***")

    async def print_progress(self):
        while True:
            self._print_progress()
            await asyncio.sleep(30)


async def run_job(
    gpu_id, doc_id, config_file, output_dir, train_script, config_id, launcher_id
):

    # print('starting job for doc_id : ', doc_id)
    # Because the file system is on the network, when creating a file one the network,
    # it may take a while for the file creation to be visible to other nodes.
    # Therefore, let's wait 15 seconds before starting the job, to see if other
    # job runners also picked this job to run. If that's the case, we forfeit this job
    # to the newest runner.
    # sleep command for io job
    await asyncio.sleep(15)
    task_output_dir = os.path.join(output_dir, doc_id)
    started_file = os.path.join(task_output_dir, f"started_{launcher_id}.txt")
    started_file_timestamp = os.path.getmtime(started_file)
    for afile in os.listdir(task_output_dir):
        if afile.startswith("started_") and afile != f"started_{launcher_id}.txt":
            other_started_file = os.path.join(task_output_dir, afile)
            other_started_file_timestamp = os.path.getmtime(other_started_file)
            if other_started_file_timestamp > started_file_timestamp:
                print(f"Another runner has started this job. Exiting.")
                return doc_id, JOB_ALREADY_STARTED

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(env.get("GPU_OFFSET", 0)) + int(gpu_id))
    command = [
        "python",
        f"{train_script}.py",
        "-c",
        config_file,
        "-k",
        f"finetune_task_name={doc_id}",
        "-k",
        f"wandb_run_name={config_id}-{doc_id}",
        "-k",
        f"output_dir={output_dir}/{doc_id}",
    ]
    print(f"Assigning DOC_ID={doc_id} to GPU_ID={gpu_id}")

    async def stream_output(stream, job_name):
        job_name = str(job_name)[:10]
        job_name = job_name.ljust(9)
        try:
            async for line in stream:
                print(f"<{job_name}>: {line.decode().strip()}")
        except asyncio.CancelledError:
            # Stream output cancelled, just return
            return

    heartbeat_file = os.path.join(output_dir, doc_id, "heartbeat.txt")

    async def doc_heartbeat(launcher_id):
        while True:
            with open(heartbeat_file, "a") as f:
                f.write(f"{time.time()} - {launcher_id}\n")
            await asyncio.sleep(30)

    # Make sure doc_id directory exists
    os.makedirs(os.path.join(output_dir, doc_id), exist_ok=True)

    heartbeat_task = asyncio.create_task(doc_heartbeat(launcher_id))

    process = await asyncio.create_subprocess_exec(
        *command,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # If this task gets cancelled, we ensure to terminate the subprocess
    try:
        await asyncio.gather(
            stream_output(process.stdout, doc_id),
            stream_output(process.stderr, doc_id),
        )
    except asyncio.CancelledError:
        # The job was cancelled, let's terminate the subprocess if it's still running
        if process.returncode is None:
            process.terminate()
            await process.wait()
        raise
    finally:
        heartbeat_task.cancel()

    return_code = await process.wait()
    print(f"Process finished with exit code: {return_code} for DOC_ID={doc_id}")

    done_file = os.path.join(output_dir, doc_id, "done.txt")
    launcher_done_file = os.path.join(output_dir, doc_id, f"done{launcher_id}.txt")

    with open(launcher_done_file, "a") as f:
        f.write(str(return_code))

    # copy the file as `done.txt`
    os.system(f"cp {launcher_done_file} {done_file}")

    return doc_id, return_code


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description="Job Launcher")
    parser.add_argument("--num-gpus", type=int, required=True, help="Number of GPUs")
    parser.add_argument("--jobs-per-gpu", type=int, required=True, help="Jobs per GPU")
    parser.add_argument(
        "--tasks-file", type=str, required=True, help="Path to the job file"
    )
    parser.add_argument("--config-id", type=str, required=True, help="Configuration ID")
    parser.add_argument(
        "--train-script",
        type=str,
        required=True,
        help="Training script filename without extension",
    )
    parser.add_argument(
        "--n-retries",
        type=int,
        default=2,
        help="Number of times to retry a crashed job",
    )
    parser.add_argument(
        "--seconds-to-crashed",
        type=int,
        default=5 * 60,
    )
    args = parser.parse_args()
    return args


class GPUMonitor:
    def __init__(self, num_gpus, jobs_per_gpu):
        self.num_gpus = num_gpus
        self.jobs_per_gpu = jobs_per_gpu
        self.gpu_usage = [0 for _ in range(num_gpus)]

    def assign_job(self):
        # Round-robin assignment: each GPU gets 1 job per round
        assert self.has_free_gpu(), "No free GPU available"
        argmin = lambda vals: min(enumerate(vals), key=lambda nv: nv[1])[0]
        gpu_id = argmin(self.gpu_usage)
        self.gpu_usage[gpu_id] += 1
        return gpu_id

    def remove_job(self, gpu_id):
        self.gpu_usage[gpu_id] -= 1

    def has_free_gpu(self):
        return any(v < self.jobs_per_gpu for v in self.gpu_usage)


async def main():
    args = parse_arguments()
    config_file = f"configs/{args.config_id}.json"
    output_dir = f"/mnt/output/kms/{args.config_id}"

    # if running locally, change the output_dir
    if os.path.exists("/data/lucas"):
        output_dir = f"/data/lucas/{output_dir}"

    os.makedirs(output_dir, exist_ok=True)
    loop = asyncio.get_running_loop()

    # Register signal handlers for graceful shutdown
    loop.add_signal_handler(signal.SIGTERM, handle_exit_signal, signal.SIGTERM, None)
    loop.add_signal_handler(signal.SIGINT, handle_exit_signal, signal.SIGINT, None)
    job_queue = JobQueue(
        args.tasks_file, output_dir, args.n_retries, args.seconds_to_crashed
    )

    gpu_monitor = GPUMonitor(args.num_gpus, args.jobs_per_gpu)
    running_tasks = []

    monitor_task = asyncio.create_task(job_queue.print_progress())
    next_job_iter = job_queue.get_next_job()
    done_training = False

    while not done_training:
        # If we received an exit signal (SIGTERM or SIGINT), shut down gracefully
        if got_exit_signal:
            logger.warning("Received exit signal, initiating graceful shutdown...")

            running_doc_ids = [doc_id for _, _, doc_id in running_tasks]
            logger.warning("All running tasks marked as crashed. Cancelling jobs.")

            # Cancel all running tasks and mark as crashed
            for t, gpu_id, doc_id in running_tasks:
                t.cancel()

            monitor_task.cancel()
            job_queue._print_progress()

            if running_tasks:
                # Wait for all tasks to be cancelled
                done, pending = await asyncio.wait(
                    [t for t, _, _ in running_tasks], return_when=asyncio.ALL_COMPLETED
                )
                running_tasks.clear()

            # delete the heartbeat file
            logger.warning("All running tasks cancelled. Exiting.")
            break

        # Round-robin assignment: each GPU gets 1 job per round
        while gpu_monitor.has_free_gpu():
            gpu_id = gpu_monitor.assign_job()
            try:
                out = next(next_job_iter)
                doc_id, previous_status = out
            except:
                done_training = True
                print(f"No more jobs to process. Training is done.")
                break

            print(f"Next job : {doc_id} - with status {previous_status}")

            if previous_status == "failed":
                # need to delete `done.txt` file
                done_file = os.path.join(output_dir, doc_id, "done.txt")
                print(f"deleting done file : {done_file}")
                try:
                    os.remove(done_file)
                except FileNotFoundError:
                    print(f"File {done_file} not found")

            # create a directory for the task ASAP
            os.makedirs(os.path.join(output_dir, doc_id), exist_ok=True)
            # create a file called `started.txt` to indicate that the task has started
            started_file = os.path.join(
                output_dir, doc_id, f"started_{job_queue.launcher_id}.txt"
            )
            with open(started_file, "a") as f:
                f.write(f"{time.time()} - {job_queue.launcher_id}\n")

            task = asyncio.create_task(
                run_job(
                    gpu_id,
                    doc_id,
                    config_file,
                    output_dir,
                    args.train_script,
                    args.config_id,
                    job_queue.launcher_id,
                )
            )
            sleep(5)
            running_tasks.append((task, gpu_id, doc_id))
            print(f"gpu usage : {gpu_monitor.gpu_usage}")

        if not running_tasks:
            monitor_task.cancel()
            print("No more jobs to process. Exiting.")
            job_queue._print_progress()
            break

        done, pending = await asyncio.wait(
            [task for task, _, _ in running_tasks],
            return_when=(
                asyncio.ALL_COMPLETED if done_training else asyncio.FIRST_COMPLETED
            ),
        )

        for completed_task in done:
            # Find which task completed
            for idx, (task, gpu_id, doc_id) in enumerate(running_tasks):
                if task == completed_task:
                    gpu_monitor.remove_job(gpu_id)
                    running_tasks.pop(idx)
                    result_doc_id, return_code = completed_task.result()
                    if return_code != JOB_ALREADY_STARTED:
                        new_return_code = job_queue.get_return_code(doc_id)
                        if new_return_code != return_code:
                            print(
                                f"Task {doc_id} has a mismatch return code: {new_return_code} != {return_code}"
                            )


if __name__ == "__main__":
    print("starting job launcher")
    asyncio.run(main())
