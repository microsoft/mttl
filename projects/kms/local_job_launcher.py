import asyncio
import json
import os
import random
import re
import signal
import time
from os.path import exists, getmtime, join
from pathlib import Path
from time import sleep

got_exit_signal = False
output_dir = None

# Some return codes
FINISHED = 0
FAILED = 1  # e.g. python ValueError
DONE_FILE_INVALID = 100
CRASHED_PREEMPTED = -1  # e.g. preemption
CRASHED_RETURN_CODE = -6  # e.g. GPU hang
JOB_ALREADY_STARTED = 1_000
JOB_ALREADY_FINISHED = 1_001

RETURN_CODE_TO_STATUS = {
    FINISHED: "finished",
    DONE_FILE_INVALID: "failed_done_file_invalid",
    FAILED: "failed",
    CRASHED_PREEMPTED: "crashed_preempted",
    CRASHED_RETURN_CODE: "crashed_return_code",
    JOB_ALREADY_STARTED: "job_already_started",
    JOB_ALREADY_FINISHED: "job_already_finished",
}

# create a logger object that logs with the time formatted as DD-MM-YYYY HH:MM:SS
import logging

logger = logging.getLogger("local_job_launcher")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s --> %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
)
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


def format_timestamp(timestamp):
    """Format a timestamp as a human-readable string"""
    return time.strftime("%d-%m-%Y %H:%M:%S", time.localtime(timestamp))


def handle_exit_signal(signum, frame):
    """Signal handler for graceful shutdown"""
    logger.warning(
        f"`local_job_launcher.py` received exit signal {signum}, initiating graceful shutdown..."
    )
    global got_exit_signal
    got_exit_signal = True


class JobQueue:
    def __init__(self, doc_ids, output_dir, n_retries, seconds_to_crashed):
        self.n_retries = n_retries
        self.output_dir = output_dir
        self.launcher_id = f"{random.randint(1000, 9999)}-{os.getpid()}"
        self.seconds_to_crashed = seconds_to_crashed

        # keep track of tasks needing execution
        self.tasks = []
        self.finished = []

        # Let's already remove finished tasks to simplify things
        for task in doc_ids:
            status = self.get_run_status(task)
            if status == "finished":
                logger.info(f"adding {task} to finished")
                self.finished.append(task)
            else:
                self.tasks.append(task)
                logger.info(f"adding {task} to tasks ({status})")

        logger.info(
            f"Launcher {self.launcher_id} initialized with {len(self.tasks)} tasks, {len(self.finished)} finished"
        )

    def task_is_done(self, task):
        done_file = join(self.output_dir, task, "done.txt")
        return exists(done_file)

    def get_return_code(self, task):
        done_file = join(self.output_dir, task, "done.txt")

        if not exists(done_file):
            logger.warning("Missing done file for task {task}")
            return DONE_FILE_INVALID

        with open(done_file, "r") as f:
            lines = f.readlines()
            try:
                if len(lines) == 0:
                    logger.warning(f"Empty done file for task {task}")
                    return DONE_FILE_INVALID

                return_codes = lines[-1].strip()
                # parse each digit (including negative)
                return_codes = [int(x) for x in re.findall(r"-?\d", return_codes)]
                return_code = return_codes[-1]

                if FINISHED in return_codes and return_code != FINISHED:
                    logger.warning(
                        f"Task {task} has completed successfully in the past, but has a non-zero return code: {return_code}"
                    )

                # process the return code a bit
                if return_code < 0:
                    return CRASHED_RETURN_CODE
                if return_code > 0:
                    return FAILED
                if return_code == 0:
                    return FINISHED

            except Exception as e:
                # copy done file to done_error{launcher}.txt
                done_error_file = join(
                    self.output_dir, task, f"done_error{self.launcher_id}.txt"
                )
                os.system(f"cp {done_file} {done_error_file}")

                logger.error(f"Error parsing task {task} `done.txt`. Return code: {e}")
                logger.error(f"Lines: {lines}")
                return_code = DONE_FILE_INVALID

        return return_code

    def count_retries(self, task):
        # check how many files are named `retry_*.txt`
        task_output_dir = join(self.output_dir, task)
        retry_files = [f for f in os.listdir(task_output_dir) if f.startswith("retry_")]
        return len(retry_files)

    def get_run_status(self, task):
        task_output_dir = join(self.output_dir, task)
        if exists(task_output_dir):
            if len(os.listdir(task_output_dir)) > 0:
                if self.task_is_done(task):
                    return RETURN_CODE_TO_STATUS[self.get_return_code(task)]

                # if directory is empty, check when it was created, and marked
                # as crashed if more than `seconds_to_crashed` seconds have passed
                if (time.time() - self.get_timestamp(task)) > self.seconds_to_crashed:
                    return RETURN_CODE_TO_STATUS[CRASHED_PREEMPTED]
                else:
                    return "running"
            else:
                return "queued"
        else:
            return "queued"

    def get_timestamp(self, task):
        task_output_dir = join(self.output_dir, task)
        files = os.listdir(task_output_dir)

        if len(files) > 0:
            return max(getmtime(join(task_output_dir, f)) for f in files)

    def get_next_job(self):

        while True:
            # shuffle the tasks
            doc_id = None
            for priority_status in ["queued", "crashed", "failed"]:

                tasks = self.tasks.copy()
                random.shuffle(tasks)

                for task in tasks:
                    status = self.get_run_status(task)
                    logger.info(
                        f"task : {task} - status : {status} - priority_status : {priority_status}"
                    )

                    if status == "finished":
                        self.finished.append(task)
                        self.tasks.remove(task)
                        continue

                    if status.startswith("crashed"):
                        doc_id = task
                        # clean out the task directory
                        logger.warning(
                            f"Crashed job {task} ({status}): {format_timestamp(self.get_timestamp(task))}"
                        )

                        # actually, let's print the name and timestamp of every file in the task directory
                        task_output_dir = join(self.output_dir, task)
                        task_files = os.listdir(task_output_dir)
                        for f in task_files:
                            timestamp = getmtime(join(task_output_dir, f))
                            logger.info(f"{f} - {format_timestamp(timestamp)}")
                        if len(task_files) == 0:
                            logger.info(f"Task {task} has Empty directory")

                    if status.startswith("failed") and priority_status == "failed":
                        # check how many retries have been done
                        n_retries = self.count_retries(task)
                        if n_retries < self.n_retries:
                            doc_id = task
                            # create a file called `retry_{timestamp}.txt` to indicate that the task is being retried
                            retry_file = join(
                                self.output_dir, task, f"retry_{time.time()}.txt"
                            )
                            with open(retry_file, "a") as f:
                                f.write(f"{time.time()} - {self.launcher_id}\n")
                        else:
                            logger.info(f"No more retries for task {task}")
                            continue

                    if status.startswith(priority_status):
                        doc_id = task
                        yield doc_id, status

                logger.info(
                    f"no more {priority_status} tasks for launcher {self.launcher_id}"
                )

            if doc_id is None:
                logger.info("No more tasks to run. Exiting.")
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
        # TODO: one last thing that would be cool to implement:
        # when the aggregate status has not changed in over THRESHOLD_AMT, exit out
        while True:
            self._print_progress()
            await asyncio.sleep(30)


async def run_job(
    gpu_id, doc_id, output_dir, python_script, config_id, launcher_id, job_queue
):

    env = os.environ.copy()

    # instead of running experiments directly in `output_dir`, we will create a temporary directory in $HOME,
    # and copy the contents to the `output_dir` after the experiment is done
    tmp_dir = join(os.environ["HOME"], f"{doc_id}_{int(time.time() * 100) % 100_000}")
    task_output_dir = join(output_dir, doc_id)

    # TODO: pass in a `create_command` method instead
    if "train" in python_script:
        env["CUDA_VISIBLE_DEVICES"] = str(int(env.get("GPU_OFFSET", 0)) + int(gpu_id))
        command = [
            "python",
            f"{python_script}.py",
            "-c",
            f"{config_id}",
            "-k",
            f"finetune_task_name={doc_id}",
            "-k",
            f"output_dir={tmp_dir}",
        ]
        print(f"Assigning DOC_ID={doc_id} to GPU_ID={gpu_id}")
    else:
        assert "eval" in python_script
        env["CUDA_VISIBLE_DEVICES"] = str(int(env.get("GPU_OFFSET", 0)) + int(gpu_id))
        command = [
            "python",
            f"{python_script}.py",
            "-c",
            f"{doc_id}",
            "-k",
            f"output_dir={tmp_dir}",
        ]

    print(f"command : {command}")

    async def stream_output(stream, job_name):
        """Pipe the output of a stream to the console"""

        job_name = str(job_name)[:30]
        job_name = job_name.ljust(min(29, len(job_name)))
        try:
            async for line in stream:
                print(f"<{job_name}>: {line.decode().strip()}")
        except asyncio.CancelledError:
            # Stream output cancelled, just return
            return

    async def doc_heartbeat(launcher_id):
        """Periodically write to a heartbeat file so the task is labelled as `running`"""

        counter = 0
        while True:
            # Simply appending to the file does not propagate across the mounted
            # file system quickly enough. Therefore, we will create a new file
            with open(join(task_output_dir, f"heartbeat{counter}.txt"), "w") as f:
                f.write(f"{format_timestamp(time.time())} - {launcher_id}\n")

            if exists(join(task_output_dir, f"heartbeat{counter-1}.txt")):
                os.remove(join(task_output_dir, f"heartbeat{counter-1}.txt"))

            counter += 1
            await asyncio.sleep(60)

    async def check_if_task_launched_somewhere_else(launcher_id):
        """Check if the task has been launched by another launcher"""

        started_file = join(task_output_dir, f"started_{launcher_id}.txt")
        started_file_timestamp = getmtime(started_file)
        await asyncio.sleep(120)

        other_started_files = [
            f
            for f in os.listdir(task_output_dir)
            if f.startswith("started_") and f != started_file
        ]
        latest_timestamp = max(
            [getmtime(join(task_output_dir, f)) for f in other_started_files]
        )

        if latest_timestamp > started_file_timestamp:
            print(f"Another runner has started job {doc_id}. Exiting.")
            print(f"Other started file: {format_timestamp(latest_timestamp)}")
            print(f"This started file: {format_timestamp(started_file_timestamp)}")
            print(
                f"Time difference in seconds : {latest_timestamp - started_file_timestamp:.1f}"
            )
            return doc_id, JOB_ALREADY_STARTED

        return doc_id, 0

    async def check_if_task_already_done():
        """Check if the task has already been done by another launcher"""

        while True:
            status = job_queue.get_run_status(doc_id)
            if status == "finished":
                print(f"Another runner has finished job {doc_id}. Exiting.")
                return doc_id, JOB_ALREADY_FINISHED

            await asyncio.sleep(120)

    # Make sure doc_id directory exists
    os.makedirs(task_output_dir, exist_ok=True)

    # creating these tasks, added to the event loop
    heartbeat_task = asyncio.create_task(doc_heartbeat(launcher_id))
    check_task = asyncio.create_task(check_if_task_launched_somewhere_else(launcher_id))
    check_done_task = asyncio.create_task(check_if_task_already_done())

    # creating the main process running the job
    process = await asyncio.create_subprocess_exec(
        *command,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=1024 * 128,
    )

    watchers = [check_task, check_done_task]
    streams = [
        asyncio.create_task(stream_output(process.stdout, doc_id)),
        asyncio.create_task(stream_output(process.stderr, doc_id)),
    ]
    try:
        wait_for = watchers + streams

        # As long as the streaming tasks are running, run the checks
        # When the streaming tasks are done, it means the process is done, so we can stop
        while all([stream_task in wait_for for stream_task in streams]):
            done, pending = await asyncio.wait(
                wait_for, return_when=asyncio.FIRST_COMPLETED
            )
            for d in done:
                if d in watchers:
                    _, ret_code = d.result()
                    if ret_code in [JOB_ALREADY_STARTED, JOB_ALREADY_FINISHED]:
                        process.terminate()
                        await process.wait()

                        # remove temporary directory
                        os.system(f"rm -rf {tmp_dir}")
                        return doc_id, ret_code

            wait_for = pending
    except asyncio.CancelledError:
        # The job was cancelled, let's terminate the subprocess if it's still running
        if process.returncode is None:
            process.terminate()
            await process.wait()
        raise
    finally:
        heartbeat_task.cancel()
        check_task.cancel()
        check_done_task.cancel()

    return_code = await process.wait()
    print(f"Process finished with exit code: {return_code} for DOC_ID={doc_id}")

    if return_code == FINISHED:
        # copy the contents of the temporary directory to the output directory
        # actually, let's move instead of copy
        os.system(f"mv {tmp_dir}/* {task_output_dir}")

    done_file = join(output_dir, doc_id, "done.txt")
    launcher_done_file = join(output_dir, doc_id, f"done{launcher_id}.txt")

    # Keep timestamps for when the job finished in launcher file
    with open(launcher_done_file, "a") as f:
        f.write(f"{format_timestamp(time.time())} - {return_code}")

    # Append to the shared `done.txt` file
    try:
        with open(done_file, "a") as f:
            f.write(str(return_code))
    except Exception as e:
        logger.error(f"Error writing to done.txt for task {doc_id}: {e}")

    return doc_id, return_code


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description="Job Launcher")
    parser.add_argument("--num-gpus", type=int, required=True, help="Number of GPUs")
    parser.add_argument("--jobs-per-gpu", type=int, required=True, help="Jobs per GPU")
    parser.add_argument(
        "--tasks-file", type=str, required=True, help="Path to the job file"
    )
    parser.add_argument("--tasks-file-split", type=str, default=None)
    parser.add_argument("--config-id", type=str, required=True, help="Configuration ID")
    parser.add_argument(
        "--python-script",
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
        default=15 * 60,
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
    global output_dir

    args = parse_arguments()
    args.eval = "eval" in args.python_script
    output_dir = f"/mnt/output/kms/{args.config_id}"
    if args.eval:
        os.environ["CONFIG_PATH"] = "eval_configs"

    # if running locally, change the output_dir
    if exists("/data/lucas"):
        output_dir = f"/data/lucas/{output_dir}"

    # get the document ids
    data = json.load(open(args.tasks_file, "r"))

    doc_ids = []
    if args.tasks_file_split:
        doc_ids = data[args.tasks_file_split]
    else:
        for value in data.values():
            doc_ids += value

    os.makedirs(output_dir, exist_ok=True)
    loop = asyncio.get_running_loop()

    # Register signal handlers for graceful shutdown
    loop.add_signal_handler(signal.SIGTERM, handle_exit_signal, signal.SIGTERM, None)
    loop.add_signal_handler(signal.SIGINT, handle_exit_signal, signal.SIGINT, None)
    job_queue = JobQueue(doc_ids, output_dir, args.n_retries, args.seconds_to_crashed)

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
                logger.info(f"No more jobs to process. Training is done.")
                break

            logger.info(f"Next job : {doc_id} - with status {previous_status}")

            if previous_status in ["failed", "crashed_return_code"]:
                # need to delete `done.txt` file
                done_file = join(output_dir, doc_id, "done.txt")
                logger.info(f"deleting done file : {done_file} ({previous_status})")
                try:
                    os.remove(done_file)
                except FileNotFoundError:
                    new_status = job_queue.get_run_status(doc_id)
                    logger.error(
                        f"File {done_file} not found. New status is {new_status}"
                    )

            # create a directory for the task ASAP
            os.makedirs(join(output_dir, doc_id), exist_ok=True)

            # create a file called `started.txt` to indicate that the task has started
            started_file = join(
                output_dir, doc_id, f"started_{job_queue.launcher_id}.txt"
            )
            with open(started_file, "a") as f:
                f.write(f"{format_timestamp(time.time())} - {job_queue.launcher_id}\n")

            # Useful to keep track of a shared `started.txt` file across different launcher
            try:
                with open(join(output_dir, doc_id, "started.txt"), "a") as f:
                    f.write(
                        f"{format_timestamp(time.time())} - {job_queue.launcher_id}\n"
                    )
            except Exception as e:
                logger.error(f"Error writing to started.txt for task {doc_id}: {e}")

            task = asyncio.create_task(
                run_job(
                    gpu_id,
                    doc_id,
                    output_dir,
                    args.python_script,
                    args.config_id,
                    job_queue.launcher_id,
                    job_queue,
                )
            )
            sleep(1)
            running_tasks.append((task, gpu_id, doc_id))
            logger.info(f"gpu usage : {gpu_monitor.gpu_usage}")

        if not running_tasks:
            monitor_task.cancel()
            logger.info("No more jobs to process. Exiting.")
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
                            logger.error(
                                f"Task {doc_id} has a mismatch return code: {new_return_code} != {return_code}"
                            )


if __name__ == "__main__":
    print("starting job launcher")
    asyncio.run(main())
