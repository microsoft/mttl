import asyncio
import json
import os
import signal
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from time import sleep

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

from mttl.models.library.backend_engine import BlobStorageEngine
from mttl.utils import logger

got_exit_signal = False


def handle_exit_signal(signum, frame):
    logger.warning(
        f"`job_launcher.py` received exit signal {signum}, initiating graceful shutdown..."
    )
    global got_exit_signal
    got_exit_signal = True


class JobQueue:
    def __init__(self, job_file, queue_file, n_retries):
        self.job_file = job_file
        self.queue_file = queue_file
        self.queue_name = self.queue_file.split("/")[-1]
        self.queue_dir = os.path.dirname(self.queue_file)

        self.launcher_id = str(uuid.uuid4())
        self.launcher_name = f"launcher-{self.launcher_id}.txt"
        self.launcher_file = os.path.join(self.queue_dir, self.launcher_name)

        self.n_retries = n_retries
        self.blob_engine = BlobStorageEngine()

        self.storage_uri, self.container = (
            self.blob_engine._parse_repo_id_to_storage_info(self.queue_file)
        )

        self.blob_service_client = self.blob_engine._get_blob_client(
            self.queue_file, use_async=True
        )

        # create container client (same for queue and heartbeat)
        self.container_client = self.blob_service_client.get_container_client(
            self.container
        )

        # create blob client for queue and heartbeat
        self.queue_client = self.blob_service_client.get_blob_client(
            container=self.container, blob=self.queue_name
        )
        # self.queue_client.acquire_lease(
        self.heartbeat_client = self.blob_service_client.get_blob_client(
            container=self.container, blob=f"launcher-{self.launcher_id}.txt"
        )

        # keep track of dead runners
        self.dead_launchers = {}

    @property
    def has_dead_launchers(self):
        return len(self.dead_launchers) > 0

    async def close(self):
        # close all azure stuff
        await self.queue_client.close()
        await self.heartbeat_client.close()
        await self.container_client.close()
        await self.blob_service_client.close()

    async def initialize(self):
        await self.initialize_queue_file()
        await self.heartbeat_client.upload_blob(str(time.time()).encode("utf-8"))

    def queue_exists(self):
        return self.blob_engine._blob_exists(self.queue_file)

    async def download_queue(self, lease=None):

        local_filename = self.blob_engine._get_local_filepath(
            self.queue_dir, self.queue_name
        )

        # postpend `self.launcher_id` before the extension
        local_filename = Path(
            str(local_filename).replace(".json", f"_{self.launcher_id}.json")
        )
        with open(file=local_filename, mode="wb") as blob_file:
            download_stream = await self.queue_client.download_blob(lease=lease)
            data = await download_stream.readall()
            blob_file.write(data)

        return json.load(open(local_filename, "r"))

    async def upload_queue(self, python_dict, lease=None):
        buffer = json.dumps(python_dict, indent=2).encode("utf-8")
        await self.queue_client.upload_blob(buffer, overwrite=True, lease=lease)

    async def initialize_queue_file(self):
        # This function ensures the queue file is in the correct format.

        if not self.queue_exists():
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

            # Convert to list of {"doc_id": ..., "status": "queued", "retries": 0}
            tasks = [
                {
                    "doc_id": doc_id,
                    "status": "queued",
                    "retries": 0,
                    "launcher_id": None,
                }
                for doc_id in doc_ids
            ]

            await self.upload_queue(tasks)

    async def get_next_job(self):
        lease = await self.get_lease(blob_client=self.queue_client)
        if not self.queue_exists():
            raise FileNotFoundError(f"Queue file {self.queue_file} not found.")

        tasks = await self.download_queue(lease=lease)
        doc_id = None

        # Find the first job that is still queued
        for task in tasks:
            if task["status"] == "queued":
                # Mark as running
                task["status"] = "running"
                task["launcher_id"] = self.launcher_id
                doc_id = task["doc_id"]
                break

        if doc_id is None:
            # Now, let's see if there are some crashed or failed jobs we can retry
            for task in tasks:
                if task["status"] in ["crashed", "failed"]:
                    if task["retries"] < self.n_retries:
                        # Mark as running
                        task["status"] = "running"
                        task["launcher_id"] = self.launcher_id
                        task["retries"] += 1
                        doc_id = task["doc_id"]
                        break

        # upload tasks
        if doc_id is not None:
            await self.upload_queue(tasks, lease=lease)

        await lease.release()
        await asyncio.sleep(1)
        return doc_id

    async def update_job_status(self, doc_id, new_status):
        lease = await self.get_lease(blob_client=self.queue_client)
        assert lease is not None

        if not isinstance(doc_id, list):
            doc_id = [doc_id]

        if not self.blob_engine._blob_exists(self.queue_file):
            return

        tasks = await self.download_queue(lease=lease)
        blobs_to_delete = set()

        self._print_progress(tasks)

        # Update the status of the job with the given doc_id
        for task in tasks:
            if task["doc_id"] in doc_id:
                if new_status == "crashed":
                    logger.warning(f"Marking {task['doc_id']} as crashed")
                else:
                    logger.info(f"Updating status of {doc_id} to {new_status}")
                task["status"] = new_status
                if new_status != "running":
                    task["launcher_id"] = None

                if len(doc_id) == 1:
                    break

            # At the same time, delete jobs from dead launchers
            if (
                task["status"] == "running"
                and task["launcher_id"] in self.dead_launchers.keys()
            ):
                blobs_to_delete.add(task["launcher_id"])
                print(
                    f"Removing running job {task['doc_id']} from dead launcher {task['launcher_id']}"
                )
                task["status"] = "crashed"
                task["launcher_id"] = None

        for l_name in list(blobs_to_delete):
            print(f"Deleting dead launcher {l_name}")
            blob = self.dead_launchers[l_name]
            try:
                await blob.delete_blob()
            except Exception as e:
                # NOTE: My understanding is that this would not be required, as only the
                # launchers responsible to jobs *not* already marked as crashed are being deleted
                # But somehow we have a race condition here, where multiple runners are attempting to
                # delete the same blob. So we need to handle this exception.
                # My guess is that the lease has expired
                if "The specified blob does not exist" in str(e):
                    print(f"Blob {l_name} already deleted")
                else:
                    raise e

        self.dead_launchers = {}

        await self.upload_queue(tasks, lease=lease)
        await lease.release()
        await asyncio.sleep(1)

    def _print_progress(self, tasks):
        total = len(tasks)
        done = len([task for task in tasks if task["status"] == "done"])
        running = len([task for task in tasks if task["status"] == "running"])
        queued = len([task for task in tasks if task["status"] == "queued"])
        failed = len([task for task in tasks if task["status"] == "failed"])
        crashed = len([task for task in tasks if task["status"] == "crashed"])
        print(
            f"***\t\tProgress: {done}/{total} done, {running} running, {queued} queued, {failed} failed, {crashed} crashed\t\t***"
        )

    async def print_progress(self):
        while True:
            tasks = await self.download_queue()
            self._print_progress(tasks)
            await asyncio.sleep(30)

    @retry(
        stop=stop_after_attempt(100),
        wait=wait_fixed(1),
        retry=retry_if_exception(
            lambda e: "There is already a lease present." in str(e)
        ),
        reraise=True,
        before=lambda rs: (
            print(f"Lease acquisition attempt {rs.attempt_number}")
            if rs.attempt_number > 1
            else None
        ),
    )
    async def get_lease(self, blob_client):
        """
        Acquire a lease on the referenced blob before operating.
        """
        try:
            lease_client = await blob_client.acquire_lease(
                lease_duration=60, lease_id=self.launcher_id
            )
            return lease_client
        except Exception as e:
            if "There is already a lease present." in str(e):
                raise e
            else:
                print("Unknown error:", e)
                raise e

    async def heartbeat(self):
        while True:
            # No need for lease, as the launcher_id is unique to this launcher

            logger.info(f"Heartbeat for {self.launcher_id}")
            await self.heartbeat_client.upload_blob(
                str(time.time()).encode("utf-8"), overwrite=True
            )
            await asyncio.sleep(30)

    async def cleanup_dead_launchers(self):
        while True:
            repo_files = [
                f.name
                async for f in self.container_client.list_blobs()
                if "launcher" in f.name and self.launcher_id not in f.name
            ]
            for fn in repo_files:
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container, blob=fn
                )
                blob_properties = await blob_client.get_blob_properties()
                last_modified = blob_properties.last_modified
                delta = time.time() - last_modified.timestamp()

                # if file last modified more than 5 minutes ago, delete it
                if delta > 360:
                    launch_id = fn.split("launcher-")[1].split(".")[0]
                    self.dead_launchers[launch_id] = blob_client

            print(f"Tracked {len(self.dead_launchers)} dead launchers")

            await asyncio.sleep(60)


async def run_job(gpu_id, doc_id, config_file, output_dir, train_file, config_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(env.get("GPU_OFFSET", 0)) + int(gpu_id))
    command = [
        "python",
        f"{train_file}.py",
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

    async def stream_output(stream, prefix, job_name):
        job_name = str(job_name)[:5]
        job_name = job_name.ljust(5)
        try:
            async for line in stream:
                print(f"{prefix} <{job_name}>: {line.decode().strip()}")
        except asyncio.CancelledError:
            # Stream output cancelled, just return
            return

    process = await asyncio.create_subprocess_exec(
        *command,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # If this task gets cancelled, we ensure to terminate the subprocess
    try:
        await asyncio.gather(
            stream_output(process.stdout, "STDOUT", doc_id),
            stream_output(process.stderr, "STDERR", doc_id),
        )
    except asyncio.CancelledError:
        # The job was cancelled, let's terminate the subprocess if it's still running
        if process.returncode is None:
            process.terminate()
            await process.wait()
        raise

    return_code = await process.wait()
    print(f"Process finished with exit code: {return_code} for DOC_ID={doc_id}")
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
        "--train-file",
        type=str,
        required=True,
        help="Training script filename without extension",
    )
    parser.add_argument(
        "--n-retries", type=int, default=0, help="Number of times to retry a failed job"
    )
    parser.add_argument(
        "--local-km-dir",
        type=str,
        default=os.getenv("LOCAL_KM_DIR", "<na>"),
        help="Local directory for knowledge modules",
    )
    parser.add_argument(
        "--mounted-km-dir",
        type=str,
        default=os.getenv("MOUNTED_KM_DIR", "<na>"),
        help="Mounted directory for knowledge modules",
    )
    parser.add_argument(
        "--storage-account",
        type=str,
        default=os.getenv("STORAGE_ACCOUNT"),
        help="Azure Storage Account",
    )
    args = parser.parse_args()
    return args


async def main():
    args = parse_arguments()
    # iterate over args and print
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    config_file = f"configs/{args.config_id}.json"
    output_dir = f"{args.mounted_km_dir}/{args.config_id}"

    # if running locally, change the output_dir
    if os.path.exists(args.local_km_dir):
        output_dir = f"/data/lucas/{output_dir}"

    loop = asyncio.get_running_loop()
    # Register signal handlers for graceful shutdown
    loop.add_signal_handler(signal.SIGTERM, handle_exit_signal, signal.SIGTERM, None)
    loop.add_signal_handler(signal.SIGINT, handle_exit_signal, signal.SIGINT, None)

    queue_file = f"{args.storage_account}/{args.config_id}/task_queue.json"
    print(f"queue file :  {queue_file}")

    # create a folder for this on the blob
    blob_engine = BlobStorageEngine()
    blob_engine.create_repo(f"{args.storage_account}/{args.config_id}", exist_ok=True)

    job_queue = JobQueue(args.tasks_file, queue_file, args.n_retries)
    print("created blob with launcher id ", job_queue.launcher_id)
    await job_queue.initialize()

    gpu_usage = {i: 0 for i in range(args.num_gpus)}
    running_tasks = []

    monitor_task = asyncio.create_task(job_queue.print_progress())
    heartbeat_task = asyncio.create_task(job_queue.heartbeat())
    cleanup_task = asyncio.create_task(job_queue.cleanup_dead_launchers())

    while True:
        # If we received an exit signal (SIGTERM or SIGINT), shut down gracefully
        if got_exit_signal:
            logger.warning(
                "Confirmed exit signal, proceeding with graceful shutdown..."
            )

            doc_ids = [doc_id for _, _, doc_id in running_tasks]
            # Since these tasks are cancelled, mark them as crashed
            await job_queue.update_job_status(doc_ids, "crashed")

            logger.warning("Marked running jobs as crashed during exit process.")

            # Cancel all running tasks
            for t, gpu_id, doc_id in running_tasks:
                t.cancel()

            # Cancel all background tasks
            monitor_task.cancel()
            heartbeat_task.cancel()
            cleanup_task.cancel()

            # Close the job queue
            await job_queue.close()

            # Wait for all tasks to be cancelled
            done, pending = await asyncio.wait(
                [t for t, _, _ in running_tasks], return_when=asyncio.ALL_COMPLETED
            )
            running_tasks.clear()
            break

        # Assign new jobs to free GPUs
        for gpu_id in range(args.num_gpus):
            while gpu_usage[gpu_id] < args.jobs_per_gpu:
                doc_id = await job_queue.get_next_job()
                if not doc_id:
                    break
                task = asyncio.create_task(
                    run_job(
                        gpu_id,
                        doc_id,
                        config_file,
                        output_dir,
                        args.train_file,
                        args.config_id,
                    )
                )
                running_tasks.append((task, gpu_id, doc_id))
                gpu_usage[gpu_id] += 1

        if not running_tasks:
            # Before exiting, let's make sure we had a chance to cleanup dead launchers
            if job_queue.has_dead_launchers:
                await job_queue.update_job_status([], "")
            else:
                monitor_task.cancel()
                heartbeat_task.cancel()
                cleanup_task.cancel()
                print("No more jobs to process. Exiting.")
                # print progress one last time
                tasks = await job_queue.download_queue()
                job_queue._print_progress(tasks)
                await job_queue.close()
                break

        if running_tasks:
            logger.info(f"waiting for {len(running_tasks)} tasks to complete")
            done, pending = await asyncio.wait(
                [task for task, _, _ in running_tasks],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for completed_task in done:
                # Find which task completed
                for idx, (task, gpu_id, doc_id) in enumerate(running_tasks):
                    if task == completed_task:
                        gpu_usage[gpu_id] -= 1
                        running_tasks.pop(idx)
                        result_doc_id, return_code = completed_task.result()
                        # Update job status based on return code
                        if return_code == 0:
                            new_status = "done"
                        elif return_code < 0:
                            new_status = "crashed"
                        else:
                            new_status = "failed"
                        await job_queue.update_job_status(result_doc_id, new_status)
                        break


if __name__ == "__main__":
    print("starting job launcher")
    asyncio.run(main())
