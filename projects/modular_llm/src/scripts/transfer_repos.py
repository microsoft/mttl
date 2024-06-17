import multiprocessing

from mttl.models.modifiers.expert_containers.expert_library import (
    BlobExpertLibrary,
    ExpertLibrary,
)


def copy_repo(origin_repo_id, target_repo_id):
    try:
        print(f"Copying {origin_repo_id} to {target_repo_id}")
        origin_repo = ExpertLibrary.get_expert_library(origin_repo_id)
        BlobExpertLibrary.from_expert_library(
            expert_lib=origin_repo,
            repo_id=target_repo_id,
            force=True,
            upload_aux_data=True,
        )
        print(f"Finished copying {origin_repo_id} to {target_repo_id}")
    except Exception as e:
        print(f"Error copying {origin_repo_id} to {target_repo_id}: {e}")


def main():
    """Set your environment variables HF_TOKEN and BLOB_SAS_TOKEN, and
    eddit the expert_repos list to include the repos you want to copy.
    """

    expert_repos = [
        # ("hf://<hf_user>/<old_repo_id>", "az://<storage_account>/<new_repo_id>")
        ("hf://hf-user/anexpert", "az://storage_account/anexpert"),
        ("hf://hf-user/anotherone", "az://storage_account/anotherone"),
    ]
    pool = multiprocessing.Pool()
    pool.starmap(copy_repo, expert_repos)
    pool.close()
    pool.join()
    # sync version
    # for origin_repo_id, target_repo_id in expert_repos:
    #     copy_repo(origin_repo_id, target_repo_id)


if __name__ == "__main__":
    main()
