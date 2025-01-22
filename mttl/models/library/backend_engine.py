import asyncio
import datetime
import glob
import logging
import os
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    HfApi,
    create_commit,
    create_repo,
    delete_repo,
    hf_hub_download,
    preupload_lfs_files,
    snapshot_download,
)

from mttl.logging import logger
from mttl.utils import remote_login


def _try_auth(*credentials, interactive=False, scope=None):
    from azure.identity import ChainedTokenCredential

    if scope is None:
        scope = "https://management.azure.com/.default"

    try:
        cred = ChainedTokenCredential(*credentials)
        return cred
    except:
        return None
    return cred


class BackendEngine(ABC):
    """The backend engine classes implement the methods for
    interacting with the different storage backends. It should
    NOT be used directly. Use the ExpertLibrary classes instead.
    The parameter `repo_id` in BackendEngine is the repository ID
    without any prefix (az://, hf://, etc.).
    """

    @abstractmethod
    def snapshot_download(self, repo_id, allow_patterns=None):
        raise NotImplementedError

    @abstractmethod
    def create_repo(self, repo_id, repo_type, exist_ok, private=True):
        raise NotImplementedError

    @abstractmethod
    def delete_repo(self, repo_id, repo_type=None):
        raise NotImplementedError

    @abstractmethod
    def create_commit(self, repo_id, operations, commit_message):
        raise NotImplementedError

    @abstractmethod
    def preupload_lfs_files(self, repo_id, additions):
        raise NotImplementedError

    @abstractmethod
    def hf_hub_download(self, repo_id, filename):
        raise NotImplementedError

    @abstractmethod
    def login(self, token: Optional[str] = None):
        raise NotImplementedError

    @abstractmethod
    def repo_info(self, repo_id):
        raise NotImplementedError

    @abstractmethod
    def list_repo_files(self, repo_id):
        raise NotImplementedError


class HuggingfaceHubEngine(BackendEngine):
    def snapshot_download(self, repo_id, allow_patterns=None):
        return snapshot_download(repo_id, allow_patterns=allow_patterns)

    def create_repo(self, repo_id, repo_type, exist_ok, private=True):
        return create_repo(
            repo_id, repo_type=repo_type, exist_ok=exist_ok, private=private
        )

    def delete_repo(self, repo_id, repo_type=None):
        delete_repo(repo_id=repo_id, repo_type=repo_type)

    def create_commit(self, repo_id, operations, commit_message):
        return create_commit(
            repo_id, operations=operations, commit_message=commit_message
        )

    def preupload_lfs_files(self, repo_id, additions):
        return preupload_lfs_files(repo_id, additions=additions)

    def hf_hub_download(self, repo_id, filename):
        return hf_hub_download(repo_id, filename=filename)

    def repo_info(self, repo_id):
        return HfApi().repo_info(repo_id)

    def login(self, token: Optional[str] = None):
        remote_login(token=token)

    def list_repo_files(self, repo_id):
        return HfApi().list_repo_files(repo_id)


class BlobStorageEngine(BackendEngine):
    def __init__(self, token: Optional[str] = None, cache_dir: Optional[str] = None):
        """Initialize the blob storage engine. The cache directory can be
        provided as an argument or through the environment variable BLOB_CACHE_DIR.
        If no cache directory is provided, the default cache directory ~/.cache/mttl is used.
        You can provide a SAS Token as an argument when login or set the environment variable BLOB_SAS_TOKEN.

        IMPORTANT: Some special characters such as underscore "_" are not allowed in the repo_id.
        Please use dashes "-" instead. For more information on the naming recommendation, see:
        https://learn.microsoft.com/en-us/rest/api/storageservices/naming-and-referencing-containers--blobs--and-metadata
        """
        super().__init__()
        self._token: str = token
        self.azure_auth: str = True
        self.cache_dir = cache_dir
        # Quiet down the azure logging
        logging.getLogger("azure").setLevel(logging.WARNING)
        self.last_modified_cache = None

    @property
    def cache_dir(self):
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, _cache_dir: Optional[str] = None) -> None:
        """If cache_dir is not provided, get it from envvar BLOB_CACHE_DIR.
        Use the default cache directory ~/.cache/mttl if not provided."""
        if _cache_dir is not None:
            self._cache_dir = Path(_cache_dir)
        if "BLOB_CACHE_DIR" in os.environ:
            self._cache_dir = os.environ["BLOB_CACHE_DIR"]
        else:
            self._cache_dir = Path.home() / ".cache" / "mttl"

    @property
    def token(self):
        if self._token:
            return self._token

        from azure.core.credentials import AccessToken
        from azure.core.exceptions import ClientAuthenticationError
        from azure.identity import (
            AzureCliCredential,
            ChainedTokenCredential,
            CredentialUnavailableError,
            EnvironmentCredential,
            ManagedIdentityCredential,
        )

        default_identity_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
        if default_identity_client_id:
            token = _try_auth(
                ManagedIdentityCredential(client_id=default_identity_client_id)
            )
        else:
            token = _try_auth(AzureCliCredential())
        return token

    def login(self, token: Optional[str] = None):
        if token is not None:
            self._token = token
            self.azure_auth = False

    def _get_blob_client(self, repo_id, use_async=False):
        storage_uri, container = self._parse_repo_id_to_storage_info(repo_id)
        if use_async:
            blob_client = AsyncBlobServiceClient(
                storage_uri + (f"/?{self.token}" if not self.azure_auth else ""),
                credential=self.token if self.azure_auth else None,
            )
        else:
            blob_client = BlobServiceClient(
                storage_uri + (f"/?{self.token}" if not self.azure_auth else ""),
                credential=self.token if self.azure_auth else None,
            )
        return blob_client

    def _get_container_client(self, repo_id, use_async=False):
        storage_uri, container = self._parse_repo_id_to_storage_info(repo_id)
        return self._get_blob_client(repo_id, use_async).get_container_client(container)

    def _last_modified(self, repo_id: str, set_cache: bool=False) -> datetime.datetime:
        """Get the last modified date of a repository."""
        
        # if cached version exists, return cache. We want to avoid repetitive calls to the API
        if self.last_modified_cache:
            return self.last_modified_cache

        else:
            try:
                last_modified =  (
                    self._get_container_client(repo_id)
                    .get_container_properties()
                    .last_modified
                )
                if set_cache:
                    self.last_modified_cache = last_modified
                return last_modified
            except ResourceNotFoundError as error:
                raise ValueError(f"Repository {repo_id} not found") from error

    def get_repository_cache_dir(self, repo_id: str) -> Path:
        """Get the cache directory for a repository. If it doesn't exist, create it.
        The directory is based on the last modified date, a snapshot of the repository.
        """
        last_modified = self._last_modified(repo_id)
        repo_cache_dir = self.cache_dir / repo_id / last_modified.isoformat()
        os.makedirs(repo_cache_dir, exist_ok=True)
        return repo_cache_dir

    def _get_local_filepath(self, repo_id, filename) -> Path:
        repo_cache_dir = self.get_repository_cache_dir(repo_id)
        return repo_cache_dir / filename

    def _parse_repo_id_to_storage_info(self, repo_id: str) -> Tuple[str, str]:
        """Extracts storage account and container from repo_id.
        Returns the container and its connection string (with SAS token)."""
        storage_account, container = repo_id.split("/")[:2]  # split at first "/"
        # The connection string is in the format:
        # https://<storage_account>.blob.core.windows.net/?<token>
        storage_uri = f"https://{storage_account}.blob.core.windows.net/"
        return storage_uri, container

    def snapshot_download(
        self, repo_id, allow_patterns: Optional[Union[List[str], str]] = None
    ) -> str:
        """Downloads the entire repository, or a subset of files if allow_patterns is provided.
        If allow_patterns is provided, paths must match at least one pattern from the allow_patterns.

        Downloads are made concurrently to speed-up the process.
        """
        repo_files = self.list_repo_files(repo_id)

        if isinstance(allow_patterns, str):
            allow_patterns = [allow_patterns]

        if allow_patterns is None:
            filtered_files = repo_files
        else:
            filtered_files = [
                repo_file
                for repo_file in repo_files
                if any(fnmatch(repo_file, r) for r in allow_patterns)
            ]

        local_filenames = asyncio.run(
            self.async_download_blobs(repo_id, filtered_files)
        )
        return str(self.get_repository_cache_dir(repo_id))

    def create_repo(self, repo_id, repo_type=None, exist_ok=True, private=True):
        """Creates a new repository. repo_type and private are ignored for blob storage."""
        try:
            _, container = self._parse_repo_id_to_storage_info(repo_id)
            self._get_blob_client(repo_id).create_container(name=container)
        except ResourceExistsError as error:
            error_message = f"A container named `{container}` already exists!"
            if exist_ok:
                logger.warning(error_message)
            else:
                raise ValueError(error_message) from error

    def delete_repo(self, repo_id, repo_type=None):
        """Deletes a repository."""
        try:
            self._get_container_client(repo_id).delete_container()
            logger.info(f"Deleted repository {repo_id}.")
        except ResourceNotFoundError:
            logger.info(f"Container {repo_id} not found.")

    def create_commit(self, repo_id, operations, commit_message="", async_mode=True):
        asyncio.run(
            self.async_create_commit(repo_id, operations, async_mode=async_mode)
        )

    async def async_create_commit(self, repo_id, operations, async_mode=False):
        tasks = []
        upload_batch = []
        for op in operations:
            if isinstance(op, CommitOperationAdd):
                upload_batch.append(op)
            elif isinstance(op, CommitOperationCopy):
                tasks.append(
                    self._async_copy_blob(
                        source_repo_id=repo_id,
                        source_filename=op.src_path_in_repo,
                        destination_repo_id=repo_id,
                        destination_filename=op.path_in_repo,
                        overwrite=True,
                    )
                )
            elif isinstance(op, CommitOperationDelete):
                tasks.append(
                    self._async_delete_blob(
                        repo_id=repo_id,
                        filename=op.path_in_repo,
                    )
                )

        # upload blobs in batch, using async!
        await self.async_upload_blobs(
            repo_id,
            filenames=[op.path_in_repo for op in upload_batch],
            buffers=[op.path_or_fileobj for op in upload_batch],
        )

    def preupload_lfs_files(self, repo_id, additions):
        # for blob storage, these operations are done in create_commit
        pass

    def hf_hub_download(self, repo_id, filename):
        local_filename = asyncio.run(self.async_download_blobs(repo_id, filename))
        return str(local_filename)

    def repo_info(self, repo_id):
        class RepoInfo:
            pass

        repo_info = RepoInfo()
        repo_info.lastModified = self._last_modified(repo_id).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        return repo_info

    def list_repo_files(self, repo_id):
        """List all files in a repository. The files might not be downloaded locally."""
        try:
            container_client = self._get_container_client(repo_id)
            return [b.name for b in container_client.list_blobs()]
        except ResourceNotFoundError as error:
            raise ValueError(f"Repository {repo_id} not found") from error

    async def async_upload_folder(
        self,
        repo_id: str,
        folder: str,
        recursive=True,
    ):
        """Uploads a folder to the repository. The folder structure is preserved.
        If recursive is True, all files in the folder and subfolders are uploaded.
        Only works for empty repositories.
        """
        if self.list_repo_files(repo_id):
            logger.warning(
                "Pushing a folder and its content to "
                f"a non empty repository. Repository ID: {repo_id}"
            )
        folder_content = glob.glob(folder + "**/**", recursive=recursive)
        relative_file_paths = []
        buffers = []
        for content in folder_content:
            if os.path.isfile(content):
                relative_file_paths.append(os.path.relpath(content, folder))
                with open(content, "rb") as f:
                    buffers.append(f.read())

        await asyncio.gather(
            self.async_upload_blobs(repo_id, relative_file_paths, buffers)
        )
        return folder

    async def async_upload_blobs(
        self,
        repo_id: str,
        filenames: Union[List[str], str],
        buffers=None,
        overwrite=True,
    ):
        is_str = isinstance(filenames, str)
        if is_str:
            filenames = [filenames]
        if buffers is None:
            buffers = [None] * len(filenames)
        else:
            if len(buffers) != len(filenames):
                raise ValueError("Filenames and buffers must have the same length.")
            
        self._last_modified(repo_id, set_cache=True) # set the cache for last_modified

        async with self._get_blob_client(
            repo_id, use_async=True
        ) as blob_service_client:
            tasks = [
                self._async_upload_blob(blob_service_client, repo_id, filename, buffer, overwrite)
                for filename, buffer in zip(filenames, buffers)
            ]
            await asyncio.gather(*tasks)

        self.last_modified_cache = None # reset the cache

        return filenames[0] if is_str else filenames
    
    async def _async_upload_blob(self, blob_service_client, repo_id, filename, buffer=None, overwrite=True):
        storage_uri, container = self._parse_repo_id_to_storage_info(repo_id)

        blob_client = blob_service_client.get_blob_client(container=container, blob=filename)

        if buffer is not None:
            await blob_client.upload_blob(buffer, overwrite=overwrite)

        else:
            local_cache = self._get_local_filepath(repo_id, filename)

            with open(file=local_cache, mode="rb") as blob_file:
                await blob_client.upload_blob(blob_file, overwrite=overwrite)

    async def async_download_blobs(
        self, repo_id: str, filesnames: Union[List[str], str]
    ) -> str:
        is_str = isinstance(filesnames, str)
        if is_str:
            filesnames = [filesnames]

        self._last_modified(repo_id, set_cache=True) # set the cache for last_modified

        async with self._get_blob_client(
            repo_id, use_async=True
        ) as blob_service_client:
            tasks = [
                self._async_download_blob(blob_service_client, repo_id, filename)
                for filename in filesnames
            ]
            local_filesnames = await asyncio.gather(*tasks)

        self.last_modified_cache = None # reset the cache

        return local_filesnames[0] if is_str else local_filesnames


    async def _async_download_blob(self, blob_service_client, repo_id, filename):
        # already cached!
        local_filename = self._get_local_filepath(repo_id, filename)
        if local_filename.exists():
            return local_filename

        storage_uri, container = self._parse_repo_id_to_storage_info(repo_id)
        blob_client = blob_service_client.get_blob_client(
            container=container, blob=filename
        )

        os.makedirs(os.path.dirname(local_filename), exist_ok=True)
        async with open(file=local_filename, mode="wb") as blob_file:
            download_stream = await blob_client.download_blob()
            data = await download_stream.readall()
            blob_file.write(data)
        return local_filename

    async def async_copy_blobs(
        self,
        source_repo_ids,
        source_filenames,
        destination_repo_ids,
        destination_filenames,
        overwrite=True,
    ):
        inputs = [
            source_repo_ids,
            source_filenames,
            destination_repo_ids,
            destination_filenames,
        ]
        # if any input is a string, convert it to a list
        inputs = [[i] if isinstance(i, str) else i for i in inputs]

        # Check that all lists have the same length
        if not all(len(i) == len(inputs[0]) for i in inputs):
            raise ValueError("All lists must have the same length.")

        tasks = [
            self._async_copy_blob(
                source_repo_id,
                source_filename,
                destination_repo_id,
                destination_filename,
                overwrite=overwrite,
            )
            for source_repo_id, source_filename, destination_repo_id, destination_filename in zip(
                inputs[0], inputs[1], inputs[2], inputs[3]
            )
        ]
        await asyncio.gather(*tasks)

    async def _async_copy_blob(
        self,
        source_repo_id,
        source_filename,
        destination_repo_id,
        destination_filename,
        overwrite=True,
    ):
        (
            source_storage_uri,
            source_container,
        ) = self._parse_repo_id_to_storage_info(source_repo_id)
        async with self._get_blob_client(
            source_repo_id, use_async=True
        ) as blob_service_client:
            source_blob_client = blob_service_client.get_blob_client(
                container=source_container, blob=source_filename
            )
            _, destination_container = self._parse_repo_id_to_storage_info(
                destination_repo_id
            )
            destination_blob_client = blob_service_client.get_blob_client(
                container=destination_container, blob=destination_filename
            )
            await destination_blob_client.upload_blob_from_url(
                source_url=source_blob_client.url, overwrite=overwrite
            )

    async def async_delete_blobs(self, repo_id: str, filesnames: Union[List[str], str]):
        if isinstance(filesnames, str):
            filesnames = [filesnames]
        tasks = [self._async_delete_blob(repo_id, filename) for filename in filesnames]
        await asyncio.gather(*tasks)

    async def _async_delete_blob(self, repo_id, filename):
        storage_uri, container = self._parse_repo_id_to_storage_info(repo_id)
        async with self._get_blob_client(
            repo_id, use_async=True
        ) as blob_service_client:
            blob_client = blob_service_client.get_blob_client(
                container=container, blob=filename
            )
            await blob_client.delete_blob()


class LocalFSEngine(BackendEngine):
    def snapshot_download(self, repo_id, allow_patterns=None):
        return repo_id

    def create_repo(self, repo_id, repo_type, exist_ok, private=True):
        os.makedirs(repo_id, exist_ok=exist_ok)

    def delete_repo(self, repo_id, repo_type=None):
        import shutil

        shutil.rmtree(repo_id)

    def create_commit(self, repo_id, operations, commit_message):
        for op in operations:
            if type(op) == CommitOperationAdd:
                with open(os.path.join(repo_id, op.path_in_repo), "wb") as f:
                    f.write(op.path_or_fileobj.read())
            elif type(op) == CommitOperationCopy:
                import shutil

                shutil.copyfile(
                    os.path.join(repo_id, op.src_path_in_repo),
                    os.path.join(repo_id, op.path_in_repo),
                )
            elif type(op) == CommitOperationDelete:
                os.remove(os.path.join(repo_id, op.path_in_repo))

    def preupload_lfs_files(self, repo_id, additions):
        pass

    def hf_hub_download(self, repo_id, filename):
        return os.path.join(repo_id, filename)

    def login(self, token: Optional[str] = None):
        pass

    def repo_info(self, repo_id):
        # return the current time into string format
        class RepoInfo:
            pass

        repo_info = RepoInfo()
        repo_info.lastModified = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return repo_info

    def list_repo_files(self, repo_id):
        return os.listdir(repo_id)


class VirtualFSEngine(LocalFSEngine):
    repos = {}

    def snapshot_download(self, repo_id, allow_patterns=None):
        raise NotImplementedError(f"`snapshot_download` not supported for virtual FS.")

    def create_repo(self, repo_id, repo_type, exist_ok, private=True):
        if repo_id in VirtualFSEngine.repos and not exist_ok:
            raise ValueError(f"Repository {repo_id} already exists.")

        VirtualFSEngine.repos[repo_id] = {}

    def delete_repo(self, repo_id, repo_type=None):
        VirtualFSEngine.repos.pop(repo_id)

    def list_repo_files(self, repo_id):
        return list(VirtualFSEngine.repos[repo_id].keys())

    def create_commit(self, repo_id, operations, commit_message):
        for op in operations:
            if type(op) == CommitOperationAdd:
                VirtualFSEngine.repos[repo_id][
                    op.path_in_repo
                ] = op.path_or_fileobj.read()
            elif type(op) == CommitOperationCopy:
                raise NotImplementedError(
                    "Copy operation not supported for virtual FS."
                )
            elif type(op) == CommitOperationDelete:
                VirtualFSEngine.repos[repo_id].pop(op.path_in_repo)

    def hf_hub_download(self, repo_id, filename):
        import io

        return io.BytesIO(VirtualFSEngine.repos[repo_id][filename])
