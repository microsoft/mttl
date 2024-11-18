from dataclasses import dataclass

from mttl.datamodule.base import DatasetConfig, DefaultCollator, DataModule
from mttl.models.library.dataset_library import DatasetLibrary
import ast


@dataclass
class OrcaDataModuleCollator(DefaultCollator):
    def __call__(self, batch):
        sources = []
        labels = []
        for item in batch:
            if type(item["messages"]) == str:
                item["messages"] = ast.literal_eval(item["messages"])

            sources.append("You are a helpful assistant.")
            labels.append(
                self.tokenizer.apply_chat_template(item["messages"], tokenize=False)
            )
        output_batch = (
            self.prepare_inputs_for_gpt_family(sources, labels)
            if self.model_family == "gpt"
            else self.prepare_inputs_for_seq2seq_family(sources, labels)
        )

        output_batch["sources_texts"] = sources
        output_batch["labels_texts"] = labels

        return output_batch


@dataclass
class OrcaDataModule(DataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_dataset(self):
        train_dataset = DatasetLibrary.pull_dataset_with_retry(
            "microsoft/orca-agentinstruct-1M-v1"
        )[self.config.finetune_task_name]

        self.train_dataset, self.dev_dataset = self.create_train_valid_split(
            train_dataset, 0.1
        )
        self.test_dataset = self.dev_dataset

        self.print_infos()

    @property
    def collate_fn(self):
        return OrcaDataModuleCollator(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            return_tensors="pt",
            model_family=self.config.model_family,
            for_generation=self.for_generation,
        )


if __name__ == "__main__":
    config = DatasetConfig(
        model="microsoft/Phi-3-mini-4k-instruct", finetune_task_name="creative_content"
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    message = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": '\n# Adapted CMake code snippet for finding a library, its include directories, and checking for a specific version\n# Additionally, handle different library names based on the platform and check for multiple components of the library\n\n```cmake\n# Define a function to check the version of the library\nfunction(check_mylib_version)\n # Assume MYLIB_VERSION_STRING is set by some mechanism after finding the library\n if (MYLIB_VERSION_STRING VERSION_LESS "2.0")\n message(WARNING "MyLib version is too old, at least version 2.0 is required.")\n set(MYLIB_VERSION_OK FALSE PARENT_SCOPE)\n else()\n set(MYLIB_VERSION_OK TRUE PARENT_SCOPE)\n endif()\nendfunction()\n\n# Potential names of the library based on the platform\nif (WIN32)\n set(MYLIB_POSSIBLE_NAMES mylib_win)\nelseif (UNIX)\n set(MYLIB_POSSIBLE_NAMES mylib_unix mylib_unix_alt)\nelse()\n set(MYLIB_POSSIBLE_NAMES mylib)\nendif()\n\n# Find the include directory and the library\nfind_path(MYLIB_INCLUDE_DIRS NAMES mylib.h)\nfind_library(MYLIB_LIBRARIES NAMES ${MYLIB_POSSIBLE_NAMES})\n\n# Handle multiple components of the library\nfind_library(MYLIB_EXTRA_LIBRARIES NAMES mylib_extra)\n\ninclude(FindPackageHandleStandardArgs)\nfind_package_handle_standard_args(MyLib DEFAULT_MSG\n MYLIB_LIBRARIES MYLIB_INCLUDE_DIRS MYLIB_EXTRA_LIBRARIES)\n\n# After finding the library, check its version\ncheck_mylib_version()\n\nif (MYLIB_INCLUDE_DIRS AND MYLIB_LIBRARIES AND MYLIB_EXTRA_LIBRARIES AND MYLIB_VERSION_OK)\n set(MYLIB_FOUND TRUE)\nendif()\n\nmark_as_advanced(MYLIB_INCLUDE_DIRS MYLIB_LIBRARIES MYLIB_EXTRA_LIBRARIES)\n\n# Output should include:\n# - Comments explaining the code execution for the scenario where mylib.h is not found,\n# but the mylib library and its extra component are found in non-standard locations.\n# - Indicate whether `MYLIB_FOUND` would be set to `TRUE` or not.\n# - Explain the version checking logic and its impact on `MYLIB_FOUND`.\n# - Discuss the handling of different library names based on the platform.\n# - Explain the importance of finding all components of the library.\n```\n\nInput:\n# Scenario where mylib.h is not found, but the mylib library and its extra component are found in non-standard locations, and the version found is 2.1.\n',
        },
        {
            "role": "assistant",
            "content": "Output:\n\nIn the given scenario, the `find_path` command is used to search for the `mylib.h` header file. Since `mylib.h` is not found, `MYLIB_INCLUDE_DIRS` will not be set, and this will be a critical failure for the `find_package_handle_standard_args` function, which expects `MYLIB_INCLUDE_DIRS` to be set to a valid path.\n\nThe `find_library` command is used to locate the `mylib` library and its extra component. The library names are determined based on the platform. For Windows, it looks for `mylib_win`, for Unix-like systems, it looks for `mylib_unix` or `mylib_unix_alt`, and for other platforms, it defaults to `mylib`. Since the libraries are found in non-standard locations, `MYLIB_LIBRARIES` and `MYLIB_EXTRA_LIBRARIES` will be set to those locations.\n\nThe `find_package_handle_standard_args` function is called to handle the results of the find commands. It checks if the required variables (`MYLIB_LIBRARIES`, `MYLIB_INCLUDE_DIRS`, `MYLIB_EXTRA_LIBRARIES`) are set. If any of these are not found, it will print a standard error message, and `MYLIB_FOUND` will not be set to `TRUE`.\n\nAfter the find commands, the `check_mylib_version` function is called to check the version of the library. Since the version found is 2.1, which is not less than 2.0, `MYLIB_VERSION_OK` will be set to `TRUE`.\n\nHowever, despite the library and its extra component being found and the version being acceptable, `MYLIB_FOUND` will not be set to `TRUE` because `MYLIB_INCLUDE_DIRS` is not set (the header file was not found). The `MYLIB_FOUND` variable is only set to `TRUE` if all components (include directories, libraries, extra libraries) are found and the version is okay.\n\nThe version checking logic is important because it ensures that the library version meets the minimum requirements for compatibility with the project. If the version is too old, it may lack necessary features or bug fixes, and the `MYLIB_VERSION_OK` variable will be set to `FALSE`, which would prevent `MYLIB_FOUND` from being set to `TRUE`.\n\nHandling different library names based on the platform is crucial because libraries can have different naming conventions on different operating systems. This flexibility allows the CMake script to find the correct library files regardless of the platform it is running on.\n\nFinding all components of the library is important because a library typically consists of both binary files (the library itself) and header files (for including in the source code). If any component is missing, the library cannot be fully utilized, and the build process may fail. Therefore, it is essential to ensure that all parts of the library are present before proceeding with the build.",
        },
    ]

    dataset = DatasetLibrary.pull_dataset_with_retry(
        "microsoft/orca-agentinstruct-1M-v1"
    )
    # output = tokenizer.apply_chat_template(message, tokenize=False)
    datamodule = OrcaDataModule(config)
    train_dataloader = datamodule.train_dataloader()
    val_dataloder = datamodule.val_dataloader()
    for batch in val_dataloder:
        print(batch)
        breakpoint()
