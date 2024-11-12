[![Tests](https://github.com/microsoft/mttl/actions/workflows/tests.yml/badge.svg)](https://github.com/microsoft/mttl/actions/workflows/tests.yml)

# MTTL - Multi-Task Transfer Learning

MTTL is a repository focusing on building LLMs that focus on model reusability, model recombination, and parameter-efficient fine-tuning (PEFT) techniques, particularly in the context of few-shot and zero-shot learning.

Check out our papers on ArXiv:

- [Arrow + MBC](https://arxiv.org/abs/2405.11157)
- [MHR](https://arxiv.org/abs/2211.03831)
- [Polytropon](https://arxiv.org/abs/2202.13914)

### About the papers

#### Towards Modular LLMs by Building and Reusing a Library of LoRAs (aka Arrow & MBC)

For the code that accompanies the paper _Towards Modular LLMs by Building and Reusing a Library of LoRAs_, please refer to the [Expert Library README](projects/modular_llm/README.md). This contains details on training and evaluating experts with Arrow.

#### Multi-Head Adapter Routing for Cross-Task Generalization (aka MHR)

For the code that accompanies the paper _Multi-Head Adapter Routing for Cross-Task Generalization_, please refer to [MHR-camera-ready](https://github.com/microsoft/mttl/tree/mhr-camera-ready).


## Transparency Notes

#### Intended uses

MTTL is intended for research use as described in the paper [Toward Modular LLMs by Building and Reusing a Library of LoRAs](https://arxiv.org/abs/2405.11157). MTTL performance in production environments has not been tested. Considerable testing and verification are needed before the concepts and code shared are used in production environments.

#### Evaluations

MTTL was evaluated on a selected set of standard NLP tasks, mostly on English data. Among these tasks are common-sense reasoning, question answering, and coding. The evaluation focused on zero-shot performance, supervised adaptation, and the effectiveness of different routing strategies and library constructions using models such as Phi-2 and Mistral. Complete details on evaluations can be found in the paper.

#### Limitations

MTTL is built on top of existing language models and LoRAs. MTTL is likely to inherit any biases, risks, or limitations of the constituent parts. For example, LLMs may inadvertently propagate biases present in their training data or produce harmful or inaccurate content. MTTL has been tested for English tasks and has not yet evaluated performance multilingual scenarios. Performance for multilingual or non-English tasks is not yet known. Since MTTL was evaluated on a selected set of standard NLP tasks, performance on tasks outside of evaluated tasks covered in the paper is not yet known

#### Safe and responsible use

Given that MTTL is used with LoRAs chosen or built by the user, it’s important for users to fully understand the behavior and safety of the LoRAs that they use. Users should verify both the accuracy and the safety for their specific configuration and scenario.

#### Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

#### Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
