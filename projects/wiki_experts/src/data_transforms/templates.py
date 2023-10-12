
from abc import abstractmethod
from mttl.dataloader.platypus_dataset_reader import InversePlatypusTemplate, PlatypusTemplate
from projects.wiki_experts.src.data_transforms.utils import INVALID_RESPONSE


class DataTransformTemplate():
    @abstractmethod
    def post_process_generation(self, output):
        pass


class QAPlatyInstructionGenerationTemplate(DataTransformTemplate):
    @classmethod
    def apply(cls, output, context=None, icl_examples=None):
        """
        We provide the context as output (response) if and only if a response is not
        present. Otherwise, we provide the context as context in addition to the previously generated response.
        """
        return InversePlatypusTemplate.apply(
            output if output is not None else context,
            input if output is not None else None,
            icl_examples,
        )

    def post_process_generation(self, output):
        return {"instruction": output}


class QAPlatyResponseGenerationTemplate(PlatypusTemplate):
    @classmethod
    def apply(cls, instruction, context=None):
        return PlatypusTemplate.apply(instruction, input=context)

    def post_process_generation(self, output):
        return {"response": output}


class OAITemplate:
    def post_process_generation(self, output):
        try:
            if "Response:" in output:
                response = output.split("Response:")[1].strip()
            else:
                raise

            instruction = output.split("Response:")[0]
            if "Instruction:" in instruction:
                instruction = instruction.split("Instruction:")[1].strip()
                data = {
                    "instruction": instruction.replace("#", ""),
                    "response": response.replace("#", ""),
                }
            else:
                raise
        except:
            data = {"instruction": INVALID_RESPONSE, "response": INVALID_RESPONSE}
        return data

    @classmethod
    def apply(cls, output, input=None, icl_examples=None):
        task_description = (
            "\nYour task is to generate clear, comprehensive, and precise instructions."
        )
        if icl_examples is not None:
            task_description += f"\n\n Here are examples of good instructions that you should imitate:\n"
            for icl_example in icl_examples:
                task_description += f"\n### Instruction:\n{icl_example}"
            task_description += "\n\n"

        task_description += "\n\nDomain context:"
        task_description += f"\n\n{output}"
        
        task_description += (
            "\nWrite an instruction suitable for the given context. "
            "Ensure it's complete, precise, and stands alone, without relying on provided context."
        )

        if icl_examples is not None:
            task_description +=  " Strive to match the style, tone, and length of the previous examples."

        task_description += "\nRemember to also provide a concise response to the generated instruction.\
            Format your ourput as follows: ### Instruction: <your instruction> ### Response: <your response>."
        return task_description
