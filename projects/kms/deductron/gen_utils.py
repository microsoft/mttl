class GenerationBackend:
    backend = None

    @classmethod
    def init(cls, backend, **kwargs):
        cls.backend = backend

        if cls.backend == "vllm":
            from vllm_utils import VLLMGenerator

            VLLMGenerator(**kwargs)
        elif cls.backend == "sgl":
            from sgl_utils import SGLGenerator

            SGLGenerator(**kwargs)
        elif cls.backend == "sglp":
            from sgl_utils import SGLGeneratorParallel

            SGLGeneratorParallel(**kwargs)
        elif cls.backend == "sglc":
            from sgl_utils import SGLGeneratorClient

            SGLGeneratorClient(**kwargs)

        print(f"Generation backend initialized: {cls.backend}")

    @classmethod
    def get(cls):
        assert cls.backend is not None, "Backend not initialized"

        if cls.backend == "vllm":
            from vllm_utils import VLLMGenerator

            return VLLMGenerator.get()
        elif cls.backend == "sgl":
            from sgl_utils import SGLGenerator

            return SGLGenerator.get()
        elif cls.backend == "sglp":
            from sgl_utils import SGLGeneratorParallel

            return SGLGeneratorParallel.get()
        elif cls.backend == "sglc":
            from sgl_utils import SGLGeneratorClient

            return SGLGeneratorClient.get()
