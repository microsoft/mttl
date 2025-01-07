class GenerationBackend:
    backend = "vllm"

    @classmethod
    def init(cls, backend, **kwargs):
        cls.backend = backend
        if cls.backend == "vllm":
            from vllm_utils import VLLMGenerator

            VLLMGenerator(**kwargs)
        elif cls.backend == "sgl":
            from sgl_utils import SGLGenerator

            SGLGenerator(**kwargs)

    @classmethod
    def get(cls):
        if cls.backend == "vllm":
            from vllm_utils import VLLMGenerator

            return VLLMGenerator.get()
        elif cls.backend == "sgl":
            from sgl_utils import SGLGenerator

            return SGLGenerator.get()
