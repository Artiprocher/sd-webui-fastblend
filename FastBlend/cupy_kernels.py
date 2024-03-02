import cupy as cp
import os
from typing import Tuple


class KernelManager:
    __instance = None
    __kernel_dict = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(KernelManager)
        return cls.__instance

    def __init__(self, source_dir: str = None, nvcc_flags: Tuple[str] = None):
        self.cuda_source_dir = (
            source_dir
            if source_dir
            else os.path.join(os.path.dirname(__file__), "csrc")
        )
        self.nvcc_flags = nvcc_flags if nvcc_flags else ("-std=c++14",)
        self.__kernel_dict = {}

    def AddKernel(self, *, filename, funcname, dirname=None, options=None):
        assert funcname not in self.__kernel_dict
        self.__kernel_dict[funcname] = cp.RawKernel(
            open(
                os.path.join(dirname if dirname else self.cuda_source_dir, filename)
            ).read(),
            funcname,
            options=options if options else self.nvcc_flags,
        )

        self.__setattr__(funcname,self.__kernel_dict[funcname])


