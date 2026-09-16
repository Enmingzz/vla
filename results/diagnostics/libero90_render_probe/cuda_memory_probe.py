import ctypes,json,os,time
from pathlib import Path
cuda=ctypes.CDLL(str(Path(os.environ['SERVER_VENV'])/'lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12'))
ptr=ctypes.c_void_p(); size=52*2**30
assert cuda.cudaMalloc(ctypes.byref(ptr),ctypes.c_size_t(size))==0
print('CUDA_READY',flush=True)
try:
    while True:
        if os.environ.get('CUDA_PROBE_BUSY')=='1':
            assert cuda.cudaMemset(ptr,0,ctypes.c_size_t(size))==0
            assert cuda.cudaDeviceSynchronize()==0
        time.sleep(.03 if os.environ.get('CUDA_PROBE_BUSY')=='1' else .2)
finally: cuda.cudaFree(ptr)
