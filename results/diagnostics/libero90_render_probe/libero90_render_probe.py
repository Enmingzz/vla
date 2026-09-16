import argparse, importlib.util, json, os, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--suite',default='libero_90')
p.add_argument('--task',type=int,default=0)
p.add_argument('--initial-state',type=int,default=0)
p.add_argument('--steps',type=int,default=400)
p.add_argument('--reserve-gib',type=float,default=0)
p.add_argument('--action-scale',type=float,default=.1)
a=p.parse_args()
if a.reserve_gib:
    import ctypes
    cudart_path=Path(os.environ['SERVER_VENV'])/'lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12'
    cuda=ctypes.CDLL(str(cudart_path))
    ptr=ctypes.c_void_p()
    code=cuda.cudaMalloc(ctypes.byref(ptr),ctypes.c_size_t(int(a.reserve_gib*2**30)))
    if code: raise RuntimeError('cudaMalloc returned '+str(code))
    print(json.dumps({'phase':'reserved','gib':a.reserve_gib}),flush=True)
sys.path.insert(0,str(Path(os.environ['OPENPI_DIR'])/'third_party/libero'))
spec=importlib.util.spec_from_file_location('official_libero_probe',Path(os.environ['OPENPI_DIR'])/'examples/libero/main.py')
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)
import numpy as np
from OpenGL import GL
suite=m.benchmark.get_benchmark_dict()[a.suite]()
print(json.dumps({'phase':'construct','suite':a.suite,'task':a.task,'name':suite.get_task(a.task).name,'backend':os.environ['MUJOCO_GL'],'cuda':os.environ.get('CUDA_VISIBLE_DEVICES'),'egl_device':os.environ.get('MUJOCO_EGL_DEVICE_ID')}),flush=True)
e,desc=m._get_libero_env(suite.get_task(a.task),224,37)
e.reset()
e.set_init_state(suite.get_task_init_states(a.task)[a.initial_state])
print(json.dumps({'phase':'ready','renderer':GL.glGetString(GL.GL_RENDERER).decode(),'vendor':GL.glGetString(GL.GL_VENDOR).decode(),'version':GL.glGetString(GL.GL_VERSION).decode()}),flush=True)
rng=np.random.default_rng(37)
start=time.monotonic()
for i in range(a.steps):
    action=[0.]*6+[-1.]
    if i>=10:
        action=(rng.uniform(-a.action_scale,a.action_scale,7)).tolist()
        action[-1]=-1.
    obs,reward,done,info=e.step(action)
    if i%20==0:
        print(json.dumps({'phase':'render','step':i,'seconds':time.monotonic()-start,'image_shape':list(obs['agentview_image'].shape)}),flush=True)
e.close()
if a.reserve_gib: cuda.cudaFree(ptr)
print(json.dumps({'phase':'complete','steps':a.steps,'seconds':time.monotonic()-start}),flush=True)
