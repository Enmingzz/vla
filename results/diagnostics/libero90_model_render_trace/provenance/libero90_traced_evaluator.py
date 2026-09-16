import json, os
from pathlib import Path
import frequency_vla.evaluator as ev
trace=Path(os.environ['FREQUENCY_TRACE_FILE'])
trace.parent.mkdir(parents=True,exist_ok=True)
def emit(row):
    with trace.open('a') as f:
        f.write(json.dumps(row)+'\n'); f.flush()
step=ev.TrackedEnv.step
def traced_step(self, action):
    import numpy as np
    index=self.tracker.current['environment_steps']
    emit({'event':'before_step','index':index,'action':np.asarray(action).tolist()})
    result=step(self,action)
    emit({'event':'after_step','index':index,'done':bool(result[2])})
    return result
ev.TrackedEnv.step=traced_step
infer=ev.CountingClient.infer
def traced_infer(self, observation):
    from OpenGL import GL
    result=infer(self,observation)
    emit({'event':'inference','index':self.tracker.current['environment_steps'],
          'actions':result['actions'].tolist(),'renderer':str(GL.glGetString(GL.GL_RENDERER))})
    return result
ev.CountingClient.infer=traced_infer
ev.main()
