# CPU scratch-read failure

The EGL pipeline's first CPU preflight (`60516007`, node `fc30560`) passed all
64 unit tests but failed to import `scipy.special._ufuncs` in the native fixture.
No GPU allocation started. The same SciPy import succeeded on the login node.

Diagnostic CPU job `60516153` revisited `fc30560` and attempted to read and hash
the installed SciPy shared libraries against their distribution RECORD before
importing them. `Path.read_bytes()` failed with
`BrokenPipeError: [Errno 108] Cannot send after transport endpoint shutdown`.
The [probe](probe.sh) and [output](probe_output.txt) preserve this evidence.

The retry excludes `fc30560`, adds a cheap binary-read check, and keeps all
dependency versions unchanged. On `fc30555`, all ten SciPy special-function
shared libraries matched their installed checksums. The exact storage-system
cause was not diagnosed; the observed failure was at the filesystem read.
