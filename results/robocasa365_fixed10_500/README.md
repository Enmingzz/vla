# Superseded submission: no GPU work ran

The first CPU installation/download job 60905099 finished those steps but its
configuration check incorrectly expected physical-width normalization arrays.
The official checkpoint stores neutral padding to 32 dimensions. The check was
corrected without changing the model, statistics, transforms or task protocol.

GPU jobs 60905771, 60905772 and 60905773 were cancelled while still pending;
none ran or consumed GPU allocation. `submission_sources.json` preserves the
original submitted source hashes. The corrected submission uses a fresh archive.
