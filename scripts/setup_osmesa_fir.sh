#!/usr/bin/env bash
# User-local EL9 renderer for the standalone LIBERO Python; no system package changes.
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
NATIVE_ROOT="$FREQUENCY_WORK/native_osmesa"
RPM_NAME=mesa-libOSMesa-25.0.7-5.el9_7.x86_64.rpm
RPM_SHA=916f7643c9e3caee7226217214266513ef0da654301e1535b728d52ea3679837
LIB_SHA=bef74da9e93e7e9557fecd7efe35eb77f3eddcd36a6c836b49153d1f4b085bf7
mkdir -p "$NATIVE_ROOT/rpms" "$NATIVE_ROOT/root"
if [[ ! -f "$NATIVE_ROOT/rpms/$RPM_NAME" ]]; then
  curl -fsSL --retry 2 --max-time 120 \
    "https://dl.rockylinux.org/vault/rocky/9.7/CRB/x86_64/os/Packages/m/$RPM_NAME" \
    -o "$NATIVE_ROOT/rpms/$RPM_NAME"
fi
printf '%s  %s\n' "$RPM_SHA" "$NATIVE_ROOT/rpms/$RPM_NAME" | sha256sum -c -
if [[ ! -f "$NATIVE_ROOT/root/usr/lib64/libOSMesa.so.8" ]]; then
  rpm2cpio "$NATIVE_ROOT/rpms/$RPM_NAME" | cpio -id --quiet --no-absolute-filenames -D "$NATIVE_ROOT/root"
fi
printf '%s  %s\n' "$LIB_SHA" "$NATIVE_ROOT/root/usr/lib64/libOSMesa.so.8" | sha256sum -c -
# The EL9 build uses the system glibc and LLVM 20; CVMFS's OSMesa uses a different glibc.
env -u PYTHONPATH -u PYTHONHOME LD_LIBRARY_PATH="$NATIVE_ROOT/root/usr/lib64" "$LIBERO_VENV/bin/python" - <<'PY'
import ctypes
ctypes.CDLL('libOSMesa.so.8')
print('Pinned OSMesa loaded successfully')
PY
