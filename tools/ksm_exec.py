#!/usr/bin/env python3
"""Exec a command with kernel samepage merging enabled for it and all its children.

prctl(PR_SET_MEMORY_MERGE, 1) (Linux >= 6.4) marks the process's memory mergeable, the flag is
inherited across fork and kept across execve, so every Wine/Proton process of a Rain World instance
is covered. Identical pages across instances (the same textures, sounds and level data, loaded the
same way) are then merged by ksmd: asset sharing between separate Unity processes, done by the
kernel. Needs KSM running: `echo 1 | sudo tee /sys/kernel/mm/ksm/run`. Without it this is a no-op.
"""
import ctypes
import os
import sys

PR_SET_MEMORY_MERGE = 67
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(PR_SET_MEMORY_MERGE, 1, 0, 0, 0) != 0:
    print(f"ksm_exec: prctl failed (errno {ctypes.get_errno()}); running without KSM", file=sys.stderr)
os.execvp(sys.argv[1], sys.argv[1:])
