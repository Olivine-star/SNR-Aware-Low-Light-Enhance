import os
import os.path as osp
import subprocess
import sys


def main():
    root = osp.dirname(osp.abspath(__file__))
    os.chdir(root)
    extra_args = sys.argv[1:]
    cmd = [sys.executable, 'train.py', '-opt', 'options/train/SDE_outdoor.yml']
    if '--launcher' not in extra_args:
        cmd.extend(['--launcher', 'none'])
    cmd.extend(extra_args)
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
