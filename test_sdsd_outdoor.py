import os
import os.path as osp
import subprocess
import sys


def main():
    root = osp.dirname(osp.abspath(__file__))
    os.chdir(root)
    cmd = [sys.executable, 'test_sdsd.py', '-opt', 'options/test/SDSD_outdoor.yml']
    cmd.extend(sys.argv[1:])
    raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
