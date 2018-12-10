'''SCRIPT TO LAUNCH MULTIPLE JOBS ON A SLURM MACHINE'''

import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use-gpu", action="store_true", default=False,
                    help="use gpu")
args = parser.parse_args()


if args.use_gpu:
    prefix = "sbatch 2:00:0 --ntasks=2 --gres=gpu:1 --mem=4G ./sbatch/run.sh "
else:
    prefix = "sbatch --time=30:0 --ntasks=2 --mem-per-cpu=1G ./sbatch/run.sh "

i = 0
for run in range(1):
    for K in (2, 5, 10, 20):
        for radius in (5, 20):
            for mu in ['--no-mu', '']:
                i += 1
                script_to_run = "python main.py --d 2 --hidden-dim 1 --hidden-layers 1 --batch-size 100 --C 8 --K {} --radius {} --algo iwae --no-RP --VR VIMCO {}".format(K, radius, mu)
                print(str(i) + ': ' + script_to_run)
                subprocess.check_output(prefix + script_to_run, shell=True)

                i += 1
                script_to_run = "python main.py --d 2 --hidden-dim 1 --hidden-layers 1 --batch-size 100 --C 8 --K {} --radius {} --algo iwae --no-RP {}".format(K, radius, mu)
                print(str(i) + ': ' + script_to_run)
                subprocess.check_output(prefix + script_to_run, shell=True)

                i += 1
                script_to_run = "python main.py --d 2 --hidden-dim 1 --hidden-layers 1 --batch-size 100 --C 8 --K {} --radius {} --algo rws {}".format(K, radius, mu)
                print(str(i) + ': ' + script_to_run)
                subprocess.check_output(prefix + script_to_run, shell=True)