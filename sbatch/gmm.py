'''SCRIPT TO LAUNCH MULTIPLE JOBS ON A SLURM MACHINE'''

import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use-gpu", action="store_true", default=False,
                    help="use gpu")
parser.add_argument("--time", default="0:30:0")
args = parser.parse_args()


if args.use_gpu:
    prefix = "sbatch --time {}--ntasks=2 --gres=gpu:1 --mem=4G ./sbatch/run.sh ".format(args.time)
else:
    prefix = "sbatch --time {} --ntasks=2 --mem-per-cpu=1G ./sbatch/run.sh ".format(args.time)

i = 0
for run in range(1):
    for K in (2, 5, 10, 20, 50, 200):
        for radius in (5, 20):
            for mu in ['--no-mu', '']:
                i += 1
                script_to_run = "python main.py --d 2 --hidden-dim 16 --hidden-layers 1 --batch-size 100 --C 8 --K {} --radius {} --algo iwae --no-RP --VR VIMCO {}".format(K, radius, mu)
                print(str(i) + ': ' + script_to_run)
                subprocess.check_output(prefix + script_to_run, shell=True)

                i += 1
                script_to_run = "python main.py --d 2 --hidden-dim 16 --hidden-layers 1 --batch-size 100 --C 8 --K {} --radius {} --algo iwae --no-RP {}".format(K, radius, mu)
                print(str(i) + ': ' + script_to_run)
                subprocess.check_output(prefix + script_to_run, shell=True)

                i += 1
                script_to_run = "python main.py --d 2 --hidden-dim 16 --hidden-layers 1 --batch-size 100 --C 8 --K {} --radius {} --algo rws {}".format(K, radius, mu)
                print(str(i) + ': ' + script_to_run)
                subprocess.check_output(prefix + script_to_run, shell=True)

                if len(mu) > 0:
                    i += 1
                    script_to_run = "python main.py --d 2 --hidden-dim 16 --hidden-layers 1 --batch-size 100 --C 8 --K {} --radius {} --algo rws {} --no-sleep".format(K, radius, mu)
                    print(str(i) + ': ' + script_to_run)
                    subprocess.check_output(prefix + script_to_run, shell=True)
