import sys
import os
import time
import subprocess

filename = sys.argv[1].split('\\')[-1].split('/')[-1]
path = sys.argv[1].replace(filename, "")
filename_no_suffix = filename[:filename.rfind(".")]
suffix = filename[filename.rfind("."):]
owd = os.getcwd()

os.system("jupyter nbconvert --to python " + sys.argv[1])
print("\033[96m {}\033[00m".format(
    "------------------------------------------------"))
print("\033[96m {}\033[00m".format(
    "Starting static analysis of the notebook..."))
st = time.time()
result = subprocess.run(
    ["node", "instrumenter.js", path + filename_no_suffix + ".py"])
ed1 = time.time()
if result.returncode:
    exit()
print("\033[96m {}\033[00m".format(
    "Static analysis completed in {:.2f} seconds.".format(ed1 - st)))

os.chdir(path)
print("\033[96m {}\033[00m".format(
    "------------------------------------------------"))
print("\033[96m {}\033[00m".format(
    "Starting dynamic analysis of the notebook..."))
result = subprocess.run(["python", filename_no_suffix + "_m.py"])
ed2 = time.time()
os.chdir(owd)
if result.returncode:
    exit()
print("\033[96m {}\033[00m".format(
    "Dynamic analysis completed in {:.2f} seconds.".format(ed2 - ed1)))

print("\033[96m {}\033[00m".format(
    "------------------------------------------------"))
print("\033[96m {}\033[00m".format("Starting generating documentation..."))
os.system("python analyzer.py " + sys.argv[1])
ed3 = time.time()
print("\033[96m {}\033[00m".format(
    "Generation completed in {:.2f} seconds.".format(ed3 - ed2)))
print("\033[96m {}\033[00m".format(
    "Script completed in {:.2f} seconds.".format(ed3 - st)))
