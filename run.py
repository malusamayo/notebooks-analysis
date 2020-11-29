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

log = open(os.path.join(path, "log.txt"), "a")

os.chdir(path)
print("\033[96m {}\033[00m".format(
    "------------------------------------------------"))
print("\033[96m {}\033[00m".format(
    "Cleaning the original notebook..."))
# clean up the python code first
with open(filename_no_suffix + ".py", "r+") as f:
    content = ""
    for line in f:
        if line.startswith("get_ipython") or line.startswith("display("):
            line = "# " + line
        content += line
    f.seek(0)
    f.truncate()
    f.write(content)
f.close()

print("\033[96m {}\033[00m".format(
    "Running the original notebook..."))
st = time.time()
result = subprocess.run(["python", filename_no_suffix + ".py"])
ed0 = time.time()
t0 = ed0 - st
os.chdir(owd)
if result.returncode:
    print("\033[91m {}\033[00m".format("Original execution failed!"))
    log.write(filename + "\t" + "Script failed\n")
    log.close()
    exit()
print("\033[96m {}\033[00m".format(
    "Original execution completed in {:.2f} seconds.".format(t0)))

print("\033[96m {}\033[00m".format(
    "------------------------------------------------"))
print("\033[96m {}\033[00m".format(
    "Starting static analysis of the notebook..."))
st = time.time()
result = subprocess.run(
    ["node", "instrumenter.js", path + filename_no_suffix + ".py"])
ed1 = time.time()
t1 = ed1 - st
if result.returncode:
    print("\033[91m {}\033[00m".format("Static analysis failed!"))
    log.write(filename + "\t" + "Static analysis failed\n")
    log.close()
    exit()
print("\033[96m {}\033[00m".format(
    "Static analysis completed in {:.2f} seconds.".format(t1)))

os.chdir(path)
print("\033[96m {}\033[00m".format(
    "------------------------------------------------"))
print("\033[96m {}\033[00m".format(
    "Starting dynamic analysis of the notebook..."))
result = subprocess.run(["python", filename_no_suffix + "_m.py"])
ed2 = time.time()
t2 = ed2 - ed1
os.chdir(owd)
if result.returncode:
    print("\033[91m {}\033[00m".format("Dynamic analysis failed!"))
    log.write(filename + "\t" + "Dynamic analysis failed\n")
    log.close()
    exit()
print("\033[96m {}\033[00m".format(
    "Dynamic analysis completed in {:.2f} seconds.".format(t2)))

print("\033[96m {}\033[00m".format(
    "------------------------------------------------"))
print("\033[96m {}\033[00m".format("Starting generating documentation..."))
result = subprocess.run(["python", "analyzer.py", sys.argv[1]])
ed3 = time.time()
t3 = ed3 - ed2
if result.returncode:
    print("\033[91m {}\033[00m".format("Doc gen failed!"))
    log.write(filename + "\t" + "Doc gen failed\n")
    log.close()
    exit()

print("\033[96m {}\033[00m".format(
'''Generation completed in {:.2f} seconds.
 Script completed in {:.2f} seconds.
 Time fraction: [{:.2f} vs {:.2f}, {:.2f}, {:.2f}] seconds
'''
.format(t3, t1+t2+t3, t0, t1, t2, t3)))
log.write(filename + "\t" + "{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:2f}\n".format(t0, t1, t2, t3, t1+t2+t3))
log.close()
