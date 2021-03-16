import sys
import os
import time
import subprocess
import argparse
import shutil

parser = argparse.ArgumentParser(description='Generate documentation for notebooks')
parser.add_argument('notebook', help='the notebook to be analyzed')
parser.add_argument('-c', '--clean', help='clean intermediate files', action="store_true")
parser.add_argument('-s', '--skip', help='skip execution of original script', action="store_true")
parser.add_argument('-k', '--keep', help='keep output of scripts', action="store_true")

args = parser.parse_args()

filename = args.notebook.split('\\')[-1].split('/')[-1]
path = args.notebook.replace(filename, "")
filename_no_suffix = filename[:filename.rfind(".")]
suffix = filename[filename.rfind("."):]
owd = os.getcwd()

log = open(os.path.join(path, "log.txt"), "a")

t = [0, 0, 0, 0]

def my_print(msg):
    print("\033[96m {}\033[00m".format(msg))

def execute_script():
    if args.skip:
        return
    os.chdir(path)
    my_print('-'*40)
    my_print("Cleaning the original notebook...")
    # clean up the python code first
    with open(filename_no_suffix + ".py", "r+", encoding='utf-8') as f:
        content = ""
        if not args.keep:
            content += "import os, sys, matplotlib \nsys.stdout = open(os.devnull, \"w\")\nmatplotlib.use('Agg')\n"
        for line in f:
            if line.startswith("get_ipython") or line.startswith("display("):
                line = "# " + line
            content += line
        f.seek(0)
        f.truncate()
        f.write(content)

    my_print("Running the original notebook...")
    st = time.time()
    result = subprocess.run(["python", filename_no_suffix + ".py"])
    ed0 = time.time()
    t[0] = ed0 - st
    os.chdir(owd)
    if result.returncode:
        print("\033[91m {}\033[00m".format("Original execution failed!"))
        log.write(filename + "\t" + "Script failed\n")
        log.close()
        exit()
    print("\033[96m {}\033[00m".format(
        "Original execution completed in {:.2f} seconds.".format(t[0])))

def static_analysis():
    print("\033[96m {}\033[00m".format('-'*40))
    print("\033[96m {}\033[00m".format(
        "Starting static analysis of the notebook..."))
    st = time.time()
    result = subprocess.run(
        ["node", "instrumenter.js", path + filename_no_suffix + ".py"])
    ed1 = time.time()
    t[1] = ed1 - st
    if result.returncode:
        print("\033[91m {}\033[00m".format("Static analysis failed!"))
        log.write(filename + "\t" + "Static analysis failed\n")
        log.close()
        exit()
    print("\033[96m {}\033[00m".format(
        "Static analysis completed in {:.2f} seconds.".format(t[1])))

def dynamic_analysis():
    os.chdir(path)
    print("\033[96m {}\033[00m".format('-'*40))
    print("\033[96m {}\033[00m".format(
        "Starting dynamic analysis of the notebook..."))
    st = time.time()
    result = subprocess.run(["python", filename_no_suffix + "_m.py"])
    ed2 = time.time()
    t[2] = ed2 - st
    os.chdir(owd)
    if result.returncode:
        print("\033[91m {}\033[00m".format("Dynamic analysis failed!"))
        log.write(filename + "\t" + "Dynamic analysis failed\n")
        log.close()
        exit()
    print("\033[96m {}\033[00m".format(
        "Dynamic analysis completed in {:.2f} seconds.".format(t[2])))

def analyze():
    print("\033[96m {}\033[00m".format('-'*40))
    print("\033[96m {}\033[00m".format("Starting generating documentation..."))
    st = time.time()
    result = subprocess.run(["python", "analyzer.py", args.notebook])
    ed3 = time.time()
    t[3] = ed3 - st
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
    .format(t[3], t[1]+t[2]+t[3], t[0], t[1], t[2], t[3])))
    log.write(filename + "\t" + "{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(t[0], t[1], t[2], t[3], t[1]+t[2]+t[3]))
    log.close()

def clean():
    try:
        shutil.rmtree(os.path.join(path, filename_no_suffix))
    except FileNotFoundError:
        print("no previous folder")
    # os.system("rm " + os.path.join(path, filename_no_suffix, "\*.json"))
    # os.system("rm " + os.path.join(path, filename_no_suffix, ".json"))
    # subprocess.run(["rmdir", os.path.join(path, filename_no_suffix)])
    if not args.clean:
        return
    subprocess.run(["rm", os.path.join(path, filename_no_suffix + ".py")])
    subprocess.run(["rm", os.path.join(path, filename_no_suffix + "_m.py")])
    subprocess.run(["rm", os.path.join(path, filename_no_suffix + "_comment.json")])

clean()
os.system("jupyter nbconvert --to python " + args.notebook)
execute_script()
static_analysis()
dynamic_analysis()
analyze()
