import sys
import os
import time
import subprocess
import argparse
import shutil

CLEANCODE = '''import os, sys, matplotlib
sys.stdout = open(os.devnull, \"w\")
matplotlib.use('Agg')
import warnings
warnings.simplefilter(action='ignore', category=Warning)
'''

parser = argparse.ArgumentParser(description='Generate documentation for notebooks')
parser.add_argument('notebook', help='the notebook to be analyzed')
parser.add_argument('-c', '--clean', help='clean intermediate files', action="store_true")
parser.add_argument('-s', '--skip', help='skip execution of original script', action="store_true")
parser.add_argument('-k', '--keep', help='keep output of scripts', action="store_true")
parser.add_argument('-i', '--ignore', help='ignore rendering html', action="store_true")

args = parser.parse_args()

SRC_PATH = "notebooks_analysis"
filename = args.notebook.split('\\')[-1].split('/')[-1]
path = args.notebook.replace(filename, "")
filename_no_suffix = filename[:filename.rfind(".")]
filename_no_suffix = filename_no_suffix[:-2] if filename_no_suffix.endswith("_m") else filename_no_suffix
suffix = filename[filename.rfind("."):]
owd = os.getcwd()

log = open(os.path.join(path, "log.txt"), "a")

t = [0, 0, 0, 0]

def print_blue(msg):
    print("\033[96m {}\033[00m".format(msg))

def print_red(msg):
    print("\033[91m {}\033[00m".format(msg))

def write_to_log(msg=""):
    log.write(filename + "\t" + "{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t".format(t[0], t[1], t[2], t[3], t[1]+t[2]+t[3]) + msg + "\n")
    log.close()

def convert():
    result = subprocess.run(["jupyter", "nbconvert", "--to", "python", args.notebook]) 
    if result.returncode:
        print_red("Notebook conversion failed!")
        log.write(filename + "\t" + "Notebook conversion failed\n")
        log.close()
        sys.exit(-5)
    result = subprocess.run(["jupyter", "nbconvert", "--to", "html", args.notebook]) 
    if result.returncode:
        print_red("Notebook conversion failed!")
        log.write(filename + "\t" + "Notebook conversion failed\n")
        log.close()
        sys.exit(-5)

def execute_script():
    print_blue('-'*40)
    print_blue("Cleaning the original notebook...")
    # clean up the python code first
    with open(path + filename_no_suffix + ".py", "r+", encoding='utf-8') as f:
        content = ""
        if not args.keep:
            content += CLEANCODE
        for line in f:
            if line.startswith("get_ipython") or line.startswith("display("):
                line = "# " + line
            content += line
        f.seek(0)
        f.truncate()
        f.write(content)
    
    if args.skip:
        return

    os.chdir(path)
    print_blue("Running the original notebook...")
    st = time.time()
    result = subprocess.run(["python", filename_no_suffix + ".py"])
    ed0 = time.time()
    t[0] = ed0 - st
    os.chdir(owd)
    if result.returncode:
        print_red("Original execution failed!")
        write_to_log("Script failed")
        sys.exit(-1)
    print_blue("Original execution completed in {:.2f} seconds.".format(t[0]))

def static_analysis():
    print_blue('-'*40)
    print_blue("Starting static analysis of the notebook...")
    st = time.time()
    result = subprocess.run(
        ["node", os.path.join(SRC_PATH, "instrumenter.js"), path + filename_no_suffix + ".py"])
    ed1 = time.time()
    t[1] = ed1 - st
    if result.returncode:
        print_red("Static analysis failed!")
        write_to_log("Static analysis failed")
        sys.exit(-2)
    print_blue("Static analysis completed in {:.2f} seconds.".format(t[1]))

def dynamic_analysis():
    os.chdir(path)
    print_blue('-'*40)
    print_blue("Starting dynamic analysis of the notebook...")
    st = time.time()
    result = subprocess.run(["python", filename_no_suffix + "_m.py"])
    ed2 = time.time()
    t[2] = ed2 - st
    os.chdir(owd)
    if result.returncode:
        print_red("Dynamic analysis failed!")
        write_to_log("Dynamic analysis failed!")
        sys.exit(-3)
    print_blue("Dynamic analysis completed in {:.2f} seconds.".format(t[2]))

def analyze():
    print_blue('-'*40)
    print_blue("Starting generating documentation...")
    st = time.time()
    result = subprocess.run(["python", os.path.join(SRC_PATH, "analyzer.py"), args.notebook])
    ed3 = time.time()
    t[3] = ed3 - st
    if result.returncode:
        print_red("Doc gen failed!")
        write_to_log("Doc gen failed!")
        sys.exit(-4)
    print_blue(
    '''Generation completed in {:.2f} seconds.
    Script completed in {:.2f} seconds.
    Time fraction: [{:.2f} vs {:.2f}, {:.2f}, {:.2f}] seconds
    '''
    .format(t[3], t[1]+t[2]+t[3], t[0], t[1], t[2], t[3]))
    write_to_log("success")

def render():
    if args.ignore:
        return
    print_blue('-'*40)
    print_blue("Converting documentation to html...")
    st = time.time()
    result = subprocess.run(["node", os.path.join(SRC_PATH, "html_convert.js"), args.notebook])
    ed4 = time.time()
    if result.returncode:
        print_red("HTML convert failed!")
        return
    print_blue("HTML rendering completed in {:.2f} seconds.".format(ed4 - st))

def clean():
    try:
        print_blue("removing folders")
        shutil.rmtree(os.path.join(path, filename_no_suffix))
    except FileNotFoundError:
        print_blue("no previous folder")
    # os.system("rm " + os.path.join(path, filename_no_suffix, "\*.json"))
    # os.system("rm " + os.path.join(path, filename_no_suffix, ".json"))
    # subprocess.run(["rmdir", os.path.join(path, filename_no_suffix)])
    if not args.clean:
        return
    subprocess.run(["rm", os.path.join(path, filename_no_suffix + ".html")])
    subprocess.run(["rm", os.path.join(path, filename_no_suffix + ".py")])
    subprocess.run(["rm", os.path.join(path, filename_no_suffix + "_m.py")])
    subprocess.run(["rm", os.path.join(path, filename_no_suffix + "_comment.json")])

if __name__ == "__main__":
    clean()
    convert()
    execute_script()
    static_analysis()
    dynamic_analysis()
    analyze()
    render()