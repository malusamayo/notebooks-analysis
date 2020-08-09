import sys
import os

filename = sys.argv[1].split('\\')[-1].split('/')[-1]
path = sys.argv[1].replace(filename, "")
filename_no_suffix = filename[:filename.rfind(".")]
suffix = filename[filename.rfind("."):]
owd = os.getcwd()

os.system("jupyter nbconvert --to python " + sys.argv[1])
os.system("node gen_runtime_info_code.js " + path + filename_no_suffix + ".py")
os.chdir(path)
os.system("python " + filename_no_suffix + "_m.py")
os.chdir(owd)
os.system("python add_runtime_info.py " + sys.argv[1] + " " + path +
          filename_no_suffix + "_log.dat")
