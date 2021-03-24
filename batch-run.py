import os, sys

def my_print(msg):
    print("\033[96m {}\033[00m".format(msg))

dir = sys.argv[1]
for file in os.listdir(dir):
    if file.endswith(".ipynb"):
        file_path = os.path.join(dir, file)
        my_print("Running " + file_path)
        os.system("python run.py " + file_path)