import os,sys
import subprocess

HEADER = '''import os, sys, re
import pandas as pd
import numpy as np
import copy as lib_copy
import collections, functools
import matplotlib, pickle, json
import nbformat
from  collections import defaultdict

script_path = os.path.realpath(__file__)
my_dir_path = os.path.dirname(os.path.realpath(__file__))

last_updates = {}
changes = defaultdict(list)
cur_cell = 0

def check_df(name, df):
	if name not in last_updates:
		changes[cur_cell].append(name)
	elif not last_updates[name].equals(df):
		changes[cur_cell].append(name)
	last_updates[name] = df.copy()

def check_changes():
	for name, v in globals().items():
		if type(v) in [pd.DataFrame]:
			check_df(name, v)
		if type(v) == list:
			for i in v:
				if type(v) in [pd.DataFrame]:
					check_df(name + f'[{i}]', v)

def write_changes(filename):
	with open(os.path.join(my_dir_path, filename+".ipynb"), encoding="UTF-8") as f:
		file_content = f.read()
	notebook = nbformat.reads(file_content, as_version=4)

	code_cells = [cell for cell in notebook.cells if cell["cell_type"] == "code"]
	code_indices = [i for i, _ in enumerate(notebook.cells) if notebook.cells[i] in code_cells]
	global changes
	changes = {code_indices[key - 1] : value for key, value in changes.items()}
	with open(os.path.join(my_dir_path, filename + "_change_log.json"), "w") as f:
		f.write(json.dumps(changes))
'''

owd = os.getcwd()


def run_script(file_path):
	filename_no_suffix = file_path.split(os.path.sep)[-1].split(".")[0]
	path = os.path.sep.join(file_path.split(os.path.sep)[:-1])
	
	ENDS = f'''
write_changes("{filename_no_suffix}")'''

	
	os.chdir(path)

	with open(filename_no_suffix + ".py", "r") as f:
		files = f.readlines()

	cur_cell = 0
	for i, l in enumerate(files):
		if l.startswith('# In['):
			files[i-1] += '\ncheck_changes()\n'
			cur_cell += 1
			files[i] += f'\ncur_cell = {cur_cell}'

	with open(filename_no_suffix + "_t.py", "w") as f:
		f.write(HEADER + ''.join(files) + ENDS)

	result = subprocess.run(["python", filename_no_suffix + "_t.py"])
	if result.returncode:
		log.write(filename_no_suffix + ".ipynb" + "\ttracking execution failed\n")
	os.chdir(owd)

if __name__ == "__main__":

	dir = sys.argv[1]
	log = open(os.path.join(dir, "log.txt"), "a")
	option = sys.argv[2] if len(sys.argv) > 2 else ""
	for file in sorted(os.listdir(dir)):
		if file.endswith(".ipynb"):
			file_path = os.path.join(dir, file)
			print(f"\033[96m running {file_path}\033[00m")
			run_script(file_path)

	log.close()