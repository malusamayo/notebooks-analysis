import nbformat
import sys
import os
dir_path = os.path.dirname(os.path.realpath(sys.argv[1]))

f = open(sys.argv[1], encoding="UTF-8")
file_content = f.read()
f.close()
f = open(sys.argv[2])
info = f.readlines()
notebook = nbformat.reads(file_content, as_version=4)
filename = sys.argv[1].split('\\')[-1].split('/')[-1]
filename_no_suffix = filename[:filename.rfind(".")]
suffix = filename[filename.rfind("."):]

### sample
'''
input: 
- x, int
- y, shape(1,2) of float
output:
- z, class MyClass
'''


def dispatch_str(items):
    if items[3].startswith("shape"):
        return items[3] + " of " + items[4]
    return items[4]


def gennerate_comments(info):
    comment_str = {}
    input_flag = False
    out_flag = False
    for line in info:
        items = line[:-1].split(";")
        if not items[0].startswith("cell") or len(items) < 5:
            continue
        num = int(items[0][5:-1])
        if num not in comment_str.keys():
            comment_str[num] = "\'\'\'\n"
            input_flag = False
            out_flag = False
        if items[1] == "IN":
            if not input_flag:
                input_flag = True
                comment_str[num] += "input\n"
            comment_str[num] += "- " + items[2] + ", " + dispatch_str(
                items) + "\n"
        elif items[1] == "OUT":
            if not out_flag:
                out_flag = True
                comment_str[num] += "output\n"
            comment_str[num] += "- " + items[2] + ", " + dispatch_str(
                items) + "\n"
        else:
            print("Error!")
    for k, v in comment_str.items():
        if v != None:
            comment_str[k] += "\'\'\'\n"
    return comment_str


comment_str = gennerate_comments(info)
# print(comment_str)

cur_cell = 0
for cell in notebook.cells:
    if cell["cell_type"] == "code":
        cur_cell += 1
        if cur_cell in comment_str.keys():
            cell["source"] = comment_str[cur_cell] + cell["source"]

nbformat.write(notebook,
               os.path.join(dir_path, filename_no_suffix + "_m" + suffix))
