import nbformat
import sys
import os
import pickle
import numpy as np
import pandas as pd
import torch
import random
import collections
from nbconvert import PythonExporter
import json
pd.set_option('display.max_columns', None)
pd.set_option('precision', 4)
np.set_printoptions(precision=4)

dir_path = os.path.dirname(os.path.realpath(sys.argv[1]))
filename = sys.argv[1].split('\\')[-1].split('/')[-1]
filename_no_suffix = filename[:filename.rfind(".")]
suffix = filename[filename.rfind("."):]

data_path = os.path.join(dir_path, filename_no_suffix + "_log.dat")
output_path = os.path.join(dir_path, filename_no_suffix + "_m" + suffix)
json_path = os.path.join(dir_path, filename_no_suffix + "_comment.json")

blanks = "\t- "

### sample
'''
input:
- x, int
- y, shape(1,2) of float
output:
- z, class MyClass
'''


def highlight_text(text):
    return "<p style='color:Tomato;'>" + text + "</p>"


def add_emphasis(table):
    for col in table:
        if col[-1] == "*":
            table[col] = table[col].map('**{}**'.format)


class Variable(object):
    def __init__(self, var, name, cellnum, outflag):
        self.var = var
        self.name = name
        self.cellnum = cellnum
        self.outflag = outflag
        self.comment = "- " + name + ", " + self.initial_comment()

    def initial_comment(self):
        var = self.var
        # if str(type(var)) == "<class 'sklearn.pipeline.Pipeline'>":
        #     return "transforms: " + str(var.steps)
        if str(type(var)) == "<class 'sklearn.utils.Bunch'>":
            return str(type(var))
        if self.outflag:
            return str(type(var)) + ", " + str(var)
        else:
            return str(type(var))

    def add_data_distribute(self):
        pass

    def check_rel(self, variable):
        return 5

    def check_copy(self, variable):
        pass

    def compare_to(self, variable):
        pass


class List(Variable):
    def __init__(self, var, name, cellnum, outflag):
        super().__init__(var, name, cellnum, outflag)
        self.comment = "- " + name + ", " + self.initial_comment()

    def initial_comment(self):
        length = len(self.var)
        return "list length of " + length + ", sample: " + str(
            self.var[:min(length, 5)])

    def check_rel(self, variable):
        rel_score = 5
        if type(variable.var) != list:
            return rel_score
        if self.name == variable.name:
            rel_score = 3
        elif len(self.var) == len(variable.var):
            rel_score = 4
        return rel_score

    def compare_to(self, variable):
        if len(self.var) == len(variable.var):
            example = [
                str(variable.var[i]) + " -> " + str(self.var[i])
                for i in range(min(len(self.var), 5))
            ]
            self.comment += "\n" + blanks + "example changes: " + example


class NdArray(Variable):
    def __init__(self, var, name, cellnum, outflag):
        super().__init__(var, name, cellnum, outflag)
        self.comment = "- " + name + ", " + self.initial_comment()

    def initial_comment(self):
        return "shape" + str(np.shape(self.var)) + " of " + str(
            np.array(self.var).dtype)

    def add_data_distribute(self):
        # blanks = " " * len("- " + self.name + ", ")
        blanks = "\t- "
        array = np.asarray(self.var)
        # only support all numerical values
        if not np.issubdtype(array.dtype, np.number):
            return
        _mean = np.mean(array)
        _variance = np.var(array)
        _max, _min = np.max(array), np.min(array)
        comment_str = "\n" + blanks + "mean: " + "%.4f" % _mean + ", variance: " + "%.4f" % _variance + ", range: ["
        if int(_min) == float(_min):
            comment_str += str(_min) + ", " + str(_max) + "]"
        else:
            comment_str += "%.4f, %.4f]" % (_min, _max)
        self.comment += comment_str

    def check_rel(self, variable):
        rel_score = 5
        if not type(variable.var) in [np.ndarray, pd.DataFrame]:
            return rel_score
        if np.shape(self.var)[0] == np.shape(variable.var)[0]:
            rel_score = 4
        return rel_score

    def check_copy(self, variable):
        if np.array_equal(self.var, variable.var):
            self.comment += "\n" + blanks
            if self.name == variable.name:
                self.comment += highlight_text("no change in the cell")
            else:
                self.comment += highlight_text("copy of " + variable.name)
            return True
        return False

    def compare_to(self, variable):
        if self.check_copy(variable):
            return
        ## check submatrix
        var_a = np.asarray(self.var)
        var_b = np.asarray(variable.var)
        if len(np.shape(var_a)) != 2 or len(np.shape(var_a)) != 2:
            return
        if np.shape(var_a)[0] == np.shape(var_b)[0]:
            if np.shape(var_a)[1] < np.shape(var_b)[1]:
                ls1 = var_a.T.tolist()
                ls2 = var_b.T.tolist()
                r1 = [element for element in ls1 if element in ls2]
                r2 = [element for element in ls2 if element in ls1]
                if r1 == r2:
                    self.comment += "\n" + blanks + highlight_text(
                        "truncated from " + variable.name)


class DataFrame(Variable):
    def __init__(self, var, name, cellnum, outflag):
        super().__init__(var, name, cellnum, outflag)
        self.change_exp = []
        self.columns = list(map(lambda x: str(x), var.columns))
        self.comment = "- " + name + ", " + self.initial_comment()

    def initial_comment(self):
        ret = "shape" + str(np.shape(self.var))
        # count column by type
        type_cnt = {}
        for t in self.var.dtypes:
            if t not in type_cnt.keys():
                type_cnt[t] = 1
            else:
                type_cnt[t] += 1
        ret += ", column types: {"
        type_ls = [str(key) + ": " + str(type_cnt[key]) for key in type_cnt]
        ret += ", ".join(type_ls) + "}"
        # ret += ", sample:\n" + str(var.head(1))
        return ret

    def add_data_distribute(self):
        array = np.asarray(self.var)
        if len(self.change_exp) > 0:
            _examples = self.change_exp
            _example_names = [
                "example_" + str(i) for i in range(len(_examples))
            ]
        else:
            max_len = min(self.var.shape[0], 5)
            _examples = [self.var.iloc[i] for i in range(max_len)]
            _example_names = ["example_" + str(i) for i in range(max_len)]

        def get_range(col):
            if np.issubdtype(col.dtype, np.number):
                return [np.min(col), np.max(col)]
            else:
                return len(col.unique())

        _type = [self.var[col].dtype for col in self.var]
        _range = [get_range(self.var[col]) for col in self.var]

        table = pd.DataFrame([_type] + _examples + [_range],
                             ["type"] + _example_names + ["range"],
                             self.columns)

        add_emphasis(table)

        def reindex_column(columns):
            ls1 = list(filter(lambda col: col[-1] == "*", columns))
            ls2 = list(filter(lambda col: col[-1] != "*", columns))
            return ls1 + ls2

        table = table.reindex(columns=reindex_column(table.columns))
        comment_str = "\n\n" + table.to_markdown()
        self.comment += comment_str

    def check_rel(self, variable):
        '''
        Score:
            0 - identical name
            1 - identical content
            2 - identical shape and type
            3 - identical shape and different type
            4 - different shape but relevant
        '''
        if type(variable.var) != pd.core.frame.DataFrame:
            return 5
        rel_score = 5
        if self.name == variable.name:
            rel_score = 0
        elif self.var.equals(variable.var):
            rel_score = 1
        elif np.shape(self.var) == np.shape(variable.var):
            if self.var.dtypes.equals(variable.var.dtypes):
                rel_score = 2
            else:
                rel_score = 3
        else:
            if np.shape(self.var)[0] == np.shape(variable.var)[0] or np.shape(
                    self.var)[1] == np.shape(variable.var)[1]:
                rel_score = 4
        return rel_score

    def check_copy(self, variable):
        if self.var.equals(variable.var):
            self.comment += "\n" + blanks
            if self.name == variable.name:
                self.comment += highlight_text("no change in the cell")
            else:
                self.comment += highlight_text("copy of " + variable.name)
            return True
        return False

    def add_change_comment(self, variable, convert, change):
        if change:
            self.comment += "\n" + blanks
            comment_str = ""
            for key in change:
                comment_str += str(
                    change[key]) + " " + str(key) + " columns changed"
            self.comment += highlight_text(comment_str)
        if convert:
            self.comment += "\n" + blanks
            comment_str = ""
            for key in convert:
                comment_str += str(convert[key]) + " " + str(
                    key[1]) + " columns converted to " + str(key[0])
            self.comment += highlight_text(comment_str)

        indices = set()
        for col in self.columns:
            if col[-1] != "*":
                continue
            col = col[:-1]
            for i in range(len(variable.var[col])):
                if str(variable.var[col][i]) != str(self.var[col][i]):
                    indices.add(i)
                    break
        row_num = self.var.shape[0]
        if row_num >= 5:
            while len(indices) < 5:
                i = random.randint(0, row_num - 1)
                indices.add(i)

        def change_str(col, idx):
            if col[-1] != "*":
                return str(self.var[col][idx])
            col = col[:-1]
            return str(variable.var[col][idx]) + " -> " + str(
                self.var[col][idx])

        if change or convert:
            for idx in indices:
                self.change_exp.append(
                    [change_str(col, idx) for col in self.columns])

    def check_difference(self, variable):
        col_a = set(self.var.columns)
        col_b = set(variable.columns)
        a_minus_b = col_a.difference(col_b)
        b_minus_a = col_b.difference(col_a)
        # if a_minus_b and b_minus_a:
        #     self.comment += "\n" + blanks
        #     comment_str = ""
        #     if len(b_minus_a) == 1:
        #         item = list(b_minus_a)[0]
        #         filter(lambda x: )
        if a_minus_b or b_minus_a:
            self.comment += "\n" + blanks
            comment_str = ""
            if a_minus_b:
                comment_str += "add columns " + str(a_minus_b)
            if b_minus_a:
                comment_str += " remove columns " + str(b_minus_a)
            self.comment += highlight_text(comment_str)
        return a_minus_b, b_minus_a

    def check_change(self, variable, diffset):
        convert = {}
        change = {}
        var_a = self.var
        var_b = variable.var
        for i in range(len(var_a.dtypes)):
            column_name = var_a.columns[i]
            if column_name in diffset:
                continue
            if var_b[column_name].dtype != var_a[column_name].dtype:
                type_pair = (var_a[column_name].dtype,
                             var_b[column_name].dtype)
                self.columns[i] += "*"
                if type_pair not in convert.keys():
                    convert[type_pair] = 1
                else:
                    convert[type_pair] += 1
            elif not var_b[column_name].equals(var_a[column_name]):
                self.columns[i] += "*"
                if var_a.dtypes[i] not in change.keys():
                    change[var_a.dtypes[i]] = 1
                else:
                    change[var_a.dtypes[i]] += 1
        self.add_change_comment(variable, convert, change)

    def compare_to(self, variable):
        if self.check_copy(variable):
            return
        # only column changed
        if np.shape(self.var)[0] == np.shape(variable.var)[0]:
            # check difference first
            a_minus_b, b_minus_a = self.check_difference(variable)
            # check convert/change in common columns
            self.check_change(variable, a_minus_b)
            for i in range(len(self.var.dtypes)):
                if self.var.columns[i] in a_minus_b:
                    self.columns[i] += "*"
        elif np.shape(self.var)[1] == np.shape(variable.var)[1]:
            if np.shape(self.var)[0] < np.shape(variable.var)[0]:
                l = len(self.var)
                if self.var.equals(variable.var.iloc[:l]) or self.var.equals(
                        variable.var.iloc[-l:]):
                    self.comment += "\n" + blanks
                    self.comment += highlight_text("truncated from " +
                                                   variable.name)


def handlecell(num, st, ed):
    # comments = ["\'\'\'"]
    comments = []
    first_in = -1
    first_out = -1
    header = "---\n"
    for i in range(st, ed + 1):
        if myvars[i].outflag == 0 and first_in == -1:
            first_in = i
            myvars[i].comment = header + "**input**\n" + myvars[i].comment
        elif myvars[i].outflag == 1 and first_out == -1:
            first_out = i
            tmp = "**output**\n" + myvars[i].comment
            myvars[i].comment = header + tmp if first_in == -1 else "\n" + tmp
    '''
    for each output variable, find the input that is closest to it
    find rel within in/out group
    '''
    if first_out != -1 and first_in != -1:
        for i in range(first_out, ed + 1):
            choose_idx = -1
            cur_score = 5
            for j in range(first_in, first_out):
                score = myvars[i].check_rel(myvars[j])
                # print(myvars[i].name, myvars[j].name, score)
                if cur_score > score:
                    score = cur_score
                    choose_idx = j
            if choose_idx != -1:
                myvars[i].compare_to(myvars[choose_idx])

    # if both output, we only check copy
    if first_out != -1:
        for i in range(first_out, ed + 1):
            for j in range(i + 1, ed + 1):
                myvars[j].check_copy(myvars[i])
        for i in range(first_out, ed + 1):
            myvars[i].add_data_distribute()

    for i in range(st, ed + 1):
        comments.append(myvars[i].comment)
    # comments.append("\'\'\'\n")
    return "\n".join(comments)


def gen_comments(labels, tmpvars):
    comment_str = {}
    max_len = len(labels)
    intervals = {}
    for i in range(max_len):
        curcell = labels[i][0]
        if curcell not in intervals.keys():
            intervals[curcell] = (i, i)
        else:
            intervals[curcell] = (intervals[curcell][0], i)
    for key in intervals:
        comment_str[key] = handlecell(key, intervals[key][0],
                                      intervals[key][1])
    return comment_str


def dispatch_gen(var, name, cellnum, outflag):
    if type(var) == list:
        return List(var, name, cellnum, outflag)
    elif type(var) in [np.ndarray, pd.Index, pd.Series]:
        return NdArray(var, name, cellnum, outflag)
    elif type(var) == pd.DataFrame:
        return DataFrame(var, name, cellnum, outflag)
    else:
        return Variable(var, name, cellnum, outflag)


def gen_func_comment(fun_name, fun_map):
    # not considering multiple return types from branches

    _type = [
        k + ": " + str(type(v)) for k, v in fun_map["saved_args"][0].items()
    ] + [type(x) for x in fun_map["saved_rets"][0]] + [""]

    # _examples = [[v for k, v in fun_map["saved_args"][i].items()] +
    #              [x for x in fun_map["saved_rets"][i]] + [""]
    #              for i in range(len(fun_map["saved_args"]))]
    # _example_names = [
    #     "example_" + str(i) for i in range(len(fun_map["saved_args"]))
    # ]
    total = sum(allexe[fun_name].values())
    coverage_examples = [
        [v for k, v in fun_map["args"][i].items()] +
        [x for x in fun_map["rets"][i]] +
        ['{:.2g}'.format(allexe[fun_name][fun_map["path"][i]] / total)]
        for i in range(len(fun_map["args"]))
    ]
    coverage_example_names = [
        "coverage_example_" + str(i) for i in range(len(fun_map["args"]))
    ]
    _columns = [
        "args[{:d}]".format(i) for i in range(len(fun_map["saved_args"][0]))
    ] + ["rets[{:d}]".format(i)
         for i in range(len(fun_map["saved_rets"][0]))] + ["percentage"]

    table = pd.DataFrame([_type] + coverage_examples,
                         ["type"] + coverage_example_names, _columns)

    comment = "'''\n[function table]\n" + str(table) + "\n'''\n"
    return comment


if __name__ == "__main__":
    with open(sys.argv[1], encoding="UTF-8") as f:
        file_content = f.read()
    notebook = nbformat.reads(file_content, as_version=4)

    # estabish map from line in .py to line in .ipynb
    lines = PythonExporter().from_notebook_node(notebook)[0].split("\n")
    code_cells = list(
        filter(lambda cell: cell["cell_type"] == "code", notebook.cells))
    code_indices = list(
        filter(lambda i: notebook.cells[i] in code_cells,
               range(len(notebook.cells))))
    begin_indices = [
        i + 3 for i in range(len(lines)) if lines[i].startswith("# In[")
    ]
    line_to_idx = {}
    for i, idx in enumerate(begin_indices):
        l = len(notebook.cells[code_indices[i]].source.split("\n"))
        for j in range(l):
            line_to_idx[idx + j] = (code_indices[i], j)

    tmpvars = []
    myvars = []
    with open(data_path, "rb") as f:
        tmpvars = pickle.load(f)
    allexe = tmpvars[-1]
    funcs = tmpvars[-2]
    labels = tmpvars[-3]
    tmpvars = tmpvars[:-3]

    for i in range(len(tmpvars)):
        myvars.append(
            dispatch_gen(tmpvars[i], labels[i][2], labels[i][0], labels[i][1]))

    comment_str = gen_comments(labels, tmpvars)

    with open(json_path) as f:
        static_comments = json.load(f)

    # add function info
    insert_map = collections.defaultdict(list)
    for fun_name, fun_map in funcs.items():
        # print(lines[fun_map["loc"] - 1])
        (i, j) = line_to_idx[fun_map["loc"] - 1]
        comment = gen_func_comment(fun_name, fun_map)
        insert_map[i].append((j, comment))

    for comment in static_comments:
        (i, j) = line_to_idx[comment[0] - 1]
        insert_map[i].append((j, "# [autodocs] " + comment[1] + "\n"))

    for key, value in insert_map.items():
        code = notebook.cells[key].source.split("\n")
        for (j, comment) in value:
            code[j] = comment + code[j]
        notebook.cells[key].source = "\n".join(code)

    # write comments to new notebooks
    cur_cell = 0
    cur_idx = 0
    insert_list = []
    for cell in notebook.cells:
        if cell["cell_type"] == "code":
            cur_cell += 1
            if cur_cell in comment_str.keys():
                comment_cell = nbformat.v4.new_markdown_cell(
                    comment_str[cur_cell])
                insert_list.append((cur_idx, comment_cell))
                cur_idx += 1
        cur_idx += 1

    for item in insert_list:
        notebook.cells.insert(item[0], item[1])

    nbformat.write(notebook, output_path)
