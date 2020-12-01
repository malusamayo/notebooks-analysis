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

sys.argv.append("notebooks/fifa_notebooks/fifa-19-deep-learning-with-keras.ipynb")

dir_path = os.path.dirname(os.path.realpath(sys.argv[1]))
filename = sys.argv[1].split('\\')[-1].split('/')[-1]
filename_no_suffix = filename[:filename.rfind(".")]
suffix = filename[filename.rfind("."):]

data_path = os.path.join(dir_path, filename_no_suffix + "_log.dat")
output_path = os.path.join(dir_path, filename_no_suffix + "_m" + suffix)
json_path = os.path.join(dir_path, filename_no_suffix + "_comment.json")
json_out_path = os.path.join(dir_path, filename_no_suffix + "_out.json")

blanks = "\t- "
postfix = "[auto]"

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
        if col.endswith(postfix):
            table[col] = table[col].map('<b>{}</b>'.format)
            # table[col] = table[col].map('**{}**'.format)


class Variable(object):
    def __init__(self, var, name, cellnum, outflag):
        self.var = var
        self.name = name
        self.cellnum = cellnum
        self.outflag = outflag
        self.json_map = {
            "type": str(type(var))[8:-2],
            "shape": "",
            "hint": "",
            "value": ""
        }
        self.comment = "- " + name + ", " + self.initial_comment()

    def initial_comment(self):
        var = self.var
        # if str(type(var)) == "<class 'sklearn.pipeline.Pipeline'>":
        #     return "transforms: " + str(var.steps)
        if str(type(var)) == "<class 'sklearn.utils.Bunch'>":
            return str(type(var))
        if self.outflag:
            self.json_map["value"] = str(var)
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
        # self.comment = "- " + name + ", " + self.initial_comment()

    def initial_comment(self):
        length = min(len(self.var), 5)
        comments = [
            dispatch_gen(self.var[i], self.name + "[" + str(i) + "]", -1,
                         -1).comment for i in range(length)
        ]
        self.json_map["value"] = comments
        self.json_map["shape"] = "(1, {})".format(str(length))
        return "list length of " + str(length) + ", sample:\n\t" + "\n\t".join(
            comments)

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
            self.json_map["value"] = str(example)
            self.comment += "\n" + blanks + "example changes: " + str(example)


class NdArray(Variable):
    def __init__(self, var, name, cellnum, outflag):
        super().__init__(var, name, cellnum, outflag)
        # self.comment = "- " + name + ", " + self.initial_comment()

    def initial_comment(self):
        self.json_map["shape"] = str(np.shape(self.var))
        self.json_map["type"] += ", dtype: " + str(np.array(self.var).dtype)
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
        comment_str = "mean: " + "%.4f" % _mean + ", variance: " + "%.4f" % _variance + ", range: ["
        if int(_min) == float(_min):
            comment_str += str(_min) + ", " + str(_max) + "]"
        else:
            comment_str += "%.4f, %.4f]" % (_min, _max)
        self.json_map["value"] = comment_str
        self.comment += "\n" + blanks + comment_str

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
                # self.json_map["hint"] += "no change in the cell; "
            else:
                self.comment += highlight_text("copy of " + variable.name)
                self.json_map["hint"] += "copy of " + variable.name + "; "
            return True
        return False

    def compare_to(self, variable):
        if self.check_copy(variable):
            return
        ## check submatrix
        var_a = np.asarray(self.var)
        var_b = np.asarray(variable.var)
        if len(np.shape(var_a)) != 2 or len(np.shape(var_b)) != 2:
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
                    self.json_map[
                        "hint"] += "truncated from " + variable.name + "； "


class DataFrame(Variable):
    def __init__(self, var, name, cellnum, outflag):
        super().__init__(var, name, cellnum, outflag)
        self.change_exp = []
        self.copy = False
        self.columns = list(map(lambda x: str(x), var.columns))
        # self.comment = "- " + name + ", " + self.initial_comment()

    def initial_comment(self):
        ret = "shape" + str(np.shape(self.var))
        # count column by type
        type_cnt = {}
        for t in self.var.dtypes:
            if t not in type_cnt.keys():
                type_cnt[t] = 1
            else:
                type_cnt[t] += 1
        col_types = ", column types: {"
        type_ls = [str(key) + ": " + str(type_cnt[key]) for key in type_cnt]
        col_types += ", ".join(type_ls) + "}"
        # ret += ", sample:\n" + str(var.head(1))
        self.json_map["shape"] = str(np.shape(self.var))
        self.json_map["type"] = "DataFrame" + col_types
        return ret + col_types

    def add_data_distribute(self):
        if self.copy:
            return
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
            if str(col.dtype) == "category":
                return len(col.unique())
            if np.issubdtype(col.dtype, np.number):
                return [np.min(col), np.max(col)]
            else:
                return len(col.unique())

        _type = [str(self.var[col].dtype) for col in self.var]
        _range = [str(get_range(self.var[col])) for col in self.var]

        table = pd.DataFrame([_type] + _examples + [_range],
                             columns=self.columns)

        table.insert(0, self.name + postfix, ["type"] + _example_names + ["range"])

        # add_emphasis(table)

        def reindex_column(columns):
            ls1 = list(filter(lambda col: col.endswith(postfix), columns))
            ls2 = list(filter(lambda col: not col.endswith(postfix), columns))
            return ls1 + ls2

        table = table.reindex(columns=reindex_column(table.columns))
        comment_str = "\n\n" + table.to_markdown()
        self.json_map["value"] = json.loads(table.to_json())
        self.comment += comment_str

    def check_rel(self, variable):
        '''
        Score:
            0 - identical name
            1 - identical content
            2 - identical shape and type
            3 - identical shape and different type
            4 - different shape but relevant
            5 - irrelevant
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
                # self.json_map["hint"] += "no change in the cell; "
                self.copy = True
            else:
                self.comment += highlight_text("copy of " + variable.name)
                self.json_map["hint"] += "copy of " + variable.name + "; "
            return True
        return False

    def add_change_comment(self, variable, convert, change, diffset):
        if change:
            self.comment += "\n" + blanks
            comment_str = ""
            for key in change:
                comment_str += str(
                    change[key]) + " " + str(key) + " columns changed"
            self.comment += highlight_text(comment_str)
            self.json_map["hint"] += comment_str + "; "
        if convert:
            self.comment += "\n" + blanks
            comment_str = ""
            for key in convert:
                comment_str += str(convert[key]) + " " + str(
                    key[1]) + " columns converted to " + str(key[0])
            self.comment += highlight_text(comment_str)
            self.json_map["hint"] += comment_str + "; "

        indices = set()
        values = set()
        for col in self.columns:
            if not col.endswith(postfix):
                continue
            col = col[:-1]
            for i in self.var.index:
                try:
                    if str(self.var[col][i]) not in values:
                        if col in diffset or str(variable.var[col][i]) != str(
                                self.var[col][i]):
                            indices.add(i)
                            values.add(str(self.var[col][i]))
                except:
                    pass
                # break after enough sample points
                if len(indices) >= 5:
                    break
        row_num = self.var.shape[0]

        # disable random choice
        # if row_num >= 5:
        # while len(indices) < 5:
        #     i = random.randint(0, row_num - 1)
        #     indices.add(i)

        def change_str(col, idx):
            if not col.endswith(postfix):
                return str(self.var[col][idx])
            col = col[:-1]
            if col in diffset:
                return str(self.var[col][idx])
            return str(variable.var[col][idx]) + " -> " + str(
                self.var[col][idx])

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
                comment_str += "add {0} columns; ".format(len(a_minus_b))
            if b_minus_a:
                comment_str += "remove {0} columns; ".format(len(b_minus_a))

            # add *s for such cols
            self.comment += highlight_text(comment_str)
            self.json_map["hint"] += comment_str

            for i in range(len(self.var.dtypes)):
                if self.var.columns[i] in a_minus_b:
                    self.columns[i] += postfix
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
            if str(var_b[column_name].dtype) != str(var_a[column_name].dtype):
                type_pair = (var_a[column_name].dtype,
                             var_b[column_name].dtype)
                self.columns[i] += postfix
                if type_pair not in convert.keys():
                    convert[type_pair] = 1
                else:
                    convert[type_pair] += 1
            elif not var_b[column_name].equals(var_a[column_name]):
                self.columns[i] += postfix
                if var_a.dtypes[i] not in change.keys():
                    change[var_a.dtypes[i]] = 1
                else:
                    change[var_a.dtypes[i]] += 1
        self.add_change_comment(variable, convert, change, diffset)

    def compare_to(self, variable):
        if self.check_copy(variable):
            return
        # only column changed
        if np.shape(self.var)[0] == np.shape(variable.var)[0]:
            # check difference first
            a_minus_b, b_minus_a = self.check_difference(variable)
            # check convert/change in common columns
            self.check_change(variable, a_minus_b)
        elif np.shape(self.var)[1] == np.shape(variable.var)[1]:
            if np.shape(self.var)[0] < np.shape(variable.var)[0]:
                l = len(self.var)
                # if self.var.equals(variable.var.iloc[:l]) or self.var.equals(
                #         variable.var.iloc[-l:]):

                self.comment += "\n" + blanks
                comment_str = "remove " + str(
                    np.shape(variable.var)[0] -
                    np.shape(self.var)[0]) + " rows from " + variable.name
                self.comment += highlight_text(comment_str)
                self.json_map["hint"] += comment_str + "; "
        if list(self.var.columns) != list(variable.columns):
            set_a = set(self.var.columns)
            set_b = set(variable.columns)
            if set_a == set_b:
                self.comment += "\n" + blanks
                self.comment += highlight_text("rearrange columns")
                self.json_map["hint"] += "rearrange columns" + "; "


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
                    cur_score = score
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

    json_map = {"input": {}, "output": {}, "summary": {}}
    for i in range(st, ed + 1):
        if myvars[i].outflag == 0:
            json_map["input"][myvars[i].name] = myvars[i].json_map
        elif myvars[i].outflag == 1:
            json_map["output"][myvars[i].name] = myvars[i].json_map
            if myvars[i].json_map["type"].startswith(
                    "DataFrame") and myvars[i].json_map["hint"] != "":
                json_map["summary"][
                    myvars[i].name] = myvars[i].json_map["hint"]

    # comments.append("\'\'\'\n")
    return "\n".join(comments), json_map


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
    json_map = {}
    for key in intervals:
        comment_str[key], inner_json_map = handlecell(key, intervals[key][0],
                                                      intervals[key][1])
        json_map[code_indices[key - 1]] = inner_json_map
    return comment_str, json_map


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

    _type = []
    for k, path_map in fun_map.items():
        if k == "loc":
            continue
        _type = [
            k + ": " + str(type(v)) for k, v in path_map["args"][0].items()
        ] + [str(type(x)) for x in path_map["rets"][0]]
        break

    total = sum([path_map["count"] for path_map in list(fun_map.values())[1:]])

    args_len, rets_len = 0, 0
    examples = []
    for k, path_map in fun_map.items():
        if k == "loc":
            continue
        args_len = max(args_len, len(path_map["args"][0]))
        rets_len = max(rets_len, len(path_map["rets"][0]))
        args_list = [[v for k, v in args.items()] for args in path_map["args"]]
        args = [[args[i] for args in args_list] for i in range(args_len)]
        rets = [[rets[i] for rets in path_map["rets"]]
                for i in range(rets_len)]
        examples.append(args + rets +
                        ['{:.2g}'.format(path_map["count"] / total)] +
                        [path_map["count"]])

    _columns = ["args[{:d}]".format(i) for i in range(args_len)
                ] + ["rets[{:d}]".format(i)
                     for i in range(rets_len)] + ["frequency", "counts"]

    table = pd.DataFrame([_type] +
                         sorted(examples, key=lambda x: x[-1], reverse=True),
                         columns=_columns)

    table.insert(0, fun_name + postfix, ["type"] +
                 ["example_" + str(i) for i in range(len(fun_map.keys()) - 1)])

    # comment = "'''\n[function table]\n" + str(table) + "\n'''\n"
    comment = ""
    json_map = json.loads(table.to_json())
    return comment, json_map


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
    funcs = tmpvars[-1]
    labels = tmpvars[-2]
    tmpvars = tmpvars[:-2]

    for i in range(len(tmpvars)):
        myvars.append(
            dispatch_gen(tmpvars[i], labels[i][2], labels[i][0], labels[i][1]))

    comment_str, json_map = gen_comments(labels, tmpvars)

    with open(json_path) as f:
        static_comments = json.load(f)

    def insert_to_map(json_map, cell_num, cat, name, value):
        if cell_num not in json_map.keys():
            json_map[cell_num] = {cat: {name: value}}
        elif cat not in json_map[cell_num].keys():
            json_map[cell_num][cat] = {name: value}
        else:
            json_map[cell_num][cat][name] = value

    # add function info
    insert_map = collections.defaultdict(list)
    for fun_name, fun_map in funcs.items():
        # print(lines[fun_map["loc"] - 1])
        (i, j) = line_to_idx[fun_map["loc"] - 1]
        comment, func_json_map = gen_func_comment(fun_name, fun_map)
        insert_to_map(json_map, i, "function", fun_name, func_json_map)
        insert_map[i].append((j, comment))

    for comment in static_comments:
        (i, j) = line_to_idx[comment[0] - 3]
        insert_to_map(json_map, i, "comment", j, comment[1])
        insert_map[i].append((j, "# [autodocs] " + comment[1] + "\n"))

    # fill not existing entries
    for key, value in json_map.items():
        cat_list = ["input", "output", "summary", "function", "comment"]
        for cat in cat_list:
            if cat not in value.keys():
                json_map[key][cat] = {}

    with open(json_out_path, "w") as f:
        f.write(json.dumps(json_map))

    # for key, value in insert_map.items():
    #     code = notebook.cells[key].source.split("\n")
    #     for (j, comment) in value:
    #         code[j] = comment + code[j]
    #     notebook.cells[key].source = "\n".join(code)

    # write comments to new notebooks
    # cur_cell = 0
    # cur_idx = 0
    # insert_list = []
    # for cell in notebook.cells:
    #     if cell["cell_type"] == "code":
    #         cur_cell += 1
    #         if cur_cell in comment_str.keys():
    #             comment_cell = nbformat.v4.new_markdown_cell(
    #                 comment_str[cur_cell])
    #             insert_list.append((cur_idx, comment_cell))
    #             cur_idx += 1
    #     cur_idx += 1

    # for item in insert_list:
    #     notebook.cells.insert(item[0], item[1])

    # nbformat.write(notebook, output_path)
