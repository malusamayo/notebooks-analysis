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
import itertools
pd.set_option('display.max_columns', None)
pd.set_option('precision', 4)
np.set_printoptions(precision=4)

sys.argv.append("notebooks/debug_example.ipynb")

dir_path = os.path.dirname(os.path.realpath(sys.argv[1]))
filename = sys.argv[1].split('\\')[-1].split('/')[-1]
filename_no_suffix = filename[:filename.rfind(".")]
suffix = filename[filename.rfind("."):]

data_path = os.path.join(dir_path, filename_no_suffix)
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


class PatternSynthesizer(object):

    '''
    df1: before, df2: after, col: the target column
    '''
    def __init__(self, df1, df2, info):
        self.df1 = df1
        self.df2 = df2
        self.cols1 = list(df1.columns)
        self.cols2 = list(df2.columns)
        self.srccols = set(info.get).intersection(self.cols1)
        self.descols = set(info.set).intersection(self.cols1)
        self.syn_stack = []
    
    def check_fillna_only(self, df1, df2, from_col, to_col):
        cmp_df = df2[to_col].compare(df1[from_col])
        return cmp_df["other"].isnull().all()

    def check_fillna(self, df1, df2, from_col, to_col):
        return df1[from_col].isnull().values.any() and not df2[to_col].isnull().values.any()
    
    def check_str(self, df, col):
        return pd.api.types.is_string_dtype(df[col])
    
    def check_int(self, df, col):
        return pd.api.types.is_integer_dtype(df[col])

    def check_float(self, df, col):
        return pd.api.types.is_float_dtype(df[col])

    def check_num(self, df, col):
        return pd.api.types.is_numeric_dtype(df[col])

    def check_cat(self, df, col):
        return pd.api.types.is_categorical_dtype(df[col])

    def check_typeconvert(self, df1, df2, from_col, to_col):
        def check_transform(f):
            try:
                if df1[from_col].map(f).equals(df2[to_col]):
                    return
            except:
                pass
            if self.check_num(df1, from_col):
                self.syn_stack.append(("num_transform", [from_col], [to_col]))
            elif self.check_str(df1, from_col):
                self.syn_stack.append(("str_transform", [from_col], [to_col]))

        # converted to str to avoid bugs when dtype == Categorical
        if str(df1[from_col].dtype) != str(df2[to_col].dtype):
            if self.check_float(df2, to_col):
                self.syn_stack.append(("float", [from_col], [to_col]))
                check_transform(float)
            elif self.check_cat(df2, to_col):
                self.syn_stack.append(("discretize", [from_col], [to_col]))
            elif self.check_int(df2, to_col):
                l = df2[to_col].unique()
                if sorted(l) == list(range(min(l), max(l)+1)):
                    self.syn_stack.append(("encode", [from_col], [to_col]))
                else:
                    self.syn_stack.append(("int", [from_col], [to_col]))
                    check_transform(int)
            else:
                self.syn_stack.append(("type_convert", [from_col], [to_col]))
            return True
        return False

    def check_column(self, df1, df2, from_col, to_col):
        
        # check the case when only different values are null values
        if self.check_fillna_only(df1, df2, from_col, to_col):
            self.syn_stack.append(("fillna", [from_col], [to_col]))
            return

        if not self.check_typeconvert(df1, df2, from_col, to_col):
            if df1[from_col].nunique() > df2[to_col].nunique():
                # [TODO] exclude fillna case
                print(len(df1[from_col].unique()), len(df2[to_col].unique()))
                self.syn_stack.append(("merge", [from_col], [to_col]))
            elif self.check_num(df1, from_col):
                self.syn_stack.append(("num_transform", [from_col], [to_col]))
            elif self.check_str(df1, from_col):
                self.syn_stack.append(("str_transform", [from_col], [to_col]))
            else:
                self.syn_stack.append(("map", [from_col], [to_col]))
        
        if self.check_fillna(df1, df2, from_col, to_col):
            self.syn_stack.append(("fillna", [from_col], [to_col]))

    def check_removecol(self, df1, df2):
        removed = [x for x in self.cols1 if x not in self.cols2]
        if removed:        
            self.cols1 = [x for x in self.cols1 if x in self.cols2]
            self.syn_stack.append(("removecol", removed))
            return True
        return False
    
    def check_rearrange(self, df1, df2):
        if self.cols1 != self.cols2 and set(self.cols1) == set(self.cols2):
            self.syn_stack.append(("rearrange", self.cols1, self.cols2))
            return True
        return False

    # [TODO] refine remove row
    def check_removerow(self, df1, df2):
        # use index to track row mappings
        if len(df2) < len(df1) and set(df2.index).issubset(set(df1.index)):
            tmp = df1.loc[~df1.index.isin(df2.index)]
            tmp_null = tmp.isnull()
            # all rows contain nan
            if tmp_null.any(axis=1).all():
                # select columns that removed rows all contain nan
                self.syn_stack.append(("remove_null", list(tmp_null.columns[tmp_null.all()])))
            else:
                self.syn_stack.append(("removerow"))
            return True
        return False

    def search(self, df1, df2):
        cols_dummy = [col for col in self.colsnew if set(df2[col].unique()).issubset({0, 1})]
        if cols_dummy:
            self.syn_stack.append(("one_hot_encoding", cols_dummy))
        # [TODO] add src cols for create; special optimization for one src col
        for col in self.colsnew - set(cols_dummy):
            if len(self.srccols) == 1:
                self.check_column(df1, df2, list(self.srccols)[0], col)
            elif self.check_num(df2, col):
                self.syn_stack.append(("num_transform", [col]))
            elif self.check_str(df2, col):
                self.syn_stack.append(("str_transform", [col]))
            else:
                self.syn_stack.append(("create", [col]))

        # [TODO] merge the same op for different cols?
        for col in self.colschange:
            self.check_column(df1, df2, col, col)

    def check(self, df1, df2):
        # [TODO] differentiate between irrelevant dfs and real transformations
        if set(self.cols1).isdisjoint(set(self.cols2)):
            return
        if len(df1) < len(df2):
            return
        if self.check_removerow(df1, df2):
            df1 = df1.loc[df2.index]
        # rows not removed -> index not subset -> irrelevant dfs
        if len(df1) > len(df2):
            return
        self.check_removecol(df1, df2)
        self.check_rearrange(df1, df2)
        self.colsnew = set(self.cols2).difference(set(self.cols1))
        self.colschange = set(col for col in self.cols1 if not df1[col].equals(df2[col]))
        print(self.colsnew, self.colschange)
        self.search(df1, df2)
        if not self.syn_stack:
            self.syn_stack.append("copy")
        return self.syn_stack

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

class Info(object):
    def __init__(self, info, cellnum):
        super().__init__()
        self.get = []
        self.set = []
        self.par = {}
        if info == None:
            return
        if str(cellnum) in info["get"]:
            self.get = info["get"][str(cellnum)]
        if str(cellnum) in info["set"]:
            self.set = info["set"][str(cellnum)]
        if str(cellnum) in info["par"]:
            self.par = info["par"][str(cellnum)]


def handlecell(myvars, st, ed, info):
    # comments = ["\'\'\'"]
    comments = []

    # find the first input and output
    flags = [var.outflag for var in myvars[st:ed+1]]
    first_in = flags.index(0) + st if 0 in flags else -1
    first_out = flags.index(1) + st if 1 in flags else -1

    # first_in = -1
    # first_out = -1
    # header = "---\n"
    # for i in range(st, ed + 1):
    #     if myvars[i].outflag == 0 and first_in == -1:
    #         first_in = i
    #         myvars[i].comment = header + "**input**\n" + myvars[i].comment
    #     elif myvars[i].outflag == 1 and first_out == -1:
    #         first_out = i
    #         tmp = "**output**\n" + myvars[i].comment
    #         myvars[i].comment = header + tmp if first_in == -1 else "\n" + tmp
    '''
    for each output variable, find the input that is closest to it
    find rel within in/out group
    '''
    if first_out != -1 and first_in != -1:
        for i in range(first_out, ed + 1):
            # choose_idx = -1
            # cur_score = 5
            for j in range(first_in, first_out):
                # score = myvars[i].check_rel(myvars[j])
                # print(myvars[i].name, myvars[j].name, score)
                if type(myvars[i].var) == pd.core.frame.DataFrame:
                    if type(myvars[j].var) == pd.core.frame.DataFrame:
                        checker = PatternSynthesizer(myvars[j].var, myvars[i].var, info)
                        l = checker.check(myvars[j].var, myvars[i].var)
                        print(myvars[i].cellnum, ":", myvars[j].name, "->", myvars[i].name, "\033[96m", l, "\033[0m")
                # if cur_score > score:
                #     cur_score = score
                #     choose_idx = j
            # if choose_idx != -1:
                # myvars[i].compare_to(myvars[choose_idx])

    # if both output, we only check copy
    # if first_out != -1:
    #     for i in range(first_out, ed + 1):
    #         for j in range(i + 1, ed + 1):
    #             myvars[j].check_copy(myvars[i])
    #     for i in range(first_out, ed + 1):
    #         myvars[i].add_data_distribute()

    # for i in range(st, ed + 1):
    #     comments.append(myvars[i].comment)

    # build json maps
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


# def gen_comments(labels, tmpvars):
#     comment_str = {}
#     max_len = len(labels)
#     intervals = {}
#     for i in range(max_len):
#         curcell = labels[i][0]
#         if curcell not in intervals.keys():
#             intervals[curcell] = (i, i)
#         else:
#             intervals[curcell] = (intervals[curcell][0], i)
#     json_map = {}
#     for key in intervals:
#         comment_str[key], inner_json_map = handlecell(key, intervals[key][0],
#                                                       intervals[key][1])
#         json_map[code_indices[key - 1]] = inner_json_map
#     return comment_str, json_map


def dispatch_gen(var, name, cellnum, outflag):
    if type(var) == list:
        return List(var, name, cellnum, outflag)
    elif type(var) in [np.ndarray, pd.Index, pd.Series]:
        return NdArray(var, name, cellnum, outflag)
    elif type(var) == pd.DataFrame:
        return DataFrame(var, name, cellnum, outflag)
    else:
        return Variable(var, name, cellnum, outflag)


# def gen_func_comment(fun_name, fun_map):
#     # not considering multiple return types from branches

#     _type = []
#     for k, path_map in fun_map.items():
#         if k == "loc":
#             continue
#         _type = [
#             k + ": " + str(type(v)) for k, v in path_map["args"][0].items()
#         ] + [str(type(x)) for x in path_map["rets"][0]]
#         break

#     total = sum([path_map["count"] for path_map in list(fun_map.values())[1:]])

#     args_len, rets_len = 0, 0
#     examples = []
#     for k, path_map in fun_map.items():
#         if k == "loc":
#             continue
#         args_len = max(args_len, len(path_map["args"][0]))
#         rets_len = max(rets_len, len(path_map["rets"][0]))
#         args_list = [[v for k, v in args.items()] for args in path_map["args"]]
#         args = [[args[i] for args in args_list] for i in range(args_len)]
#         rets = [[rets[i] for rets in path_map["rets"]]
#                 for i in range(rets_len)]
#         examples.append(args + rets +
#                         ['{:.2g}'.format(path_map["count"] / total)] +
#                         [path_map["count"]])

#     _columns = ["args[{:d}]".format(i) for i in range(args_len)
#                 ] + ["rets[{:d}]".format(i)
#                      for i in range(rets_len)] + ["frequency", "counts"]

#     table = pd.DataFrame([_type] +
#                          sorted(examples, key=lambda x: x[-1], reverse=True),
#                          columns=_columns)

#     table.insert(0, fun_name + postfix, ["type"] +
#                  ["example_" + str(i) for i in range(len(fun_map.keys()) - 1)])

#     # comment = "'''\n[function table]\n" + str(table) + "\n'''\n"
#     comment = ""
#     json_map = json.loads(table.to_json())
#     return comment, json_map

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
    # begin_indices = [
    #     i + 3 for i in range(len(lines)) if lines[i].startswith("# In[")
    # ]
    # line_to_idx = {}
    # for i, idx in enumerate(begin_indices):
    #     l = len(notebook.cells[code_indices[i]].source.split("\n"))
    #     for j in range(l):
    #         line_to_idx[idx + j] = (code_indices[i], j)

    static_comments = {}
    with open(json_path) as f:
        json_tmp_list = json.load(f)
        for [idx, content] in json_tmp_list:
            static_comments[idx] = content

    json_map = {}
    info = {}
    # funcs = {}
    with open(os.path.join(data_path, "info.json"), 'r') as j:
        info = json.loads(j.read())


    for file in os.listdir(data_path):
        myvars = []
        if file == "info.json" or file.endswith("_f.dat"):
            continue
        # if file.endswith("_f.dat"):
        #     with open(os.path.join(data_path, file), "rb") as f:
        #         try:
        #             funcs = pickle.load(f)
        #         except:
        #             pass
        else:
            with open(os.path.join(data_path, file), "rb") as f:
                try:
                    vars = pickle.load(f)
                except AttributeError as e:
                    print(e)
                    continue
                for i in range(len(vars)):
                    try:
                        myvars.append(
                            dispatch_gen(vars[i][0], vars[i][1][2], vars[i][1][0], vars[i][1][1]))
                    except:
                        pass
                # comments = static_comments[vars[0][1][0]] if vars[0][1][0] in static_comments.keys() else []
                _, json_map[code_indices[vars[0][1][0] - 1]] = handlecell(myvars, 0, len(vars)-1, Info(info, vars[0][1][0]))
                

    # fill not existing entries
    for key, value in json_map.items():
        cat_list = ["input", "output", "summary", "function", "comment"]
        for cat in cat_list:
            if cat not in value.keys():
                json_map[key][cat] = {}

    with open(json_out_path, "w") as f:
        f.write(json.dumps(json_map))

    # with open(data_path, "rb") as f:
    #     tmpvars = pickle.load(f)
    # funcs = tmpvars[-1]
    # labels = tmpvars[-2]
    # tmpvars = tmpvars[:-2]

    # for i in range(len(tmpvars)):
    #     myvars.append(
    #         dispatch_gen(tmpvars[i], labels[i][2], labels[i][0], labels[i][1]))

    # comment_str, json_map = gen_comments(labels, tmpvars)

    # format: [[cellnum, comment] or [funcname, cellnum]]
    # comment should be used later, along with cell number

    # def insert_to_map(json_map, cell_num, cat, name, value):
    #     if cell_num not in json_map.keys():
    #         json_map[cell_num] = {cat: {name: value}}
    #     elif cat not in json_map[cell_num].keys():
    #         json_map[cell_num][cat] = {name: value}
    #     else:
    #         json_map[cell_num][cat][name] = value

    # # add function info
    # insert_map = collections.defaultdict(list)
    # for fun_name, fun_map in funcs.items():
    #     # print(lines[fun_map["loc"] - 1])
    #     # affected by "-s"
    #     # (i, j) = line_to_idx[fun_map["loc"] -3]
    #     fun_name_no_idx = fun_name[:fun_name.rfind("_")]
    #     cell_num = [v for k, v in static_comments.items() if k == fun_name_no_idx]
    #     assert(len(cell_num) == 1)
    #     comment, func_json_map = gen_func_comment(fun_name, fun_map)
    #     insert_to_map(json_map, cell_num[0], "function", fun_name, func_json_map)
    #     # insert_map[i].append((j, comment))

    # for comment in static_comments:
    #     (i, j) = line_to_idx[comment[0] - 3]
    #     insert_to_map(json_map, i, "comment", j, comment[1])
    #     insert_map[i].append((j, "# [autodocs] " + comment[1] + "\n"))

    

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
