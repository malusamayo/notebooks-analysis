import nbformat
import sys
import os
import pickle
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('precision', 4)
np.set_printoptions(precision=4)

dir_path = os.path.dirname(os.path.realpath(sys.argv[1]))
filename = sys.argv[1].split('\\')[-1].split('/')[-1]
filename_no_suffix = filename[:filename.rfind(".")]
suffix = filename[filename.rfind("."):]

table_str = "\n------------------------------- Data Info Table -------------------------------\n"

### sample
'''
input:
- x, int
- y, shape(1,2) of float
output:
- z, class MyClass
'''


class Variable(object):
    def __init__(self, var, name, cellnum, outflag, idx):
        self.var = var
        self.name = name
        self.cellnum = cellnum
        self.outflag = outflag
        self.idx = idx
        self.comment = "- " + name + ", " + self.initial_comment()

    def initial_comment(self):
        var = self.var
        # if str(type(var)) == "<class 'sklearn.pipeline.Pipeline'>":
        #     return "transforms: " + str(var.steps)
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

    def add_rel_comment(self, variable):
        pass


class NdArray(Variable):
    def __init__(self, var, name, cellnum, outflag, idx):
        super().__init__(var, name, cellnum, outflag, idx)
        self.comment = "- " + name + ", " + self.initial_comment()

    def initial_comment(self):
        return "shape" + str(np.shape(self.var)) + " of " + str(
            np.array(self.var).dtype)

    def add_data_distribute(self):
        blanks = " " * len("- " + self.name + ", ")
        array = np.asarray(self.var)
        # only support all numerical values
        if not np.issubdtype(array.dtype, np.number):
            return
        _mean = np.mean(array)
        _variance = np.var(array)
        _max, _min = np.max(array), np.min(array)
        comment_str = "\n" + blanks + "mean: " + "%.4f" % _mean + ", variance: " + "%.4f" % _variance + ", range: [" + str(
            _min) + ", " + str(_max) + "]"
        self.comment += comment_str


class DataFrame(Variable):
    def __init__(self, var, name, cellnum, outflag, idx):
        super().__init__(var, name, cellnum, outflag, idx)
        self.convert = None
        self.columns = list(var.columns)
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
        for key in type_cnt:
            ret = ret + ", " + str(type_cnt[key]) + " " + str(key) + " columns"
        # ret += ", sample:\n" + str(var.head(1))
        return ret

    def add_data_distribute(self):
        array = np.asarray(self.var)
        _example = array[0]
        if self.convert != None:
            _example = self.convert
        if not np.issubdtype(array.dtype, np.number):
            table = pd.DataFrame([_example], ["example"], self.columns)
            self.comment += table_str + str(table) + table_str
            return
        _mean = np.mean(array, 0)
        _variance = np.var(array, 0)
        _max, _min = np.max(array, 0), np.min(array, 0)
        _range = [[_min[i], _max[i]] for i in range(len(_max))]
        table = pd.DataFrame([_example, _mean, _variance, _range],
                             ["example", "mean", "variance", "range"],
                             self.columns)
        # table = self.var.describe().drop(["count", "25%", "50%", "75%"])
        # _range = [[[table.loc["min"][i], table.loc["max"][i]]
        #            for i in range(len(table.columns))], ["range"],
        #           table.columns]
        comment_str = table_str + str(table) + table_str
        self.comment += comment_str

    def check_copy(self, variable):
        if self.outflag != variable.outflag or self.outflag == 0:
            return
        if self.var.equals(variable.var):
            self.comment += ", copy of " + variable.name

    def check_rel(self, variable):
        '''
        Score:
            0 - identical name
            1 - identical content
            2 - identical shape and type
            3 - identical shape and different type
            4 - different shape but subcontent
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

        return rel_score

    def check_content(self, variable):
        change = {}
        var_a = self.var
        var_b = variable.var
        for i in range(len(var_a.dtypes)):
            if not var_a[var_a.columns[i]].equals(var_b[
                    var_b.columns[i]]) and var_a.dtypes[i] == var_b.dtypes[i]:
                self.columns[i] += "*"
                if var_a.dtypes[i] not in change.keys():
                    change[var_a.dtypes[i]] = 1
                else:
                    change[var_a.dtypes[i]] += 1
        for key in change:
            self.comment += ", " + str(change[key]) + " " + str(
                key) + " columns of " + variable.name + " changed"

    def check_convert(self, variable):
        convert = {}
        var_a = self.var
        var_b = variable.var
        assert (type(var_a) == pd.core.frame.DataFrame)
        for i in range(len(var_a.dtypes)):
            if var_a.dtypes[i] != var_b.dtypes[i] and var_a.columns[
                    i] == var_b.columns[i]:
                type_pair = (var_a.dtypes[i], var_b.dtypes[i])
                self.columns[i] += "*"
                if type_pair not in convert.keys():
                    convert[type_pair] = 1
                else:
                    convert[type_pair] += 1
        blanks = " " * len("- " + self.name)
        self.comment += "\n" + blanks
        for key in convert:
            self.comment += ", " + str(convert[key]) + " " + str(
                key[1]
            ) + " columns of " + variable.name + " converted to " + str(key[0])
        self.convert = [
            str(np.asarray(var_b)[0][i]) + " -> " +
            str(np.asarray(var_a)[0][i]) for i in range(len(var_a.dtypes))
        ]

    def add_rel_comment(self, variable):
        # print(self.name, variable.name)
        if self.var.equals(variable.var):
            if self.name == variable.name:
                self.comment += ", no change in the cell"
            else:
                self.comment += ", copy of " + variable.name
        elif np.shape(self.var) == np.shape(variable.var):
            if self.var.dtypes.equals(variable.var.dtypes):
                self.check_content(variable)
            else:
                self.check_convert(variable)
                self.check_content(variable)
                # table = pd.DataFrame([_example], ["convert"], var_a.columns)
                # self.comment += "\n" + str(table)


def handlecell(num, st, ed):
    comments = ["\'\'\'"]
    first_in = -1
    first_out = -1
    for i in range(st, ed + 1):
        if myvars[i].outflag == 0 and first_in == -1:
            first_in = i
            myvars[i].comment = "input\n" + myvars[i].comment
        elif myvars[i].outflag == 1 and first_out == -1:
            first_out = i
            myvars[i].comment = "output\n" + myvars[i].comment
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
                myvars[i].add_rel_comment(myvars[choose_idx])

    # if both output, we only check copy
    if first_out != -1:
        for i in range(first_out, ed + 1):
            for j in range(i + 1, ed + 1):
                myvars[j].check_copy(myvars[i])
        for i in range(first_out, ed + 1):
            myvars[i].add_data_distribute()

    for i in range(st, ed + 1):
        comments.append(myvars[i].comment)
    comments.append("\'\'\'\n")
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


if __name__ == "__main__":
    f = open(sys.argv[1], encoding="UTF-8")
    file_content = f.read()
    f.close()
    notebook = nbformat.reads(file_content, as_version=4)

    tmpvars = []
    myvars = []
    with open(sys.argv[2], "rb") as f:
        tmpvars = pickle.load(f)
    labels = tmpvars[-1]
    tmpvars = tmpvars[:-1]

    for i in range(len(tmpvars)):
        if type(tmpvars[i]) == np.ndarray or type(
                tmpvars[i]) == pd.Index or type(tmpvars[i]) == pd.Series:
            myvars.append(
                NdArray(tmpvars[i], labels[i][2], labels[i][0], labels[i][1],
                        i))
        elif type(tmpvars[i]) == pd.DataFrame:
            myvars.append(
                DataFrame(tmpvars[i], labels[i][2], labels[i][0], labels[i][1],
                          i))
        else:
            myvars.append(
                Variable(tmpvars[i], labels[i][2], labels[i][0], labels[i][1],
                         i))

    comment_str = gen_comments(labels, tmpvars)

    # write comments to new notebooks
    cur_cell = 0
    for cell in notebook.cells:
        if cell["cell_type"] == "code":
            cur_cell += 1
            if cur_cell in comment_str.keys():
                cell["source"] = comment_str[cur_cell] + cell["source"]

    nbformat.write(notebook,
                   os.path.join(dir_path, filename_no_suffix + "_m" + suffix))
