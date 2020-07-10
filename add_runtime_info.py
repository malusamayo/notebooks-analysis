import nbformat
import sys
import os
import pickle
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

dir_path = os.path.dirname(os.path.realpath(sys.argv[1]))
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


def handle_dataframe(var):
    ret = "shape" + str(np.shape(var))
    # count column by type
    type_cnt = {}
    for t in var.dtypes:
        if t not in type_cnt.keys():
            type_cnt[t] = 1
        else:
            type_cnt[t] += 1
    for key in type_cnt:
        ret = ret + ", " + str(type_cnt[key]) + " " + str(key) + " columns"
    # ret += ", sample:\n" + str(var.head(1))
    return ret


def dispatch_str(var):
    if isinstance(var, np.ndarray) or isinstance(var, pd.Index) or isinstance(
            var, pd.core.series.Series):
        return "shape" + str(np.shape(var)) + " of " + str(np.array(var).dtype)
    if str(type(var)) == "<class 'pandas.core.frame.DataFrame'>":
        return handle_dataframe(var)
    if str(type(var)) == "<class 'sklearn.pipeline.Pipeline'>":
        return "transforms: " + str(var.steps)
    return str(type(var))


class Variable(object):
    def __init__(self, var, name, cellnum, outflag, idx):
        self.var = var
        self.name = name
        self.cellnum = cellnum
        self.outflag = outflag
        self.idx = idx
        self.comment = "- " + name + ", " + dispatch_str(var)

    def add_comment(self):
        pass

    def add_data_distribute(self, dataframe_flag):
        blanks = " " * len("- " + self.name + ", ")
        array = np.asarray(self.var)
        # only support all numerical values
        if not np.issubdtype(array.dtype, np.number):
            return
        if not dataframe_flag:
            _mean = np.mean(array)
            _variance = np.var(array)
            _max, _min = np.max(array), np.min(array)
            comment_str = "\n" + blanks + "mean: " + _mean + " variance: " + _variance + " range: (" + _max + ", " + _min + ")"
            self.comment += comment_str
        else:
            _mean = np.mean(array, 0)
            _variance = np.var(array, 0)
            _max, _min = np.max(array, 0), np.min(array, 0)
            _range = [[_min[i], _max[i]] for i in range(len(_max))]
            table = pd.DataFrame([_mean, _variance, _range],
                                 ["mean", "variance", "range"],
                                 self.var.columns)
            comment_str = "\n------------------------------- Data Info Table -------------------------------\n" + str(
                table
            ) + "\n------------------------------- Data Info Table -------------------------------"
            self.comment += comment_str

    def check_sub_array(self, variable):
        return False
        shape_a = np.shape(self.var)
        shape_b = np.shape(variable.var)
        if shape_a[0] < shape_b[0] or shape_a[1] < shape_b[1]:
            return False
        array_a = np.asarray(self.var)
        array_b = np.asarray(variable.var)

    def check_rel_dataframe(self, variable):
        '''
        Score:
            0 - identical name
            1 - identical content
            2 - identical shape and type
            3 - identical shape and different type
            4 - different shape but subcontent
        '''
        if type(self.var) != pd.core.frame.DataFrame or type(
                variable.var) != pd.core.frame.DataFrame:
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
        elif self.check_sub_array(variable):
            rel_score = 4
        return rel_score

    def add_rel_comment(self, variable):
        # print(self.name, variable.name)
        if self.var.equals(variable.var):
            if self.name == variable.name:
                self.comment += ", no change in the cell"
            else:
                self.comment += ", copy of " + variable.name
        elif np.shape(self.var) == np.shape(variable.var):
            if self.var.dtypes.equals(variable.var.dtypes):
                # TODO
                pass
            else:
                convert = {}
                var_a = self.var
                var_b = variable.var
                for i in range(len(var_a.dtypes)):
                    if var_a.dtypes[i] != var_b.dtypes[i]:
                        type_pair = (var_a.dtypes[i], var_b.dtypes[i])
                        if type_pair not in convert.keys():
                            convert[type_pair] = 1
                        else:
                            convert[type_pair] += 1
                blanks = " " * len("- " + self.name)
                self.comment += "\n" + blanks
                for key in convert:
                    self.comment += ", " + str(convert[key]) + " " + str(
                        key[0]
                    ) + " columns of " + variable.name + " converted to " + str(
                        key[1])


def handlecell(num, st, ed):
    comments = []
    comments.append("\'\'\'")
    input_flag = False
    out_flag = False
    first_in = -1
    first_out = -1
    for i in range(st, ed + 1):
        if myvars[i].outflag == 0 and not input_flag:
            input_flag = True
            first_in = i
            myvars[i].comment = "input\n" + myvars[i].comment
        elif myvars[i].outflag == 1 and not out_flag:
            out_flag = True
            first_out = i
            myvars[i].comment = "output\n" + myvars[i].comment

    # if first_in != -1:
    #     rend = ed
    #     if first_out != -1:
    #         rend = first_out
    #     for i in range(first_in, rend + 1):
    #         for j in range(first_in - 1, -1, -1):
    #             if myvars[j].name == myvars[i].name:
    #                 myvars[i].comment = myvars[j].comment
    #                 break
    '''
    for each output variable, find the input that is closest to it
    find rel within in/out group
    '''
    if first_out != -1 and first_in != -1:
        for i in range(first_out, ed + 1):
            choose_idx = -1
            cur_score = 5
            for j in range(first_in, first_out):
                score = myvars[i].check_rel_dataframe(myvars[j])
                # print(myvars[i].name, myvars[j].name, score)
                if cur_score > score:
                    score = cur_score
                    choose_idx = j
            if choose_idx != -1:
                myvars[i].add_rel_comment(myvars[choose_idx])

    for i in range(first_out, ed + 1):
        if type(myvars[i].var) == pd.core.frame.DataFrame:
            myvars[i].add_data_distribute(True)
        elif type(myvars[i].var) == np.ndarray or type(myvars[i].var) == list:
            myvars[i].add_data_distribute(False)

    for i in range(st, ed + 1):
        for j in range(i + 1, ed + 1):
            # if both input/output, we only check copy
            if myvars[i].outflag == myvars[j].outflag:
                if myvars[i].outflag == 0:
                    continue
                var_a = myvars[i].var
                var_b = myvars[j].var
                if type(var_a) == pd.core.frame.DataFrame:
                    if np.shape(var_a) == np.shape(var_b):
                        # find copy
                        if var_a.equals(var_b):
                            myvars[j].comment += ", copy of " + myvars[i].name

                continue

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
        myvars.append(
            Variable(tmpvars[i], labels[i][2], labels[i][0], labels[i][1], i))

    comment_str = gen_comments(labels, tmpvars)
    # print(comment_str)

    cur_cell = 0
    for cell in notebook.cells:
        if cell["cell_type"] == "code":
            cur_cell += 1
            if cur_cell in comment_str.keys():
                cell["source"] = comment_str[cur_cell] + cell["source"]

    nbformat.write(notebook,
                   os.path.join(dir_path, filename_no_suffix + "_m" + suffix))
