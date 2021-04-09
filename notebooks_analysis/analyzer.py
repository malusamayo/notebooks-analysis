import nbformat
import sys
import os
import pickle
import numpy as np
import pandas as pd
import torch
import random
import collections
from nbconvert import PythonExporter, HTMLExporter
import json, copy
import itertools
import queue
import warnings
from model import DataFrame, dispatch_gen 

pd.set_option('display.max_columns', None)
pd.set_option('precision', 4)
warnings.simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(precision=4)

# sys.argv.append("notebooks/debug_example.ipynb")

dir_path = os.path.dirname(os.path.realpath(sys.argv[1]))
filename = sys.argv[1].split('\\')[-1].split('/')[-1]
filename_no_suffix = filename[:filename.rfind(".")]
filename_no_suffix = filename_no_suffix[:-2] if filename_no_suffix.endswith("_m") else filename_no_suffix
suffix = filename[filename.rfind("."):]

data_path = os.path.join(dir_path, filename_no_suffix)
output_path = os.path.join(dir_path, filename_no_suffix + "_m" + suffix)
nb_path = os.path.join(dir_path, filename_no_suffix + ".ipynb")
html_path = os.path.join(dir_path, filename_no_suffix + ".html")
comment_path = os.path.join(dir_path, filename_no_suffix + "_comment.json")
json_out_path = os.path.join(data_path, "result.json")

blanks = "\t- "
postfix = "[auto]"
access_path = []

def print_error(msg):
    print("\033[91m", msg, "\033[0m")


class Pattern(object):
    COMPUTE = "compute"
    FILLNA = "fillna"
    MERGE = "merge"
    STRAN = "str_transform"
    SUBSTR = "substr"
    NTRAN = "num_transform"
    CONV = "type_convert"
    FLOAT = "float"
    STR = "str"
    TIME = "datetime64"
    CAT = "category"
    INT = "int"
    ENCODE = "encode"
    ONEHOT = "one_hot_encoding"
    COSTS = {COMPUTE:15, FILLNA:2, MERGE:2, STRAN:3, SUBSTR: 2,
        NTRAN:3, CONV:3, FLOAT:2, STR:2, TIME: 2,
        CAT:2, INT:2, ENCODE:2, ONEHOT:2}
    DSL = list(COSTS.keys())

    def __init__(self, pattern=""):
        self.patterns = []
        self.cost = 0
        if pattern != "":
            self.add(pattern)

    def add(self, pattern):
        self.patterns.append(pattern)
        self.cost += Pattern.COSTS[pattern]
    
    def addAll(self, patterns):
        self.patterns += patterns
        self.cost += sum([Pattern.COSTS[p] for p in patterns])

    def copy(self):
        cp = Pattern()
        cp.addAll(self.patterns.copy())
        return cp
    
    def __gt__(self, other):
        return self.cost > other.cost

    def __eq__(self, other):
        return self.cost == other.cost

    def __ge__(self, other):
        return self.cost >= other.cost


class PatternSynthesizer(object):

    '''
    df1: before, df2: after, col: the target column
    '''
    def __init__(self, DF1, DF2, info, in_vars=[]):
        self.df1 = DF1.var if type(DF1) == DataFrame else DF1
        self.df2 = DF2.var if type(DF2) == DataFrame else DF2
        self.df1_name = DF1.name if type(DF1) == DataFrame else ""
        self.df2_name = DF2.name if type(DF2) == DataFrame else ""
        self.cellnum = DF1.cellnum
        # rename columns to str type
        self.cols1 = list(self.df1.columns.astype(str))
        self.cols2 = list(self.df2.columns.astype(str))
        self.df1.rename(str, axis = 1, inplace = True)
        self.df2.rename(str, axis = 1, inplace = True)
        self.srccols = [col for col in info.get if col in self.cols1]
        self.other_src = {x.name: x.var for x in in_vars if type(x.var) in [pd.Series, pd.DataFrame] and x != DF1}
        self.descols = [col for col in info.set if col in self.cols1] 
        self.partition = collections.defaultdict(list)
        if self.df1_name in info.par:
            self.partition = copy.deepcopy(info.par[self.df1_name])
        self.syn_stack = []
        self.summary = collections.defaultdict(list)
        self.markers = {}
        self.table = {}
    
    def synthesis_append(self, pattern, from_col, to_col):
        self.syn_stack.append((pattern, from_col, to_col))
        if pattern == "rearrange":
            self.summary["other_patterns"].append({pattern: 
                ','.join(from_col) + '|' + ','.join(to_col)})
        elif len(to_col) > 0:
            self.summary[','.join(from_col) + '|' + ','.join(to_col)].append(pattern)
        else:
            self.summary["other_patterns"].append({pattern: ','.join(from_col)})

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
    
    def check_datetime(self, df, col):
        return pd.api.types.is_datetime64_dtype(df[col])

    def prune_DSL(self, df1, df2, from_col, to_col):
        res = [Pattern.COMPUTE]
        hint = {"convert": False, Pattern.FILLNA:False}

        def check_encode(df1, df2, from_col, to_col):
            l = sorted(df2[to_col].unique())
            if len(l) == max(l) + 1 - min(l):
                hint["encode"] = True
                if len(l) <= 2:
                    res.append(Pattern.ONEHOT)
                else:
                    res.append(Pattern.ENCODE)

        if str(df1[from_col].dtype) != str(df2[to_col].dtype):
            hint["convert"] = True
            if self.check_float(df2, to_col):
                res.append(Pattern.FLOAT)
                check_encode(df1, df2, from_col, to_col)
            elif self.check_cat(df2, to_col):
                res.append(Pattern.CAT)
            elif self.check_int(df2, to_col):                
                res.append(Pattern.INT)
                check_encode(df1, df2, from_col, to_col)
            elif self.check_str(df2, to_col):
                res.append(Pattern.STR)
            elif self.check_datetime(df2, to_col):
                res.append(Pattern.TIME)
            else:
                res.append(Pattern.CONV)
        else:
            if self.check_num(df2, to_col):
                check_encode(df1, df2, from_col, to_col)

        # carefully handling na in a column
        if len(df1[from_col].unique()) > len(df2[to_col].unique()):
            res.append(Pattern.MERGE)
        elif self.check_num(df1, from_col):
            res.append(Pattern.NTRAN)
        elif self.check_str(df1, from_col):
            res.append(Pattern.STRAN)
            # res.append(Pattern.SUBSTR)
        
        if self.check_fillna(df1, df2, from_col, to_col):
            hint[Pattern.FILLNA] = True
            res.append(Pattern.FILLNA)

        return res, hint

    def validate(self, df1, df2, from_col, to_col, patterns, hint):
        
        CONVERT = {Pattern.CONV, Pattern.FLOAT, Pattern.STR, Pattern.CAT, Pattern.INT, Pattern.ENCODE, Pattern.ONEHOT, Pattern.TIME}
        SYM = "SYMBOLIC"
        ANY =  "any"

        # basic pruning
        if Pattern.COMPUTE in patterns:
            return 1

        if hint["convert"] and not (CONVERT & set(patterns)):
            return -1

        if "encode" in hint and Pattern.ENCODE not in patterns and Pattern.ONEHOT not in patterns:
            return -1
        
        if hint[Pattern.FILLNA] and Pattern.FILLNA not in patterns:
            return 0
        
        # helper functions
        def typeof(df, col):
            if self.check_float(df, col):
                return Pattern.FLOAT
            elif self.check_cat(df, col):
                return Pattern.CAT
            elif self.check_int(df, col):
                return Pattern.INT
            elif self.check_str(df, col):
                return Pattern.STR
            elif self.check_datetime(df, col):
                return Pattern.TIME
            else:
                return ANY
        
        def convertable(f, f_col):
            try:
                res = f_col.astype(type_before).astype(f)
                # if f == int:
                #     res = f_col.astype(float).astype(f)
                # else:
                #     res = f_col.astype(f)
                return True, res
            except:
                return False, None
        
        
        tmp = df1[from_col].copy().astype(str)
        target  = df2[to_col]
        type_before = typeof(df1, from_col)
        constraints = {"type": type_before, "substr": False}

        for p in patterns[::-1]:
            if p == Pattern.FILLNA:
                tmp[tmp == 'nan'] = SYM
                constraints["na_filled"] = True
            if p in [Pattern.INT, Pattern.FLOAT, Pattern.STR, Pattern.CAT, Pattern.TIME]:            
                cmp_idx = ~tmp.str.startswith(SYM)
                flag, res = convertable(p, tmp[cmp_idx])
                if flag:
                    tmp[cmp_idx] = res.astype(str)
                    constraints["type"] = p
                else:
                    return 0
            elif p in [Pattern.ENCODE, Pattern.ONEHOT]:
                # add stronger constraints for one-hot? how to deal with nan?
                tmp2idx = {x:SYM + '_' + str(i) for i, x in enumerate(tmp.unique())}
                tmp2idx[SYM] = SYM
                tmp = tmp.map(tmp2idx)
                if p == Pattern.ENCODE:
                    constraints["cont-int"] = True
                elif p == Pattern.ONEHOT:
                    constraints["one-hot"] = True
                # constraints["substr"] = False
            elif p == Pattern.CONV:
                tmp.loc[:] = SYM
                constraints["type"] = ANY
                # constraints["substr"] = False
            elif p == Pattern.MERGE:
                tmp.loc[:] = SYM
                constraints["unique_before"] = len(df1[from_col].unique())
                # constraints["substr"] = False
            elif p in [Pattern.STRAN, Pattern.SUBSTR] and constraints["type"] == Pattern.STR:
                tmp.loc[:] = SYM
                # constraints["substr"] = False
                # if p == Pattern.SUBSTR:
                #     constraints["substr"] = True
            elif p == Pattern.NTRAN and constraints["type"] in [Pattern.INT, Pattern.FLOAT, ANY]:
                tmp.loc[:] = SYM
                # constraints["substr"] = False
        
        sym_idx = tmp.str.startswith(SYM, na=False)
        cmp_idx = ~sym_idx
        if (tmp[cmp_idx].astype(str) != target[cmp_idx].astype(str)).any():
            return 0
        
        # validate constraints
        if "na_filled" in constraints:
            if target.isnull().any():
                return -1
        
        if "unique_before" in constraints:
            if constraints["unique_before"] <= len(df2[to_col].unique()): 
                return -1

        if "cont-int" in constraints or "one-hot" in constraints:
            l = sorted(df2[to_col].unique())
            tmp = tmp[tmp[sym_idx] != SYM]
            if "cont-int" in constraints:
                if len(l) != max(l) + 1 - min(l):
                    return -1
                for v in l:
                    if tmp[df2[to_col] == v].nunique() > 1:
                        return -1
            if "one-hot" in constraints:
                if len(l) > 2:
                    return -1
                for v in l:
                    if tmp[df2[to_col] == v].nunique() <= 1:
                        return 1
                return 0
        
        # if constraints["substr"]:
        #     df1[from_col].map(df2[to_col].astype(str))

        return 1
        

    # per column searching
    def check_column(self, df1, df2, from_col, to_col):
        worklist = queue.PriorityQueue()
        cur_DSL, hint = self.prune_DSL(df1, df2, from_col, to_col)
        for pattern in cur_DSL:
            p = Pattern(pattern)
            worklist.put(p)
        
        top = Pattern(Pattern.COMPUTE)

        while not worklist.empty():
            cur = worklist.get()
            if cur >= top:
                break
            ret = int(self.validate(df1, df2, from_col, to_col, cur.patterns, hint))
            if ret == -1:
                continue
            if ret == 1:
                top = cur
                continue
            for pattern in cur_DSL:
                if pattern not in cur.patterns:
                    p = cur.copy()
                    p.add(pattern)
                    worklist.put(p)
        
        return top

    def check_copy(self, df1, df2):
        if df1.equals(df2):
            self.synthesis_append("copy", [], [])
            return True
        return False

    def check_concat(self, df1, df2):
        if len(df1) == len(df2) and df1.shape[1] < df2.shape[1] and set(df1.columns).issubset(set(df2.columns)) and self.other_src:
            candidates = []
            cols = set(df1.columns)
            cols2 = set(df2.columns)
            for name, var in self.other_src.items():
                if len(var) != len(df2):
                    continue
                if type(var) == pd.DataFrame:
                    if set(var.columns).issubset(cols2):
                        candidates.append(name)
                        cols.update(var.columns)
                elif type(var) == pd.Series:
                    # what happen if series have no name? default column name: 0, 1, ...
                    if var.name in cols2: 
                        candidates.append(name)
                        cols.add(var.name)
            if cols == cols2:
                df_test = pd.concat([df1] + [self.other_src[var_name] for var_name in candidates], axis=1)[df2.columns]
                if df_test.equals(df2):
                    self.synthesis_append("concat_col", [self.df1_name] + candidates, [])
                    return True
        elif len(df1) < len(df2) and df1.shape[1] == df2.shape[1] and set(df1.index).issubset(set(df2.index)) and self.other_src:
            candidates = []
            idxes = set(df1.index)
            idxes2 = set(df2.index)
            for name, var in self.other_src.items():
                if type(var) == pd.DataFrame:
                    if set(var.index).issubset(idxes2):
                        candidates.append(name)
                        idxes.update(var.index)
            if idxes == idxes2:
                df_test = pd.concat([df1] + [self.other_src[var_name] for var_name in candidates], axis=0)
                df_test.reindex(df2.index)
                if df_test.equals(df2):
                    self.synthesis_append("concat_row", [self.df1_name] + candidates, [])
                    return True
        return False


    def check_removecol(self, df1, df2):
        self.removedcols = [x for x in self.cols1 if x not in self.cols2]
        if self.removedcols:        
            self.cols1 = [x for x in self.cols1 if x in self.cols2]
            self.synthesis_append("removecol", self.removedcols, [])
            return True
        return False
    
    def check_rearrange(self, df1, df2):
        if self.cols1 != self.cols2 and set(self.cols1) == set(self.cols2):
            self.synthesis_append("rearrange", self.cols1, self.cols2)
            return True
        return False

    def check_removerow(self, df1, df2):
        # [TODO] add other cases
        self.removedrow = pd.DataFrame()
        # use index to track row mappings
        if len(df2) < len(df1) and set(df2.index).issubset(set(df1.index)):
            removed = df1.loc[~df1.index.isin(df2.index)]
            self.removedrow = removed
            left = df1.loc[df1.index.isin(df2.index)]
            removed_null = removed.isnull()
            left_null = left.isnull()
            # all removed rows contain nan
            if removed_null.any(axis=1).all():
                # select columns that some removed rows contain nan & remaining rows contain no nan
                any_nan = set(removed_null.columns[removed_null.any()])
                no_nan = set(left_null.columns[~left_null.any()])
                self.synthesis_append("removerow_null", [str(len(removed))] + list(any_nan & no_nan), [])
            elif len(left.merge(removed)) == len(left):
                self.synthesis_append("removerow_dup", [str(len(removed))], [])
            else:
                self.synthesis_append("removerow", [str(len(removed))], [])
            return True
        return False

    def gen_defaut_partition(self, df1, df2):
        MAGIC_BOUND = 25
        paths = collections.defaultdict(list)
        for col in self.colsnew:
            if df2[col].nunique() > MAGIC_BOUND:
                continue
            for i in df2.index:
                paths[i].append([str(df2[col].at[i]), "default_"+ col])
        for col in self.colschange:
            # look at diff
            if df2[col].compare(df1[col])["self"].nunique() > MAGIC_BOUND:
                continue
            for i in df2.index:
                try:
                    if type(df2[col].at[i]) == pd.Series:
                        if df2[col].at[i].equals(df1[col].at[i]):
                            paths[i].append(["DUMMY", "default_"+ col])
                        else:
                            paths[i].append([str(df2[col].at[i]), "default_"+ col])
                    else:
                        if df2[col].at[i] == df1[col].at[i]:
                            paths[i].append(["DUMMY", "default_"+ col])
                        else:
                            paths[i].append([str(df2[col].at[i]), "default_"+ col])
                except:
                    print_error("bugs when generating partition")
                    return
        if len({str(i) for i in paths.values()}) > MAGIC_BOUND:
            return
        for k, v in paths.items():
            self.partition[str(v)].append(k)
        # print(self.partition.keys())

    def get_src(self, col):
        src = []
        if self.srccols:
            set_idxes = [i for i, acc in enumerate(access_path) if acc[0] == col and acc[1] == self.cellnum and acc[3] == True]
            get_idxes = [i for i, acc in enumerate(access_path) if acc[0] in self.srccols and acc[1] <= self.cellnum and acc[3] == False]
            # set_idxes = [i for i, acc in enumerate(access_path) if acc[0] == col and acc[1] == self.cellnum and acc[2] == True]
            # get_idxes = [i for i, acc in enumerate(access_path) if acc[0] in self.srccols and acc[1] <= self.cellnum and acc[2] == False]
            for set_idx in set_idxes:
                filtered_get_idxes = [i for i in get_idxes if i<=set_idx]
                # set could be intialized without src
                if filtered_get_idxes:
                    lineno = access_path[filtered_get_idxes[-1]][2]
                    src_candidates = [access_path[i][0] for i in filtered_get_idxes if access_path[i][2] == lineno]
                    for candiate in src_candidates:
                        if candiate not in src:
                            src.append(candiate)
                # get_idx = set_idx - 1
                # while get_idx >= 0:
                #     if get_idx in get_idxes and access_path[get_idx][0] not in src:
                #         src.append(access_path[get_idx][0])
                #         break
                #     get_idx -= 1
        if not src:
            src = self.srccols
        return src

    def search(self, df1, df2):
        # early detection of one-hot encoding:
        one_hots = [col for col in self.colsnew if set(df2[col].unique()).issubset({0,1})]
        col_left = [col for col in self.colsnew]
        
        all_src = []
        
        while one_hots:        
            s = pd.Series(np.zeros(len(df2)), dtype=int, index = df2.index)
            candidates = []
            for col in one_hots:
                if (s.loc[df2[col] == 1] == 0).all():
                    s += df2[col]
                    candidates.append(col)
                if (s == 1).all():
                    break
            if (s == 1).all():
                one_hots = [col for col in one_hots if col not in candidates]
                col_left = [col for col in col_left if col not in candidates]
                all_src_candidaes = []  
                for col in candidates:
                    src = self.get_src(col)
                    all_src_candidaes += [col for col in src if col not in all_src_candidaes]
                all_src += all_src_candidaes
                if not all_src_candidaes:
                    all_src_candidaes.append(self.df1_name)
                self.synthesis_append(Pattern.ONEHOT, all_src_candidaes, candidates)
            else:
                break                    

        if "index" in col_left and pd.Series(df1.index).equals(df2["index"]):
            self.synthesis_append("reset_index", [], [])
            col_left.remove("index")

        for col in col_left:
            src = self.get_src(col)
            all_src += src
                            
            patterns = collections.defaultdict(list)
            for src_col in src:
                top = self.check_column(df1, df2, src_col, col)
                patterns[tuple(top.patterns)].append(src_col)
            
            for src_col, src_series in self.other_src.items():
                if type(src_series) == pd.Series and src_series.index.equals(df2.index):
                    top = self.check_column(pd.DataFrame({src_col: src_series}), df2, src_col, col)
                    patterns[tuple(top.patterns)].append(src_col)

            for p_ls, src in patterns.items():
                p_ls = list(p_ls)
                for p in p_ls:
                    self.synthesis_append(p, src, [col])
            
            # no src col     
            if not patterns:
                self.synthesis_append(Pattern.COMPUTE, [self.df1_name], [col])

        if self.colsnew:
            self.srccols = [col for col in self.srccols if col in all_src]

        for col in self.colschange:
            top = self.check_column(df1, df2, col, col)    
            for p in top.patterns:
                self.synthesis_append(p, [col], [col])
        
        # generate default partition
        if not self.partition:
            self.gen_defaut_partition(df1, df2)


    def check(self, df1, df2):
        # ignore irrelevant dfs
        # if set(self.cols1).isdisjoint(set(self.cols2)):
        #     return

        # check copy and concat first
        if self.check_copy(df1, df2) or self.check_concat(df1, df2):
            return self.summary

        # ignore adding rows
        if len(df1) < len(df2):
            return False

        # handle easy op: removerow, removecol, rearrange
        if self.check_removerow(df1, df2):
            # if index is reset this might lead to error?
            df1 = df1.loc[df1.index.isin(df2.index)]
        # rows not removed -> index not subset -> irrelevant dfs
        if len(df1) > len(df2):
            return False
        if not df1.index.equals(df2.index):
            if df1.index.sort_values().equals(df2.index.sort_values()):
                self.synthesis_append("rearrange_row", [], [])
                df1 = df1.reindex(df2.index)
            else:
                print_error(f"{self.cellnum}: mismatched index for {self.df1_name} and {self.df2_name}")
                return False
        self.check_removecol(df1, df2)
        self.check_rearrange(df1, df2)

        # locate created/changed columns
        self.colsnew = [col for col in self.cols2 if col not in self.cols1]
        self.colschange = [col for col in self.cols1 if not df1[col].equals(df2[col])]
        # print(self.colsnew, self.colschange)
        if self.colsnew or self.colschange:
            self.search(df1, df2)
        if self.syn_stack:
            self.gen_table(df1, df2)
        # elif not self.colsnew and not self.colschange:
        #     self.synthesis_append("copy", [], [])
        return self.summary

    def gen_table(self, df1, df2):
        df = df2.copy()
        
        for col in self.removedcols:
            df[col] = df1[col]

        # generate extra info
        def get_unique(col):
            return str(len(col.unique()))
        def get_range(col):
            if str(col.dtype) == Pattern.CAT:
                return ""
            if np.issubdtype(col.dtype, np.number):
                return [np.min(col), np.max(col)]
            else:
                return ""
        _type = {col:str(df[col].dtype) for col in df if col not in self.colschange}
        _range = {col: str(get_range(df[col])) for col in df if col not in self.colschange}
        _unique = {col: str(get_unique(df[col])) for col in df if col not in self.colschange}

        # build data changes for columns
        for col in self.colschange:
            _type[col] = str(df1[col].dtype) + "->" + str(df[col].dtype)
            _range[col] = str(get_range(df1[col])) + "->" + str(get_range(df[col]))
            _unique[col] = str(get_unique(df1[col])) + "->" + str(get_unique(df[col]))
            df[col] = df1[col].astype(str) + ['->']*len(df1) +  df[col].astype(str)
        for col in df.columns:
            if self.check_cat(df, col):
                df[col] = df[col].astype(str)

        
        # sort examples 
        new_df = pd.DataFrame()
        if self.partition:
            # sort self.partition first by frequency
            self.partition = dict(sorted(self.partition.items(), key=lambda item: len(item[1]), reverse=True))
            for k, l in dict(self.partition).items():
                self.markers[k] = len(new_df)
                # removerow -> some items in l might not be in df.index
                new_df = new_df.append(df.loc[df.index.isin(l)])
        else:
            self.markers["[[0, 'empty']]"] = len(new_df)
            new_df = df

        if not self.removedrow.empty:
            self.markers["[[0, 'removed']]"] = len(new_df)
            new_df = pd.concat([new_df, self.removedrow])[new_df.columns]
        
        df = pd.concat([pd.DataFrame([_type, _unique, _range]), new_df], ignore_index=True)

        # rearrange cols to make changed/new cols first
        colsleft = [col for col in df.columns if col not in self.colsnew + self.colschange + self.removedcols]
        colssrc = [col for col in colsleft if col in self.srccols]
        colsleft = [col for col in colsleft if col not in self.srccols]
        df = df.reindex(columns = self.removedcols + self.colschange + self.colsnew + colssrc + colsleft)

        def rename(col):
            if col in self.colschange:
                return col + "*" + postfix
            elif col in self.colsnew:
                return col + "+" + postfix
            elif col in self.removedcols:
                return col + "-" + postfix
            elif col in self.srccols:
                return col + ">" + postfix
            return col

        df.rename(rename, axis =1 ,inplace = True)

        # print(self.markers)
        self.table = json.loads(df.to_json())
        

class Info(object):
    def __init__(self, info, cellnum):
        super().__init__()
        self.get = []
        self.set = []
        self.par = collections.defaultdict(lambda : collections.defaultdict(list))
        if info == None:
            return
        if str(cellnum) in info["get"]:
            self.get = info["get"][str(cellnum)]
        if str(cellnum) in info["set"]:
            self.set = info["set"][str(cellnum)]
        if str(cellnum) in info["par"]:
            self.par = info["par"][str(cellnum)]

def score(df1, df2):
    diff_len = len(set(df1.columns) - set(df2.columns))
    return abs(df1.shape[0] - df2.shape[0] + 1) * (diff_len + 1) ** 2

def handlecell(myvars, st, ed, info):
    # comments = ["\'\'\'"]
    comments = []
    cell_num = myvars[st].cellnum

    ins = [var for var in myvars[st:ed+1] if not var.outflag]
    outs = [var for var in myvars[st:ed+1] if var.outflag]

    # build json maps
    json_map = {"input": {}, "output": {}, "summary": {}, "partition": {}, "table":{}}
    for in_var in ins:
        json_map["input"][in_var.name] = in_var.json_map
    for out_var in outs:
        json_map["output"][out_var.name] = out_var.json_map

    '''
    for each output variable, find the input that is closest to it
    find rel within in/out group
    '''
    for out_var in outs:
        if type(out_var.var) == pd.DataFrame:
            s_map = {x: score(x.var, out_var.var) for x in ins if type(x.var)==pd.DataFrame}
            if s_map:
                (in_var, _) = min(s_map.items(), key=lambda x: x[1])
                checker = PatternSynthesizer(in_var, out_var, info, ins)
                result = checker.check(in_var.var, out_var.var)
                if result:
                    flow = ' '.join([in_var.name, "->", out_var.name])
                    json_map["summary"][flow] = dict(checker.summary)
                    if checker.table:
                        json_map["partition"][flow] = checker.markers
                        json_map["table"][flow] = checker.table
                    print(cell_num, ":", flow, "\033[96m", 
                        dict(checker.summary), len(checker.markers), "\033[0m")           

    return "\n".join(comments), json_map

if __name__ == "__main__":
    with open(nb_path, encoding="UTF-8") as f:
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

    # load static comments
    # static_comments = {}
    # with open(json_path) as f:
    #     json_tmp_list = json.load(f)
    #     for [idx, content] in json_tmp_list:
    #         static_comments[idx] = content

    json_map = {}
    info = {}
    # funcs = {}
    with open(os.path.join(data_path, "info.json"), 'r') as j:
        info = json.loads(j.read())
    access_path = info["graph"]


    for file in sorted(os.listdir(data_path)):
        myvars = []
        if file == "info.json" or file.endswith("_f.dat"):
            continue
        elif file.endswith(".dat"):
            with open(os.path.join(data_path, file), "rb") as f:
                try:
                    vars = pickle.load(f)
                except:    
                    print_error("error when pickle from " + file)
                    continue
                for i in range(len(vars)):
                    try:
                        myvars.append(
                            dispatch_gen(vars[i][0], vars[i][1][2], vars[i][1][0], vars[i][1][1]))
                        if type(vars[i][0]) == list and len(vars[i][0]) <= 3:
                            for j in range(len(vars[i][0])):
                                myvars.append(
                                    dispatch_gen(vars[i][0][j], vars[i][1][2] + f"[{j}]", vars[i][1][0], vars[i][1][1]))
                    except:
                        print_error("error when dispatch var " + vars[i][1][2])
                        pass
                # comments = static_comments[vars[0][1][0]] if vars[0][1][0] in static_comments.keys() else []
                comment, json_map = handlecell(myvars, 0, len(myvars)-1, Info(info, vars[0][1][0]))
                # notebook.cells[code_indices[vars[0][1][0] - 1]].source = f"'''\n{comment}\n'''\n" + notebook.cells[code_indices[vars[0][1][0] - 1]].source
                
                # distributed
                with open(os.path.join(data_path, f"result_{code_indices[vars[0][1][0] - 1]}.json"), "w") as f:
                    f.write(json.dumps(json_map))
                # with open(os.path.join(data_path, f"code_{code_indices[vars[0][1][0] - 1]}.py"), "w") as f:
                #     f.write(notebook.cells[code_indices[vars[0][1][0] - 1]].source)

    # fill not existing entries
    # for key, value in json_map.items():
    #     cat_list = ["input", "output", "summary", "partition", "table"]
    #     for cat in cat_list:
    #         if cat not in value.keys():
    #             json_map[key][cat] = {}

    # with open(json_out_path, "w") as f:
    #     f.write(json.dumps(json_map))


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

    # for comment in static_comments:
    #     (i, j) = line_to_idx[comment[0] - 3]
    #     insert_to_map(json_map, i, "comment", j, comment[1])
    #     insert_map[i].append((j, "# [autodocs] " + comment[1] + "\n"))
