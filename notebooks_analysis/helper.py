import os, sys, re
from numpy.lib.function_base import append
import pandas as pd
import numpy as np
import copy as lib_copy
import inspect, collections, functools
import matplotlib, pickle, json

from pandas.core.series import Series
# my_labels = []
my_dir_path = os.path.dirname(os.path.realpath(__file__))
ignore_types = [
    "<class 'module'>", "<class 'type'>", "<class 'function'>",
    "<class 'matplotlib.figure.Figure'>"
]
TRACE_INTO = []



# TYPE_1_FUN = ["capitalize", "casefold", "lower", "replace", "title", "upper"]
# TYPE_2_FUN = ["rsplit", "split", "splitlines"]

matplotlib.use('Agg')

# global variables for information saving
store_vars = collections.defaultdict(list)
cur_cell = 0
cur_exe = []
get__keys = collections.defaultdict(list)
set__keys = collections.defaultdict(list)
id2name = {}
cur_get = []
graph = collections.defaultdict(list)
# noop = lambda *args, **kwargs: None

# def ddict():
#     return collections.defaultdict(ddict)


# def ddict2dict(d):
#     for k, v in d.items():
#         if isinstance(v, dict):
#             d[k] = ddict2dict(v)
#     return dict(d)


# funcs = ddict()

def my_store_info(info, var):
    if str(type(var)) in ignore_types:
        return
    if type(var) in [pd.DataFrame]:
        id2name[id(var.index)] = info[2]
    store_vars[info[0]].append((wrap_copy(var), info))


def wrap_copy(var):
    try:
        return lib_copy.deepcopy(var)
    except NotImplementedError:
        return "NOT COPIED"
    except TypeError:
        return "NOT COPIED"
    except SystemError:
        return "NOT COPIED"


def func_info_saver(line):
    def inner_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if func.__name__ not in TRACE_INTO and func.__name__ != '<lambda>':
                return func(*args, **kwargs)
            pathTracker.next_iter()

            # name = func.__name__ + "_" + str(line)
            # args_name = tuple(inspect.signature(func).parameters)
            # arg_dict = dict(zip(args_name, args))
            # arg_dict.update(kwargs)
            # funcs[name]["loc"] = line

            # convert arg of str to MyStr
            new_args = []
            for arg in list(args):
                if type(arg) == str:
                    new_args.append(MyStr(arg))
                else:
                    new_args.append(arg)
            args = tuple(new_args)

            # should make sure it is inside map/apply
            rets = func(*args, **kwargs)
            # convert back to str
            if type(rets) == MyStr:
                rets = str(rets)
            
            pathTracker.update_ls(cur_exe)
            # path_per_row[wrapper.cnt] += cur_exe

            # if tuple(cur_exe) not in funcs[name].keys():
            #     funcs[name][tuple(cur_exe)]["count"] = 0
            #     funcs[name][tuple(cur_exe)]["args"] = []
            #     funcs[name][tuple(cur_exe)]["rets"] = []

            # funcs[name][tuple(cur_exe)]["count"] += 1
            # if funcs[name][tuple(cur_exe)]["count"] <= 10:
            #     funcs[name][tuple(cur_exe)]["args"].append(wrap_copy(arg_dict))
            #     funcs[name][tuple(cur_exe)]["rets"].append([wrap_copy(rets)])
            cur_exe.clear()
            return rets

        return wrapper

    return inner_decorator

# should converted to str when return
class MyStr(str):
    # cnt = 0
    def __new__(cls, content):
        return super().__new__(cls, content)
    
    def replace(self, __old: str, __new: str, __count=-1) -> str:
        ret = super().replace(__old, __new, __count)
        if self == ret:
            pathTracker.update(0)
        else:
            pathTracker.update(1)
        return MyStr(ret)
    
    def split(self, sep=None, maxsplit=-1):
        ret = super().split(sep, maxsplit)
        pathTracker.update(len(ret))
        return [MyStr(x) for x in ret]

    def strip(self, __chars=None) :
        ret = super().strip(__chars)
        pathTracker.update(int(self != ret))
        return MyStr(ret)
    
    def lower(self):
        ret = super().lower()
        pathTracker.update(int(self != ret))
        return MyStr(ret)

    def upper(self):
        ret = super().upper()
        pathTracker.update(int(self != ret))
        return MyStr(ret)

def if_expr_wrapper(expr):
    pathTracker.update(int(expr))
    return expr

class LibDecorator(object):
    def __init__(self):
        super().__init__()
        pd.DataFrame.__getitem__ = self.get_decorator(pd.DataFrame.__getitem__)
        pd.DataFrame.__setitem__ = self.set_decorator(pd.DataFrame.__setitem__)
        pd.core.indexing._LocationIndexer.__setitem__ = self.index_set_decorator(pd.core.indexing._LocationIndexer.__setitem__)
        pd.core.indexing._ScalarAccessIndexer.__setitem__ = self.index_set_decorator(pd.core.indexing._ScalarAccessIndexer.__setitem__)
        pd.Series.replace = self.replace_decorator(pd.Series.replace)
        pd.Series.fillna = self.fillna_decorator(pd.Series.fillna)
        pd.DataFrame.fillna = self.fillna_decorator(pd.DataFrame.fillna)
        pd.Series.map  = self.map_decorator(pd.Series.map)
        pd.Series.apply  = self.apply_decorator(pd.Series.apply)
        pd.DataFrame.apply  = self.df_apply_decorator(pd.DataFrame.apply)
        pd.Series.str.split = self.str_split_decorator(pd.Series.str.split)

        # reset index when appending rows
        pd.concat = self.concat_decorator(pd.concat)
        pd.DataFrame.merge = self.merge_decorator(pd.DataFrame.merge)
    
    def replace_decorator(self, wrapped_method):
        def f(x, key, value, regex):
            pathTracker.next_iter()
            if regex:
                try:
                    if type(key) == list:
                        for i, pat in enumerate(key):
                            if bool(re.search(pat, x)):
                                pathTracker.update(i)
                                return
                        pathTracker.update(-1)
                    elif bool(re.search(key, x)):
                        pathTracker.update(1)
                    else:
                        pathTracker.update(0)
                except:
                    pathTracker.update(-2) # error
            elif type(key) == list:
                pathTracker.update(key.index(x) if x in key else -1)
            else:
                if x == key:
                    pathTracker.update(1)
                else:
                    pathTracker.update(0)
        def decorate(self, to_replace=None, value=None, inplace=False, limit=None, regex=False, method="pad"):
            if to_replace != None:
                self.map(lambda x: f(x, to_replace, value, regex))
            return wrapped_method(self, to_replace, value, inplace, limit, regex, method)
        return decorate

    def fillna_decorator(self, wrapped_method):
        def f(x, value):
            pathTracker.next_iter()
            if pd.api.types.is_numeric_dtype(type(x)) and np.isnan(x):
                pathTracker.update(1)
            else:
                pathTracker.update(0)

        def decorate(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
            if type(self) == pd.DataFrame:
                pathTracker.reset(self.index)
                for i, v in enumerate(self.isnull().sum(axis=1)):
                    pathTracker.next_iter()
                    pathTracker.update(v)
            else:
                self.map(lambda x: f(x, value))
            if inplace:
                cur_get.clear()
            return wrapped_method(self, value, method, axis, inplace, limit, downcast)
        return decorate

    def str_split_decorator(self, wrapped_method):
        def f(x, pat, n):
            pathTracker.next_iter()
            try:
                ret = x.split(pat, n)
                pathTracker.update(len(ret))
            except AttributeError:
                pathTracker.update(-2) # x not str
        def decorate(self, pat=None, n=-1, expand=False):
            self._parent.map(lambda x: f(x, pat, n))
            return wrapped_method(self, pat, n, expand)
        return decorate

    def map_decorator(self, wrapped_method):
        def f(x, d):
            pathTracker.next_iter()
            pathTracker.update(list(d).index(x) if x in d else -1)
        def decorate(self, arg, na_action=None):
            # should do init work here
            pathTracker.reset(self.index)
            if type(arg) == dict:
                self.map(lambda x: f(x, arg))
            return wrapped_method(self, arg, na_action)
        return decorate

    def apply_decorator(self, wrapped_method):
        def decorate(self, func, convert_dtype=True, args=(), **kwds):
            pathTracker.reset(self.index)
            if kwds:
                return wrapped_method(self, func, convert_dtype, args, kwds=kwds)
            else:
                return wrapped_method(self, func, convert_dtype, args)
        return decorate


    def df_apply_decorator(self, wrapped_method):
        def decorate(self, func, axis=0, raw=False, result_type=None, args=(), **kwds):
            pathTracker.reset(self.index)
            if kwds:
                return wrapped_method(self, func, axis, raw, result_type, args, kwds=kwds)
            else:
                return wrapped_method(self, func, axis, raw, result_type, args)
        return decorate

    def concat_decorator(self, wrapped_method):
        def decorate(objs, axis=0, join="outer", ignore_index = False, keys=None, levels=None, names=None, verify_integrity = False, sort = False, copy = True):
            if axis == 0:
                ignore_index = True
            return wrapped_method(objs, axis, join, ignore_index, keys, levels, names, verify_integrity, sort, copy)
        return decorate

    def merge_decorator(self, wrapped_method):
        def decorate(other, ignore_index=False, verify_integrity=False, sort=False):
            ignore_index = True
            return wrapped_method(other, ignore_index, verify_integrity, sort)
        return decorate

    def get_decorator(self, method):     
        def append(key, ls):
            if pd.core.dtypes.common.is_hashable(key) and key not in ls:
                ls.append(key)
        def decorate(self, key):
            if type(key) == list:
                for item in key:
                    append(item, get__keys[cur_cell])
                    append(item, cur_get)
            else:
                append(key, get__keys[cur_cell])
                append(key, cur_get)
            return method(self, key)
        return decorate
    def set_decorator(self, method):
        def append(key, ls):
            if pd.core.dtypes.common.is_hashable(key) and key not in ls:
                ls.append(key)
        def decorate(self, key, value):
            if type(key) == list:
                for item in key:
                    append(item, set__keys[cur_cell])
                    graph[item] += cur_get
            else:
                append(key, set__keys[cur_cell])
                graph[key] += cur_get
            cur_get.clear()
            return method(self, key, value)
        return decorate
    def index_set_decorator(self, method):
        def append(key, ls):
            if pd.core.dtypes.common.is_hashable(key) and key not in ls:
                ls.append(key)
        def decorate(self, key, value):
            if hasattr(self, "obj") and type(self.obj) == pd.Series:
                append(self.obj.name, set__keys[cur_cell])
                graph[self.obj.name] += cur_get
            cur_get.clear()
            return method(self, key, value)
        return decorate

class PathTracker(object):
    def __init__(self) -> None:
        super().__init__()
        self.paths = collections.defaultdict(lambda: collections.defaultdict(list))
        self.partitions = {}
        sys.settrace(self.trace_calls)

    def reset(self, index):
        self.index = index
        self.id = id(index)
        if self.id in id2name:
            self.id = id2name[self.id]
        self.iter = iter(index)
        self.cur_idx = -1

    def next_iter(self):
        # try:
        self.cur_idx = next(self.iter)
        # except StopIteration:
        #     self.cur_idx = next(iter(self.index))

    def update(self, new_path):
        self.paths[self.id][self.cur_idx].append(new_path)

    def update_ls(self, new_paths):
        self.paths[self.id][self.cur_idx] += new_paths

    def to_partition(self):
        if not self.paths:
            return
        row_eq = {}
        for i, path in self.paths.items():
            row_eq[i] = collections.defaultdict(list)
            for k, v in path.items():
                row_eq[i][str(tuple(v))].append(k)
        self.partitions[cur_cell] = row_eq
        self.paths.clear()
        cur_get.clear()

    def trace_lines(self, frame, event, arg):
        if event != 'line':
            return
        co = frame.f_code
        func_name = co.co_name
        line_no = frame.f_lineno
        filename = co.co_filename
        cur_exe.append(line_no)


    def trace_calls(self, frame, event, arg):
        if event != 'call':
            return
        co = frame.f_code
        func_name = co.co_name
        try:
            if func_name not in TRACE_INTO:
                return
        except TypeError:
            print(func_name, TRACE_INTO)
        line_no = frame.f_lineno
        return self.trace_lines

libDec = LibDecorator()
pathTracker = PathTracker()