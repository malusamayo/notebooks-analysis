import os, sys, re
from numpy.lib.function_base import append
import pandas as pd
import numpy as np
import copy as lib_copy
import inspect, collections, functools
import matplotlib, pickle, json
store_vars = collections.defaultdict(list)
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
noop = lambda *args, **kwargs: None
cur_exe = []


def ddict():
    return collections.defaultdict(ddict)


def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)


funcs = ddict()


def trace_lines(frame, event, arg):
    if event != 'line':
        return
    co = frame.f_code
    func_name = co.co_name
    line_no = frame.f_lineno
    filename = co.co_filename
    cur_exe.append(line_no)


def trace_calls(frame, event, arg):
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
    return trace_lines


sys.settrace(trace_calls)


def my_store_info(info, var):
    if str(type(var)) in ignore_types:
        return
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
            if func.__name__ not in TRACE_INTO:
                return func(*args, **kwargs)
            try:
                wrapper.cnt = (wrapper.cnt + 1) % maxrow
            except:
                wrapper.cnt = 0
            MyStr.cnt = wrapper.cnt

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

            rets = func(*args, **kwargs)
            # convert back to str
            if type(rets) == MyStr:
                rets = str(rets)
            
            path_per_row[wrapper.cnt] += cur_exe

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

# global variables for information saving
cur_cell = 0
maxrow = 1

get__keys = collections.defaultdict(list)
set__keys = collections.defaultdict(list)
path_per_row = collections.defaultdict(list)
partitions = {}

# should converted to str when return
class MyStr(str):
    cnt = 0
    def __new__(cls, content):
        return super().__new__(cls, content)
    
    def replace(self, __old: str, __new: str, __count=-1) -> str:
        ret = super().replace(__old, __new, __count)
        if self == ret:
            path_per_row[MyStr.cnt].append(0)
        else:
            path_per_row[MyStr.cnt].append(1)
        return MyStr(ret)
    
    def split(self, sep=None, maxsplit=-1):
        ret = super().split(sep, maxsplit)
        path_per_row[MyStr.cnt].append(len(ret))
        return [MyStr(x) for x in ret]

    def strip(self, __chars=None) :
        ret = super().strip(__chars)
        path_per_row[MyStr.cnt].append(int(self != ret))
        return MyStr(ret)
    
    def lower(self):
        ret = super().lower()
        path_per_row[MyStr.cnt].append(int(self != ret))
        return MyStr(ret)

    def upper(self):
        ret = super().upper()
        path_per_row[MyStr.cnt].append(int(self != ret))
        return MyStr(ret)

class LibDecorator(object):
    def __init__(self):
        super().__init__()
        pd.DataFrame.__getitem__ = self.get_decorator(pd.DataFrame.__getitem__)
        pd.DataFrame.__setitem__ = self.set_decorator(pd.DataFrame.__setitem__)
        pd.Series.replace = self.replace_decorator(pd.Series.replace)
        pd.Series.fillna = self.fillna_decorator(pd.Series.fillna)
        pd.DataFrame.fillna = self.fillna_decorator(pd.DataFrame.fillna)
        pd.Series.map  = self.map_decorator(pd.Series.map)
        pd.Series.str.split = self.str_split_decorator(pd.Series.str.split)
    
    def replace_decorator(self, wrapped_method):
        def f(x, key, value, regex):
            try:
                f.cnt = (f.cnt + 1) % maxrow
            except:
                f.cnt = 0
            if regex:
                try:
                    if type(key) == list:
                        for i, pat in enumerate(key):
                            if bool(re.search(pat, x)):
                                path_per_row[f.cnt].append(i)
                                return
                        path_per_row[f.cnt].append(-1)
                    elif bool(re.search(key, x)):
                        path_per_row[f.cnt].append(1)
                    else:
                        path_per_row[f.cnt].append(0)
                except:
                    path_per_row[f.cnt].append(-2) # error
            elif type(key) == list:
                path_per_row[f.cnt].append(key.index(x) if x in key else -1)
            else:
                if x == key:
                    path_per_row[f.cnt].append(1)
                else:
                    path_per_row[f.cnt].append(0)
        def decorate(self, to_replace=None, value=None, inplace=False, limit=None, regex=False, method="pad"):
            if to_replace != None:
                self.map(lambda x: f(x, to_replace, value, regex))
            return wrapped_method(self, to_replace, value, inplace, limit, regex, method)
        return decorate

    def fillna_decorator(self, wrapped_method):
        def f(x, value):
            try:
                f.cnt = (f.cnt + 1) % maxrow
            except:
                f.cnt = 0
            if pd.api.types.is_numeric_dtype(type(x)) and np.isnan(x):
                path_per_row[f.cnt].append(1)
            else:
                path_per_row[f.cnt].append(0)

        def decorate(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
            if type(self) == pd.DataFrame:
                update_maxrow(len(self))
                for i, v in enumerate(self.isnull().sum(axis=1)):
                    path_per_row[i].append(v)
            else:
                self.map(lambda x: f(x, value))
            return wrapped_method(self, value, method, axis, inplace, limit, downcast)
        return decorate

    def str_split_decorator(self, wrapped_method):
        def f(x, pat, n):
            try:
                f.cnt = (f.cnt + 1) % maxrow
            except:
                f.cnt = 0
            try:
                ret = x.split(pat, n)
                path_per_row[f.cnt].append(len(ret))
            except AttributeError:
                path_per_row[f.cnt].append(-2) # x not str
        def decorate(self, pat=None, n=-1, expand=False):
            self._parent.map(lambda x: f(x, pat, n))
            return wrapped_method(self, pat, n, expand)
        return decorate

    def map_decorator(self, wrapped_method):
        def f(x, d):
            try:
                f.cnt = (f.cnt + 1) % maxrow
            except:
                f.cnt = 0
            path_per_row[f.cnt].append(list(d).index(x) if x in d else -1)
        def decorate(self, arg, na_action=None):
            update_maxrow(len(self))
            if type(arg) == dict:
                self.map(lambda x: f(x, arg))
            return wrapped_method(self, arg, na_action)
        return decorate


    def get_decorator(self, method):     
        def append(key, ls):
            if pd.core.dtypes.common.is_hashable(key) and key not in ls:
                ls.append(key)
        def decorate(self, key):
            if type(key) == list:
                for item in key:
                    append(item, get__keys[cur_cell])
            else:
                append(key, get__keys[cur_cell])
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
            else:
                append(key, set__keys[cur_cell])
            return method(self, key, value)
        return decorate

# we now update maxrow before each map/apply
def update_maxrow(l):
    global maxrow
    maxrow = l

def set_partition():
    if not path_per_row:
        return
    row_eq = collections.defaultdict(list)
    for k, v in path_per_row.items():
        row_eq[str(tuple(v))].append(k)
    partitions[cur_cell] = row_eq
    path_per_row.clear()

def update_path(idx, path):
    path_per_row[idx].append(path)

libDec = LibDecorator()