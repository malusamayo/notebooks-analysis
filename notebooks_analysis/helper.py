import os, sys, re
import pandas as pd
import numpy as np
import copy as lib_copy
import collections, functools
import matplotlib, pickle, json
from inspect import getframeinfo, stack

script_path = os.path.realpath(__file__)
my_dir_path = os.path.dirname(os.path.realpath(__file__))
ignore_types = [
    "<class 'module'>", "<class 'type'>", "<class 'function'>",
    "<class 'matplotlib.figure.Figure'>", "<class 'tensorflow.python.keras.engine.sequential.Sequential'>"
]
reset_index_types = [
    "<class 'pandas.core.indexes.range.RangeIndex'>", "<class 'pandas.core.indexes.numeric.Int64Index'>"
]

TRACE_INTO = []

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
setter2lines = collections.defaultdict(set)
getter2lines = collections.defaultdict(set)
line2cell = {}
# noop = lambda *args, **kwargs: None

def my_store_info(info, var):
    if str(type(var)) in ignore_types:
        return
    if type(var) in [pd.DataFrame] and info[1] == 0:
        if str(type(var.index)) in reset_index_types:
            var.reset_index(inplace=True, drop=True)
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

            try:
                pathTracker.next_iter()
            except:
                # don't track apply/map of other objects
                pathTracker.clean()
                return func(*args, **kwargs)

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
            
            if cur_exe:
                pathTracker.update(lib_copy.copy(cur_exe), func.__name__)
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
        pathTracker.update(int(self != ret), "replace")
        return MyStr(ret)
    
    def split(self, sep=None, maxsplit=-1):
        ret = super().split(sep, maxsplit)
        pathTracker.update(len(ret), "split")
        return [MyStr(x) for x in ret]

    def strip(self, __chars=None) :
        ret = super().strip(__chars)
        pathTracker.update(int(self != ret), "strip")
        return MyStr(ret)
    
    def lower(self):
        ret = super().lower()
        pathTracker.update(int(self != ret), "lower")
        return MyStr(ret)

    def upper(self):
        ret = super().upper()
        pathTracker.update(int(self != ret), "upper")
        return MyStr(ret)

def if_expr_wrapper(expr):
    if pathTracker.cur_idx >= 0:
        pathTracker.update(int(expr), "if_expr")
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
        # pd.concat = self.concat_decorator(pd.concat) (disabled due to bugs)
        # pd.DataFrame.merge = self.merge_decorator(pd.DataFrame.merge)
    
    def replace_decorator(self, wrapped_method):
        def f(x, key, value, regex):
            pathTracker.next_iter()
            if type(value) in [list, tuple, range]:
                if regex:
                    for i, pat in enumerate(key):
                        try:
                            if bool(re.search(pat, x)):
                                pathTracker.update(i, "replace_ls")
                                return
                        except:
                            pathTracker.update(-2, "replace_ls") # error
                            return
                    pathTracker.update(-1, "replace_ls")
                else:
                    pathTracker.update(key.index(x) if x in key else -1, "replace_ls")
            elif type(key) == list:
                if regex:
                    try:
                        pathTracker.update(int(any(re.search(item, x) for item in key)), "replace")
                    except:
                        pathTracker.update(-2, "replace") # error
                else:
                    pathTracker.update(int(x in key), "replace")
            else:
                if regex:
                    try:
                        pathTracker.update(int(re.search(key, x)), "replace")
                    except:
                        pathTracker.update(-2, "replace") # error
                else:
                    pathTracker.update(int(x != key), "replace")
        def decorate(self, to_replace=None, value=None, inplace=False, limit=None, regex=False, method="pad"):
            if to_replace != None and type(to_replace) != dict:
                self.map(lambda x: f(x, to_replace, value, regex))
            return wrapped_method(self, to_replace, value, inplace, limit, regex, method)
        return decorate

    def fillna_decorator(self, wrapped_method):
        def f(x, value):
            pathTracker.next_iter()
            if pd.api.types.is_numeric_dtype(type(x)) and np.isnan(x):
                pathTracker.update(1, "fillna")
            else:
                pathTracker.update(0, "fillna")

        def decorate(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
            if type(self) == pd.DataFrame:
                pathTracker.reset(self.index)
                for i, v in enumerate(self.isnull().sum(axis=1)):
                    pathTracker.next_iter()
                    pathTracker.update(int(v), "fillna")
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
                pathTracker.update(len(ret), "split")
            except AttributeError:
                pathTracker.update(-2, "split") # x not str
        def decorate(self, pat=None, n=-1, expand=False):
            self._parent.map(lambda x: f(x, pat, n))
            return wrapped_method(self, pat, n, expand)
        return decorate

    def map_decorator(self, wrapped_method):
        def f(x, d):
            pathTracker.next_iter()
            pathTracker.update(list(d).index(x) if x in d else -1, "map_dict")
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
            if axis == 1 or axis == 'columns':
                pathTracker.reset(self.index)
            else:
                pathTracker.clean()
            if kwds:
                return wrapped_method(self, func, axis, raw, result_type, args, kwds=kwds)
            else:
                return wrapped_method(self, func, axis, raw, result_type, args)
        return decorate

    # def concat_decorator(self, wrapped_method):
    #     def decorate(objs, axis=0, join="outer", ignore_index = False, keys=None, levels=None, names=None, verify_integrity = False, sort = False, copy = True):
    #         if axis == 0:
    #             ignore_index = True
    #         return wrapped_method(objs, axis, join, ignore_index, keys, levels, names, verify_integrity, sort, copy)
    #     return decorate

    # def merge_decorator(self, wrapped_method):
    #     def decorate(other, ignore_index=False, verify_integrity=False, sort=False):
    #         ignore_index = True
    #         return wrapped_method(other, ignore_index, verify_integrity, sort)
    #     return decorate

    def get_decorator(self, method):     
        def append(key, ls):
            if pd.core.dtypes.common.is_hashable(key) and key not in ls:
                ls.append(key)
        def decorate(self, key):
            caller = getframeinfo(stack()[1][0])
            line2cell[caller.lineno] = cur_cell
            if script_path.endswith(caller.filename):  
                if type(key) == list:
                    for item in key:
                        append(item, get__keys[cur_cell])
                        append(item, cur_get)
                        getter2lines[item].add(caller.lineno)
                elif type(key) == str:
                    append(key, get__keys[cur_cell])
                    append(key, cur_get)
                    getter2lines[key].add(caller.lineno)
            return method(self, key)
        return decorate
    def set_decorator(self, method):
        def append(key, ls):
            if pd.core.dtypes.common.is_hashable(key) and key not in ls:
                ls.append(key)
        def decorate(self, key, value):
            caller = getframeinfo(stack()[1][0])
            line2cell[caller.lineno] = cur_cell
            if script_path.endswith(caller.filename):
                if type(key) == list:
                    for item in key:
                        append(item, set__keys[cur_cell])
                        graph[item] += cur_get
                        setter2lines[item].add(caller.lineno)
                elif type(key) == str:
                    append(key, set__keys[cur_cell])
                    graph[key] += cur_get            
                    setter2lines[key].add(caller.lineno)
                cur_get.clear()
            return method(self, key, value)
        return decorate
    def index_set_decorator(self, method):
        def append(key, ls):
            if pd.core.dtypes.common.is_hashable(key) and key not in ls:
                ls.append(key)
        def index_model(key, index):
            if len(key) != len(index):
                return
            pathTracker.reset(index)
            for i, v in enumerate(key):
                pathTracker.next_iter()
                pathTracker.update(int(v), "loc/at")
        def decorate(self, key, value):
            caller = getframeinfo(stack()[1][0])
            line2cell[caller.lineno] = cur_cell
            if script_path.endswith(caller.filename):  
                if hasattr(self, "obj") and type(self.obj) == pd.Series:
                    append(self.obj.name, set__keys[cur_cell])
                    graph[self.obj.name] += cur_get
                    setter2lines[self.obj.name].add(caller.lineno)
                    # maybe we could model scalr/slice?
                    if type(key) == pd.Series and key.dtype == bool:
                        index_model(key, self.obj.index)
                if hasattr(self, "obj") and type(self.obj) == pd.DataFrame:
                    if type(key) == tuple and type(key[0]) == pd.Series and key[0].dtype == bool:
                        index_model(key[0], self.obj.index)
                cur_get.clear()
            return method(self, key, value)
        return decorate

class PathTracker(object):
    def __init__(self) -> None:
        super().__init__()
        self.paths = collections.defaultdict(lambda: collections.defaultdict(list))
        self.partitions = {}
        self.cur_idx = -1
        sys.settrace(self.trace_calls)

    def reset(self, index):
        self.index = index
        self.id = id(index)
        if self.id in id2name:
            self.id = id2name[self.id]
        self.iter = iter(index)
        self.cur_idx = 0
    
    def clean(self):
        self.iter = iter(())
        self.cur_idx = -1

    def next_iter(self):
        # try:
        self.cur_idx = next(self.iter)
        # except StopIteration:
        #     self.cur_idx = next(iter(self.index))

    def update(self, new_path, func_name):
        self.paths[self.id][self.cur_idx].append([new_path, func_name])

    def to_partition(self):
        if not self.paths:
            return
        row_eq = {}
        for i, path in self.paths.items():
            row_eq[i] = collections.defaultdict(list)
            for k, v in path.items():
                row_eq[i][str(v)].append(k)
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
            return
        line_no = frame.f_lineno
        return self.trace_lines

libDec = LibDecorator()
pathTracker = PathTracker()