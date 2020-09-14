import os
import pickle
import copy
import inspect, collections, functools
import sys
store_vars = []
my_labels = []
funcs = collections.defaultdict(lambda: collections.defaultdict(list))
my_dir_path = os.path.dirname(os.path.realpath(__file__))
ignore_types = ["<class 'module'>", "<class 'type'>"]
copy_types = [
    "<class 'folium.plugins.marker_cluster.MarkerCluster'>",
    "<class 'matplotlib.axes._subplots.AxesSubplot'>"
]
TRACE_INTO = []
TYPE_1_FUN = ["capitalize", "casefold", "lower", "replace", "title", "upper"]
TYPE_2_FUN = ["rsplit", "split", "splitlines"]

noop = lambda *args, **kwargs: None
func_coverage = collections.defaultdict(set)
cur_exe = []
all_exe = collections.defaultdict(lambda: collections.defaultdict(int))


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
    if func_name not in TRACE_INTO:
        return
    line_no = frame.f_lineno
    return trace_lines


sys.settrace(trace_calls)


def my_store_info(info, var):
    if str(type(var)) in ignore_types:
        return
    my_labels.append(info)
    if str(type(var)) in copy_types:
        store_vars.append(copy.copy(var))
    else:
        store_vars.append(copy.deepcopy(var))


def func_info_saver(line):
    def inner_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func.__name__ + "_" + str(line)
            args_name = tuple(inspect.signature(func).parameters)
            arg_dict = dict(zip(args_name, args))
            arg_dict.update(kwargs)
            funcs[name]["loc"] = line
            rets = func(*args, **kwargs)
            # diff = cur_exe.difference(func_coverage[name])
            # if len(diff) > 0:
            #     print('cover new line ' + str(diff))
            #     func_coverage[name] |= diff
            all_exe[name][tuple(cur_exe)] += 1
            if all_exe[name][tuple(cur_exe)] == 1:
                funcs[name]["path"].append(copy.deepcopy(tuple(cur_exe)))
                funcs[name]["args"].append(copy.deepcopy(arg_dict))
                funcs[name]["rets"].append(copy.deepcopy([rets]))
            cur_exe.clear()
            if len(funcs[name]["saved_args"]) < 5:
                funcs[name]["saved_args"].append(copy.deepcopy(arg_dict))
                funcs[name]["saved_rets"].append(copy.deepcopy([rets]))
            return rets

        return wrapper

    return inner_decorator


def cov(f):
    @functools.wraps(f)
    def cov_wrapper_1(*args, **kwargs):
        ret = f(*args, **kwargs)
        if (ret == args[0]):
            noop
        else:
            noop
        return ret

    def cov_wrapper_2(*args, **kwargs):
        ret = f(*args, **kwargs)
        if (len(ret) <= 1):
            noop
        else:
            noop
        return ret

    if f.__name__ in TYPE_1_FUN:
        return cov_wrapper_1
    elif f.__name__ in TYPE_2_FUN:
        return cov_wrapper_2
    else:
        return f