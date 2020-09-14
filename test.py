import inspect
import collections
import functools

funcs = collections.defaultdict(lambda: collections.defaultdict(list))

import trace
import sys

TRACE_INTO = ['say_whee']
func_coverage = collections.defaultdict(set)
cur_exe = set()


def trace_lines(frame, event, arg):
    if event != 'line':
        return
    co = frame.f_code
    func_name = co.co_name
    line_no = frame.f_lineno
    filename = co.co_filename
    cur_exe.add(line_no)
    # print('  %s line %s' % (func_name, line_no))


def trace_calls(frame, event, arg):
    if event != 'call':
        return
    co = frame.f_code
    func_name = co.co_name
    if func_name not in TRACE_INTO:
        return
    line_no = frame.f_lineno
    # print('Call to %s on line %s' % (func_name, line_no))
    # Trace into this function
    return trace_lines


# sys.settrace(trace_calls)


def cov(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        print(args[0])
        ret = f(*args, **kwargs)
        if (ret == args[0]):
            print("same")
        else:
            print("no")
        return ret

    return wrapper


cov(str.replace)("x y", " ", " ")


# def func_info_saver(line):
def inner_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        name = func.__name__ + "_" + str(id(func))
        args_name = tuple(inspect.signature(func).parameters)
        arg_dict = dict(zip(args_name, args))
        arg_dict.update(kwargs)

        rets = func(*args, **kwargs)
        diff = cur_exe.difference(func_coverage[name])
        if len(diff) > 0:
            print('cover new line ' + str(diff))
            func_coverage[name] |= diff
        cur_exe.clear()

        if len(diff) > 0:
            funcs[name]["args"].append(arg_dict)
            funcs[name]["rets"].append(rets)
        elif len(funcs[name]["saved_args"]) < 5:
            funcs[name]["saved_args"].append(arg_dict)
            funcs[name]["saved_rets"].append(rets)

        return rets

    return wrapper

    # return inner_decorator


@inner_decorator
def say_whee(x, y, z=3):
    print("Whee!" + x + y * z)
    if z == 3:
        z = x
    else:
        z = y


say_whee("Me", "hh", z=4)

say_whee("Me", "hh")
import pandas as pd
df = pd.DataFrame([[1, 2], [3, 4]])
f = inner_decorator(lambda x: x * 2)
g = "a b".split()
df[0] = df[0].map(f)

# for x, y in funcs.items():
#     print(x)
#     for s, t in y.items():
#         print(s, t)
# print(dict(funcs))
# print(func_coverage)