import inspect
import collections
import functools

funcs = collections.defaultdict(lambda: collections.defaultdict(list))


# def func_info_saver(line):
def inner_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        name = func.__name__ + "_" + str(id(func))
        args_name = tuple(inspect.signature(func).parameters)
        arg_dict = dict(zip(args_name, args))
        arg_dict.update(kwargs)
        # funcs[name]["loc"] = line
        if len(funcs[name]["args"]) < 5:
            funcs[name]["args"].append(arg_dict)
        rets = func(*args, **kwargs)
        if len(funcs[name]["rets"]) < 5:
            funcs[name]["rets"].append(rets)
        return rets

    return wrapper

    # return inner_decorator


@inner_decorator
def say_whee(x, y, z=3):
    print("Whee!" + x + y * z)


say_whee("Me", "hh", z=4)
import pandas as pd
df = pd.DataFrame([[1, 2], [3, 4]])
f = inner_decorator(lambda x: x * 2)
df[0] = df[0].map(f)

for x, y in funcs.items():
    print(x)
    for s, t in y.items():
        print(s, t)
print(dict(funcs))