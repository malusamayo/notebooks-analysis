from analyzer import PatternSynthesizer, Info
import pandas as pd

def test_case1():
    df1 = pd.read_csv('./notebooks/input/train.csv')
    df2 = df1.copy()
    df2 = df1[df1.isnull().any(axis=1)]
    df2['Sex'] = df2['Sex'].map({'male':0, 'female':1})
    checker = PatternSynthesizer(df1, df2, Info(None, None))
    l = checker.check(df1, df2)
    print("df1", "->","df2", "\033[96m", l, "\033[0m")

test_case1()