"use strict";
var py = require("../../notebooks-analysis");
var fs = require('fs');
const { printNode, RefSet } = require("../../notebooks-analysis");

let replace_strs = [];

const str_fun = ["capitalize", "casefold", "lower", "replace", "title", "upper", "center",
    "count", "endswith", "find", "index", "isalpha",
    "isascii", "isdecimal", "isdigital", "isidentifier", "islower", "isnumeric",
    "isprintable", "isspace", "istitle", "isupper", "ljust", "lstrip", "partition",
    "rfind", "rindex", "rjust", "rpartition", "rstrip", "strip", "startswith", "zfill", "join",
    "rsplit", "split", "splitlines"]

let changed = false;

function find_col(node) {
    switch (node.type) {
        case 'assert':
            find_col(node.cond);
            break;
        case 'assign': {
            node.targets.forEach(x => find_col(x));
            node.sources.forEach(x => find_col(x));
            break;
        }
        case 'binop': {
            find_col(node.left);
            find_col(node.right);
            break;
        }
        case 'call': {
            find_col(node.func);
            node.args.forEach(x => find_col(x.actual));
            break;
        }
        case 'class': {
            node.code.forEach(stmt => find_col(stmt));
            break;
        }
        // case 'decorator':
        //     return ('@' +
        //         node.decorator +
        //         (node.args ? '(' + commaSep(node.args) + ')' : ''));
        // case 'decorate':
        //     return (tabs +
        //         lines(node.decorators, tabLevel) +
        //         printTabbed(node.def, tabLevel));
        case 'def': {
            node.code.forEach(stmt => find_col(stmt, true));
            break;
        }
        case 'dot': {
            find_col(node.value);
            if (node.value.type == "name" &&
                find_col.pyTypeof.get(node.value.id) == "dataframe")
                find_col.cols.add(node.name)
            break;
        }
        case 'else': {
            node.code.forEach(x => find_col(x));
            break;
        }
        case 'for': {
            node.code.forEach(x => find_col(x));
            if (node.else)
                node.else.forEach(x => find_col(x));
            break;
        }
        case 'if': {
            // no need to replace condition?
            // find_col(node.cond);
            node.code.forEach(x => find_col(x));
            if (node.elif) {
                node.elif.forEach(elif => elif.code.forEach(x => find_col(x)))
            }
            if (node.else) {
                node.else.code.forEach(x => find_col(x));
            }
            break;
        }
        // case 'ifexpr': {
        //     break;
        // }
        case 'index': {
            find_col(node.value);
            node.args.forEach(x => find_col(x));
            if (node.value.type == "name" &&
                find_col.pyTypeof.get(node.value.id) == "dataframe" &&
                node.args.length == 1 &&
                node.args[0].type == "literal")
                find_col.cols.add(node.args[0].value)
            break;
        }
        case 'lambda': {
            find_col(node.code);
            break;
        }
        // case 'list':
        //     return '[' + node.items.map(function (item) { return printNode(item); }).join(comma) + ']';
        // case 'module':
        //     return lines(node.code, tabLevel);
        // case 'name':
        //     return node.id;
        // case 'nonlocal':
        //     return tabs + 'nonlocal ' + node.names.join(comma);
        case 'return': {
            node.values.forEach(v => find_col(v));
            break;
        }
        // case 'set':
        //     return '{' + commaSep(node.entries) + '}';
        // case 'slice':
        //     return ((node.start ? printNode(node.start) : '') +
        //         ':' +
        //         (node.stop ? printNode(node.stop) : '') +
        //         (node.step ? ':' + printNode(node.step) : ''));
        // case 'starred':
        //     find_col(node.value);
        // case 'try':
        //     return (tabs +
        //         'try:' +
        //         lines(node.code, tabLevel + 1) +
        //         (node.excepts
        //             ? node.excepts.map(function (ex) {
        //                 return tabs +
        //                     'except ' +
        //                     (ex.cond
        //                         ? printNode(ex.cond) + (ex.name ? ' as ' + ex.name : '')
        //                         : '') +
        //                     ':' +
        //                     lines(ex.code, tabLevel + 1);
        //             })
        //             : '') +
        //         (node.else ? tabs + 'else:' + lines(node.else, tabLevel + 1) : '') +
        //         (node.finally
        //             ? tabs + 'finally:' + lines(node.finally, tabLevel + 1)
        //             : ''));
        // case 'tuple':
        //     node.items.forEach(x => find_col(x));
        case 'unop': {
            find_col(node.operand);
            break;
        }
        case 'while':
        case 'with': {
            node.code.forEach(x => find_col(x));
            break;
        }
        // case 'yield':
        //     return (tabs +
        //         'yield ' +
        //         (node.from ? printNode(node.from) : '') +
        //         (node.value ? commaSep(node.value) : ''));
    }
    return [];
}

function traverse(node, in_def) {
    switch (node.type) {
        case 'assert':
            traverse(node.cond, in_def);
            break;
        case 'assign': {
            node.targets.forEach(x => traverse(x, in_def));
            node.sources.forEach(x => traverse(x, in_def));
            break;
        }
        case 'binop': {
            traverse(node.left, in_def);
            traverse(node.right, in_def);
            break;
        }
        case 'call': {
            let [id, ret] = traverse(node.func, in_def);
            node.args.forEach(x => traverse(x, in_def));
            if (ret) {
                changed = true;
                node.func = {
                    id: id,
                    type: 'name'
                }
                node.args.unshift({
                    actual: ret,
                    type: 'arg'
                })
            }
            break;
        }
        case 'class': {
            node.code.forEach(stmt => traverse(stmt, in_def));
            break;
        }
        // case 'decorator':
        //     return ('@' +
        //         node.decorator +
        //         (node.args ? '(' + commaSep(node.args) + ')' : ''));
        // case 'decorate':
        //     return (tabs +
        //         lines(node.decorators, tabLevel) +
        //         printTabbed(node.def, tabLevel));
        case 'def': {
            node.code.forEach(stmt => traverse_and_replace(stmt, true));
            break;
        }
        case 'dot': {
            traverse(node.value, in_def);
            // imperfect solutions, dynamic type checking of nested expressions could be expensive
            if (str_fun.includes(node.name)) {
                return [`cov(type(${printNode(node.value).replace("cov", "")}).${node.name})`, node.value];
            }
            break;
        }
        case 'else': {
            node.code.forEach(x => traverse_and_replace(x, in_def));
            break;
        }
        case 'for': {
            node.code.forEach(x => traverse_and_replace(x, in_def));
            if (node.else)
                node.else.forEach(x => traverse_and_replace(x, in_def));
            break;
        }
        case 'if': {
            // no need to replace condition?
            // traverse(node.cond);
            node.code.forEach(x => traverse_and_replace(x, in_def));
            if (node.elif) {
                node.elif.forEach(elif => elif.code.forEach(x => traverse_and_replace(x, in_def)))
            }
            if (node.else) {
                node.else.code.forEach(x => traverse_and_replace(x, in_def));
            }
            break;
        }
        // case 'ifexpr':
        //     return (printNode(node.then) +
        //         ' if ' +
        //         printNode(node.test) +
        //         ' else ' +
        //         printNode(node.else));
        case 'index': {
            traverse(node.value);
            node.args.forEach(x => traverse(x, in_def));
            break;
        }
        // case 'lambda':
        //     return ('lambda ' +
        //         node.args.map(printParam).join(comma) +
        //         ': ' +
        //         printNode(node.code));
        // case 'list':
        //     return '[' + node.items.map(function (item) { return printNode(item); }).join(comma) + ']';
        // case 'module':
        //     return lines(node.code, tabLevel);
        // case 'name':
        //     return node.id;
        // case 'nonlocal':
        //     return tabs + 'nonlocal ' + node.names.join(comma);
        // case 'return':
        //     return tabs + 'return ' + (node.values ? commaSep(node.values) : '');
        // case 'set':
        //     return '{' + commaSep(node.entries) + '}';
        // case 'slice':
        //     return ((node.start ? printNode(node.start) : '') +
        //         ':' +
        //         (node.stop ? printNode(node.stop) : '') +
        //         (node.step ? ':' + printNode(node.step) : ''));
        // case 'starred':
        //     traverse(node.value);
        // case 'try':
        //     return (tabs +
        //         'try:' +
        //         lines(node.code, tabLevel + 1) +
        //         (node.excepts
        //             ? node.excepts.map(function (ex) {
        //                 return tabs +
        //                     'except ' +
        //                     (ex.cond
        //                         ? printNode(ex.cond) + (ex.name ? ' as ' + ex.name : '')
        //                         : '') +
        //                     ':' +
        //                     lines(ex.code, tabLevel + 1);
        //             })
        //             : '') +
        //         (node.else ? tabs + 'else:' + lines(node.else, tabLevel + 1) : '') +
        //         (node.finally
        //             ? tabs + 'finally:' + lines(node.finally, tabLevel + 1)
        //             : ''));
        // case 'tuple':
        //     node.items.forEach(x => traverse(x));
        case 'unop': {
            traverse(node.operand, in_def);
            break;
        }
        case 'while': {
            // traverse(node.cond);
            node.code.forEach(x => traverse_and_replace(x, in_def));
            break;
        }
        // case 'with':
        //     return (tabs +
        //         'with ' +
        //         node.items.map(function (w) { return w.with + (w.as ? ' as ' + w.as : ''); }).join(comma) +
        //         ':' +
        //         lines(node.code, tabLevel + 1));
        // case 'yield':
        //     return (tabs +
        //         'yield ' +
        //         (node.from ? printNode(node.from) : '') +
        //         (node.value ? commaSep(node.value) : ''));
    }
    return [];
}

function traverse_and_replace(stmt, in_def) {
    traverse(stmt, in_def);
    if (changed && in_def) {
        console.log(printNode(stmt));
        replace_strs.push([stmt.location.first_line, stmt.location.last_line, [printNode(stmt)]]);
        changed = false;
    }
}

function wrap_methods(tree) {
    for (let [i, stmt] of tree.code.entries()) {
        let in_def = (stmt.type == "def");
        traverse(stmt, in_def);
    }
    return replace_strs;
}

function collect_defs(code) {
    let def_list = []
    for (let [i, stmt] of code.entries()) {
        if (stmt.type == "def") {
            def_list.push([stmt.name, stmt.location.first_line]);
            def_list.concat(collect_defs(stmt.code));
        } else if (stmt.type == "class") {
            def_list.concat(collect_defs(stmt.code));
        }
    }
    return def_list;
}

function collect_cols(stmt, pyTypeof) {
    find_col.cols = new Set();
    find_col.pyTypeof = pyTypeof;
    if (stmt.type == 'assign')
        stmt.sources.forEach(x => find_col(x))
    else
        find_col(stmt);
    return find_col.cols;
}

exports.collect_defs = collect_defs;
exports.wrap_methods = wrap_methods;
exports.collect_cols = collect_cols;