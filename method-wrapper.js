"use strict";
var py = require("../python-program-analysis");
var fs = require('fs');
const { printNode, RefSet } = require("../python-program-analysis");
const { trace } = require("console");

let replace_strs = [];

const str_fun = ["capitalize", "casefold", "lower", "replace", "title", "upper", "center",
    "count", "endswith", "find", "index", "isalpha",
    "isascii", "isdecimal", "isdigital", "isidentifier", "islower", "isnumeric",
    "isprintable", "isspace", "istitle", "isupper", "ljust", "lstrip", "partition",
    "rfind", "rindex", "rjust", "rpartition", "rstrip", "strip", "startswith", "zfill", "join",
    "rsplit", "split", "splitlines"]

let changed = false;
let in_def = false;


function traverse(node) {
    switch (node.type) {
        case 'assert':
            traverse(node.cond);
            break;
        case 'assign': {
            node.targets.forEach(x => traverse(x));
            node.sources.forEach(x => traverse(x));
            break;
        }
        case 'binop': {
            traverse(node.left);
            traverse(node.right);
            break;
        }
        case 'call': {
            let [id, ret] = traverse(node.func);
            node.args.forEach(x => traverse(x));
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
            node.code.forEach(stmt => traverse(stmt));
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
            in_def = true;
            node.code.forEach(stmt => traverse_and_replace(stmt));
            in_def = false;
            break;
        }
        case 'dot': {
            traverse(node.value);
            if (str_fun.includes(node.name)) {
                return [`cov(str.${node.name})`, node.value];
            }
            break;
        }
        case 'else': {
            node.code.forEach(x => traverse_and_replace(x));
            break;
        }
        case 'for': {
            node.code.forEach(x => traverse_and_replace(x));
            if (node.else)
                node.else.forEach(x => traverse_and_replace(x));
            break;
        }
        case 'if': {
            // no need to replace condition?
            // traverse(node.cond);
            node.code.forEach(x => traverse_and_replace(x));
            if (node.elif) {
                node.elif.forEach(elif => elif.code.forEach(x => traverse_and_replace(x)))
            }
            if (node.else) {
                node.else.code.forEach(x => traverse_and_replace(x));
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
            node.args.forEach(x => traverse(x));
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
            traverse(node.operand);
            break;
        }
        case 'while': {
            // traverse(node.cond);
            node.code.forEach(x => traverse_and_replace(x));
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

function traverse_and_replace(stmt) {
    traverse(stmt);
    if (changed && in_def) {
        console.log(printNode(stmt));
        replace_strs.push([stmt.location.first_line, stmt.location.last_line, [printNode(stmt)]]);
        changed = false;
    }
}

function wrap_methods(tree) {
    for (let [i, stmt] of tree.code.entries()) {
        traverse(stmt);
    }
    return replace_strs;
}

exports.wrap_methods = wrap_methods;