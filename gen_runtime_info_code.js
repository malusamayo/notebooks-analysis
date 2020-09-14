"use strict";
var py = require("../python-program-analysis");
var fs = require('fs');
const { printNode, RefSet } = require("../python-program-analysis");
const { matchesProperty, map, head } = require("lodash");
const { printArg } = require("./dist/es5/printNode");
const { ADDRCONFIG } = require("dns");
const { assert } = require("console");
const { collapseTextChangeRangesAcrossMultipleVersions } = require("typescript");
let args = process.argv.slice(2);
let path = args[0];
//const path = './python-examples/python/';
let filename = path.split('\\').pop().split('/').pop();
let filename_no_suffix = filename.substring(0, filename.lastIndexOf('.'));
let suffix = filename.substring(filename.lastIndexOf('.'));
let dir = path.replace(filename, '');
let text = fs.readFileSync(path).toString();
let lineToCell = new Map();
let ins = new Map();
let outs = new Map();
let replace_strs = [];
let head_str = fs.readFileSync("headstr.py").toString();
let def_list = [];
let branch_insert_list = [];

let TYPE_1_FUN = ["capitalize", "casefold", "lower", "replace", "title", "upper"];
let TYPE_2_FUN = ["center", "count", "endswith", "find", "index", "isalpha",
    "isascii", "isdecimal", "isdigital", "isidentifier", "islower", "isnumeric",
    "isprintable", "isspace", "istitle", "isupper", "ljust", "lstrip", "partition",
    "rfind", "rindex", "rjust", "rpartition", "rstrip", "strip", "startswith", "zfill"];
let TYPE_3_FUN = ["rsplit", "split", "splitlines"]

let write_str =
    "store_vars.append(my_labels)\n" +
    "store_vars.append(dict(funcs))\n" +
    "store_vars.append(dict(all_exe))\n" +
    "f = open(os.path.join(my_dir_path, \"" + filename_no_suffix +
    "_log.dat\"), \"wb\")\n" +
    "pickle.dump(store_vars, f)\n" +
    "f.close()\n";

function init_lineToCell() {
    let lines = text.split("\n");
    let max_line = lines.length;
    let cur_cell = 0;
    for (let i = 0; i < max_line; i++) {
        if (lines[i].startsWith('# In['))
            cur_cell++;
        if (lines[i].startsWith("#"))
            continue;
        lineToCell.set(i + 1, cur_cell);
    }
    // console.log(lines);
}

function add(map, key, value) {
    if (map.get(key) == undefined)
        map.set(key, []);
    if (map.get(key).find(x => x == value) == undefined)
        map.get(key).push(value);
}

// add vars from external input or used for plotting 
function add_extra_vars(tree) {
    for (let stmt of tree.code) {
        // if (stmt.type == "assign") {
        //     // external input: x = pd.read_csv()
        //     for (let [i, src] of stmt.sources.entries()) {
        //         if (src.type == "call" && src.func.name == "read_csv") {
        //             add(ins, lineToCell.get(stmt.location.first_line), stmt.targets[i].id)
        //         }
        //     }
        // }

        // add plotting vars
        if (stmt.type == "call") {
            if (stmt.func.name == "plot") {
                let cell = lineToCell.get(stmt.location.first_line);
                if (stmt.func.value.type == "index")
                    add(outs, cell, stmt.func.value.value.id);
                else if (stmt.func.value.id == "plt") {
                    add(outs, cell, stmt.args[0].actual.id);
                    add(outs, cell, stmt.args[1].actual.id);
                } else
                    add(outs, cell, stmt.func.value.id);
            }
            if (["factorplot", "countplot", "barplot"].includes(stmt.func.name)) {
                for (let arg of stmt.args) {
                    if ("keyword" in arg && arg.keyword.id == "data") {
                        add(outs, lineToCell.get(stmt.location.first_line), arg.actual.id);
                    }
                }
            }
        }
    }
}

// recursively find whether current subtree contains a node of certain type
function contain_type(node, type) {
    if (node == undefined)
        return undefined;
    if (node.type == type)
        return printNode(node);
    if (node.targets != undefined) {
        for (let des of node.targets) {
            let res = contain_type(des, type);
            if (res != undefined)
                return res;
        }
    }
    if (node.sources != undefined) {
        for (let src of node.sources) {
            let res = contain_type(src, type);
            if (res != undefined)
                return res;
        }
    }
    if (node.args != undefined) {
        for (let arg of node.args) {
            let res = contain_type(arg.actual, type);
            if (res != undefined)
                return res;
        }
    }
    return undefined;
}

function value_type_handler(type, node) {
    if (type == "index") {
        assert(node.args.length == 1);
        assert(node.args[0].type == "literal");
        let col = node.args[0].value;
        return "[" + col.replace(/['"]+/g, '') + "]";
    } else if (type == "dot") {
        return "[" + node.name.replace(/['"]+/g, '') + "]";
    }
}

// function decompose_func(node, name) {
//     if (node.type == "call" && node.func.type == "dot") {
//         let res = decompose_func(node.func.value, name);
//         if (res.length >= 1)
//             node.value = name;
//         res.push(name.id + " = " + printNode(node));
//         return res;
//     } else if (node.type == "index") {
//         let res = decompose_func(node.value, name);
//         if (res.length >= 1)
//             node.value = name;
//         res.push(name.id + " = " + printNode(node));
//         return res;
//     } else if (node.type == "name") {
//         return [];
//     }
// }

function static_analyzer(tree) {
    let static_comments = new Map();
    for (let [i, stmt] of tree.code.entries()) {
        // console.log(printNode(stmt));
        let lambda = contain_type(stmt, "lambda");
        if (lambda != undefined) {
            // should also record/convert lambda function later
            let lambda_rep = "func_info_saver(" + stmt.location.first_line + ")(" + lambda + ")";
            let stmt_str = printNode(stmt);
            stmt_str = stmt_str.replace(lambda, lambda_rep);
            replace_strs.push([stmt.location.first_line, stmt.location.last_line, [stmt_str]]);
        }
        if (stmt.type == "assign") {
            // external input: x = pd.read_csv()
            for (let [i, src] of stmt.sources.entries()) {
                // x[y] = x1[y1].map(...) || x.y = x1.y1.map(...)
                if (src.type == "call" && src.func.name == "map") {
                    let value_type = ["index", "dot"];
                    if (value_type.includes(stmt.targets[i].type)
                        && value_type.includes(src.func.value.type)) {
                        let src_col = value_type_handler(src.func.value.type, src.func.value);
                        let des_col = value_type_handler(stmt.targets[i].type, stmt.targets[i]);
                        let comment = ""
                        // same/different literal
                        if (src_col == des_col)
                            comment = "modify column " + des_col + " using map"
                        else
                            comment = "create column " + des_col + " from " + src_col
                        static_comments.set(stmt.location.first_line,
                            comment);
                    }
                }
                if (src.type == "call" && src.func.name == "apply") {
                    let value_type = ["index", "dot"];
                    if (value_type.includes(stmt.targets[i].type)) {
                        let des_col = value_type_handler(stmt.targets[i].type, stmt.targets[i]);
                        static_comments.set(stmt.location.first_line,
                            "create column " + des_col + " from whole row");
                    }
                }
                // x = pd.get_dymmies()
                if (src.type == "call" && src.func.name == "get_dummies") {
                    static_comments.set(stmt.location.first_line,
                        "encoding in dummy variables");
                }
                // x1, x2, y1, y2 = train_test_split()
                if (src.type == "call" && src.func.name == "train_test_split") {
                    static_comments.set(stmt.location.first_line,
                        "spliting data to train set and test set");
                }
                // x = df.select_dtypes().columns
                if (src.type == "dot" && src.name == "columns") {
                    if (src.value.type == "call" && src.value.func.name == "select_dtypes")
                        static_comments.set(stmt.location.first_line,
                            "select columns of specific data types");
                }
                // x.at[] = ... || x.loc[] = ...
                if (stmt.targets[i].type == "index"
                    && ["at", "loc"].includes(stmt.targets[i].value.name)) {
                    static_comments.set(stmt.location.first_line,
                        "re-write the column");
                }
            }
        } else if (stmt.type == "call") {
            // x.fillna()
            if (stmt.func.name == "fillna") {
                static_comments.set(stmt.location.first_line,
                    "fill missing values");
            }
        } else if (stmt.type == "def") {
            // record defined funcions
            def_list.push(stmt.name);
            // for (let def_stmt of stmt.code) {
            //     if (def_stmt.type == "assign") {
            //         // only consider x = y.f()
            //         if (def_stmt.sources.length == 1) {
            //             let src = def_stmt.sources[0];
            //             let des = def_stmt.targets[0];
            //             if (des.type == "name" && src.type == "call") {
            //                 let func_name = "";
            //                 let src_name = "";
            //                 if (src.func.type == "dot")
            //                     func_name = src.func.name;
            //                 else if (src.func.type == "name")
            //                     func_name = src.func.id;
            //                 if (src.func.value != undefined && src.func.value.type == "name")
            //                     src_name = src.func.value.id;
            //                 let res = decompose_func(src, des);
            //                 if (res != undefined && res.length > 1) {
            //                     replace_strs.push([def_stmt.location.first_line, def_stmt.location.last_line, res]);
            //                 }
            //                 console.log(func_name);
            //                 if (TYPE_3_FUN.includes(func_name) || TYPE_1_FUN.includes(func_name))
            //                     branch_insert_list.push([def_stmt.location.first_line, def_stmt.location.last_line,
            //                         func_name, src_name, des.id]);
            //             }
            //         }
            //     }
            // }
        }
    }
    console.log(static_comments)
    return static_comments;
}

let flag = false;
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
                flag = true;
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
        // case 'class':
        //     return (tabs +
        //         'class ' +
        //         node.name +
        //         (node.extends ? '(' + commaSep(node.extends) + ')' : '') +
        //         ':' +
        //         lines(node.code, tabLevel + 1));
        // case 'decorator':
        //     return ('@' +
        //         node.decorator +
        //         (node.args ? '(' + commaSep(node.args) + ')' : ''));
        // case 'decorate':
        //     return (tabs +
        //         lines(node.decorators, tabLevel) +
        //         printTabbed(node.def, tabLevel));
        case 'def': {
            node.code.forEach(stmt => {
                traverse(stmt);
                if (flag) {
                    console.log(printNode(stmt));
                    replace_strs.push([stmt.location.first_line, stmt.location.last_line, [printNode(stmt)]]);
                    flag = false;
                }
            });
            break;
        }
        case 'dot': {
            traverse(node.value);
            if (TYPE_1_FUN.includes(node.name) || TYPE_3_FUN.includes(node.name)) {
                return [`cov(str.${node.name})`, node.value];
            }
            if (node.name == "g") {
                return ["cov(lib.g)", node.value];
            }
            break;
            // return printNode(node.value) + '.' + node.name;
        }
        // case 'else':
        //     node.code.forEach(x => traverse(x));
        // case 'for': {
        //     node.code.forEach(x => traverse(x));
        //     if (node.else)
        //         node.else.forEach(x => traverse(x));
        // }
        // case 'if':
        //     return (tabs +
        //         'if ' +
        //         printNode(node.cond) +
        //         ':' +
        //         lines(node.code, tabLevel + 1) +
        //         (node.elif
        //             ? node.elif.map(function (elif) {
        //                 return tabs +
        //                     'elif ' +
        //                     elif.cond +
        //                     ':' +
        //                     lines(elif.code, tabLevel + 1);
        //             })
        //             : '') +
        //         (node.else ? tabs + 'else:' + lines(node.else.code, tabLevel + 1) : ''));
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
        // case 'literal':
        //     return typeof node.value === 'string' && node.value.indexOf('\n') >= 0
        //         ? '""' + node.value + '""'
        //         : node.value.toString();
        // case 'module':
        //     return lines(node.code, tabLevel);
        // case 'name':
        //     return node.id;
        // case 'nonlocal':
        //     return tabs + 'nonlocal ' + node.names.join(comma);
        // case 'raise':
        //     return tabs + 'raise ' + printNode(node.err);
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
        // case 'unop':
        //     traverse(node.operand);
        // case 'while': {
        //     traverse(node.cond);
        //     node.code.forEach(x => traverse(x));
        // }
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

function wrap_methods(tree) {
    for (let [i, stmt] of tree.code.entries()) {
        traverse(stmt);
        if (flag) {
            console.log(printNode(stmt));
            replace_strs.push([stmt.location.first_line, stmt.location.last_line, [printNode(stmt)]]);
            flag = false;
        }
    }
}

function compute_flow_vars(code) {
    let tree = py.parse(code);
    // console.log(py.walk(tree).map(function (node) { return node.type; }));
    let cfg = new py.ControlFlowGraph(tree);
    // console.log(cfg.blocks);
    const analyzer = new py.DataflowAnalyzer();
    const flows = analyzer.analyze(cfg).dataflows;
    let line_in = new Map();
    let line_out = new Map();
    for (let flow of flows.items) {
        let fromLine = flow.fromNode.location.first_line;
        let toLine = flow.toNode.location.first_line;
        // use interSec to avoid missing in/out var bugs
        let defs = analyzer.getDefs(flow.fromNode, new RefSet()).items.map(x => x.name);
        let uses = analyzer.getUses(flow.toNode).items.map(x => x.name);
        let interSec = defs.filter(x => uses.includes(x));

        interSec.forEach(x => {
            add(line_in, toLine, x);
            add(line_out, fromLine, x);
        })
        // add in/out vars to cells
        if (flow.fromRef !== undefined) {
            // console.log(fromLine + "->" + toLine + " " + flow.fromNode.type + " " + flow.toNode.type + " " + flow.fromRef.name);
            add(line_in, toLine, flow.toRef.name);
            add(line_out, fromLine, flow.fromRef.name);
        }

        if (lineToCell.get(fromLine) < lineToCell.get(toLine)) {
            // console.log(fromLine + "->" + toLine + " " + flow.fromNode.type + " " + flow.toNode.type + " " + flow.toRef.name);
            // ignore import and funtion def
            if (["import", "def", "from"].includes(flow.fromNode.type))
                continue;
            interSec.forEach(x => {
                add(ins, lineToCell.get(toLine), x);
                add(outs, lineToCell.get(fromLine), x);
            })
            // console.log(analyzer.getUses(flow.toNode));
            add(ins, lineToCell.get(toLine), flow.toRef.name);
            add(outs, lineToCell.get(fromLine), flow.fromRef.name);
        }
        // console.log(flow.fromRef.name + "--------------" + flow.toRef.name)
        // console.log(py.printNode(flow.fromNode) +
        //     "\n -----------------> \n" + py.printNode(flow.toNode) + "\n");
    }
    add_extra_vars(tree);
    let comments = static_analyzer(tree);
    wrap_methods(tree);
    console.log(ins);
    console.log(outs);
    console.log(comments)
    return comments;
}

// type 1 == OUT, type 0 == IN
function print_info(cell, v, type) {
    return "my_store_info((" + cell + ", " + type + ", \"" + v + "\"), " + v + ")\n";
}

function insert_print_stmt(code) {
    let lines = code.split("\n");
    let max_line = lines.length;
    let cur_cell = 0;
    lines[0] = lines[0] + head_str;
    for (let item of replace_strs) {
        let idx = item[0] - 1;
        let space = " ".repeat((lines[idx].length - lines[idx].trimLeft().length))
        lines[item[0] - 1] = space + item[2].join("\n" + space);
        for (let i = item[0]; i < item[1]; i++)
            lines[i] = ""
    }
    for (let i = 0; i < max_line; i++) {
        if (lines[i].startsWith('# In[')) {
            if (outs.get(cur_cell) !== undefined)
                outs.get(cur_cell).forEach(x => lines[i - 1] += print_info(cur_cell, x, 1));
            cur_cell++;
            if (ins.get(cur_cell) !== undefined)
                ins.get(cur_cell).forEach(x => lines[i] += print_info(cur_cell, x, 0));
        }
        if (lines[i].startsWith("#"))
            continue;
        // deal with corner case
        if (lines[i].startsWith("get_ipython"))
            lines[i] = "";
        if (lines[i].startsWith("from __future__")) {
            lines[0] = lines[i] + lines[0];
            lines[i] = "";
        }
        // deal with functions
        let space = " ".repeat((lines[i].length - lines[i].trimLeft().length))
        if (lines[i].trim().startsWith("def")) {
            lines[i] = space + "@func_info_saver(" + (i + 1) + ")\n" + lines[i]
        }
    }
    lines[max_line - 1] += write_str;
    return lines.join("\n");
}


// let tree = py.parse("x = ticket.split(' ')");
// wrap_methods(tree);
// for (let [i, stmt] of tree.code.entries()) {
//     // console.log(stmt);
//     console.log(printNode(stmt));
// }

init_lineToCell();
let comments = compute_flow_vars(text);
// set up trace functions
let def_str = "TRACE_INTO = ['cov_wrapper_1','cov_wrapper_2'," + def_list.map(x => "'" + x + "'").join(",") + "]\n";
head_str = head_str.split("\n")
head_str[14] = def_str
head_str = head_str.join("\n") + "\n"
// insert save stmt
let modified_text = insert_print_stmt(text);
// save static comment
fs.writeFileSync(dir + filename_no_suffix + "_comment.json", JSON.stringify([...comments]));
fs.writeFileSync(dir + filename_no_suffix + "_m" + suffix, modified_text);
