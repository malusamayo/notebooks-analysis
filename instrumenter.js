"use strict";
var py = require("../python-program-analysis");
var fs = require('fs');
const { printNode, RefSet } = require("../python-program-analysis");
const { matchesProperty, map, head } = require("lodash");
const { printArg } = require("./dist/es5/printNode");
const { ADDRCONFIG } = require("dns");
const { assert } = require("console");
const { collapseTextChangeRangesAcrossMultipleVersions } = require("typescript");
const { wrap_methods, collect_defs, collect_cols } = require('../python-program-analysis/method-wrapper');

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
let head_str = fs.readFileSync("helper.py").toString();
let def_list = [];

let pyTypeof = new Map();

const trace_into_line = head_str.split("\n").findIndex(x => x.startsWith("TRACE_INTO"));
let write_str =
    `
tmp_dir_path = os.path.join(my_dir_path, "${filename_no_suffix}")
if not os.path.isdir(tmp_dir_path):
    os.mkdir(tmp_dir_path)
for idx, vars in store_vars.items():
    with open(os.path.join(tmp_dir_path, "${filename_no_suffix}_" + format(idx, '03') + ".dat"), "wb") as f:
        pickle.dump(vars, f)
# with open(os.path.join(tmp_dir_path, "${filename_no_suffix}_f.dat"), "wb") as f:
#     pickle.dump(ddict2dict(funcs), f)
with open(os.path.join(tmp_dir_path, "info.json"), "w") as f:
    f.write(json.dumps({"get": get__keys, "set": set__keys, "par": pathTracker.partitions}))
`

function init_lineToCell() {
    let lines = text.split("\n");
    let max_line = lines.length;
    let cur_cell = 0;
    for (let i = 0; i < max_line; i++) {
        if (lines[i].startsWith('# In['))
            cur_cell++;
        if (lines[i].startsWith("#"))
            continue;
        if (lines[i].trim() == "")
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
        return node;
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

function static_analyzer(tree) {
    let static_comments = new Map();
    let df_construct = ["DataFrame", "read_csv"];
    let df_funcs = ["append", "concat", "copy", "drop", "get_dummies"];
    let old_key = -1;
    let lambda_id = 0;
    let cell_cols = new Set();

    // simple type inference
    function infer_types(stmt) {
        if (stmt.type == "assign" && stmt.targets.length == stmt.sources.length) {
            for (let [i, src] of stmt.sources.entries()) {
                if (src.type == "dict")
                    pyTypeof.set(stmt.targets[i].id, "dict");
                // df = pd.read_csv() ...
                if (src.type == "call" && df_construct.includes(src.func.name)) {
                    pyTypeof.set(stmt.targets[i].id, "dataframe");
                }
                // df = df.func()
                if (src.type == "call" && df_funcs.includes(src.func.name)) {
                    if (src.func.value.type == "name" &&
                        pyTypeof.get(src.func.value.id) == "dataframe")
                        pyTypeof.set(stmt.targets[i].id, "dataframe");
                }
                // df2 = df
                if (src.type == "name" && pyTypeof.get(src.id) == "dataframe") {
                    pyTypeof.set(stmt.targets[i].id, "dataframe");
                }
                if (stmt.type == "def") {
                    pyTypeof.set(stmt.name, "func");
                }
            }
        }
    }

    function value_type_handler(type, node) {
        if (type == "index") {
            assert(node.args.length == 1);
            if (node.args[0].type == "literal") {
                let col = node.args[0].value;
                return "[" + col.replace(/['"]+/g, '') + "]";
            } else if (node.args[0].type == "name") {
                return "[" + node.args[0].id + "]";
            }
        } else if (type == "dot") {
            return "[" + node.name.replace(/['"]+/g, '') + "]";
        }
    }

    function map_handler(stmt, src, des) {
        let value_type = ["index", "dot"];
        if (value_type.includes(des.type)
            && value_type.includes(src.func.value.type)) {
            let src_col = value_type_handler(src.func.value.type, src.func.value);
            let des_col = value_type_handler(des.type, des);
            let comment = ""
            // same/different literal
            if (src_col == des_col)
                comment = "[map/apply],modify column " + des_col + " using map/apply"
            else
                comment = "[map/apply],create column " + des_col + " from " + src_col
            if (pyTypeof.get(src.args[0].actual.id) == 'dict')
                comment += " with dict"
            if (src.args[0].actual.type == 'lambda') {
                // console.log(printNode(src.args[0].actual));
                let args = src.args[0].actual.args.map(x => x.name);
                let code = src.args[0].actual.code;
                let def_code = "";
                if (code.type == "ifexpr") {
                    def_code = ["def lambda_" + lambda_id + "(" + args.join(", ") + "):",
                    "if " + printNode(code.test) + ":", "\treturn " + printNode(code.then),
                        "else:", "\treturn " + printNode(code.else)];
                } else {
                    def_code = ["def lambda_" + lambda_id + "(" + args.join(", ") + "):", "return " + printNode(code)];
                }
                // console.log(def_code.join("\n\t"));

                src.args[0].actual = { id: "lambda_" + lambda_id, location: src.args[0].actual.location, type: "name" };
                replace_strs.push([stmt.location.first_line, stmt.location.last_line, [def_code.join("\n\t") + "\n" + printNode(stmt)]]);
                def_list.push("lambda_" + lambda_id)
                lambda_id++;
            } else if (src.args[0].actual.type == 'name') {
                def_list.push(src.args[0].actual.id);
            }
            return comment;
        }
    }

    for (let [_, stmt] of tree.code.entries()) {
        infer_types(stmt);
        let cols = collect_cols(stmt, pyTypeof);
        cols.forEach(value => cell_cols.add(value.replace(/['"]+/g, '')));

        // console.log(printNode(stmt));

        // lambda function tracking cancelled

        // let lambda = contain_type(stmt, "lambda");
        // if (lambda != undefined) {
        //     // should also record/convert lambda function later
        //     // lambda.name = "lambda_" + lambda.location.first_line;
        //     // lambda.type = "def";
        //     // lambda.params = lambda.args;
        //     // lambda.code = [lambda.code, lambda.code];
        //     // console.log(printNode(lambda));
        //     let lambda_rep = "func_info_saver(" + stmt.location.first_line + ")(" + lambda + ")";
        //     let stmt_str = printNode(stmt);
        //     stmt_str = stmt_str.replace(lambda, lambda_rep);
        //     replace_strs.push([stmt.location.first_line, stmt.location.last_line, [stmt_str]]);
        // }
        let key = lineToCell.get(stmt.location.first_line);
        if (key != old_key) {
            if (old_key != -1 && cell_cols.size > 0) {
                add(static_comments, old_key,
                    "[used]," + Array.from(cell_cols).join(","))
                cell_cols.clear()
            }
            old_key = key;
        }
        if (stmt.type == "assign" && stmt.targets.length == stmt.sources.length) {
            // external input: x = pd.read_csv()
            for (let [i, src] of stmt.sources.entries()) {
                // x[y] = x1[y1].map(...) || x.y = x1.y1.map(...)
                if (src.type == "call" && src.func.name == "map") {
                    let res = map_handler(stmt, src, stmt.targets[i]);
                    if (res)
                        add(static_comments, key, res);
                }
                if (src.type == "call" && src.func.name == "apply") {
                    let res = map_handler(stmt, src, stmt.targets[i]);
                    if (res)
                        add(static_comments, key, res);
                    else {
                        let value_type = ["index", "dot"];
                        if (value_type.includes(stmt.targets[i].type)) {
                            let des_col = value_type_handler(stmt.targets[i].type, stmt.targets[i]);
                            add(static_comments, key,
                                "[apply],create column " + des_col + " from whole row");
                        }
                    }
                }
                // x = pd.get_dymmies()
                if (src.type == "call" && src.func.name == "get_dummies") {
                    add(static_comments, key,
                        "[get_dummies],encoding in dummy variables");
                }
                // x1, x2, y1, y2 = train_test_split()
                if (src.type == "call" && src.func.name == "train_test_split") {
                    add(static_comments, key,
                        "[train_test_split],spliting data to train set and test set");
                }
                // x = df.select_dtypes().columns
                if (src.type == "dot" && src.name == "columns") {
                    if (src.value.type == "call" && src.value.func.name == "select_dtypes")
                        add(static_comments, key,
                            "[select_dtypes],select columns of specific data types");
                }
                // x.at[] = ... || x.loc[] = ...
                if (stmt.targets[i].type == "index" && stmt.targets[i].value.type == "dot"
                    && ["at", "loc"].includes(stmt.targets[i].value.name)) {
                    add(static_comments, key,
                        "[at/loc],re-write the column");
                }
            }
        } else if (stmt.type == "call") {
            // x.fillna()
            if (stmt.func.name == "fillna") {
                add(static_comments, key,
                    "[fillna],fill missing values");
            }
        }
    }
    return static_comments;
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

            // contain local var with same name??
            interSec.forEach(x => {
                add(ins, lineToCell.get(toLine), x);
                add(outs, lineToCell.get(fromLine), x);
            })

            // console.log(interSec);
            // console.log(flow.toRef.name);
            // console.log(flow.fromRef.name);
            // console.log(analyzer.getUses(flow.toNode));
            add(ins, lineToCell.get(toLine), flow.toRef.name);
            add(outs, lineToCell.get(fromLine), flow.fromRef.name);
        }
        // console.log(flow.fromRef.name + "--------------" + flow.toRef.name)
        // console.log(py.printNode(flow.fromNode) +
        //     "\n -----------------> \n" + py.printNode(flow.toNode) + "\n");
    }
    // add_extra_vars(tree); // bugs here
    let comments = static_analyzer(tree);
    let defs = collect_defs(tree.code);
    defs.forEach(item => {
        comments.set(item[0], lineToCell.get(item[1]))
    });
    // def_list = def_list.map(item => item[0]);

    // disable coverage replacement now
    // replace_strs = replace_strs.concat(wrap_methods(tree));
    console.log(ins);
    console.log(outs);
    console.log(comments)
    return comments;
}

// type 1 == OUT, type 0 == IN
function print_info(cell, v, type) {
    return `
try:
    my_store_info((${cell}, ${type}, "${v}"), ${v})
except NameError:
    pass
`
}

function insert_print_stmt(code) {
    let lines = code.split("\n");
    let max_line = lines.length;
    let cur_cell = 0;
    lines[0] = head_str + lines[0];
    for (let item of replace_strs) {
        let idx = item[0] - 1;
        let space = " ".repeat((lines[idx].length - lines[idx].trimLeft().length))
        lines[item[0] - 1] = space + item[2].join("\n" + space);
        for (let i = item[0]; i < item[1]; i++)
            lines[i] = ""
    }
    for (let i = 0; i < max_line; i++) {
        if (lines[i].startsWith('# In[')) {
            lines[i - 1] += "pathTracker.to_partition()\n"
            if (outs.get(cur_cell) !== undefined)
                outs.get(cur_cell).forEach(x => lines[i - 1] += print_info(cur_cell, x, 1));
            cur_cell++;
            if (ins.get(cur_cell) !== undefined) {
                ins.get(cur_cell).forEach(x => lines[i] += print_info(cur_cell, x, 0));
                // lines[i] += "update_maxrow([" + ins.get(cur_cell).join(", ") + "])\n";
            }
            // update cur_cell before execution
            lines[i] += "cur_cell = " + cur_cell + "\n";
        }
        if (lines[i].startsWith("#"))
            continue;
        // deal with corner case
        if (lines[i].startsWith("get_ipython") || lines[i].startsWith("display("))
            lines[i] = "";
        if (lines[i].startsWith("from __future__")) {
            lines[0] = lines[i] + lines[0];
            lines[i] = "";
        }
        // deal with functions
        let space = " ".repeat((lines[i].length - lines[i].trimLeft().length))
        if (lines[i].trim().startsWith("def ")) {
            lines[i] = space + "@func_info_saver(" + (i + 1) + ")\n" + lines[i]
        }
    }
    lines[max_line - 1] += write_str;
    return lines.join("\n");
}


// let tree = py.parse(
//     `data_Simple['Unit'] = data_Simple['Value'].str[-1]
// data_Simple['Value (M)'] = np.where(data_Simple['Unit'] == '0', 0, 
//                                            data_Simple['Value'].str[1:-1].replace(r'[a-zA-Z]',''))
// data_Simple['Value (M)'] = data_Simple['Value (M)'].astype(float)
// data_Simple['Value (M)'] = np.where(data_Simple['Unit'] == 'M', 
//                                            data_Simple['Value (M)'], 
//                                            data_Simple['Value (M)']/1000)
// data_Simple = data_Simple.drop('Unit', 1)
// `);
// let _ = static_analyzer(tree);
// console.log(_)
// for (let [i, stmt] of tree.code.entries()) {
//     // console.log(stmt);
//     console.log(printNode(stmt));
// }

// let code = "train_test_data = [train, test]\nfor dataset in train_test_data:\n" +
//     "    train_test_data = dataset['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)\n" +
//     "sex_mapping = {\"male\": 0, \"female\": 1}\nfor dataset in train_test_data:\n" +
//     "    dataset['Sex'] = dataset['Sex'].map(sex_mapping)\n"
// let tree = py.parse(code);
// let cfg = new py.ControlFlowGraph(tree);
// // console.log(cfg.blocks);
// const analyzer = new py.DataflowAnalyzer();
// const flows = analyzer.analyze(cfg).dataflows;
// for (let flow of flows.items) {
//     let fromLine = flow.fromNode.location.first_line;
//     let toLine = flow.toNode.location.first_line;
//     console.log(fromLine, toLine);
// }

init_lineToCell();
let comments = compute_flow_vars(text);
// set up trace functions
let def_str = "TRACE_INTO = [" + def_list.map(x => "'" + x + "'").join(",") + "]\n";
head_str = head_str.split("\n")
head_str[trace_into_line] = def_str
head_str = head_str.join("\n") + "\n"
// insert save stmt
let modified_text = insert_print_stmt(text);
// save static comment
fs.writeFileSync(dir + filename_no_suffix + "_comment.json", JSON.stringify([...comments]));
fs.writeFileSync(dir + filename_no_suffix + "_m" + suffix, modified_text);
