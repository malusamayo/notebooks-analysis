"use strict";
var py = require("../python-program-analysis");
var fs = require('fs');
const { printNode, RefSet } = require("../python-program-analysis");
const { matchesProperty } = require("lodash");
const { printArg } = require("./dist/es5/printNode");
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
let FUNC_STR =
    "import os\n" +
    "my_dir_path = os.path.dirname(os.path.realpath(__file__))\n" +
    "log = \"This is a log file of input/output vars of each cell.\\n\"\n" +
    "def print_info(x):\n" +
    "    import numpy as np\n" +
    "    res = \"\"\n" +
    "    if isinstance(x, list) or isinstance(x, np.ndarray):\n" +
    "        res = \"shape\" + str(np.shape(x)) + \";\" + str(np.array(x).dtype)\n" +
    "    elif isinstance(x, dict):\n" +
    "        res = str(len(x)) + \";\" + str(type(x))\n" +
    "    else:\n" +
    "        res = str(x) + \";\" + str(type(x))\n" +
    "    return res.replace(\"\\n\", \"\")";
let HEAD_STR =
    "import os\n" +
    "import pickle\n" +
    "import copy\n" +
    "store_vars = []\n" +
    "my_labels = []\n" +
    "my_dir_path = os.path.dirname(os.path.realpath(__file__))\n";
let write_str =
    "store_vars.append(my_labels)\n" +
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

function compute_flow_vars(code) {
    let tree = py.parse(code);
    // console.log(py.walk(tree).map(function (node) { return node.type; }));
    let cfg = new py.ControlFlowGraph(tree);
    // console.log(cfg.blocks);
    const analyzer = new py.DataflowAnalyzer();
    const flows = analyzer.analyze(cfg).dataflows;
    for (let flow of flows.items) {
        let fromLine = flow.fromNode.location.first_line;
        let toLine = flow.toNode.location.first_line;
        // add in/out vars to cells
        // if (flow.fromRef !== undefined)
        //     console.log(fromLine + "->" + toLine + " " + flow.fromNode.type + " " + flow.toNode.type + " " + flow.fromRef.name);

        if (lineToCell.get(fromLine) < lineToCell.get(toLine)) {
            // console.log(fromLine + "->" + toLine + " " + flow.fromNode.type + " " + flow.toNode.type + " " + flow.toRef.name);
            // ignore import and funtion def
            if (flow.fromNode.type == "import" || flow.fromNode.type == "def")
                continue;
            let defs = analyzer.getDefs(flow.fromNode, new RefSet()).items.map(x => x.name)
            // console.log(fromLine + " " + toLine);
            // console.log(defs);
            let uses = analyzer.getUses(flow.toNode).items.map(x => x.name)
            // console.log(uses);
            let interSec = defs.filter(x => uses.includes(x));
            // console.log(interSec);
            interSec.forEach(x => {
                add(ins, lineToCell.get(toLine), x);
                add(outs, lineToCell.get(fromLine), x)
            })
            // console.log(analyzer.getUses(flow.toNode));
            add(ins, lineToCell.get(toLine), flow.toRef.name);
            add(outs, lineToCell.get(fromLine), flow.fromRef.name);
        }
        // console.log(flow.fromRef.name + "--------------" + flow.toRef.name)
        // console.log(py.printNode(flow.fromNode) +
        //     "\n -----------------> \n" + py.printNode(flow.toNode) + "\n");
    }
    console.log(ins);
    console.log(outs);
}

// type 1 == OUT, type 0 == IN
function print_info(cell, v, type) {
    // let head_info = "log = log + \"cell[" + cell + "];" + type + ";" + v + ";\" + ";
    // let tail_info = " + \"\\n\"\n";
    // if (v.includes("[") == true)
    //     return "try:\n    " + head_info + "print_info(" + v + ")" + tail_info +
    //         "except IndexError:\n    " + head_info + "\"IndexError!\"" + tail_info;
    // return head_info + "print_info(" + v + ")" + tail_info;
    return "my_labels.append((" + cell + ", " + type + ", \"" + v + "\"))\n"
        + "store_vars.append(copy.deepcopy(" + v + "))\n";
}

function insert_print_stmt(code) {
    let lines = code.split("\n");
    let max_line = lines.length;
    let cur_cell = 0;
    lines[0] = lines[0] + HEAD_STR;
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
    }
    lines[max_line - 1] += write_str;
    return lines.join("\n");
}

init_lineToCell();
compute_flow_vars(text);
let modified_text = insert_print_stmt(text);
fs.writeFileSync(dir + filename_no_suffix + "_m" + suffix, modified_text);