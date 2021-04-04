"use strict";

const testFolder = './notebooks/titanic_notebooks/';
const { dir } = require('console');
const { writeFileSync, readdirSync, readFileSync } = require('fs');
const path = require('path');
const { pathsToModuleNameMapper } = require('ts-jest');
const { createImportSpecifier } = require('typescript');

let STARTS = `<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN">
<HTML>
<HEAD>
<META NAME="GENERATOR" CONTENT="Java2HTML Version 1.5.1">
<TITLE>Pattern Overview</TITLE>
<LINK REL="stylesheet" TYPE="text/css" HREF="stylesheet.css" TITLE="Style">
</HEAD>
<BODY>
<FONT size="+1" CLASS="FrameHeadingFont">Pattern Overview</FONT>
<BR>`
let ENDS = "</HTML>"

const getDirectories = source =>
	readdirSync(source, { withFileTypes: true })
		.filter(dirent => dirent.isDirectory())
		.map(dirent => dirent.name)

let dirs = getDirectories(testFolder).map(x => path.join('notebooks', 'titanic_notebooks', x));

console.log(dirs);

let summary = {}

function add_pattern(pattern, filepath) {
	if (pattern in summary) {
		summary[pattern].push(filepath);
	} else {
		summary[pattern] = []
		summary[pattern].push(filepath);
	}
}

function generateSummary(patterns, filepath) {
	if ("other_patterns" in patterns) {
		patterns.other_patterns.forEach((pattern, _) => {
			Object.keys(pattern).forEach(k => {
				if (k != "copy")
					add_pattern(k, filepath);
			})
		});
	}
	Object.keys(patterns).forEach(col_str => {
		let cols = col_str.split('|');
		if (cols.length <= 1)
			return false;
		let pattern = patterns[col_str].join('(') + "(" + ")".repeat(patterns[col_str].length);
		add_pattern(pattern, filepath);
	});
}

dirs.forEach(dir => {
	let nb = dir.split(path.sep).slice(-1) + ".html";
	readdirSync(dir).filter(file => file.startsWith("result")).sort().forEach(file => {
		let data = JSON.parse(readFileSync(path.join(dir, file)));
		let cell_num = file.match(/\d+/)[0];

		Object.keys(data.summary).forEach(k => {
			generateSummary(data.summary[k], nb + "#" + cell_num);
		})
	});
});


// console.log(summary);

let output_html = STARTS;

Object.keys(summary).forEach(pattern => {
	output_html += "<h3>" + pattern + "</h3><br>\n";
	summary[pattern].forEach(file_name => {
		output_html += '<FONT CLASS="FrameItemFont"><A HREF="' + file_name + '" TARGET="SourceFrame">' + file_name + '</A></FONT><BR>\n'
	})
})

output_html += ENDS;

writeFileSync(path.join("html_outputs", "index.html"), output_html);

// let summaries = new Map();
// let funcs = [];

// jsonfiles.forEach(file => {
//   let v = JSON.parse(fs.readFileSync(testFolder + file).toString());
//   summaries.set(file, [])
//   Object.entries(v).forEach(cell => {
//     if (Object.entries(cell[1].summary).length > 0) {
//       let arr = summaries.get(file);
//       Object.entries(cell[1].summary).forEach(inv => arr.push(inv));
//       summaries.set(file, arr);
//     }
//     if (Object.entries(cell[1].function).length > 0) {
//       funcs.push(cell[1].function);
//     }
//     // summaries.push(cell[1].summary);
//   })
// })

// let length_arr = [];

// for (let [key, value] of summaries) {
//   // console.log(key + ' = ' + value)
//   summaries.set(key, value.filter(x => {
//     return !x[1].includes("no change");
//   }));
// }

// // summaries.forEach(key, item => {
// //   // console.log(key)
// //   // let res = item.filter(x => {
// //   //   return !x[1].includes("no change");
// //   // });
// //   // item = res;
// //   // summaries.set()
// //   // item.forEach(x => {
// //   //   console.log(x[1]);
// //   //   if (x[1].includes("no change"))
// //   //     item
// //   // })
// // })

// let cntType = { "removeCol": 0, "addCol": 0, "removeRow": 0, "convert": 0, "change": 0, "copy": 0, "truncate": 0, "rearrange": 0 }

// let pattern = Object.entries(cntType).map(x => x[0]);
// pattern = pattern.slice(3);
// console.log(pattern)

// summaries.forEach(item => {
//   item.forEach(x => {
//     // console.log(x[1]);
//     if (x[1].includes("remove") && x[1].includes("columns"))
//       cntType["removeCol"] += 1;
//     if (x[1].includes("add") && x[1].includes("columns"))
//       cntType["addCol"] += 1;
//     if (x[1].includes("remove") && x[1].includes("rows"))
//       cntType["removeRow"] += 1;
//     pattern.forEach(str => {
//       if (x[1].includes(str))
//         cntType[str] += 1;
//     })
//   })
//   length_arr.push(item.length);
// })

// funcs.forEach(item => {
//   Object.entries(item).forEach(x => {
//     console.log(x[0], x[1].counts)
//   }
//   )
// })
// console.log(funcs.length)

// console.log(cntType);
// console.log(length_arr)
// console.log(length_arr.reduce((x, y) => x + y, 0))
// console.log(summaries);
