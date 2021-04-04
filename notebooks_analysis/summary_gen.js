"use strict";

const testFolder = process.argv[2];
const { dir } = require('console');
const { writeFileSync, readdirSync, readFileSync, existsSync } = require('fs');
const path = require('path');
const execSync = require('child_process').execSync;

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

let script = ["notebooks_analysis", "html_convert.js"].join(path.sep);


const getDirectories = source =>
	readdirSync(source, { withFileTypes: true })
		.filter(dirent => dirent.isDirectory())
		.map(dirent => dirent.name)

let dirs = getDirectories(testFolder).map(x => path.join(testFolder, x));

console.log(dirs);

let summary = {}
let all_cells = []

function add_pattern(pattern, filepath) {
	if (!(pattern in summary)) {
		summary[pattern] = []
	}
	if (!summary[pattern].includes(filepath)) {
		summary[pattern].push(filepath);
	}
	if (!all_cells.includes(filepath)) {
		all_cells.push(filepath);
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
	let dir_ls = dir.split(path.sep);
	let nb_html = dir_ls.slice(-1) + ".html";
	dir_ls[dir_ls.length - 1] += ".ipynb";
	let nb_path = dir_ls.join(path.sep);
	if (existsSync(nb_path)) {
		// re-generate documents
		if (process.argv.length > 3 && process.argv[3] == "-e") {
			execSync(["jupyter", "nbconvert", "--to", "html", nb_path, "--output-dir", "html_outputs"].join(" "));
			console.log("Adding documentation...")
			execSync(["node", script, nb_path].join(" "));
			console.log("Generating summary...")
		}
		// generate summary
		readdirSync(dir).filter(file => file.startsWith("result")).sort().forEach(file => {
			let data = JSON.parse(readFileSync(path.join(dir, file)));
			let cell_num = file.match(/\d+/)[0].padStart(3, "0");

			Object.keys(data.summary).forEach(k => {
				generateSummary(data.summary[k], nb_html + "#" + cell_num);
			})
		});
	}
});

let output_html = STARTS;

// sampling code
output_html += `
<script>
function sample(all_cells) {
  let sampled_cells = []
  while (sampled_cells.length < 10) {
	const file_name = all_cells[Math.floor(Math.random() * all_cells.length)];
	if (!sampled_cells.includes(file_name))
	  sampled_cells.push(file_name);
  }
  return sampled_cells;
}
let cells = ${JSON.stringify(all_cells)};
</script>
<h1>Sampled Cells:</h1>
<button
onclick="let sp = sample(cells).sort();Array.from(document.getElementById('sampled_cells').children).filter(x => x.nodeName !='BR').forEach((x, i)=> {let new_x = document.createElement('a'); new_x.href=sp[i]; new_x.innerText=sp[i]; new_x.target = 'SourceFrame'; x.outerHTML = new_x.outerHTML;});">
Resample</button><br>
`


// sampling cells
function sample(all_cells) {
	let sampled_cells = []
	while (sampled_cells.length < 10) {
		const file_name = all_cells[Math.floor(Math.random() * all_cells.length)];
		if (!sampled_cells.includes(file_name))
			sampled_cells.push(file_name);
	}
	return sampled_cells;
}

let sampled_cells = sample(all_cells);
output_html += `<div id="sampled_cells">`;
sampled_cells.sort().forEach(file_name => {
	output_html += '<FONT CLASS="FrameItemFont"><A HREF="' + file_name + '" TARGET="SourceFrame">' + file_name + '</A></FONT><BR>\n';
});
output_html += '</div>';

// pattern list
output_html += "<br><h1>" + "Full Pattern List:" + "</h1><br>";
Object.keys(summary).forEach(pattern => {
	output_html += "<h3>" + pattern + "</h3><br>";
	summary[pattern].forEach(file_name => {
		output_html += '<FONT CLASS="FrameItemFont"><A HREF="' + file_name + '" TARGET="SourceFrame">' + file_name + '</A></FONT><BR>\n'
	})
})

output_html += ENDS;

writeFileSync(path.join("html_outputs", "index.html"), output_html);
