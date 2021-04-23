"use strict";

const { dir } = require('console');
const { writeFileSync, readdirSync, readFileSync, existsSync, read } = require('fs');
const path = require('path');
const JSON5 = require('json5')
const execSync = require('child_process').execSync;
const { exit } = require('process');

let collapse_style = `<style>
.collapsible {
  background-color: #777;
  color: white;
  cursor: pointer;
  padding: 18px;
  width: 50%;
  border: none;
  text-align: left;
  outline: none;
  font-size: 30px;
}

.active, .collapsible:hover {
  background-color: #555;
}

.content {
  padding: 0 18px;
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.2s ease-out;
  background-color: #f1f1f1;
}
</style>`

let STARTS = `<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN">
<HTML>
<HEAD>
<META NAME="GENERATOR" CONTENT="Java2HTML Version 1.5.1">
<TITLE>Pattern Overview</TITLE>
<LINK REL="stylesheet" TYPE="text/css" HREF="stylesheet.css" TITLE="Style">
<script src="https://kit.fontawesome.com/e787e986bb.js" crossorigin="anonymous"></script>
${collapse_style}
</HEAD>
<BODY>`

let ENDS = "</BODY></HTML>"

let collapsible_script = `<script>
var coll = document.getElementsByClassName("collapsible");
var i;

for (i = 0; i < coll.length; i++) {
	coll[i].addEventListener("click", function() {
		this.classList.toggle("active");
		var content = this.nextElementSibling;
		if (content.style.maxHeight){
			content.style.maxHeight = null;
		} else {
			content.style.maxHeight = content.scrollHeight + "px";
		} 
	});
}
</script>`

let script = ["notebooks_analysis", "html_convert.js"].join(path.sep);

const getDirectories = source =>
	readdirSync(source, { withFileTypes: true })
		.filter(dirent => dirent.isDirectory())
		.map(dirent => dirent.name)

const titanic_folder = ["notebooks", "titanic"].join(path.sep);
const fifa_folder = ["notebooks", "fifa19"].join(path.sep);
const google_folder = ["notebooks", "google"].join(path.sep);
const airbnb_folder = ["notebooks", "airbnb"].join(path.sep);


let summary = {};
let statistics = {};
let clusters = {};
let cell_info = {};
let tracked_cells = [];
let sample_num = 20;

let counts = {
	"modified": [],
	"new": [],
	"removecol": [],
	"rearrange": [],
	"removerow": [],
	"removerow_null": [],
	"removerow_dup": [],
	"rearrange_row": []
};
let func_counts = {
	"replace": [],
	"strip": [],
	"upper": [],
	"lower": [],
	"if_expr": [],
	"loc/at": [],
	"empty": [],
	"removed": [],
	"replace_ls": [],
	"split": [],
	"fillna": [],
	"extract": [],
	"map_dict": [],
	"get_dummies": [],
	"set": [],
	"dropna": [],
	"others": []
}
let pat2cols = {}
let nb2pat = {}

function add_count(pattern, filepath) {
	if (pattern in counts && !counts[pattern].includes(filepath)) {
		counts[pattern].push(filepath);
	}
}

function add_func_counts(func, filepath) {
	if (func in func_counts) {
		func_counts[func].push(filepath);
	} else if (!func.startsWith("default_")) {
		func_counts["others"].push(filepath);
	}
}

function add_col(pattern, col) {
	if (!(pattern in pat2cols)) {
		pat2cols[pattern] = []
	}
	pat2cols[pattern].push(col);
}

function update_np2pat(pattern, notebook) {
	if (!(notebook in nb2pat)) {
		nb2pat[notebook] = []
	}
	if (!nb2pat[notebook].includes(pattern)) {
		nb2pat[notebook].push(pattern);
	}
}

function add_pattern(pattern, filepath) {
	if (!(pattern in summary)) {
		summary[pattern] = []
	}
	if (!summary[pattern].includes(filepath)) {
		summary[pattern].push(filepath);
	}
}

function update_statistics(pattern) {
	if (!(pattern in statistics)) {
		statistics[pattern] = 0;
	}
	statistics[pattern] += 1
}

function update_clusters(num, filepath) {
	if (!(num in clusters)) {
		clusters[num] = []
	}
	clusters[num].push(filepath);
}

function generateSummary(flow, patterns, filepath) {
	if ("other_patterns" in patterns) {
		patterns.other_patterns.forEach((pattern, _) => {
			Object.keys(pattern).forEach(k => {
				if (k != "copy") {
					cell_info[filepath]["changed"] = true;
					add_pattern(k, filepath);
					update_statistics(k);
				}
				add_count(k, filepath);
			})
		});
	}
	Object.keys(patterns).forEach(col_str => {
		let cols = col_str.split('|');
		if (cols.length <= 1)
			return false;
		if (cols[0] == cols[1])
			add_count("modified", filepath);
		else
			add_count("new", filepath);
		let pattern = patterns[col_str].join('(') + "(" + ")".repeat(patterns[col_str].length);
		cols[1].split(',').forEach(x => add_col(pattern, x))
		add_pattern(pattern, filepath);
		for (const p of patterns[col_str]) {
			update_statistics(p, filepath);
			update_np2pat(p, filepath.substring(0, filepath.indexOf(".html")))
		}
	});
}

// sampling cells
function sample(tracked_cells, sample_num) {
	let sampled_cells = []
	while (sampled_cells.length < sample_num) {
		const file_name = tracked_cells[Math.floor(Math.random() * tracked_cells.length)];
		if (!sampled_cells.includes(file_name))
			sampled_cells.push(file_name);
	}
	return sampled_cells;
}

function read_data(testFolder) {

	let dirs = getDirectories(testFolder).map(x => path.join(testFolder, x));

	console.log(dirs);
	console.log(testFolder.split(path.sep)[1] + ".html");

	dirs.forEach(dir => {
		let dir_ls = dir.split(path.sep);
		let nb_html = dir_ls.slice(-1) + ".html";
		dir_ls[dir_ls.length - 1] += ".ipynb";
		let nb_path = dir_ls.join(path.sep);
		let log_path = dir_ls.join(path.sep).replace(".ipynb", "_change_log.json");
		if (existsSync(nb_path)) {
			// re-generate documents
			// if (process.argv.length > 3 && process.argv[3] == "-e") {
			// 	execSync(["jupyter", "nbconvert", "--to", "html", nb_path, "--output-dir", "html_outputs"].join(" "));
			// 	console.log("Adding documentation...")
			// 	execSync(["node", script, nb_path].join(" "));
			// 	console.log("Generating summary...")
			// }

			// generate summary
			let cell_input = {};
			let cell_output = {};
			let tracked = {}

			readdirSync(dir).filter(file => file.startsWith("result")).sort().forEach(file => {
				let data = JSON.parse(readFileSync(path.join(dir, file)));
				let cell_num = file.match(/\d+/)[0].padStart(3, "0");

				cell_input[cell_num] = Object.keys(data.input).filter(k => data.input[k].type.startsWith("DataFrame"));
				cell_output[cell_num] = Object.keys(data.output);
				tracked[cell_num] = Object.keys(data.summary).map(x => x.split(" -> ")[1]);

				// Object.keys(data.partition).forEach(k => {
				// 	let arr = Object.keys(data.partition[k]);
				// 	let res = JSON5.parse(arr[0])[0].slice(-1)[0];
				// 	let l = res.startsWith("default") ? 1 : arr.length;
				// 	update_clusters(l, filepath);
				// })
				Object.keys(data.summary).forEach(k => {
					let var_name = "[" + k.split(" -> ")[1] + "]";
					let filepath = nb_html + "#" + cell_num + var_name;

					if (!(filepath in cell_info)) {
						cell_info[filepath] = {
							"SA-ignored": false,
							"incomplete-doc": false,
							"no-doc": false,
							"tracked": false,
							"changed": false
						};
					}

					cell_info[filepath]["tracked"] = true;

					generateSummary(k, data.summary[k], filepath);

					// cluster
					if (k in data.partition) {
						let arr = Object.keys(data.partition[k]);
						let res = JSON5.parse(arr[0]);
						let l = res[0].slice(-1)[0].startsWith("default") ? 1 : arr.length;
						update_clusters(l, filepath);
						res.forEach(x => add_func_counts(x[1], filepath));

					}
				})
			});

			let changes = JSON.parse(readFileSync(log_path));
			Object.keys(changes).forEach(cell_num => {
				let cell_num_padded = String(cell_num).padStart(3, "0");
				changes[cell_num].forEach(var_name => {
					let filepath = nb_html + "#" + cell_num_padded + "[" + var_name + "]";
					if (!(filepath in cell_info)) {
						cell_info[filepath] = {
							"SA-ignored": false,
							"incomplete-doc": false,
							"no-doc": false,
							"tracked": false
						};
					}
					cell_info[filepath]["changed"] = true;
					if (!(cell_num_padded in cell_input) || cell_input[cell_num_padded].length == 0
						|| !cell_output[cell_num_padded].includes(var_name)) {
						cell_info[filepath]["SA-ignored"] = true;
					}
				})
			})
		}
	});
}

function append_item(file_name) {
	return '<FONT CLASS="FrameItemFont"><A HREF="' + file_name.substring(0, file_name.indexOf("[")) + '" TARGET="SourceFrame">' + file_name + '</A></FONT><BR>\n';
}

let avg = []

function gen_plots(summary, statistics, clusters, cell_info, testFolder = "") {
	let output_html = "";

	// not tracked list
	let changed = new Set(Object.keys(cell_info).filter(x => cell_info[x]["changed"] == true));
	// console.log(changed.size, new Set(Array.from(changed).map(file_name => file_name.substring(0, file_name.indexOf("[")))))
	// let tracked = Object.keys(cell_info).filter(x => cell_info[x]["tracked"] == true).map(x => x.substring(0, x.lastIndexOf("#")));
	// const map = tracked.reduce((acc, e) => acc.set(e, (acc.get(e) || 0) + 1), new Map());
	// let times = readFileSync(testFolder.split(path.sep).concat(["log.txt"]).join(path.sep), "utf8").split("\n").filter(x => x != "");
	// // console.log(map)
	// for (const line of times) {
	// 	let items = line.split("\t");
	// 	console.log(items[0], items[4] / map.get(items[0].replace(".ipynb", ".html")))
	// 	avg.push(items[4] / map.get(items[0].replace(".ipynb", ".html")))
	// }
	// let average = (array) => array.reduce((a, b) => a + b) / array.length;
	// console.log(average(avg), avg.length)
	let not_tracked = new Set(Object.keys(cell_info).filter(x => cell_info[x]["tracked"] == false))
	let no_SA = new Set(Object.keys(cell_info).filter(x => cell_info[x]["SA-ignored"] == true));

	// let incomp = new Set(Object.keys(cell_info).filter(x => cell_info[x]["incomplete-doc"] == true && cell_info[x]["no-doc"] == false));
	// let nodoc = new Set(Object.keys(cell_info).filter(x => cell_info[x]["no-doc"] == true));

	// console.log(Array.from(changed).filter(x => cell_info[x]["no-doc"] == false && cell_info[x]["tracked"] == false));


	output_html += `<br><button class="collapsible">No-doc Vars: ${not_tracked.size}/${changed.size}</button>` + `<div class="content"><ul>`;
	// nodoc
	let SA_not_tracked = new Set([...not_tracked].filter(x => no_SA.has(x)));
	let other_not_tracked = new Set([...not_tracked].filter(x => !no_SA.has(x)));
	output_html += `<h2>Vars without doc due to static analysis: ${SA_not_tracked.size}/${not_tracked.size}</h2>`
	SA_not_tracked.forEach(file_name => {
		output_html += append_item(file_name);
	})
	output_html += `<h2>Vars without doc due to other reasons: ${other_not_tracked.size}/${not_tracked.size}</h2>`
	other_not_tracked.forEach(file_name => {
		output_html += append_item(file_name);
	})
	output_html += "</ul></div>";

	let stat_arr = Object.keys(statistics).map(key => [key, statistics[key], String(statistics[key])]).sort((a, b) => b[1] - a[1]);
	let cluster_arr = Object.keys(clusters).map(key => [key, clusters[key].length]).filter(k => k[0] < 20);
	cluster_arr.unshift(["branchnum", "count"]);
	cluster_arr.push([">=20", Object.keys(clusters).filter(k => k >= 20).map(k => clusters[k].length).reduce((a, b) => a + b, 0)]);

	output_html += `
	<h1> Statistics: </h1>

	<script src="https://www.gstatic.com/charts/loader.js"></script>
	<div id="barchart"></div>
	<div id="piechart"></div>

	<script type="text/javascript">
	google.charts.load('current', {
		packages: ['corechart']
	}).then(function () {
		var data = new google.visualization.DataTable();
		data.addColumn('string', 'Patterns');
		data.addColumn('number', '# of Patterns');

		// add annotation column role
		data.addColumn({ type: 'string', role: 'annotation' });

		data.addRows(${JSON.stringify(stat_arr)});

		var options = {
			title: 'Patterns Distribution',
			legend: {
				position: 'top',
				alignment: 'start'
			},
			width: 1250,
			height: 420
		};

		var chart = new google.visualization.ColumnChart(
			document.getElementById('barchart')
		);
		chart.draw(data, options);

		var data2 = google.visualization.arrayToDataTable(${JSON.stringify(cluster_arr)});

		var options = { 'title': 'Branch Distribution', 'width': 750, 'height': 400 };

		var chart = new google.visualization.PieChart(document.getElementById('piechart'));
		chart.draw(data2, options);
	});
</script>`

	// jump to clusters
	output_html += `Jump to branches: `
	Object.keys(clusters).forEach(num => { output_html += `<a href="#branch_${num}">${num}</a>, ` })

	// pattern list
	output_html += "<h1>" + "Full Pattern List:" + "</h1><ul>";
	Object.keys(summary).map(k => [k, summary[k].length]).sort((a, b) => b[1] - a[1]).forEach(k => {
		output_html += `<li><a href="#${k[0]}">${k[0]}</a>: #${k[1]}</li>`;
	})
	output_html += "</ul>";

	// output_html += `<br><button class="collapsible">Cells by Pattern</button>` + `<div class="content">`
	Object.keys(summary).forEach(pattern => {
		output_html += `<br><br><a id=${pattern}><h3>` + pattern + "</h3></a>";
		summary[pattern].forEach(file_name => {
			output_html += append_item(file_name);
		})
	})
	// output_html += "</div>";

	// branch list
	// output_html += `<br><button class="collapsible">Cells by Branches</button>` + `<div class="content">`
	output_html += "<br><br><h1>" + "Full Branches List:" + "</h1>";
	Object.keys(clusters).forEach(num => {
		output_html += `<br><h3><a id=branch_${num}>` + num + "</h3></a>";
		clusters[num].forEach(file_name => {
			append_item(file_name);
		})
	})
	// output_html += "</div>";
	output_html += collapsible_script;

	return output_html;
}


function gen_html_page(testFolder) {
	read_data(testFolder);

	let output_html = STARTS;
	tracked_cells = Object.keys(cell_info).filter(x => cell_info[x]["changed"] == true);
	sample_num = Math.min(sample_num, tracked_cells.length)

	// sampling code
	output_html += `
<a href="index.html"><h3><i class="fas fa-backward"></i> Go to MainPage</h3></a>
<script>
function sample(tracked_cells) {
  let sampled_cells = [];
  while (sampled_cells.length < ${sample_num}) {
	const file_name = tracked_cells[Math.floor(Math.random() * tracked_cells.length)];
	if (!sampled_cells.includes(file_name))
	  sampled_cells.push(file_name);
  }
  return sampled_cells;
}
let cells = ${JSON.stringify(tracked_cells)};
</script>
<h1>Sampled Cells (${sample_num}/${tracked_cells.length}):</h1>
<button
onclick="let sp = sample(cells).sort();Array.from(document.getElementById('sampled_cells').children).filter(x => x.nodeName !='BR').forEach((x, i)=> {let new_x = document.createElement('a'); new_x.href=sp[i]; new_x.innerText=sp[i]; new_x.target = 'SourceFrame'; x.outerHTML = new_x.outerHTML;});">
Resample
</button><br>
`


	let sampled_cells = sample(tracked_cells, sample_num);
	output_html += `<div id="sampled_cells">`;
	sampled_cells.sort().forEach(file_name => {
		output_html += append_item(file_name);
	});
	output_html += '</div>';

	// resample scripts
	output_html += `<script>
		let sp = sample(cells).sort();
		Array.from(document.getElementById('sampled_cells').children).filter(x => x.nodeName != 'BR').forEach((x, i) => { 
			let new_x = document.createElement('a'); 
			new_x.href = sp[i]; 
			new_x.innerText = sp[i]; 
			new_x.target = 'SourceFrame'; 
			x.outerHTML = new_x.outerHTML; 
		});
	</script>`

	// add statistics
	output_html += gen_plots(summary, statistics, clusters, cell_info, testFolder);

	output_html += ENDS;

	writeFileSync(path.join("html_outputs",
		testFolder.split(path.sep)[1] + ".html"), output_html);
}


function gen_index_page() {
	let output_html = `<HTML>

<HEAD>
  <META NAME="GENERATOR" CONTENT="Java2HTML Version 1.5.1">
  <TITLE>Patterns MainPage</TITLE>
  <LINK REL="stylesheet" TYPE="text/css" HREF="stylesheet.css" TITLE="Style">
  <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
	${collapse_style}
</HEAD>

<BODY>
  <center>
    <h1>
      <FONT class="w3-sans-serif"><A HREF="titanic.html" TARGET="SourceFrame">titanic (#${folder2cellnum[titanic_folder]})</A></FONT><BR>
      <FONT CLASS="w3-sans-serif"><A HREF="fifa19.html" TARGET="SourceFrame">fifa19 (#${folder2cellnum[fifa_folder]})</A></FONT><BR>
      <FONT CLASS="w3-sans-serif"><A HREF="google.html" TARGET="SourceFrame">google (#${folder2cellnum[google_folder]})</A></FONT><BR>
      <FONT CLASS="w3-sans-serif"><A HREF="airbnb.html" TARGET="SourceFrame">airbnb (#${folder2cellnum[airbnb_folder]})</A></FONT><BR>
    </h1>
  </center>
`

	let all_tracked_cells = Object.keys(all_cell_info).filter(x => all_cell_info[x]["changed"] == true);
	let sampled_cells = sample(all_tracked_cells, 100);
	output_html += `<h1>Sampled 100 Cells:</h1>
	<h2>Inter Rated 20 Cells</h2>
	<div id="sampled_cells">`;
	sampled_cells.forEach((file_name, idx) => {
		output_html += append_item(file_name);
		if (idx == 19) {
			output_html += "<hr>\n";
			output_html += `<br><button class="collapsible">Other 80 Cells</button>` + `<div class="content">`;
		}
	});
	output_html += '</div></div>';

	output_html += gen_plots(all_summary, all_statistics, all_clusters, all_cell_info);

	output_html += ENDS;
	writeFileSync(path.join("html_outputs", "index.html"), output_html);
}


let folder2cellnum = {};

let all_statistics = {};
let all_clusters = {};
let all_summary = {};
let all_cell_info = {};

function convert(testFolder) {

	let dirs = getDirectories(testFolder).map(x => path.join(testFolder, x));

	console.log(dirs);
	console.log(testFolder.split(path.sep)[1] + ".html");

	dirs.forEach(dir => {
		let dir_ls = dir.split(path.sep);
		dir_ls[dir_ls.length - 1] += ".ipynb";
		let nb_path = dir_ls.join(path.sep);
		if (existsSync(nb_path)) {
			// re-generate documents
			execSync(["jupyter", "nbconvert", "--to", "html", nb_path, "--output-dir", "html_outputs"].join(" "));
			console.log("Adding documentation...")
			execSync(["node", script, nb_path].join(" "));
		}
	});
}

// for (const testFolder of [titanic_folder, fifa_folder, google_folder, airbnb_folder]) {
// 	convert(testFolder);
// }
// process.exit();
// gen_html_page(titanic_folder);
// process.exit();

for (const testFolder of [titanic_folder, fifa_folder, google_folder, airbnb_folder]) {
	gen_html_page(testFolder);
	folder2cellnum[testFolder] = tracked_cells.length;

	for (const pattern in statistics) {
		if (!(pattern in all_statistics)) {
			all_statistics[pattern] = 0;
		}
		all_statistics[pattern] += statistics[pattern];
	}

	for (const num in clusters) {
		if (!(num in all_clusters)) {
			all_clusters[num] = []
		}
		all_clusters[num] = all_clusters[num].concat(clusters[num]);
	}

	for (const pattern in summary) {
		if (!(pattern in all_summary)) {
			all_summary[pattern] = []
		}
		all_summary[pattern] = all_summary[pattern].concat(summary[pattern]);
	}

	for (const filepath in cell_info) {
		all_cell_info[filepath] = cell_info[filepath]
		// if (!(pattern in all_cell_info)) {
		// 	all_summary[pattern] = []
		// }
		// all_summary[pattern] = all_summary[pattern].concat(summary[pattern]);
	}

	summary = {};
	clusters = {};
	statistics = {};
	cell_info = {};
}

// for (const pattern in counts) {
// 	console.log(pattern, counts[pattern].length);
// }
// for (const pattern in pat2cols) {
// 	console.log(pattern, pat2cols[pattern].length);
// }
// let arr = Object.keys(nb2pat).map(x => Number(nb2pat[x].length)).sort()
// console.log(nb2pat)
// for (let i in arr) {
// 	console.log(i, arr[i]);
// }
// for (const func in func_counts) {
// 	console.log(func, func_counts[func].length);
// }
// let inverse = {};
// let len_per = {}
// for (const pattern in all_summary) {
// 	for (const cell in all_summary[pattern]) {
// 		if (!(cell in inverse)) {
// 			inverse[cell] = [];
// 		}
// 		inverse[cell].push(pattern)
// 	}
// }
// for (const cell in inverse) {
// 	if (!(inverse[cell].length in len_per)) {
// 		len_per[inverse[cell].length] = []
// 	}
// 	len_per[inverse[cell].length].push(cell)
// }
// for (const l in len_per) {
// 	console.log(l, len_per[l].length);
// }
gen_index_page();