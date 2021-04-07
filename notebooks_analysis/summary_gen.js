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
<script src="https://kit.fontawesome.com/e787e986bb.js" crossorigin="anonymous"></script>
</HEAD>
<BODY>`
let ENDS = "</BODY></HTML>"

let script = ["notebooks_analysis", "html_convert.js"].join(path.sep);

const getDirectories = source =>
	readdirSync(source, { withFileTypes: true })
		.filter(dirent => dirent.isDirectory())
		.map(dirent => dirent.name)


let summary = {};
let statistics = {};
let clusters = {}
let all_cells = [];
let sample_num = 20;

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
	if (!clusters[num].includes(filepath)) {
		clusters[num].push(filepath);
	}
}

function generateSummary(patterns, filepath) {
	if ("other_patterns" in patterns) {
		patterns.other_patterns.forEach((pattern, _) => {
			Object.keys(pattern).forEach(k => {
				if (k != "copy") {
					add_pattern(k, filepath);
					update_statistics(k);
				}
			})
		});
	}
	Object.keys(patterns).forEach(col_str => {
		let cols = col_str.split('|');
		if (cols.length <= 1)
			return false;
		let pattern = patterns[col_str].join('(') + "(" + ")".repeat(patterns[col_str].length);
		add_pattern(pattern, filepath);
		for (const p of patterns[col_str]) {
			update_statistics(p);
		}
	});
}

function gen_html_page(testFolder) {

	let dirs = getDirectories(testFolder).map(x => path.join(testFolder, x));

	console.log(dirs);
	console.log(testFolder.split(path.sep)[1] + ".html");


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

				Object.keys(data.partition).forEach(k => {
					update_clusters(Object.keys(data.partition[k]).length, nb_html + "#" + cell_num);
				})
				Object.keys(data.summary).forEach(k => {
					generateSummary(data.summary[k], nb_html + "#" + cell_num);
				})
			});
		}
	});

	let output_html = STARTS;
	sample_num = Math.max(sample_num, all_cells.length)

	// sampling code
	output_html += `
<a href="index.html"><h3><i class="fas fa-backward"></i> Go to MainPage</h3></a>
<script>
function sample(all_cells) {
  let sampled_cells = [];
  while (sampled_cells.length < ${sample_num}) {
	const file_name = all_cells[Math.floor(Math.random() * all_cells.length)];
	if (!sampled_cells.includes(file_name))
	  sampled_cells.push(file_name);
  }
  return sampled_cells;
}
let cells = ${JSON.stringify(all_cells)};
</script>
<h1>Sampled Cells (${sample_num}/${all_cells.length}):</h1>
<button
onclick="let sp = sample(cells).sort();Array.from(document.getElementById('sampled_cells').children).filter(x => x.nodeName !='BR').forEach((x, i)=> {let new_x = document.createElement('a'); new_x.href=sp[i]; new_x.innerText=sp[i]; new_x.target = 'SourceFrame'; x.outerHTML = new_x.outerHTML;});">
Resample
</button><br>
`

	// sampling cells
	function sample(all_cells) {
		let sampled_cells = []
		while (sampled_cells.length < sample_num) {
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

	// add statistics
	let stat_arr = Object.keys(statistics).map(key => [key, statistics[key], String(statistics[key])]).sort((a, b) => b[1] - a[1]);
	let cluster_arr = Object.keys(clusters).map(key => [key, clusters[key].length]).filter(k => k[0] < 20);
	cluster_arr.unshift(["branchnum", "count"]);
	cluster_arr.push([">=20", Object.keys(clusters).filter(k => k >= 20).map(k => clusters[k].length).reduce((a, b) => a + b, 0)]);
	output_html += `
	<script>
		let sp = sample(cells).sort();
		Array.from(document.getElementById('sampled_cells').children).filter(x => x.nodeName != 'BR').forEach((x, i) => { 
			let new_x = document.createElement('a'); 
			new_x.href = sp[i]; 
			new_x.innerText = sp[i]; 
			new_x.target = 'SourceFrame'; 
			x.outerHTML = new_x.outerHTML; 
		});
	</script>

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
			width: 1000,
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
</script>
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

	Object.keys(summary).forEach(pattern => {
		output_html += `<br><br><a id=${pattern}><h3>` + pattern + "</h3></a>";
		summary[pattern].forEach(file_name => {
			output_html += '<FONT CLASS="FrameItemFont"><A HREF="' + file_name + '" TARGET="SourceFrame">' + file_name + '</A></FONT><BR>\n'
		})
	})

	// branch list
	output_html += "<br><br><h1>" + "Full Branches List:" + "</h1>";
	Object.keys(clusters).forEach(num => {
		output_html += `<br><h3><a id=branch_${num}>` + num + "</h3></a>";
		clusters[num].forEach(file_name => {
			output_html += '<FONT CLASS="FrameItemFont"><A HREF="' + file_name + '" TARGET="SourceFrame">' + file_name + '</A></FONT><BR>\n'
		})
	})

	output_html += ENDS;

	writeFileSync(path.join("html_outputs",
		testFolder.split(path.sep)[1] + ".html"), output_html);
}

let titanic_folder = ["notebooks", "titanic"].join(path.sep);
let fifa_folder = ["notebooks", "fifa19"].join(path.sep);
let google_folder = ["notebooks", "google"].join(path.sep);
let airbnb_folder = ["notebooks", "airbnb"].join(path.sep);

gen_html_page(testFolder);

// for (const testFolder of [titanic_folder, fifa_folder, google_folder, airbnb_folder]) {
// 	gen_html_page(testFolder);
// 	summary = {};
// 	all_cells = [];
//  statistics = {};
// }