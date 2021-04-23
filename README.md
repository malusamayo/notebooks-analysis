# notebooks-analysis

A tool for generating documentation information for jupyter notebooks via program analysis & synthesis. 

## How to use the tool
1. Run `npm install` and `npm run build`
2. Run `python run.py your_notebook.ipynb`

## Trouble Shooting
1. npm install bugs: please refer to https://stackoverflow.com/a/64043900/13881716

## Project Structure
.
│
├── src: static analysis tool code
│
│
├── notebooks_analysis
│   ├── analyzer.py: synthesis engine
│   │ 
│   ├── detect_changes.py: detect variable changes, used for evaluation
│   │ 
│   ├── helper.py: tracing code
│   │ 
│   ├── html_convert.js: convert a notebook and generated docs to an html page
│   │ 
│   ├── instrumenter.js: instrument tracing code and variable saving code
│   │ 
│   ├── log_analysis.js: analyze runtime overhead, used for evaluation
│   │ 
│   ├── model.py: model common objects
│   │ 
│   ├── summary_gen.py: generate summary pages for a set of notebooks and docs, used for evaluation
│   │ 
│   └── synthesizer_test.py: testing code for synthesis
│   
├── batch-run.py: run script for notebooks in a folder
│
└── run.py: run script