"use strict";

const testFolder = './notebooks/titanic_notebooks/';
const fs = require('fs');
const { createImportSpecifier } = require('typescript');

let jsonfiles = [];

fs.readdirSync(testFolder).forEach(file => {
  if (file.endsWith("out.json")) {
    jsonfiles.push(file);
  }
});

console.log(jsonfiles);

let summaries = new Map();
let funcs = [];

jsonfiles.forEach(file => {
  let v = JSON.parse(fs.readFileSync(testFolder + file).toString());
  summaries.set(file, [])
  Object.entries(v).forEach(cell => {
    if (Object.entries(cell[1].summary).length > 0) {
      let arr = summaries.get(file);
      Object.entries(cell[1].summary).forEach(inv => arr.push(inv));
      summaries.set(file, arr);
    }
    if (Object.entries(cell[1].function).length > 0) {
      funcs.push(cell[1].function);
    }
    // summaries.push(cell[1].summary);
  })
})

let length_arr = [];

for (let [key, value] of summaries) {
  // console.log(key + ' = ' + value)
  summaries.set(key, value.filter(x => {
    return !x[1].includes("no change");
  }));
}

// summaries.forEach(key, item => {
//   // console.log(key)
//   // let res = item.filter(x => {
//   //   return !x[1].includes("no change");
//   // });
//   // item = res;
//   // summaries.set()
//   // item.forEach(x => {
//   //   console.log(x[1]);
//   //   if (x[1].includes("no change"))
//   //     item
//   // })
// })

let cntType = { "removeCol": 0, "addCol": 0, "removeRow": 0, "convert": 0, "change": 0, "copy": 0, "truncate": 0, "rearrange": 0 }

let pattern = Object.entries(cntType).map(x => x[0]);
pattern = pattern.slice(3);
console.log(pattern)

summaries.forEach(item => {
  item.forEach(x => {
    // console.log(x[1]);
    if (x[1].includes("remove") && x[1].includes("columns"))
      cntType["removeCol"] += 1;
    if (x[1].includes("add") && x[1].includes("columns"))
      cntType["addCol"] += 1;
    if (x[1].includes("remove") && x[1].includes("rows"))
      cntType["removeRow"] += 1;
    pattern.forEach(str => {
      if (x[1].includes(str))
        cntType[str] += 1;
    })
  })
  length_arr.push(item.length);
})

funcs.forEach(item => {
  Object.entries(item).forEach(x => {
    console.log(x[0], x[1].counts)
  }
  )
})
console.log(funcs.length)

// console.log(cntType);
// console.log(length_arr)
// console.log(length_arr.reduce((x, y) => x + y, 0))
// console.log(summaries.length);
// console.log(summaries);