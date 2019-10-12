import { ExecutionLogSlicer } from '../log-slicer';
import { Location, LogCell, DataflowAnalyzer } from '..';
import { expect } from 'chai';

function loc(line0: number, col0: number, line1 = line0 + 1, col1 = 0): Location {
	return { first_line: line0, first_column: col0, last_line: line1, last_column: col1 };
}

function makeLog(lines: string[]) {
	const cells = lines.map((text, i) => new LogCell({ text: text, executionCount: i + 1 }));
	const logSlicer = new ExecutionLogSlicer(new DataflowAnalyzer());
	cells.forEach(cell => logSlicer.logExecution(cell));
	return logSlicer;
}

describe('log-slicer', () => {

	it("does jim's demo", () => {
		const logSlicer = makeLog([
			/*[1]*/  "import pandas as pd",
			/*[2]*/  "Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4'], 'Price': [22000,25000,27000,35000]}\n" +
			"df = pd.DataFrame(Cars,columns= ['Brand', 'Price'])",
			/*[3]*/  "def check(df, size=11):\n" +
			"    print(df)",
			/*[4]*/  "print(df)",
			/*[5]*/  "x = df['Brand'].values"
		]);
		const lastCell = logSlicer.cellExecutions[logSlicer.cellExecutions.length - 1].cell;
		const slice = logSlicer.sliceLatestExecution(lastCell.persistentId);
		expect(slice).to.exist;
		expect(slice.cellSlices).to.exist;
		const cellCounts = slice.cellSlices.map(cell => cell.cell.executionCount);
		[1, 2, 5].forEach(c => expect(cellCounts).to.include(c));
		[3, 4].forEach(c => expect(cellCounts).to.not.include(c));
	});

	describe("getDependentCells", () => {

		it("handles simple in-order", () => {
			const lines = [
				"x = 3",
				"y = x+1"
			];
			const logSlicer = makeLog(lines);
			const deps = logSlicer.getDependentCells(logSlicer.cellExecutions[0].cell.executionEventId);
			expect(deps).to.exist;
			expect(deps).to.have.length(1);
			expect(deps[0].text).to.equal(lines[1]);
		});

		it("handles variable redefinition", () => {
			const lines = [
				"x = 3",
				"y = x+1",
				"x = 4",
				"y = x*2",
			];
			const logSlicer = makeLog(lines);
			const deps = logSlicer.getDependentCells(logSlicer.cellExecutions[0].cell.executionEventId);
			expect(deps).to.exist;
			expect(deps).to.have.length(1);
			expect(deps[0].text).to.equal(lines[1]);
			const deps2 = logSlicer.getDependentCells(logSlicer.cellExecutions[2].cell.executionEventId);
			expect(deps2).to.exist;
			expect(deps2).to.have.length(1);
			expect(deps2[0].text).to.equal(lines[3]);
		});

		it("handles no deps", () => {
			const lines = [
				"x = 3\nprint(x)",
				"y = 2\nprint(y)",
			];
			const logSlicer = makeLog(lines);
			const deps = logSlicer.getDependentCells(logSlicer.cellExecutions[0].cell.executionEventId);
			expect(deps).to.exist;
			expect(deps).to.have.length(0);
		});

		it("works transitively", () => {
			const lines = [
				"x = 3",
				"y = x+1",
				"z = y-1"
			];
			const logSlicer = makeLog(lines);
			const deps = logSlicer.getDependentCells(logSlicer.cellExecutions[0].cell.executionEventId);
			expect(deps).to.exist;
			expect(deps).to.have.length(2);
			const deplines = deps.map(d => d.text);
			expect(deplines).includes(lines[1]);
			expect(deplines).includes(lines[2]);
		});

		it("includes all defs within cells", () => {
			const lines = [
				"x = 3\nq = 2",
				"y = x+1",
				"z = q-1"
			];
			const logSlicer = makeLog(lines);
			const deps = logSlicer.getDependentCells(logSlicer.cellExecutions[0].cell.executionEventId);
			expect(deps).to.exist;
			expect(deps).to.have.length(2);
			const deplines = deps.map(d => d.text);
			expect(deplines).includes(lines[1]);
			expect(deplines).includes(lines[2]);
		});

		it("handles cell re-execution", () => {
			const lines = [
				"x = 2\nprint(x)",
				"y = x+1\nprint(y)",
				"q = 2"
			];
			const cells = lines.map((text, i) => new LogCell({ text: text, executionCount: i + 1 }));
			cells.push(new LogCell(Object.assign({}, cells[0],
				{ text: "x = 20\nprint(x)", executionCount: cells.length + 1 })));
			const logSlicer = new ExecutionLogSlicer(new DataflowAnalyzer());
			cells.forEach(cell => logSlicer.logExecution(cell));
			const rerunFirst = logSlicer.cellExecutions[3].cell.executionEventId;
			const deps = logSlicer.getDependentCells(rerunFirst);
			expect(deps).to.exist;
			expect(deps).to.have.length(1);
			expect(deps[0].text).equals(lines[1]);
		});

		it("handles cell re-execution no-op", () => {
			const lines = [
				"x = 2\nprint(x)",
				"y = 3\nprint(y)",
				"q = 2"
			];
			const cells = lines.map((text, i) => new LogCell({ text: text, executionCount: i + 1 }));
			cells.push(new LogCell(Object.assign({}, cells[0],
				{ text: "x = 20\nprint(x)", executionCount: cells.length + 1 })));
			const logSlicer = new ExecutionLogSlicer(new DataflowAnalyzer());
			cells.forEach(cell => logSlicer.logExecution(cell));
			const deps = logSlicer.getDependentCells(logSlicer.cellExecutions[3].cell.executionEventId);
			expect(deps).to.exist;
			expect(deps).to.have.length(0);
		});

		it("return result in topo order", () => {
			const lines = [
				"x = 1",
				"y = 2*x",
				"z = x*y"
			];
			const cells = lines.map((text, i) => new LogCell({ text, id: i.toString(), persistentId: i.toString(), executionCount: i + 1 }));
			cells.push(new LogCell(Object.assign({}, cells[0],
				{ text: "x = 2", executionCount: cells.length + 1 })));
			cells.push(new LogCell(Object.assign({}, cells[1],
				{ text: "y = x*2", executionCount: cells.length + 1 })));
			cells.push(new LogCell(Object.assign({}, cells[2],
				{ text: "z = y*x", executionCount: cells.length + 1 })));
			cells.push(new LogCell(Object.assign({}, cells[0],
				{ text: "x = 3", executionCount: cells.length + 1 })));
			const logSlicer = new ExecutionLogSlicer(new DataflowAnalyzer());
			cells.forEach(cell => logSlicer.logExecution(cell));
			const lastEvent = logSlicer.cellExecutions[logSlicer.cellExecutions.length - 1].cell.executionEventId;
			const deps = logSlicer.getDependentCells(lastEvent);
			expect(deps).to.exist;
			expect(deps).to.have.length(2);
			expect(deps[0].text).equals('y = x*2');
			expect(deps[1].text).equals('z = y*x');
		});

		it("can be called multiple times", () => {
			const lines = [
				"x = 1",
				"y = 2*x",
				"z = x*y"
			];
			const cells = lines.map((text, i) => new LogCell({ text, id: i.toString(), persistentId: i.toString(), executionCount: i + 1 }));
			const logSlicer = new ExecutionLogSlicer(new DataflowAnalyzer());
			cells.forEach(cell => logSlicer.logExecution(cell));
			const deps = logSlicer.getDependentCells(logSlicer.cellExecutions[0].cell.executionEventId);
			expect(deps).to.exist;
			expect(deps).to.have.length(2);
			expect(deps[0].text).equals('y = 2*x');
			expect(deps[1].text).equals('z = x*y');

			logSlicer.logExecution(new LogCell(Object.assign({}, cells[0],
				{ text: "x = 2", executionCount: cells.length + 1 })));
			logSlicer.logExecution(new LogCell(Object.assign({}, cells[1],
				{ text: "y = x*2", executionCount: cells.length + 1 })));
			logSlicer.logExecution(new LogCell(Object.assign({}, cells[2],
				{ text: "z = y*x", executionCount: cells.length + 1 })));
			logSlicer.logExecution(new LogCell(Object.assign({}, cells[0],
				{ text: "x = 3", executionCount: cells.length + 1 })));
			const lastEvent = logSlicer.cellExecutions[logSlicer.cellExecutions.length - 1].cell.executionEventId;
			const deps2 = logSlicer.getDependentCells(lastEvent);
			expect(deps2).to.exist;
			expect(deps2).to.have.length(2);
			expect(deps2[0].text).equals('y = x*2');
			expect(deps2[1].text).equals('z = y*x');
		});

		it("handles api calls", () => {
			const lines = [
				"from matplotlib.pyplot import scatter\nfrom sklearn.cluster import KMeans\nfrom sklearn import datasets",
				"data = datasets.load_iris().data[:,2:4]\npetal_length, petal_width = data[:,1], data[:,0]",
				"k=3",
				"clusters = KMeans(n_clusters=k).fit(data).labels_",
				"scatter(petal_length, petal_width, c=clusters)"
			];
			const cells = lines.map((text, i) => new LogCell({ text: text, executionCount: i + 1 }));
			cells.push(new LogCell(Object.assign({}, cells[2],
				{ text: "k=4", executionCount: cells.length + 1 })));
				
			const logSlicer = new ExecutionLogSlicer(new DataflowAnalyzer());
			cells.forEach(cell => logSlicer.logExecution(cell));

			const lastEvent = logSlicer.cellExecutions[logSlicer.cellExecutions.length - 1].cell.executionEventId;
			const deps = logSlicer.getDependentCells(lastEvent);
			expect(deps).to.exist;
			expect(deps).to.have.length(2);
			const sliceText = deps.map(c => c.text);
			expect(sliceText).to.include(lines[3]);
			expect(sliceText).to.include(lines[4]);
		});
	});

});
