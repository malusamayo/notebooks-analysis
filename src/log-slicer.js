"use strict";
var __spreadArrays = (this && this.__spreadArrays) || function () {
    for (var s = 0, i = 0, il = arguments.length; i < il; i++) s += arguments[i].length;
    for (var r = Array(s), k = 0, i = 0; i < il; i++)
        for (var a = arguments[i], j = 0, jl = a.length; j < jl; j++, k++)
            r[k] = a[j];
    return r;
};
exports.__esModule = true;
exports.ExecutionLogSlicer = exports.SlicedExecution = exports.CellExecution = void 0;
var cellslice_1 = require("./cellslice");
var program_builder_1 = require("./program-builder");
var set_1 = require("./set");
var slice_1 = require("./slice");
/**
 * A record of when a cell was executed.
 */
var CellExecution = /** @class */ (function () {
    function CellExecution(cell, executionTime) {
        this.cell = cell;
        this.executionTime = executionTime;
    }
    /**
     * Update this method if at some point we only want to save some about a CellExecution when
     * serializing it and saving history.
     */
    CellExecution.prototype.toJSON = function () {
        return JSON.parse(JSON.stringify(this));
    };
    return CellExecution;
}());
exports.CellExecution = CellExecution;
/**
 * A slice over a version of executed code.
 */
var SlicedExecution = /** @class */ (function () {
    function SlicedExecution(executionTime, cellSlices) {
        this.executionTime = executionTime;
        this.cellSlices = cellSlices;
    }
    SlicedExecution.prototype.merge = function () {
        var slicedExecutions = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            slicedExecutions[_i] = arguments[_i];
        }
        var cellSlices = {};
        var mergedCellSlices = [];
        for (var _a = 0, _b = slicedExecutions.concat(this); _a < _b.length; _a++) {
            var slicedExecution = _b[_a];
            for (var _c = 0, _d = slicedExecution.cellSlices; _c < _d.length; _c++) {
                var cellSlice = _d[_c];
                var cell = cellSlice.cell;
                if (!cellSlices[cell.executionEventId]) {
                    var newCellSlice = new cellslice_1.CellSlice(cell.deepCopy(), new slice_1.LocationSet(), cellSlice.executionTime);
                    cellSlices[cell.executionEventId] = newCellSlice;
                    mergedCellSlices.push(newCellSlice);
                }
                var mergedCellSlice = cellSlices[cell.executionEventId];
                mergedCellSlice.slice = mergedCellSlice.slice.union(cellSlice.slice);
            }
        }
        return new SlicedExecution(new Date(), // Date doesn't mean anything for the merged slice.
        mergedCellSlices.sort(function (a, b) { return a.cell.executionCount - b.cell.executionCount; }));
    };
    return SlicedExecution;
}());
exports.SlicedExecution = SlicedExecution;
/**
 * Makes slice on a log of executed cells.
 */
var ExecutionLogSlicer = /** @class */ (function () {
    /**
     * Construct a new execution log slicer.
     */
    function ExecutionLogSlicer(dataflowAnalyzer) {
        this.dataflowAnalyzer = dataflowAnalyzer;
        this.executionLog = [];
        /**
         * Signal emitted when a cell's execution has been completely processed.
         */
        this.executionLogged = [];
        this.programBuilder = new program_builder_1.ProgramBuilder(dataflowAnalyzer);
    }
    /**
     * Log that a cell has just been executed. The execution time for this cell will be stored
     * as the moment at which this method is called.
     */
    ExecutionLogSlicer.prototype.logExecution = function (cell) {
        var cellExecution = new CellExecution(cell, new Date());
        this.addExecutionToLog(cellExecution);
    };
    /**
     * Use logExecution instead if a cell has just been run to annotate it with the current time
     * as the execution time. This function is intended to be used only to initialize history
     * when a notebook is reloaded. However, any method that eventually calls this method will
     * notify all observers that this cell has been executed.
     */
    ExecutionLogSlicer.prototype.addExecutionToLog = function (cellExecution) {
        this.programBuilder.add(cellExecution.cell);
        this.executionLog.push(cellExecution);
        this.executionLogged.forEach(function (callback) { return callback(cellExecution); });
    };
    /**
     * Reset the log, removing log records.
     */
    ExecutionLogSlicer.prototype.reset = function () {
        this.executionLog = [];
        this.programBuilder.reset();
    };
    /**
     * Get slice for the latest execution of a cell.
     */
    ExecutionLogSlicer.prototype.sliceLatestExecution = function (cellId, seedLocations) {
        // XXX: This computes more than it has to, performing a slice on each execution of a cell
        // instead of just its latest computation. Optimize later if necessary.
        return this.sliceAllExecutions(cellId, seedLocations).pop();
    };
    /**
     * Get slices of the necessary code for all executions of a cell.
     * Relevant line numbers are relative to the cell's start line (starting at first line = 0).
     */
    ExecutionLogSlicer.prototype.sliceAllExecutions = function (cellId, seedLocations) {
        var _this = this;
        // Make a map from cells to their execution times.
        var cellExecutionTimes = new Map(this.executionLog.map(function (e) { return [e.cell.executionEventId, e.executionTime]; }));
        return this.executionLog
            .filter(function (execution) {
            return execution.cell.persistentId == cellId && execution.cell.executionCount;
        })
            .map(function (execution) {
            // Build the program up to that cell.
            var program = _this.programBuilder.buildTo(execution.cell.executionEventId);
            if (!program) {
                return null;
            }
            // Set the seed locations for the slice.
            if (!seedLocations) {
                // If seed locations weren't specified, slice the whole cell.
                // XXX: Whole cell specified by an unreasonably large character range.
                seedLocations = new slice_1.LocationSet({
                    first_line: 1,
                    first_column: 1,
                    last_line: 10000,
                    last_column: 10000
                });
            }
            // Set seed locations were specified relative to the last cell's position in program.
            var lastCellLines = program.cellToLineMap[execution.cell.executionEventId];
            var lastCellStart = Math.min.apply(Math, lastCellLines.items);
            var adjustedSeedLocations = seedLocations.mapSame(function (loc) { return ({
                first_line: lastCellStart + loc.first_line - 1,
                first_column: loc.first_column,
                last_line: lastCellStart + loc.last_line - 1,
                last_column: loc.last_column
            }); });
            // Slice the program
            var sliceLocations = slice_1.slice(program.tree, adjustedSeedLocations, _this.dataflowAnalyzer).items.sort(function (loc1, loc2) { return loc1.first_line - loc2.first_line; });
            // Get the relative offsets of slice lines in each cell.
            var cellSliceLocations = {};
            var cellOrder = [];
            sliceLocations.forEach(function (location) {
                var sliceCell = program.lineToCellMap[location.first_line];
                var sliceCellLines = program.cellToLineMap[sliceCell.executionEventId];
                var sliceCellStart = Math.min.apply(Math, sliceCellLines.items);
                if (cellOrder.indexOf(sliceCell) == -1) {
                    cellOrder.push(sliceCell);
                }
                var adjustedLocation = {
                    first_line: location.first_line - sliceCellStart + 1,
                    first_column: location.first_column,
                    last_line: location.last_line - sliceCellStart + 1,
                    last_column: location.last_column
                };
                if (!cellSliceLocations[sliceCell.executionEventId]) {
                    cellSliceLocations[sliceCell.executionEventId] = new slice_1.LocationSet();
                }
                cellSliceLocations[sliceCell.executionEventId].add(adjustedLocation);
            });
            var cellSlices = cellOrder.map(function (sliceCell) {
                return new cellslice_1.CellSlice(sliceCell, cellSliceLocations[sliceCell.executionEventId], cellExecutionTimes[sliceCell.executionEventId]);
            });
            return new SlicedExecution(execution.executionTime, cellSlices);
        })
            .filter(function (s) { return s != null && s != undefined; });
    };
    Object.defineProperty(ExecutionLogSlicer.prototype, "cellExecutions", {
        get: function () {
            return this.executionLog;
        },
        enumerable: false,
        configurable: true
    });
    /**
     * Get the cell program (tree, defs, uses) for a cell.
     */
    ExecutionLogSlicer.prototype.getCellProgram = function (executionEventId) {
        return this.programBuilder.getCellProgram(executionEventId);
    };
    /**
     * Returns the cells that directly or indirectly use variables
     * that are defined in the given cell. Result is in
     * topological order.
     * @param executionEventId a cell in the log
     */
    ExecutionLogSlicer.prototype.getDependentCells = function (executionEventId) {
        var program = this.programBuilder.buildFrom(executionEventId);
        var sameCell = this.programBuilder.getCellProgramsWithSameId(executionEventId);
        var lines = new set_1.NumberSet();
        sameCell.forEach(function (cp) {
            return (lines = lines.union(program.cellToLineMap[cp.cell.executionEventId]));
        });
        var seedLocations = new (slice_1.LocationSet.bind.apply(slice_1.LocationSet, __spreadArrays([void 0], lines.items.map(function (line) { return ({
            first_line: line,
            first_column: 0,
            last_line: line,
            last_column: 1
        }); }))))();
        var sliceLocations = slice_1.slice(program.tree, seedLocations, undefined, slice_1.SliceDirection.Forward).items;
        return new (set_1.Set.bind.apply(set_1.Set, __spreadArrays([void 0, function (c) { return c.persistentId; }], sliceLocations.map(function (loc) { return program.lineToCellMap[loc.first_line]; }))))().items.filter(function (c) { return c.executionEventId !== executionEventId; });
    };
    return ExecutionLogSlicer;
}());
exports.ExecutionLogSlicer = ExecutionLogSlicer;
