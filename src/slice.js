"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __spreadArrays = (this && this.__spreadArrays) || function () {
    for (var s = 0, i = 0, il = arguments.length; i < il; i++) s += arguments[i].length;
    for (var r = Array(s), k = 0, i = 0; i < il; i++)
        for (var a = arguments[i], j = 0, jl = a.length; j < jl; j++, k++)
            r[k] = a[j];
    return r;
};
exports.__esModule = true;
exports.sliceLines = exports.slice = exports.SliceDirection = exports.LocationSet = void 0;
var python_parser_1 = require("./python-parser");
var control_flow_1 = require("./control-flow");
var data_flow_1 = require("./data-flow");
var set_1 = require("./set");
function lineRange(loc) {
    return set_1.range(loc.first_line, loc.last_line + (loc.last_column ? 1 : 0));
}
var LocationSet = /** @class */ (function (_super) {
    __extends(LocationSet, _super);
    function LocationSet() {
        var items = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            items[_i] = arguments[_i];
        }
        return _super.apply(this, __spreadArrays([function (l) {
                return [l.first_line, l.first_column, l.last_line, l.last_column].toString();
            }], items)) || this;
    }
    return LocationSet;
}(set_1.Set));
exports.LocationSet = LocationSet;
function within(inner, outer) {
    var leftWithin = outer.first_line < inner.first_line ||
        (outer.first_line == inner.first_line &&
            outer.first_column <= inner.first_column);
    var rightWithin = outer.last_line > inner.last_line ||
        (outer.last_line == inner.last_line &&
            outer.last_column >= inner.last_column);
    return leftWithin && rightWithin;
}
function isPositionBetween(line, column, start_line, start_column, end_line, end_column) {
    var afterStart = line > start_line || (line == start_line && column >= start_column);
    var beforeEnd = line < end_line || (line == end_line && column <= end_column);
    return afterStart && beforeEnd;
}
function intersect(l1, l2) {
    return (isPositionBetween(l1.first_line, l1.first_column, l2.first_line, l2.first_column, l2.last_line, l2.last_column) ||
        isPositionBetween(l1.last_line, l1.last_column, l2.first_line, l2.first_column, l2.last_line, l2.last_column) ||
        within(l1, l2) ||
        within(l2, l1));
}
var SliceDirection;
(function (SliceDirection) {
    SliceDirection[SliceDirection["Forward"] = 0] = "Forward";
    SliceDirection[SliceDirection["Backward"] = 1] = "Backward";
})(SliceDirection = exports.SliceDirection || (exports.SliceDirection = {}));
/**
 * More general slice: given locations of important syntax nodes, find locations of all relevant
 * definitions. Locations can be mapped to lines later.
 * seedLocations are symbol locations.
 */
function slice(ast, seedLocations, dataflowAnalyzer, direction) {
    if (direction === void 0) { direction = SliceDirection.Backward; }
    dataflowAnalyzer = dataflowAnalyzer || new data_flow_1.DataflowAnalyzer();
    var cfg = new control_flow_1.ControlFlowGraph(ast);
    var dfa = dataflowAnalyzer.analyze(cfg).dataflows;
    // Include at least the full statements for each seed.
    var acceptLocation = function (loc) { return true; };
    var sliceLocations = new LocationSet();
    if (seedLocations) {
        var seedStatementLocations_1 = findSeedStatementLocations(seedLocations, cfg);
        acceptLocation = function (loc) { return seedStatementLocations_1.some(function (seedStmtLoc) { return intersect(seedStmtLoc, loc); }); };
        sliceLocations = new (LocationSet.bind.apply(LocationSet, __spreadArrays([void 0], seedStatementLocations_1.items)))();
    }
    var lastSize;
    do {
        lastSize = sliceLocations.size;
        var _loop_1 = function (flow) {
            var _a = direction === SliceDirection.Backward ?
                [flow.fromNode.location, flow.toNode.location] :
                [flow.toNode.location, flow.fromNode.location], start = _a[0], end = _a[1];
            if (acceptLocation(end)) {
                sliceLocations.add(end);
            }
            if (sliceLocations.some(function (loc) { return within(end, loc); })) {
                sliceLocations.add(start);
            }
        };
        for (var _i = 0, _a = dfa.items; _i < _a.length; _i++) {
            var flow = _a[_i];
            _loop_1(flow);
        }
    } while (sliceLocations.size > lastSize);
    return sliceLocations;
}
exports.slice = slice;
function findSeedStatementLocations(seedLocations, cfg) {
    var seedStatementLocations = new LocationSet();
    seedLocations.items.forEach(function (seedLoc) {
        for (var _i = 0, _a = cfg.blocks; _i < _a.length; _i++) {
            var block = _a[_i];
            for (var _b = 0, _c = block.statements; _b < _c.length; _b++) {
                var statement = _c[_b];
                if (intersect(seedLoc, statement.location)) {
                    seedStatementLocations.add(statement.location);
                }
            }
        }
    });
    return seedStatementLocations;
}
/**
 * Slice: given a set of lines in a program, return lines it depends on.
 * OUT OF DATE: use slice() instead of sliceLines().
 */
function sliceLines(code, relevantLineNumbers) {
    var ast = python_parser_1.parse(code);
    var cfg = new control_flow_1.ControlFlowGraph(ast);
    var dataflowAnalyzer = new data_flow_1.DataflowAnalyzer();
    var dfa = dataflowAnalyzer.analyze(cfg).dataflows;
    var lastSize;
    do {
        lastSize = relevantLineNumbers.size;
        for (var _i = 0, _a = dfa.items; _i < _a.length; _i++) {
            var flow = _a[_i];
            var fromLines = lineRange(flow.fromNode.location);
            var toLines = lineRange(flow.toNode.location);
            var startLines = toLines;
            var endLines = fromLines;
            if (!relevantLineNumbers.intersect(startLines).empty) {
                relevantLineNumbers = relevantLineNumbers.union(endLines);
            }
        }
    } while (relevantLineNumbers.size > lastSize);
    return relevantLineNumbers;
}
exports.sliceLines = sliceLines;
