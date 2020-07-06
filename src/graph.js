"use strict";
var __spreadArrays = (this && this.__spreadArrays) || function () {
    for (var s = 0, i = 0, il = arguments.length; i < il; i++) s += arguments[i].length;
    for (var r = Array(s), k = 0, i = 0; i < il; i++)
        for (var a = arguments[i], j = 0, jl = a.length; j < jl; j++, k++)
            r[k] = a[j];
    return r;
};
exports.__esModule = true;
exports.Graph = void 0;
var _1 = require(".");
var Graph = /** @class */ (function () {
    function Graph(getIdentifier) {
        this.getIdentifier = getIdentifier;
        this.outgoing = new Map();
        this.incoming = new Map();
        this._nodes = new _1.Set(getIdentifier);
    }
    Graph.prototype.addEdge = function (fromNode, toNode) {
        this._nodes.add(fromNode);
        this._nodes.add(toNode);
        if (!this.outgoing.has(fromNode)) {
            this.outgoing.set(fromNode, new _1.Set(this.getIdentifier));
        }
        if (!this.incoming.has(toNode)) {
            this.incoming.set(toNode, new _1.Set(this.getIdentifier));
        }
        this.outgoing.get(fromNode).add(toNode);
        this.incoming.get(toNode).add(fromNode);
    };
    Object.defineProperty(Graph.prototype, "nodes", {
        // tslint:disable-next-line: prefer-array-literal
        get: function () { return this._nodes.items; },
        enumerable: false,
        configurable: true
    });
    Graph.prototype.topoSort = function () {
        var _this = this;
        var sorted = [];
        var work = new (_1.Set.bind.apply(_1.Set, __spreadArrays([void 0, this.getIdentifier], this.nodes.filter(function (n) { return !_this.incoming.has(n); }))))();
        var _loop_1 = function () {
            var n = work.pop();
            sorted.push(n);
            if (this_1.outgoing.has(n)) {
                this_1.outgoing.get(n).items.forEach(function (m) {
                    _this.outgoing.get(n).remove(m);
                    _this.incoming.get(m).remove(n);
                    if (_this.incoming.get(m).empty) {
                        work.add(m);
                    }
                });
            }
        };
        var this_1 = this;
        while (!work.empty) {
            _loop_1();
        }
        return sorted;
    };
    return Graph;
}());
exports.Graph = Graph;
