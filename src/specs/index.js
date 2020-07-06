"use strict";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
exports.__esModule = true;
exports.DefaultSpecs = void 0;
var builtins = require("./__builtins__.json");
var random = require("./random.json");
var matplotlib = require("./matplotlib.json");
var pandas = require("./pandas.json");
var sklearn = require("./sklearn.json");
var numpy = require("./numpy.json");
exports.DefaultSpecs = __assign(__assign(__assign(__assign(__assign(__assign({}, builtins), random), matplotlib), pandas), sklearn), numpy);
