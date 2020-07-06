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
exports.range = exports.NumberSet = exports.StringSet = exports.Set = void 0;
var Set = /** @class */ (function () {
    // Two items A and B are considered the same value iff getIdentifier(A) === getIdentifier(B).
    function Set(getIdentifier) {
        var items = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            items[_i - 1] = arguments[_i];
        }
        this.getIdentifier = getIdentifier;
        this._items = {};
        this.getIdentifier = getIdentifier;
        this._items = {};
        this.add.apply(this, items);
    }
    Object.defineProperty(Set.prototype, "size", {
        get: function () {
            return this.items.length;
        },
        enumerable: false,
        configurable: true
    });
    Set.prototype.add = function () {
        var _this = this;
        var items = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            items[_i] = arguments[_i];
        }
        items.forEach(function (item) { return (_this._items[_this.getIdentifier(item)] = item); });
    };
    Set.prototype.remove = function (item) {
        if (this.has(item)) {
            delete this._items[this.getIdentifier(item)];
        }
    };
    Set.prototype.pop = function () {
        if (this.empty) {
            throw 'empty';
        }
        var someKey = Object.keys(this._items)[0];
        var result = this._items[someKey];
        this.remove(result);
        return result;
    };
    Set.prototype.has = function (item) {
        return this._items[this.getIdentifier(item)] != undefined;
    };
    Object.defineProperty(Set.prototype, "items", {
        get: function () {
            var _this = this;
            return Object.keys(this._items).map(function (k) { return _this._items[k]; });
        },
        enumerable: false,
        configurable: true
    });
    Set.prototype.equals = function (that) {
        return (this.size == that.size && this.items.every(function (item) { return that.has(item); }));
    };
    Object.defineProperty(Set.prototype, "empty", {
        get: function () {
            return Object.keys(this._items).length == 0;
        },
        enumerable: false,
        configurable: true
    });
    Set.prototype.union = function () {
        var _a;
        var those = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            those[_i] = arguments[_i];
        }
        return new (Set.bind.apply(Set, __spreadArrays([void 0, this.getIdentifier], (_a = this.items).concat.apply(_a, those.map(function (that) { return that.items; })))))();
    };
    Set.prototype.intersect = function (that) {
        return new (Set.bind.apply(Set, __spreadArrays([void 0, this.getIdentifier], this.items.filter(function (item) { return that.has(item); }))))();
    };
    Set.prototype.filter = function (predicate) {
        return new (Set.bind.apply(Set, __spreadArrays([void 0, this.getIdentifier], this.items.filter(predicate))))();
    };
    Set.prototype.map = function (getIdentifier, transform) {
        return new (Set.bind.apply(Set, __spreadArrays([void 0, getIdentifier], this.items.map(transform))))();
    };
    Set.prototype.mapSame = function (transform) {
        return new (Set.bind.apply(Set, __spreadArrays([void 0, this.getIdentifier], this.items.map(transform))))();
    };
    Set.prototype.some = function (predicate) {
        return this.items.some(predicate);
    };
    Set.prototype.minus = function (that) {
        return new (Set.bind.apply(Set, __spreadArrays([void 0, this.getIdentifier], this.items.filter(function (x) { return !that.has(x); }))))();
    };
    Set.prototype.take = function () {
        if (this.empty) {
            throw 'cannot take from an empty set';
        }
        var first = parseInt(Object.keys(this._items)[0]);
        var result = this._items[first];
        this.remove(result);
        return result;
    };
    Set.prototype.product = function (that) {
        var _this = this;
        return new (Set.bind.apply(Set, __spreadArrays([void 0, function (_a) {
                var x = _a[0], y = _a[1];
                return _this.getIdentifier(x) + that.getIdentifier(y);
            }], flatten.apply(void 0, this.items.map(function (x) {
            return flatten(that.items.map(function (y) { return [x, y]; }));
        })))))();
    };
    return Set;
}());
exports.Set = Set;
var StringSet = /** @class */ (function (_super) {
    __extends(StringSet, _super);
    function StringSet() {
        var items = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            items[_i] = arguments[_i];
        }
        return _super.apply(this, __spreadArrays([function (s) { return s; }], items)) || this;
    }
    return StringSet;
}(Set));
exports.StringSet = StringSet;
var NumberSet = /** @class */ (function (_super) {
    __extends(NumberSet, _super);
    function NumberSet() {
        var items = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            items[_i] = arguments[_i];
        }
        return _super.apply(this, __spreadArrays([function (n) { return n.toString(); }], items)) || this;
    }
    return NumberSet;
}(Set));
exports.NumberSet = NumberSet;
function range(min, max) {
    var numbers = [];
    for (var i = min; i < max; i++) {
        numbers.push(i);
    }
    return new (NumberSet.bind.apply(NumberSet, __spreadArrays([void 0], numbers)))();
}
exports.range = range;
function flatten() {
    var items = [];
    for (var _i = 0; _i < arguments.length; _i++) {
        items[_i] = arguments[_i];
    }
    return [].concat.apply([], items);
}
