import * as ast from './python-parser';
import { Block, ControlFlowGraph } from './control-flow';
import { Set, StringSet } from './set';
import { SliceConfiguration } from './slice-config';

/**
 * Use a shared dataflow analyzer object for all dataflow analysis / querying for defs and uses.
 * It caches defs and uses for each statement, which can save time.
 * For caching to work, statements must be annotated with a cell's ID and execution count.
 */
export class DataflowAnalyzer {
	constructor(sliceConfiguration?: SliceConfiguration) {
		this._sliceConfiguration = sliceConfiguration || [];
	}

	private _statementLocationKey(statement: ast.SyntaxNode) {
		return (
			statement.location.first_line +
			',' +
			statement.location.first_column +
			',' +
			statement.location.last_line +
			',' +
			statement.location.last_column
		);
	}

	getDefsUses(
		statement: ast.SyntaxNode,
		symbolTable?: SymbolTable
	): IDefUseInfo {
		symbolTable = symbolTable || { moduleNames: new StringSet() };
		let cacheKey = this._statementLocationKey(statement);
		if (cacheKey != null) {
			if (this._defUsesCache.hasOwnProperty(cacheKey)) {
				return this._defUsesCache[cacheKey];
			}
		}
		let defSet = this.getDefs(statement, symbolTable);
		let useSet = this.getUses(statement, symbolTable);
		let result = { defs: defSet, uses: useSet };
		if (cacheKey != null) {
			this._defUsesCache[cacheKey] = result;
		}
		return result;
	}

	analyze(
		cfg: ControlFlowGraph,
		sliceConfiguration?: SliceConfiguration,
		namesDefined?: StringSet
	): DataflowAnalysisResult {
		sliceConfiguration = sliceConfiguration || [];
		let symbolTable: SymbolTable = { moduleNames: new StringSet() };
		const workQueue: Block[] = cfg.blocks.reverse();
		let undefinedRefs = new RefSet();

		let defsForLevelByBlock: {
			[level: string]: { [blockId: number]: RefSet };
		} = {};
		for (let level of Object.keys(ReferenceType)) {
			defsForLevelByBlock[level] = {};
			for (let block of workQueue) {
				defsForLevelByBlock[level][block.id] = new RefSet();
			}
		}

		let dataflows = new Set<Dataflow>(getDataflowId);

		while (workQueue.length) {
			const block = workQueue.pop();

			let oldDefsForLevel: { [level: string]: RefSet } = {};
			let defsForLevel: { [level: string]: RefSet } = {};
			for (let level of Object.keys(ReferenceType)) {
				oldDefsForLevel[level] = defsForLevelByBlock[level][block.id];
				// incoming definitions are come from predecessor blocks
				defsForLevel[level] = oldDefsForLevel[level].union(
					...cfg
						.getPredecessors(block)
						.map(block => defsForLevelByBlock[level][block.id])
						.filter(s => s != undefined)
				);
			}

			// TODO: fix up dataflow computation within this block: check for definitions in
			// defsWithinBlock first; if found, don't look to defs that come from the predecessor.
			for (let statement of block.statements) {
				// Note that defs includes both definitions and mutations and variables
				let { defs: definedHere, uses: usedHere } = this.getDefsUses(
					statement,
					symbolTable
				);

				// Sort definitions and uses into references.
				let statementRefs: { [level: string]: RefSet } = {};
				for (let level of Object.keys(ReferenceType)) {
					statementRefs[level] = new RefSet();
				}
				for (let def of definedHere.items) {
					statementRefs[def.level].add(def);
					if (TYPES_WITH_DEPENDENCIES.indexOf(def.level) != -1) {
						undefinedRefs.add(def);
					}
				}
				for (let use of usedHere.items) {
					statementRefs[ReferenceType.USE].add(use);
					undefinedRefs.add(use);
				}

				// Get all new dataflow dependencies.
				let newFlows = new Set<Dataflow>(getDataflowId);
				for (let level of Object.keys(ReferenceType)) {
					// For everything that's defined coming into this block, if it's used in this block, save connection.
					let result = createFlowsFrom(
						statementRefs[level],
						defsForLevel[level],
						statement
					);
					let flowsCreated = result[0].items;
					let defined = result[1];
					newFlows.add(...flowsCreated);
					for (let ref of defined.items) {
						undefinedRefs.remove(ref);
					}
				}
				dataflows = dataflows.union(newFlows);

				for (let level of Object.keys(ReferenceType)) {
					// 🙄 it doesn't really make sense to update the "use" set for a block but whatever
					defsForLevel[level] = updateDefsForLevel(
						defsForLevel[level],
						level,
						statementRefs,
						DEPENDENCY_RULES
					);
				}
			}

			// Check to see if definitions have changed. If so, redo the successor blocks.
			for (let level of Object.keys(ReferenceType)) {
				if (!oldDefsForLevel[level].equals(defsForLevel[level])) {
					defsForLevelByBlock[level][block.id] = defsForLevel[level];
					for (let succ of cfg.getSuccessors(block)) {
						if (workQueue.indexOf(succ) < 0) {
							workQueue.push(succ);
						}
					}
				}
			}
		}

		// Check to see if any of the undefined names were defined coming into the graph. If so,
		// don't report them as being undefined.
		if (namesDefined) {
			for (let ref of undefinedRefs.items) {
				if (namesDefined.items.some(n => n == ref.name)) {
					undefinedRefs.remove(ref);
				}
			}
		}

		return {
			flows: dataflows,
			undefinedRefs: undefinedRefs,
		};
	}

	getDefs(statement: ast.SyntaxNode, symbolTable: SymbolTable): RefSet {
		let defs = new RefSet();
		if (!statement) return defs;

		/*
		 * Assume by default that all names passed in as arguments to a function
		 * an all objects that a function is called on are modified by that function call,
		 * unless otherwise specified in the slice configuration.
		 */
		let callNamesListener = new CallNamesListener(
			this._sliceConfiguration,
			statement
		);
		ast.walk(statement, callNamesListener);
		defs.add(...callNamesListener.defs.items);

		let defAnnotationsListener = new DefAnnotationListener(statement);
		ast.walk(statement, defAnnotationsListener);
		defs = defs.union(defAnnotationsListener.defs);

		switch (statement.type) {
			case ast.IMPORT: {
				const modnames = statement.names.map(i => i.name || i.path);
				symbolTable.moduleNames.add(...modnames);
				defs.add(
					...statement.names.map(nameNode => {
						return {
							type: SymbolType.IMPORT,
							level: ReferenceType.DEFINITION,
							name: nameNode.name || nameNode.path,
							location: nameNode.location,
							statement: statement,
						};
					})
				);
				break;
			}
			case ast.FROM: {
				/*
				 * TODO(andrewhead): Discover definitions of symbols from wildcards, like {@code from <pkg> import *}.
				 */
				let modnames: Array<string> = [];
				if (statement.imports.constructor === Array) {
					modnames = statement.imports.map(i => i.name || i.path);
					symbolTable.moduleNames.add(...modnames);
					defs.add(
						...statement.imports.map(i => {
							return {
								type: SymbolType.IMPORT,
								level: ReferenceType.DEFINITION,
								name: i.name || i.path,
								location: i.location,
								statement: statement,
							};
						})
					);
				}
				break;
			}
			case ast.ASSIGN: {
				let targetsDefListener = new TargetsDefListener(statement);
				if (statement.targets) {
					for (let target of statement.targets) {
						ast.walk(target, targetsDefListener);
					}
				}
				/*
				 * If `statement.op` is defined, then it is an augassign (e.g., +=). Any variables defined
				 * on the left-hand side are actually getting updated.
				 */
				if (statement.op) {
					for (let targetDef of targetsDefListener.defs.items) {
						if (targetDef.level === ReferenceType.DEFINITION) {
							targetDef.level = ReferenceType.UPDATE;
						}
					}
				}
				defs = defs.union(targetsDefListener.defs);
				break;
			}
			case ast.DEF: {
				defs.add({
					type: SymbolType.FUNCTION,
					level: ReferenceType.DEFINITION,
					name: statement.name,
					location: statement.location,
					statement: statement,
				});
				break;
			}
			case ast.CLASS: {
				defs.add({
					type: SymbolType.CLASS,
					level: ReferenceType.DEFINITION,
					name: statement.name,
					location: statement.location,
					statement: statement,
				});
			}
		}
		return defs;
	}

	getUses(statement: ast.SyntaxNode, _: SymbolTable): RefSet {
		let uses = new RefSet();

		switch (statement.type) {
			// TODO: should we collect when importing with FROM from something else that was already imported...
			case ast.ASSIGN: {
				// XXX: Is this supposed to union with funcArgs?
				const targetNames = gatherNames(statement.targets);
				const targets = new RefSet(
					...targetNames.items.map(([name, node]) => {
						return {
							type: SymbolType.VARIABLE,
							level: ReferenceType.USE,
							name: name,
							location: node.location,
							statement: statement,
						};
					})
				);
				const sourceNames = gatherNames(statement.sources);
				const sources = new RefSet(
					...sourceNames.items.map(([name, node]) => {
						return {
							type: SymbolType.VARIABLE,
							level: ReferenceType.USE,
							name: name,
							location: node.location,
							statement: statement,
						};
					})
				);
				uses = uses.union(sources).union(statement.op ? targets : new RefSet());
				break;
			}
			case ast.DEF:
				let defCfg = new ControlFlowGraph(statement);
				let argNames = new StringSet(
					...statement.params
						.map(p => p.name)
						.filter(n => n != undefined)
				);
				let undefinedRefs = this.analyze(
					defCfg,
					this._sliceConfiguration,
					argNames
				).undefinedRefs;
				uses = undefinedRefs.filter(r => r.level == ReferenceType.USE);
				break;
			case ast.CLASS:
				break;
			default: {
				const usedNames = gatherNames(statement);
				uses = new RefSet(
					...usedNames.items.map(([name, node]) => {
						return {
							type: SymbolType.VARIABLE,
							level: ReferenceType.USE,
							name: name,
							location: node.location,
							statement: statement,
						};
					})
				);
				break;
			}
		}

		return uses;
	}

	private _sliceConfiguration: SliceConfiguration;
	private _defUsesCache: { [statementLocation: string]: IDefUseInfo } = {};
}

export interface Dataflow {
	fromNode: ast.SyntaxNode;
	toNode: ast.SyntaxNode;
}

export enum ReferenceType {
	DEFINITION = 'DEFINITION',
	UPDATE = 'UPDATE',
	USE = 'USE',
}

export enum SymbolType {
	VARIABLE,
	CLASS,
	FUNCTION,
	IMPORT,
	MUTATION,
	MAGIC,
}

export interface Ref {
	type: SymbolType;
	level: ReferenceType;
	name: string;
	location: ast.Location;
	statement: ast.SyntaxNode;
}

export class RefSet extends Set<Ref> {
	constructor(...items: Ref[]) {
		super(r => r.name + r.level + locString(r.location), ...items);
	}
}

function locString(loc: ast.Location): string {
	return `${loc.first_line}:${loc.first_column}-${loc.last_line}:${loc.last_column}`;
}

export function sameLocation(
	loc1: ast.Location,
	loc2: ast.Location
): boolean {
	return (
		loc1.first_column === loc2.first_column &&
		loc1.first_line === loc2.first_line &&
		loc1.last_column === loc2.last_column &&
		loc1.last_line === loc2.last_line
	);
}

function getNameSetId([name, node]: [string, ast.SyntaxNode]) {
	if (!node.location) console.log('***', node);
	return `${name}@${locString(node.location)}`;
}

class NameSet extends Set<[string, ast.SyntaxNode]> {
	constructor(...items: [string, ast.SyntaxNode][]) {
		super(getNameSetId, ...items);
	}
}

function gatherNames(node: ast.SyntaxNode | ast.SyntaxNode[]): NameSet {
	if (Array.isArray(node)) {
		return new NameSet().union(...node.map(gatherNames));
	} else {
		return new NameSet(
			...ast
				.walk(node)
				.filter(e => e.type == ast.NAME)
				.map((e: ast.Name): [string, ast.SyntaxNode] => [e.id, e])
		);
	}
}

interface IDefUseInfo {
	defs: RefSet;
	uses: RefSet;
}

interface SymbolTable {
	// ⚠️ We should be doing full-blown symbol resolution, but meh 🙄
	moduleNames: StringSet;
}

/**
 * Tree walk listener for collecting manual def annotations.
 */
class DefAnnotationListener implements ast.WalkListener {
	constructor(statement: ast.SyntaxNode) {
		this._statement = statement;
	}

	onEnterNode(node: ast.SyntaxNode, type: string) {
		if (type == ast.LITERAL) {
			let literal = node as ast.Literal;

			// If this is a string, try to parse a def annotation from it
			if (typeof literal.value == 'string' || literal.value instanceof String) {
				let string = literal.value;
				let jsonMatch = string.match(/"defs: (.*)"/);
				if (jsonMatch && jsonMatch.length >= 2) {
					let jsonString = jsonMatch[1];
					let jsonStringUnescaped = jsonString.replace(/\\"/g, '"');
					try {
						let defSpecs = JSON.parse(jsonStringUnescaped);
						for (let defSpec of defSpecs) {
							this.defs.add({
								type: SymbolType.MAGIC,
								level: ReferenceType.DEFINITION,
								name: defSpec.name,
								location: {
									first_line: defSpec.pos[0][0] + node.location.first_line,
									first_column: defSpec.pos[0][1],
									last_line: defSpec.pos[1][0] + node.location.first_line,
									last_column: defSpec.pos[1][1],
								},
								statement: this._statement,
							});
						}
					} catch (e) { }
				}
			}
		}
	}

	private _statement: ast.SyntaxNode;
	readonly defs: RefSet = new RefSet();
}

/**
 * Tree walk listener for collecting names used in function call.
 */
class CallNamesListener implements ast.WalkListener {
	constructor(
		sliceConfiguration: SliceConfiguration,
		statement: ast.SyntaxNode
	) {
		this._sliceConfiguration = sliceConfiguration;
		this._statement = statement;
	}

	onEnterNode(
		node: ast.SyntaxNode,
		type: string,
		ancestors: ast.SyntaxNode[]
	) {
		if (type == ast.CALL) {
			let callNode = node as ast.Call;
			let functionNameNode: ast.SyntaxNode;
			let functionName: string;
			if (callNode.func.type == ast.DOT) {
				functionNameNode = callNode.func.name;
				functionName = functionNameNode.toString();
			} else {
				functionNameNode = callNode.func as ast.Name;
				functionName = functionNameNode.id;
			}

			let skipRules = this._sliceConfiguration
				.filter(config => config.functionName == functionName)
				.filter(config => {
					if (!config.objectName) return true;
					if (
						callNode.func.type == ast.DOT &&
						callNode.func.value.type == ast.NAME
					) {
						let instanceName = (callNode.func.value as ast.Name).id;
						return config.objectName == instanceName;
					}
					return false;
				});

			if (callNode.func.type == ast.DOT) {
				let skipObject = false;
				for (let skipRule of skipRules) {
					if (skipRule.doesNotModify.indexOf('OBJECT') !== -1) {
						skipObject = true;
						break;
					}
				}
				if (!skipObject && callNode.func.value !== undefined) {
					this._subtreesToProcess.push(callNode.func.value);
				}
			}

			for (let i = 0; i < callNode.args.length; i++) {
				let arg = callNode.args[i];
				let skipArg = false;
				for (let skipRule of skipRules) {
					for (let skipSpec of skipRule.doesNotModify) {
						if (typeof skipSpec === 'number' && skipSpec === i) {
							skipArg = true;
							break;
						} else if (typeof skipSpec === 'string') {
							if (
								skipSpec === 'ARGUMENTS' ||
								(arg.keyword && (arg.keyword as ast.Name).id === skipSpec)
							) {
								skipArg = true;
								break;
							}
						}
					}
					if (skipArg) break;
				}
				if (!skipArg) {
					this._subtreesToProcess.push(arg.actual);
				}
			}
		}

		if (type == ast.NAME) {
			for (let ancestor of ancestors) {
				if (this._subtreesToProcess.indexOf(ancestor) !== -1) {
					this.defs.add({
						type: SymbolType.MUTATION,
						level: ReferenceType.UPDATE,
						name: (node as ast.Name).id,
						location: node.location,
						statement: this._statement,
					});
					break;
				}
			}
		}
	}

	private _sliceConfiguration: SliceConfiguration;
	private _statement: ast.SyntaxNode;
	private _subtreesToProcess: ast.SyntaxNode[] = [];
	readonly defs: RefSet = new RefSet();
}

/**
 * Tree walk listener for collecting definitions in the target of an assignment.
 */
class TargetsDefListener implements ast.WalkListener {
	constructor(statement: ast.SyntaxNode) {
		this._statement = statement;
	}

	onEnterNode(
		node: ast.SyntaxNode,
		type: string,
		ancestors: ast.SyntaxNode[]
	) {
		if (type == ast.NAME) {
			let level = ReferenceType.DEFINITION;
			if (ancestors.some(a => a.type == ast.DOT)) {
				level = ReferenceType.UPDATE;
			} else if (ancestors.some(a => a.type == ast.INDEX)) {
				level = ReferenceType.UPDATE;
			}
			this.defs.add({
				type: SymbolType.VARIABLE,
				level: level,
				location: node.location,
				name: (node as ast.Name).id,
				statement: this._statement,
			});
		}
	}

	private _statement: ast.SyntaxNode;
	readonly defs: RefSet = new RefSet();
}

function getDataflowId(df: Dataflow) {
	if (!df.fromNode.location)
		console.log('*** FROM', df.fromNode, df.fromNode.location);
	if (!df.toNode.location) console.log('*** TO', df.toNode, df.toNode.location);
	return `${locString(df.fromNode.location)}->${locString(df.toNode.location)}`;
}

function createFlowsFrom(
	fromSet: RefSet,
	toSet: RefSet,
	fromStatement: ast.SyntaxNode
): [Set<Dataflow>, Set<Ref>] {
	let refsDefined = new RefSet();
	let newFlows = new Set<Dataflow>(getDataflowId);
	for (let from of fromSet.items) {
		for (let to of toSet.items) {
			if (to.name == from.name) {
				refsDefined.add(from);
				newFlows.add({ fromNode: to.statement, toNode: fromStatement });
			}
		}
	}
	return [newFlows, refsDefined];
}

let DEPENDENCY_RULES = [
	// "from" depends on all reference types in "to"
	{
		from: ReferenceType.USE,
		to: [ReferenceType.UPDATE, ReferenceType.DEFINITION],
	},
	{
		from: ReferenceType.UPDATE,
		to: [ReferenceType.DEFINITION],
	},
];

let TYPES_WITH_DEPENDENCIES = DEPENDENCY_RULES.map(r => r.from);

let KILL_RULES = [
	// Which types of references "kill" which other types of references?
	// In general, the rule of thumb here is, if x depends on y, x kills y, because anything that
	// depends on x will now depend on y transitively.
	// If x overwrites y, x also kills y.
	// The one case where a variable doesn't kill a previous variable is the global configuration, because
	// it neither depends on initializations or updates, nor clobbers them.
	{
		level: ReferenceType.DEFINITION,
		kills: [ReferenceType.DEFINITION, ReferenceType.UPDATE],
	},
	{
		level: ReferenceType.UPDATE,
		kills: [ReferenceType.DEFINITION, ReferenceType.UPDATE],
	},
];

function updateDefsForLevel(
	defsForLevel: RefSet,
	level: string,
	newRefs: { [level: string]: RefSet },
	dependencyRules: { from: ReferenceType; to: ReferenceType[] }[]
) {
	let genSet = new RefSet();
	let levelDependencies = dependencyRules.filter(r => r.from == level).pop();
	for (let level of Object.keys(ReferenceType)) {
		newRefs[level].items.forEach(ref => {
			if (levelDependencies && levelDependencies.to.indexOf(ref.level) != -1) {
				genSet.add(ref);
			}
		});
	}
	const killSet = defsForLevel.filter(def => {
		let found = false;
		genSet.items.forEach(gen => {
			if (gen.name == def.name) {
				let killRules = KILL_RULES.filter(r => r.level == gen.level).pop();
				if (killRules && killRules.kills.indexOf(def.level) != -1) {
					found = true;
				}
			}
		});
		return found;
	});
	return defsForLevel.minus(killSet).union(genSet);
}

export type DataflowAnalysisResult = {
	flows: Set<Dataflow>;
	undefinedRefs: RefSet;
};
