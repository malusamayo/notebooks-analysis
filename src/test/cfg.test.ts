import { ControlFlowGraph } from "../control-flow";
import { Def, parse } from "../python-parser";

describe("ControlFlowGraph", () => {
  function makeCfg(...codeLines: string[]): ControlFlowGraph {
    let code = codeLines.concat("").join("\n"); // add newlines to end of every line.
    return new ControlFlowGraph(parse(code));
  }

  it("builds the right successor structure for try-except", () => {
    let cfg = makeCfg("try:", "    return 0", "except:", "    return 1");
    let handlerHead = cfg.blocks.filter(b => b.hint == "handlers").pop();
    expect(cfg.getPredecessors(handlerHead).pop().hint).toBe("try body");
  });

  it("builds a cfg for a function body", () => {
    let ast = parse(
      [
        "def foo(n):",
        "    if n < 4:",
        "        return 1",
        "    else:",
        "        return 2"
      ].join("\n")
    );
    expect(ast.code).toHaveLength(1);
    expect(ast.code[0].type).toBe("def");
    const cfg = new ControlFlowGraph(ast.code[0] as Def);
    expect(cfg.blocks).toBeDefined();
    expect(cfg.blocks).toHaveLength(6);
  });
});
