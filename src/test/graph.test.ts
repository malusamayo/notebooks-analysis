import { Graph } from "../graph";

describe("graph", () => {
  it("handles empty", () => {
    const g = new Graph<string>(s => s);
    expect(g.nodes).toBeDefined();
    expect(g.nodes).toHaveLength(0);
  });

  it("tracks nodes", () => {
    const g = new Graph<string>(s => s);
    g.addEdge("a", "b");
    g.addEdge("b", "c");
    const ns = g.nodes;
    expect(ns).toBeDefined();
    expect(ns).toHaveLength(3);
    expect(ns).toContain("a");
    expect(ns).toContain("b");
    expect(ns).toContain("c");
  });

  it("sorts forests", () => {
    const g = new Graph<string>(s => s);
    g.addEdge("a", "b");
    g.addEdge("c", "d");
    const s = g.topoSort();
    expect(s).toBeDefined();
    expect(s).toHaveLength(4);
    expect(s).toContain("a");
    expect(s).toContain("b");
    expect(s).toContain("c");
    expect(s).toContain("d");
    // can't say exact order
    expect(s.indexOf("a")).toBeLessThan(s.indexOf("b"));
    expect(s.indexOf("c")).toBeLessThan(s.indexOf("d"));
  });

  it("sorts dags", () => {
    const g = new Graph<string>(s => s);
    g.addEdge("a", "b");
    g.addEdge("b", "c");
    g.addEdge("a", "c");
    const s = g.topoSort();
    expect(s).toBeDefined();
    expect(s).toHaveLength(3);
    // must be in this order
    expect(s[0]).toBe("a");
    expect(s[1]).toBe("b");
    expect(s[2]).toBe("c");
  });
});
