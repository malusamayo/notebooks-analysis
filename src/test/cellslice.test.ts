import { CellSlice } from "../cellslice";
import { LocationSet } from "../slice";
import { TestCell } from "./testcell";

describe("CellSlice", () => {
  it("yields a text slice based on a set of locations", () => {
    let cellSlice = new CellSlice(
      new TestCell(["a = 1", "b = 2", "c = 3", "d = 4", ""].join("\n"), 1),
      new LocationSet(
        { first_line: 1, first_column: 0, last_line: 1, last_column: 5 },
        { first_line: 2, first_column: 4, last_line: 3, last_column: 4 }
      )
    );
    expect(cellSlice.textSlice).toBe(["a = 1", "2", "c = "].join("\n"));
  });

  it("yields entire lines if requested", () => {
    let cellSlice = new CellSlice(
      new TestCell(["a = 1", "b = 2", "c = 3", "d = 4", ""].join("\n"), 1),
      new LocationSet(
        { first_line: 1, first_column: 0, last_line: 1, last_column: 5 },
        { first_line: 2, first_column: 4, last_line: 3, last_column: 4 }
      )
    );
    expect(cellSlice.textSliceLines).toBe(
      ["a = 1", "b = 2", "c = 3"].join("\n")
    );
  });
});
