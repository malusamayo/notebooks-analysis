import { Signal } from '@phosphor/signaling';
import { Cell } from './cell';
import { CellSlice } from './cellslice';
import { DataflowAnalyzer } from './data-flow';
import { CellProgram, ProgramBuilder } from './program-builder';
import { LocationSet, slice } from './slice';

/**
 * A record of when a cell was executed.
 */
export class CellExecution {
  readonly cell: Cell;
  readonly executionTime: Date;

  constructor(cell: Cell, executionTime: Date) {
    this.cell = cell;
    this.executionTime = executionTime;
  }

  /**
   * Update this method if at some point we only want to save some about a CellExecution when
   * serializing it and saving history.
   */
  toJSON(): any {
    return JSON.parse(JSON.stringify(this));
  }
}

/**
 * A slice over a version of executed code.
 */
export class SlicedExecution {
  constructor(public executionTime: Date, public cellSlices: CellSlice[]) {}

  merge(...slicedExecutions: SlicedExecution[]): SlicedExecution {
    let cellSlices: { [cellExecutionEventId: string]: CellSlice } = {};
    let mergedCellSlices = [];
    for (let slicedExecution of slicedExecutions.concat(this)) {
      for (let cellSlice of slicedExecution.cellSlices) {
        let cell = cellSlice.cell;
        if (!cellSlices[cell.executionEventId]) {
          let newCellSlice = new CellSlice(
            cell.deepCopy(),
            new LocationSet(),
            cellSlice.executionTime
          );
          cellSlices[cell.executionEventId] = newCellSlice;
          mergedCellSlices.push(newCellSlice);
        }
        let mergedCellSlice = cellSlices[cell.executionEventId];
        mergedCellSlice.slice = mergedCellSlice.slice.union(cellSlice.slice);
      }
    }
    return new SlicedExecution(
      new Date(), // Date doesn't mean anything for the merged slice.
      mergedCellSlices.sort(
        (a, b) => a.cell.executionCount - b.cell.executionCount
      )
    );
  }
}

/**
 * Makes slice on a log of executed cells.
 */
export class ExecutionLogSlicer {
  public _executionLog: CellExecution[] = [];
  public _programBuilder: ProgramBuilder;
  private _dataflowAnalyzer: DataflowAnalyzer;

  /**
   * Signal emitted when a cell's execution has been completely processed.
   */
  readonly executionLogged = new Signal<this, CellExecution>(this);

  /**
   * Construct a new execution log slicer.
   */
  constructor(dataflowAnalyzer: DataflowAnalyzer) {
    this._dataflowAnalyzer = dataflowAnalyzer;
    this._programBuilder = new ProgramBuilder(dataflowAnalyzer);
  }

  /**
   * Log that a cell has just been executed. The execution time for this cell will be stored
   * as the moment at which this method is called.
   */
  public logExecution(cell: Cell) {
    let cellExecution = new CellExecution(cell, new Date());
    this.addExecutionToLog(cellExecution);
  }

  /**
   * Use logExecution instead if a cell has just been run to annotate it with the current time
   * as the execution time. This function is intended to be used only to initialize history
   * when a notebook is reloaded. However, any method that eventually calls this method will
   * notify all observers that this cell has been executed.
   */
  public addExecutionToLog(cellExecution: CellExecution) {
    this._programBuilder.add(cellExecution.cell);
    this._executionLog.push(cellExecution);
    this.executionLogged.emit(cellExecution);
  }

  /**
   * Reset the log, removing log records.
   */
  public reset() {
    this._executionLog = [];
    this._programBuilder.reset();
  }

  /**
   * Get slice for the latest execution of a cell.
   */
  public sliceLatestExecution(
    cell: Cell,
    seedLocations?: LocationSet
  ): SlicedExecution {
    // XXX: This computes more than it has to, performing a slice on each execution of a cell
    // instead of just its latest computation. Optimize later if necessary.
    return this.sliceAllExecutions(cell, seedLocations).pop();
  }

  /**
   * Get slices of the necessary code for all executions of a cell.
   * Relevant line numbers are relative to the cell's start line (starting at first line = 0).
   */
  public sliceAllExecutions(
    cell: Cell,
    pSeedLocations?: LocationSet
  ): SlicedExecution[] {
    // Make a map from cells to their execution times.
    let cellExecutionTimes: { [cellExecutionEventId: string]: Date } = {};
    for (let execution of this._executionLog) {
      cellExecutionTimes[execution.cell.executionEventId] =
        execution.executionTime;
    }

    return this._executionLog
      .filter(execution => execution.cell.persistentId == cell.persistentId)
      .filter(execution => execution.cell.executionCount != undefined)
      .map(execution => {
        // Build the program up to that cell.
        let program = this._programBuilder.buildTo(
          execution.cell.executionEventId
        );
        if (program == null) return null;

        // Set the seed locations for the slice.
        let seedLocations;
        if (pSeedLocations) {
          seedLocations = pSeedLocations;
          // If seed locations weren't specified, slice the whole cell.
          // XXX: Whole cell specified by an unreasonably large character range.
        } else {
          seedLocations = new LocationSet({
            first_line: 1,
            first_column: 1,
            last_line: 10000,
            last_column: 10000,
          });
        }

        // Set seed locations were specified relative to the last cell's position in program.
        let lastCellLines =
          program.cellToLineMap[execution.cell.executionEventId];
        let lastCellStart = Math.min(...lastCellLines.items);
        seedLocations = new LocationSet(
          ...seedLocations.items.map(loc => {
            return {
              first_line: lastCellStart + loc.first_line - 1,
              first_column: loc.first_column,
              last_line: lastCellStart + loc.last_line - 1,
              last_column: loc.last_column,
            };
          })
        );

        // Slice the program
        let sliceLocations = slice(
          program.tree,
          seedLocations,
          this._dataflowAnalyzer
        ).items.sort((loc1, loc2) => loc1.first_line - loc2.first_line);

        // Get the relative offsets of slice lines in each cell.
        let cellSliceLocations: {
          [executionEventId: string]: LocationSet;
        } = {};
        let cellOrder: Cell[] = [];
        sliceLocations.forEach(location => {
          let sliceCell = program.lineToCellMap[location.first_line];
          let sliceCellLines =
            program.cellToLineMap[sliceCell.executionEventId];
          let sliceCellStart = Math.min(...sliceCellLines.items);
          if (cellOrder.indexOf(sliceCell) == -1) {
            cellOrder.push(sliceCell);
          }
          let adjustedLocation = {
            first_line: location.first_line - sliceCellStart + 1,
            first_column: location.first_column,
            last_line: location.last_line - sliceCellStart + 1,
            last_column: location.last_column,
          };
          if (!cellSliceLocations[sliceCell.executionEventId]) {
            cellSliceLocations[sliceCell.executionEventId] = new LocationSet();
          }
          cellSliceLocations[sliceCell.executionEventId].add(adjustedLocation);
        });

        let cellSlices = cellOrder.map(
          (sliceCell): CellSlice => {
            let executionTime = undefined;
            if (cellExecutionTimes[sliceCell.executionEventId]) {
              executionTime = cellExecutionTimes[sliceCell.executionEventId];
            }
            return new CellSlice(
              sliceCell,
              cellSliceLocations[sliceCell.executionEventId],
              executionTime
            );
          }
        );
        return new SlicedExecution(execution.executionTime, cellSlices);
      })
      .filter(s => s != null && s != undefined);
  }

  get cellExecutions(): ReadonlyArray<CellExecution> {
    return this._executionLog;
  }

  /**
   * Get the cell program (tree, defs, uses) for a cell.
   */
  getCellProgram(cell: Cell): CellProgram {
    return this._programBuilder.getCellProgram(cell);
  }
}
