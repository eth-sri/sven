/**
 * @name Check for CWE-119 in this example
 * @description 
 * @kind problem
 * @problem.severity warning
 * @id custom-cpp/cwe-119-mitre-eg-3
 */

import cpp
import semmle.code.cpp.controlflow.Guards
import semmle.code.cpp.rangeanalysis.SimpleRangeAnalysis
import semmle.code.cpp.controlflow.SSA

from ArrayExpr ae

where

    //the index to array access is out of bounds (range analysis)
    ( lowerBound(ae.getArrayOffset()) <0 
      or
      upperBound(ae.getArrayOffset()) >3 )

    // the index to array access is not protected by any guard condition
    and not (
        //the array expression is part of a while or for loop
        exists( ForStmt for_stmt, WhileStmt while_stmt |
            ae.getEnclosingElement*() = for_stmt
            or ae.getEnclosingElement*()= while_stmt
        )

        //there is a guard condition that has the index of array expression
        and exists (GuardCondition gc|
            ae.getArrayOffset().toString().matches( gc.getAChild().toString() )  
            //and gc.getEnclosingElement() = rel_op.getEnclosingElement()
        )

    )

select ae, "cwe-119 found in "+ ae.getFile().getShortName()+ " at line" + ae.getLocation().getEndLine()