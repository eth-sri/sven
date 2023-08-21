/**
 * @name Check for CWE-798 in codeql-eg-HardcodedCredentials-b
 * @description 
 * @kind problem
 * @problem.severity warning
 * @id custom-cpp/cwe-798-codeql-eg-hardcodedcredentials-b
 */

import python
import semmle.python.dataflow.new.DataFlow
import semmle.python.dataflow.new.TaintTracking

string getACredentialRegex() {
    result = "(?i).*pass(wd|word|code|phrase)(?!.*question).*" 
    //or
    //result = "(?i).*(puid|username|userid).*" or
    //result = "(?i).*(cert)(?!.*(format|name)).*"
  }

from Compare cmp, Expr literal
where 
    literal instanceof StrConst
    and
    cmp.getOp(0) instanceof Eq
    and
    (
      (
        cmp.getComparator(0) =literal
        and
        cmp.getLeft().toString().regexpMatch(getACredentialRegex())
        )
      or
      (
        cmp.getLeft() =literal
        and
        cmp.getComparator(0).toString().regexpMatch(getACredentialRegex())
        )
    )
select cmp, "cwe-798: "+cmp.toString()+ " at line "+ cmp.getLocation().getStartLine()+
" is a password comparison with a string literal"