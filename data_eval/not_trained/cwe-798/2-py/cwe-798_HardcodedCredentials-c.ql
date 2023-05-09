/**
 * @name Check for CWE-798 in codeql-eg-HardcodedCredentials-c
 * @description
 * @kind problem
 * @problem.severity warning
 * @id custom-cpp/cwe-798-codeql-eg-hardcodedcredentials-c
 */

import python

string getACredentialRegex() {
  result = "(?i).*pass(wd|word|code|phrase)(?!.*question).*" or
  result = "(?i).*(puid|username|userid|user).*" or
  result = "(?i).*(cert)(?!.*(format|name)).*"
}

from Keyword k, Expr literal

where

    k.getArg().toString().regexpMatch(getACredentialRegex())
    and
    literal instanceof StrConst
    and
    k.getValue()=literal

select k, k.getArg().toString()
