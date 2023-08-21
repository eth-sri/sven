wget https://github.com/github/codeql-cli-binaries/releases/download/v2.11.1/codeql-linux64.zip
unzip codeql-linux64.zip
git clone --depth=1 --branch codeql-cli-2.11.1 https://github.com/github/codeql.git codeql/codeql-repo
codeql/codeql pack download codeql-cpp codeql-python@0.6.2 codeql/ssa codeql/tutorial codeql/regex codeql/util
cp data_eval/trained/cwe-190/1-c/ArithmeticTainted.ql codeql/codeql-repo/cpp/ql/src/Security/CWE/CWE-190/ArithmeticTainted.ql