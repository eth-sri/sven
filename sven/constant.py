SEC_LABEL = 'sec'
VUL_LABEL = 'vul'
BINARY_LABELS = [SEC_LABEL, VUL_LABEL]

ALL_VUL_TYPES = ['cwe-089', 'cwe-125', 'cwe-078', 'cwe-476', 'cwe-416', 'cwe-022', 'cwe-787', 'cwe-079', 'cwe-190']
PY_VUL_TYPES = ['cwe-022', 'cwe-078', 'cwe-079', 'cwe-089']
CPP_VUL_TYPES = ['cwe-125', 'cwe-190', 'cwe-416', 'cwe-476', 'cwe-787']
DOP_VUL_TYPES = ['cwe-089']
NOTTRAINED_VUL_TYPES = ['cwe-119', 'cwe-502', 'cwe-732', 'cwe-798']
COMPOSITE_VUL_TYPES = ['cwe-022-079', 'cwe-078-089']

VAL_SCENARIOS = {
    ('cwe-078', '2-py'),
    ('cwe-089', '2-py'),
    ('cwe-125', '2-c'),
    ('cwe-190', '2-c'),
    ('cwe-022', '2-py'),
    ('cwe-787', '2-c'),
}

TEST_SCENARIOS = [
    ['cwe-089', '0-py', '1-py'],
    ['cwe-125', '0-c', '1-c'],
    ['cwe-078', '0-py', '1-py'],
    ['cwe-476', '0-c', '2-c'],
    ['cwe-416', '0-c'],
    ['cwe-022', '0-py', '1-py'],
    ['cwe-787', '0-c', '1-c'],
    ['cwe-079', '0-py', '1-py'],
    ['cwe-190', '0-c', '1-c'],
    ['cwe-416', '1-c'],
]

DOP_SCENARIOS = [
    ('cwe-089', 'con'),
    ('cwe-089', 'm-1'),
    ('cwe-089', 'm-2'),
    ('cwe-089', 'm-3'),
    ('cwe-089', 'm-4'),
    ('cwe-089', 'd-1'),
    ('cwe-089', 'd-2'),
    ('cwe-089', 'd-3'),
    ('cwe-089', 'd-4'),
    ('cwe-089', 'd-5'),
    ('cwe-089', 'd-6'),
    ('cwe-089', 'd-7'),
    ('cwe-089', 'c-1'),
    ('cwe-089', 'c-2'),
    ('cwe-089', 'c-3'),
    ('cwe-089', 'c-4'),
    ('cwe-089', 'c-5'),
]

NOTTRAINED_SCENARIOS = [
    ['cwe-119', '0-c', '1-c', '2-c'],
    ['cwe-502', '0-py', '1-py', '2-py'],
    ['cwe-732', '0-c', '1-c', '2-py'],
    ['cwe-798', '0-py', '1-py', '2-py'],
]

PROMPTS = [
    'The following code is secure\n',
    'The following code is vulnerable\n',
]

MODEL_DIRS = {
    '350m': 'Salesforce/codegen-350M-multi',
    '2b': 'Salesforce/codegen-2B-multi',
    '6b': 'Salesforce/codegen-6B-multi',
}