SEC_LABEL = 'sec'
VUL_LABEL = 'vul'
BINARY_LABELS = [SEC_LABEL, VUL_LABEL]

CWES_TRAINED = [
    'cwe-089',
    'cwe-125',
    'cwe-078',
    'cwe-476',
    'cwe-416',
    'cwe-022',
    'cwe-787',
    'cwe-079',
    'cwe-190',
]

CWES_TRAINED_SUBSET = [
    'cwe-089',
    'cwe-125',
    'cwe-078',
    'cwe-022',
    'cwe-787',
    'cwe-079'
]

CWES_PROMPTS = ['cwe-089']

CWES_GEN_1 = [
    'cwe-119',
    'cwe-502',
    'cwe-732',
    'cwe-798'
]

CWES_GEN_2 = [
    'cwe-020',
    'cwe-094',
    'cwe-116',
    'cwe-117',
    'cwe-209',
    'cwe-215',
    'cwe-312',
    'cwe-327',
    'cwe-377',
    'cwe-611',
    'cwe-643',
    'cwe-777',
    'cwe-918',
]

CWES_DICT = {
    'trained': CWES_TRAINED,
    'trained_subset': CWES_TRAINED_SUBSET,
    'prompts': CWES_PROMPTS,
    'gen_1': CWES_GEN_1,
    'gen_2': CWES_GEN_2,
}

VAL_SCENARIOS = {
    ('cwe-078', '2-py'),
    ('cwe-089', '2-py'),
    ('cwe-125', '2-c'),
    ('cwe-190', '2-c'),
    ('cwe-022', '2-py'),
    ('cwe-787', '2-c'),
}

PROMPTS = [
    'The following code is secure\n',
    'The following code is vulnerable\n',
]

MODEL_DIRS = {
    '350m': 'Salesforce/codegen-350M-multi',
    '2b': 'Salesforce/codegen-2B-multi',
    '6b': 'Salesforce/codegen-6B-multi',
    'incoder': 'facebook/incoder-6B',
    'santa': 'bigcode/santacoder',
}