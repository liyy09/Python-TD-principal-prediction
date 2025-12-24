# set features
features = ['ncloc', 'lines', 'files', 'classes', 'functions', 'comment_lines',
            'bugs', 'vulnerabilities', 'code_smells', 'uncovered_lines',
            'duplicated_lines', 'duplicated_blocks', 'duplicated_files', 'complexity',
            'LSC', 'LBC', 'LC', 'LTCE', 'CCC', 'LMC', 'MNC', 
            'software_quality_maintainability_remediation_effort']

# set dataset path
project='textual'
path = f'{project}_dataset.csv'

# set prediction period
start = 7
end = start+1