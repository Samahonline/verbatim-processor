# config.py

# Keywords to identify verbatim columns
VERBATIM_KEYWORDS = [
    'verbatim', 'text', 'comment', 'response', 'answer',
    'open', 'openend', 'qualitative', 'feedback', 'remark',
    'opinion', 'suggestion', 'describe', 'explain', 'why',
    'other', 'specify', 'additional'
]

# Patterns for ID/intnr columns
ID_COLUMN_PATTERNS = [
    'intnr', 'interview', 'respondent', 'id', 'caseid', 'resp_id',
    'respondent_id', 'case_id', 'participant'
]

# Excel output settings
MAX_SHEET_NAME_LENGTH = 31
DEFAULT_COLUMN_WIDTHS = {
    'id_column': 15,
    'verbatim_column': 50
}