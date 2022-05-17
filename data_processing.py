import glob
import os
import pandas as pd

sections = {
    'Chief Complaint:\n': 'cc',
    'CHIEF COMPLAINT:': 'cc',
    'Major Surgical or Invasive Procedure:\n': 'procedure',
    'MAJOR PROCEDURES:': 'procedure',
    'History of Present Illness:\n': 'hpi',
    'HISTORY OF PRESENT ILLNESS:': 'hpi',
    'Past Medical History:\n': 'mh',
    'PAST MEDICAL HISTORY:': 'mh',
    'Brief Hospital Course:\n': 'course',
    'HOSPITAL COURSE:': 'course',
    'Discharge Disposition:\n': 'discharge_disp',
    'DISPOSITION:': 'discharge_disp',
    'Discharge Diagnosis:\n': 'discharge_diag',
    'DISCHARGE DIAGNOSIS:': 'discharge_diag',
}


def get_notes_from_dir(directory):
    notes = glob.glob(directory+'*.txt')
    return notes


def extract_note_sections(file, identifier):
    textout = {}
    sections_tup = tuple(sections.keys())
    current_section = None
    with open(file) as f:
        for line in f:
            if line.startswith(sections_tup):
                print(line)
                current_section = [v for i, v in sections.items() if line.startswith(i)][0]
                textout[current_section] = line
            elif line in '\n':
                current_section = None
            elif current_section:
                textout[current_section] = textout[current_section] + line.replace('\n', ' ')

    textout['id'] = identifier
    return textout


if __name__ == '__main__':
    notes = get_notes_from_dir('./data/training_20180910/')
    alldocs = []
    for note in notes:
        alldocs.append(extract_note_sections(note, os.path.split(note)[1].split('.')[0]))

    # x = extract_note_sections('./data/training_20180910/100883.txt', '100883')

    docs = pd.DataFrame(alldocs)
    docs.to_csv('./data/procnotes.csv', index=False)
