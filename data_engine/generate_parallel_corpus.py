"""
Generates a parallel corpus from the EDUB-GT Annotations:
    A language is the image captions.
    The other language is the previous caption of each sentence.
"""

base_path = '/media/HDD_2TB/DATASETS/EDUB-SegDesc/GT/'

txt_files = base_path + 'text.clean.txt'
dest_files = base_path + 'training.'

file = open(txt_files, mode='r')

file_prevs = open(dest_files + 'prev', mode='w')
file_curr = open(dest_files + 'curr', mode='w')

prev_id = 'Segment1'
caps_txt = []
prev_caps = ['None']
i = 0
for line in file:
    id_text = line.split(",")
    id = id_text[0]
    text = ' '.join(id_text[1:]).strip()
    if id == prev_id:
        caps_txt.append(text)
    elif id == 'Segment1':
        prev_id = id
        prev_caps = ['None']
        caps_txt.append(text)
        for curr_cap in caps_txt:
            for prev_cap in prev_caps:
                file_prevs.write(prev_cap + '\n')
                file_curr.write(curr_cap + '\n')
                i += 1
    else:
        caps_txt.append(text)
        for curr_cap in caps_txt:
            for prev_cap in prev_caps:
                file_prevs.write(prev_cap + '\n')
                file_curr.write(curr_cap + '\n')
                i += 1

        prev_id = id
        prev_caps = caps_txt
        caps_txt = []
