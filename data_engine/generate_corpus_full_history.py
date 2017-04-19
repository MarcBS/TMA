"""
the file id_seg_cap.txt has been generated with the folloing script

awk '{print substr(FILENAME, 1, length(FILENAME)-4) "," $0}' * > ../id_seg_cap.txt

and its format is:
    file_id, segment_number, caption
"""

base_path = '/media/HDD_2TB/DATASETS/EDUB-SegDesc/GT/'

txt_files = base_path + 'id_seg_cap.txt'
dest_files = base_path + 'captions.id.full_history.txt'

file = open(txt_files, mode='r')
dest_file = open(dest_files + 'curr', mode='w')

separator = '----'
space_sym = ' <pad> '

prev_id = 'Segment1'
caps_txt = []
prev_caps = []
j = 0
for line in file:
    id_text = line.split(",")
    user_id = id_text[0]
    segment_id = id_text[1]
    text = ' '.join(id_text[2:]).strip()
    j += 1
    if j % 1000 == 0:
        print "Processed", j, "lines"
    if segment_id == prev_id:
        caps_txt.append(text)

        # for prev_cap in prev_caps:
        #    caps_txt.append(prev_cap + space_sym  + text)
    elif segment_id == 'Segment1':  # Start of day
        prev_id = segment_id
        i = 0
        for curr_cap in caps_txt:
            dest_file.write(user_id + '_' + segment_id + '#' + str(i) + separator + curr_cap + '\n')
            i += 1
        prev_caps = caps_txt
    else:
        # Different segment
        # We combine
        prev_id = segment_id
        # for prev_cap in prev_caps:
        #    prev_caps2.append(prev_cap + space_sym + cap)
        caps_txt = []
        caps_txt.append(text)
        i = 0
        for prev_cap in prev_caps:
            for curr_cap in caps_txt:
                dest_file.write(
                    user_id + '_' + segment_id + '#' + str(i) + separator + prev_cap + space_sym + curr_cap + '\n')
            i += 1
        prev_caps = [prev_cap + space_sym + curr_cap for curr_cap in caps_txt for prev_cap in prev_caps]
