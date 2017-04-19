import glob
import os

import xlrd

# Split the existent data in train, val and test
data_path = '/media/HDD_3TB/DATASETS/EDUB-SegDesc'

# input data paths
in_descriptions_path = 'GT/descriptions'
in_segments_path = 'GT/segmentations'
in_images_path = 'Images'  # <in_images_path>/<day_name>/<img_name>.jpg
imgs_format = '.jpg'

# output data paths
out_features_path = 'Features'  # <set_split>_<out_features_name>_all_frames.csv & <set_split>_<out_features_name>_all_frames_counts.txt
out_descriptions_path = 'Annotations'
out_image_lists_path = 'Annotations'  # <set_split>_imgs_list.txt & <set_split>_imgs_counts.txt

# Get day_sets for each data split
sets = dict()
for s in ['train', 'val', 'test']:
    sets[s] = []
    with open(data_path + '/' + out_descriptions_path + '/' + s + '_list_final.txt', 'r') as list_file:
        prev_set = -1
        for line in list_file:
            line = line.rstrip('\n')
            line = line.split('_')
            if line[0] != prev_set:
                sets[s].append(line[0])
            prev_set = line[0]

# Get segments' IDs with errors
errors = dict()
for s in ['train', 'val', 'test']:
    errors[s] = dict()
    for day_split in sets[s]:
        errors[s][day_split] = []
        with open(data_path + '/' + in_descriptions_path + '/' + day_split + '.txt', 'r') as list_file:
            for line in list_file:
                line = line.rstrip('\n').split(',')
                segm_id = int(line[0][7:])
                desc = ','.join(line[1:])
                desc = desc.strip().lower()
                if desc == 'error':
                    errors[s][day_split].append(segm_id)

# Get events of correct segments
for s in ['train', 'val', 'test']:

    file_imgs = open(data_path + '/' + out_image_lists_path + '/' + s + '_imgs_list.txt', 'w')
    file_counts = open(data_path + '/' + out_image_lists_path + '/' + s + '_imgs_counts.txt', 'w')

    for day_split in sets[s]:
        possible_names = ['/GT_' + day_split + '.xls', '/GT_' + day_split + '.xlsx', '/' + day_split + '.xls',
                          '/' + day_split + '.xlsx']
        exists = False
        i = 0
        while not os.path.isfile(data_path + '/' + in_segments_path + possible_names[i]):
            i += 1
        file = xlrd.open_workbook(data_path + '/' + in_segments_path + possible_names[i])
        sheet = file.sheet_by_index(0)

        count_segments = 1
        these_events = []
        empty = False
        i = 2  # 1st row with info
        while not empty:
            try:
                evt = sheet.cell(i, 1).value.split()
                if len(evt) == 1:
                    evt = sheet.cell(i, 1).value.split('-')
                if evt:
                    if count_segments not in errors[s][day_split]:  # avoid segments with errors (dark/blurry images)
                        these_events.append([evt[0].strip(), evt[1].strip()])
                else:
                    empty = True
                i += 1
                count_segments += 1
            except:
                empty = True

        # Get list of images
        these_images = glob.glob(data_path + '/' + in_images_path + '/' + day_split + '/*' + imgs_format)
        final_these_images = []
        for im in these_images:
            final_these_images.append(im.split('/')[-1].split('.')[0])
        final_these_images = sorted(final_these_images)

        for e in these_events:
            if e[1] not in final_these_images:
                e[1] = '0' + e[1]
            if e[0] not in final_these_images:
                e[0] = '0' + e[0]

            fin_idx = final_these_images.index(e[1]) + 1
            ini_idx = final_these_images.index(e[0])
            current_event_imgs = final_these_images[ini_idx:fin_idx]

            # Store in files
            this_count = 0
            for imid in current_event_imgs:
                file_imgs.write(in_images_path + '/' + day_split + '/' + imid + imgs_format + '\n')
                this_count += 1
            file_counts.write(str(this_count) + '\n')

    file_imgs.close()
    file_counts.close()

print 'DONE!'
