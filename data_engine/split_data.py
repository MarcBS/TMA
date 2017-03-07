import numpy as np
import xlrd
import glob
import os

# Split the existent data in train, val and test
data_path = '/media/HDD_3TB/DATASETS/EDUB-SegDesc'
split_prop = {'train': 0.7,
             'val': 0.15,
             'test': 0.15,
            }
sets_names = ['Estefania1', 'Estefania2', 'Estefania3', 'Estefania4', 'Estefania5',
              'Gabriel1', 'Gabriel2', 'Gabriel3', 'Gabriel4',
              'MAngeles1', 'MAngeles2', 'MAngeles3', 'MAngeles4',
              'Marc1', 'Marc2', 'Marc3', 'Marc4', 'Marc5', 'Marc6', 'Marc7', 'Marc8', 'Marc9',
              'Marc10', 'Marc11', 'Marc12', 'Marc13', 'Marc14', 'Marc15', 'Marc16', 'Marc17', 'Marc18',
              'MarcC1',
              'Mariella', 'Mariella2', 'Mariella3',
              'Maya1', 'Maya2', 'Maya3', 'Maya4', 'Maya5', 'Maya6', 'Maya7', 'Maya8',
              'Maya9', 'Maya10', 'Maya11', 'Maya12', 'Maya13', 'Maya14',
              'Pedro1', 'Pedro2', 'Pedro3', 'Pedro4',
              #'Txell1'
              'Petia1', 'Petia2',
              ]

# input data paths
in_features_path = 'Features/Features_original'  # <name>/<in_features_name>.csv
in_descriptions_path = 'GT/descriptions'  # <name>.txt
in_segments_path = 'GT/segmentations'  # GT_<name>.xls(x)
in_images_path = 'Images'  # <name>/<image_name>.jpg
in_features_name = 'GoogleNet_ImageNet'
format = '.jpg'
# list of non-informative images stored in <in_features_path>/NonInfo/<noninformative_prefix>.csv
# leave empty for not using it
in_noninfo_path = 'Features/NonInfo'
noninformative_prefix = ''

# output data paths
out_features_path = 'Features'   # <set_split>_<out_features_name>_all_frames.csv & <set_split>_<out_features_name>_all_frames_counts.txt
out_descriptions_path = 'Annotations' # captions.id.en & <set_split>_list.txt
out_features_name = 'ImageNet'
separator = '----'


####################################

# generate data splits
available_sets = len(sets_names)
randomized = np.random.choice(sets_names, available_sets, replace=False)

#randomized = np.array(sets_names)

sets = dict()
picked_so_far = 0
for s,p in split_prop.iteritems():
    last_picked = np.ceil(picked_so_far+available_sets*p)
    sets[s] = randomized[picked_so_far:last_picked]
    picked_so_far = last_picked

# read images
images = dict()
for n,s in sets.iteritems():
    for set in s:
        images[set] = []
        these_images = glob.glob(data_path +'/'+ in_images_path +'/'+ set +'/*'+format)
        for im in these_images:
            images[set].append(im.split('/')[-1].split('.')[0])
        images[set] = sorted(images[set])

# read segmentations
events = dict()
for n,s in sets.iteritems():
    for set in s:
        possible_names = ['/GT_' +set+ '.xls', '/GT_' +set+ '.xlsx', '/' +set+ '.xls', '/' +set+ '.xlsx']
        exists = False
        i = 0
        while not os.path.isfile(data_path +'/'+ in_segments_path + possible_names[i]):
            i += 1
        file = xlrd.open_workbook(data_path +'/'+ in_segments_path + possible_names[i])
        sheet = file.sheet_by_index(0)

        these_events = []
        empty = False
        i = 2 # 1st row with info
        while not empty:
            try:
                evt = sheet.cell(i,1).value.split()
                if len(evt) == 1:
                    evt = sheet.cell(i, 1).value.split('-')
                if evt:
                    these_events.append([evt[0].strip(), evt[1].strip()])
                else:
                    empty = True
                i += 1
            except:
                empty = True
        events[set] = these_events


# get frames counts from segments and images lists
counts = dict()
for n,s in sets.iteritems():
    counts[n] = []
    for set in s:
        counts[set] = []
        prev = -1
        for e in events[set]:
            if e[1] not in images[set]:
                e[1] = '0' + e[1]
            if e[0] not in images[set]:
                e[0] = '0' + e[0]

            if prev != -1 and images[set].index(e[0]) - images[set].index(prev) > 1:
                raise Exception(images[set].index(e[0]), images[set].index(prev))
            c = images[set].index(e[1]) - images[set].index(e[0]) + 1
            prev = e[1]

            counts[set].append(c)
            counts[n].append(c)

        assert np.sum(counts[set]) == len(images[set])


# get descriptions for each segment
to_remove = dict()
caption_general = open(data_path +'/'+ out_descriptions_path +'/'+'captions_final.id.en', 'w')
for n,s in sets.iteritems():
    split_file = open(data_path +'/'+ out_descriptions_path +'/'+n+'_list_final.txt', 'w')
    to_remove[n] = dict()
    for set in s:
        to_remove[n][set] = []
        with open(data_path +'/'+ in_descriptions_path +'/'+ set +'.txt', 'r') as desc_file:
            prev_segm = -1
            count = 0
            segm_count = 0
            segm_count_show = 0
            for cline,line in enumerate(desc_file):
                if line:
                    line = line.rstrip('\n').split(',')
                    segm = line[0]
                    desc = ','.join(line[1:])
                    desc = desc.strip().lower()
                    if desc == 'error':
                        to_remove[n][set].append(segm_count)
                    else:
                        if prev_segm != segm:
                            segm_count_show += 1
                            split_file.write(set + '_Segment_' + str(segm_count_show) + '\n')
                            count = 0
                        caption_general.write(set + '_Segment_' + str(segm_count_show)
                                              + '#' + str(count) + separator + desc + '\n')
                        count += 1
                    assert segm[:7] == 'Segment', set +', line '+str(cline)
                    if prev_segm != segm:
                        if prev_segm == -1:
                            assert int(segm[7:]) == 1
                        else:
                            assert int(segm[7:]) == int(prev_segm[7:])+1, set +', line '+str(cline)+ ': ' + str(int(segm[7:])) + ' != ' + str(int(prev_segm[7:])+1)
                        segm_count += 1
                    prev_segm = segm
            try:
                int(segm[7:])
            except:
                raise Exception(set +' wrong Segment identifier: '+segm)
            assert segm_count == int(segm[7:]), set + ': ' + str(segm_count) + ' != ' + segm[7:]
            assert len(counts[set]) == segm_count, set + ': ' + str(segm_count) + ' != ' + str(len(counts[set]))

    split_file.close()
caption_general.close()

# get features for each data splits
for n,s in sets.iteritems():
    feats_file = open(data_path + '/' + out_features_path + '/' + n +'_'+out_features_name + '_all_frames.csv', 'w')
    counts_file = open(data_path + '/' + out_features_path + '/' + n + '_' + out_features_name + '_all_frames_counts.txt', 'w')
    for set in s:
        these_removed = to_remove[n][set]
        these_counts = counts[set]
        feats_set = open(data_path + '/' + in_features_path + '/' + set +'/' +in_features_name + '.csv', 'r')
        if noninformative_prefix:
            noninfo_file = open(data_path + '/' + in_noninfo_path +'/' +noninformative_prefix +'_'+set+ '.csv', 'r')
        for ic,count in enumerate(these_counts):
            new_count = 0
            these_feats = []
            for c in range(count):
                line = feats_set.next().rstrip('\n')
                is_informative = True
                if noninformative_prefix:
                    noninfo_line = noninfo_file.next().rstrip('\n')
                    if float(noninfo_line.split(',')[0]) >= 0.5: # checks if the current frame is non-informative and discards it
                        is_informative = False
                if is_informative:
                    these_feats.append(line)
                    new_count +=1
            if ic not in these_removed:
                for feat in these_feats:
                    feats_file.write(feat+'\n')
                counts_file.write(str(new_count)+'\n')

        if noninformative_prefix:
            noninfo_file.close()
        feats_set.close()
    feats_file.close()
    counts_file.close()


print 'DONE!'