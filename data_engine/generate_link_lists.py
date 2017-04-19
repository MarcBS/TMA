## Parameters

base_path = '/media/HDD_3TB/DATASETS/EDUB-SegDesc/'

path_files = 'Annotations'
without_noninfo = True

# Names of the different samples
#   All samples belonging to the same day must accomplish the following requirements:
#       - Be referenced continuously, without mixing with other days
#       - Be stored in chronological order
#       - Include the day identifier at the beginning of the line separated by the symbol '_'
#   Example:
#       Day1_video_1
#       Day1_video_2
#       Day1_video_3
#       Day2_video_1
#       Day2_video_2
####

if without_noninfo:
    suffix = '_without_noninfo'
else:
    suffix = ''

train = 'train_list_final' + suffix + '.txt'
val = 'val_list_final' + suffix + '.txt'
test = 'test_list_final' + suffix + '.txt'

# Outputs
train_out = 'train_link_samples' + suffix + '.txt'
val_out = 'val_link_samples' + suffix + '.txt'
test_out = 'test_link_samples' + suffix + '.txt'

#################################

## Code

# Generate temporal links between samples which belong to the same day
for fin, fout in zip([train, val, test], [train_out, val_out, test_out]):

    with open(base_path + '/' + path_files + '/' + fin, 'r') as fi, open(base_path + '/' + path_files + '/' + fout,
                                                                         'w') as fo:
        prev_day_name = ''
        lines_counter = -1
        for line in fi:
            day_name = line.split('_')[0]
            if day_name == prev_day_name:
                fo.write(str(lines_counter) + '\n')
                lines_counter += 1
            else:
                fo.write('-1\n')
                lines_counter += 1

            prev_day_name = day_name

print 'Done'
