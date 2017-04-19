import numpy as np


def main():
    # base_path = '/media/HDD_2TB/DATASETS/MSVD/'
    base_path = '/media/HDD_3TB/DATASETS/EDUB-SegDesc/'

    without_noninfo = True

    path_files = 'Annotations'

    # Inputs
    # text = 'captions.id.en'
    if without_noninfo:
        text = 'captions_final_without_noninfo.id.en'
    else:
        text = 'captions_final.id.en'
    separator = '----'

    # train = 'train_list.txt'
    # val =   'val_list.txt'
    # test =  'test_list.txt'

    if without_noninfo:
        train = 'train_list_final_without_noninfo.txt'
        val = 'val_list_final_without_noninfo.txt'
        test = 'test_list_final_without_noninfo.txt'

        # Outputs
        train_out = 'train_descriptions_without_noninfo.txt'
        val_out = 'val_descriptions_without_noninfo.txt'
        test_out = 'test_descriptions_without_noninfo.txt'

        train_out_counts = 'train_descriptions_counts_without_noninfo.npy'
        val_out_counts = 'val_descriptions_counts_without_noninfo.npy'
        test_out_counts = 'test_descriptions_counts_without_noninfo.npy'

    else:
        train = 'train_list_final.txt'
        val = 'val_list_final.txt'
        test = 'test_list_final.txt'

        # Outputs
        train_out = 'train_descriptions.txt'
        val_out = 'val_descriptions.txt'
        test_out = 'test_descriptions.txt'

        train_out_counts = 'train_descriptions_counts.npy'
        val_out_counts = 'val_descriptions_counts.npy'
        test_out_counts = 'test_descriptions_counts.npy'

    #################################

    # Code

    text = path_files + '/' + text
    splits = [path_files + '/' + train, path_files + '/' + val, path_files + '/' + test]
    splits_out = [path_files + '/' + train_out, path_files + '/' + val_out, path_files + '/' + test_out]
    splits_counts = [path_files + '/' + train_out_counts, path_files + '/' + val_out_counts,
                     path_files + '/' + test_out_counts]

    # read video names
    img_splits = [[], [], []]
    for i, s in enumerate(splits):
        with open(base_path + s, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                img_splits[i].append(line)

    # print img_splits


    # read descriptions and assign them to a split
    desc_splits = []
    counts_splits = []
    for i_s, s in enumerate(splits):
        desc_splits.append([[] for i in range(len(img_splits[i_s]))])
        counts_splits.append([0 for i in range(len(img_splits[i_s]))])
    with open(base_path + text, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            line = line.split('#')
            img = line[0]
            line = line[1].split(separator)
            desc = line[1]

            found = False
            i = 0
            while (not found and i < len(splits)):
                if (img in img_splits[i]):
                    found = True
                    idx = img_splits[i].index(img)
                    desc_splits[i][idx].append(desc)
                    counts_splits[i][idx] += 1
                i += 1

            if (not found):
                print 'Warning: Video ' + img + ' does not exist in lists'

    # write descriptions in separate files
    for f, d in zip(splits_out, desc_splits):
        f = open(base_path + f, 'w')
        for im in d:
            for desc in im:
                f.write(desc + '\n')
        f.close()

    # store description counts for each video
    for c, s in zip(counts_splits, splits_counts):
        np.save(base_path + s, c)

    print 'Done'


main()
