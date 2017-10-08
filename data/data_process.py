import os
import sys
import re
import codecs
import json
import random
import logging
import imghdr
import csv
import heapq
import ahocorasick
import numpy as np


# reload(sys)
# sys.setdefaultencoding('utf-8')

image_dataset_dir = '/raid5data/dplearn/taobao/crawler_tbimg/mimages'

data_dir = '/home/deepinsight/tongzhen/data-set/mogu_embedding'
data_file = 'mogujie_r.json'

save_dir = '/home/deepinsight/tongzhen/data-set/mogu_embedding'

cid_attr_file = 'mogu_category_attrs.json'
mogu_valid_json = 'mogu_valid.json'
image_path_file = 'image_path.txt'

# image_dataset_dir = '/home/tze/Tmp/taobao_stn'
#
# data_dir = '/home/tze'
# data_file = 'mogujie_r.json'
#
# save_dir = '/home/tze'
#
# cid_attr_file = 'mogu_category_attrs.json'
# mogu_valid_json = 'mogu_valid.json'
# image_path_file = 'image_path.txt'

src_dir = '/raid5data/dplearn/taobao/crawler_tbimg/mimages'

NUM_CID = 245
NUM_ATTR_KEY = 128

num_class_by_hand = 237
num_attr_key = 121
num_attr_val = 1160
num_color_val = 75
avg_images_per_id = 5

train_set_keep_prob = 0.7


def parse_json(item):
    data = json.loads(item)
    cid_str = data['cid']
    iid_str = data['iid']
    im_list = data['im_list']
    title_str = data['title']
    prop_list = data['props']
    attr_dict = data['attributes']

    return cid_str, iid_str, im_list, title_str, prop_list, attr_dict


# 1
def _json_filter():
    with open(os.path.join(data_dir, data_file), 'r') as f, open(os.path.join(save_dir, mogu_valid_json), 'w') as g:
        for line in f:
            data = json.loads(line)
            if _is_valid_json(data):
                g.write(line)


def _is_valid_json(data):
    return ('cid' in data and
            'iid' in data and
            'im_list' in data and
            'title' in data and
            'props' in data and
            'attributes' in data)


# 2
def _preprocess_attrs_props():
    cid_attr_key = {}
    props_counts = {}
    with open(os.path.join(save_dir, mogu_valid_json), 'r') as f:
        for line in f:
            data = json.loads(line)
            cid_str = data['cid']
            attr_dict = data['attributes']
            prop_list = data['props']

            if cid_str not in cid_attr_key:
                cid_attr_key[cid_str] = set(attr_dict.keys())
            else:
                cid_attr_key[cid_str].update(attr_dict.keys())

            for prop in prop_list:
                if prop not in props_counts:
                    props_counts[prop] = 1
                else:
                    props_counts[prop] += 1

    with codecs.open(os.path.join(save_dir, cid_attr_file), 'w', 'utf-8') as f:
        for cid, attr_key in cid_attr_key.items():
            f.write(json.dumps({cid: list(attr_key)}, ensure_ascii=False) + '\n')

    with codecs.open(os.path.join(save_dir, 'mogu_cid.txt'), 'w', 'utf-8') as f:
        for cid in cid_attr_key:
            f.write(cid + '\n')

    sorted_prop = sorted(props_counts.items(), key=lambda kv: kv[1], reverse=True)
    with codecs.open(os.path.join(save_dir, 'mogu_props_count.txt'), 'w', 'utf-8') as f:
        for prop, count in sorted_prop:
            f.write('{}:{}\n'.format(prop, count))


# 3
def _get_global_attr_key():
    attr_key = {}
    with open(os.path.join(save_dir, mogu_valid_json), 'r') as f:
        for line in f:
            data = json.loads(line)
            attr_dict = data['attributes']

            for k in attr_dict:
                if k not in attr_key:
                    attr_key[k] = 1
                else:
                    attr_key[k] += 1

    sorted_attr_key = sorted(attr_key.items(), key=lambda k: k[1], reverse=True)
    attr_set = set()
    for idx, kv in enumerate(sorted_attr_key):
        if idx < NUM_ATTR_KEY:
            attr_set.add(kv[0])
        else:
            break
    with codecs.open(os.path.join(save_dir, 'final_mogu_attr_keys.txt'), 'w', 'utf-8') as f:
        for k in attr_set:
            f.write(k + '\n')

    cid_attr = {}
    with open(os.path.join(save_dir, cid_attr_file), 'r') as f:
        for line in f:
            data = json.loads(line)
            for cid_str, attr_key_list in data.items():
                cid_attr[cid_str] = []
                for k in attr_key_list:
                    if k in attr_set:
                        cid_attr[cid_str].append(k)

    with codecs.open(os.path.join(save_dir, 'final_mogu_cid_attr_key.json'), 'w', 'utf-8') as f:
        for cid, key_list in cid_attr.items():
            f.write(json.dumps({cid: key_list}, ensure_ascii=False) + '\n')

    cut_off_freq = 15
    with codecs.open(os.path.join(save_dir, 'final_attr_key_counts.txt'), 'w', 'utf-8') as f:
        for attr, count in sorted_attr_key:
            if count > cut_off_freq:
                f.write('{}:{}\n'.format(attr, count))


def get_image_path():
    image_folder_name = [folder for folder in os.listdir(image_dataset_dir)
                         if os.path.isdir(os.path.join(image_dataset_dir, folder))]

    image_path_list = []
    for folder in image_folder_name:
        image_folder = os.path.join(image_dataset_dir, folder)
        for image in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image)
            if os.path.isfile(image_path) and imghdr.what(image_path) == 'jpeg':
                image_path_list.append(image_path)

    with open(os.path.join(save_dir, image_path_file), 'w') as f:
        for path in image_path_list:
            f.write(path + '\n')


def create_selected_attr_key_val():
    attr_keys = set()
    with codecs.open(os.path.join(save_dir, 'final_mogu_attr_keys_byhand.txt'), 'r', 'utf-8') as f:
        for line in f:
            attr_keys.add(line.strip())

    attr_key_val_dict = {}
    with open(os.path.join(save_dir, mogu_valid_json), 'r') as f:
        for line in f:
            data = json.loads(line)
            attr_dict = data['attributes']
            for attr_k, attr_str in attr_dict.items():
                if attr_k in attr_keys:
                    if attr_k not in attr_key_val_dict:
                        attr_key_val_dict[attr_k] = {}

                    attr_str_filter = _clean_text(attr_str)
                    for attr_val in attr_str_filter:
                        if attr_val in attr_key_val_dict[attr_k]:
                            attr_key_val_dict[attr_k][attr_val] += 1
                        else:
                            attr_key_val_dict[attr_k][attr_val] = 1

    attr_val_counts_thresh = 10
    attr_val_share_thresh = 0.975
    for attr_k, attr_val_dict in attr_key_val_dict.items():
        for val_k, val_counts in attr_val_dict.items():
            if val_counts < attr_val_counts_thresh:
                attr_val_dict.pop(val_k)

    for attr_k, attr_val_dict in attr_key_val_dict.items():
        sorted_attr_val_dict = sorted(attr_val_dict.items(), key=lambda kv: kv[1], reverse=True)
        val_kv = list(zip(*sorted_attr_val_dict))
        sum_val_counts = sum(val_kv[1])
        val_counts_share = sum_val_counts * attr_val_share_thresh

        val_counter = 0.
        for val_k, val_counts in sorted_attr_val_dict:
            if val_counter < val_counts_share:
                val_counter += val_counts
            else:
                attr_val_dict.pop(val_k)

    with codecs.open(os.path.join(save_dir, 'final_mogu_attr_key_val_counts.json'), 'w', 'utf-8') as f, \
            codecs.open(os.path.join(save_dir, 'final_mogu_selected_attr.json'), 'w', 'utf-8') as g:
        label_counter = 0
        attr_key_val_label_dict = {}
        for attr_k, attr_val_dict in attr_key_val_dict.items():
            f.write(json.dumps({attr_k: attr_val_dict}, ensure_ascii=False) + '\n')

            attr_val_label_dict = {}
            for attr_val_key in attr_val_dict.keys():
                attr_val_label_dict[attr_val_key] = label_counter
                label_counter += 1
            attr_key_val_label_dict[attr_k] = attr_val_label_dict

        g.write(json.dumps(attr_key_val_label_dict, ensure_ascii=False))


def process_color_attr():
    # the unicode of 'color' of chinese character
    color_attr_in_chinese = u'\u989c\u8272'
    color_attr_val_dict = {}
    with open(os.path.join(save_dir, mogu_valid_json), 'r') as f:
        for line in f:
            data = json.loads(line)
            attr_dict = data['attributes']
            attr_k = attr_dict.keys()

            if color_attr_in_chinese in attr_k:
                color_attr_str = attr_dict[color_attr_in_chinese]
                color_attr_val = _clean_text(color_attr_str)
                for val in color_attr_val:
                    if val in color_attr_val_dict:
                        color_attr_val_dict[val] += 1
                    else:
                        color_attr_val_dict[val] = 1

    sorted_color_val = sorted(color_attr_val_dict.items(), key=lambda kv: kv[1], reverse=True)

    attr_val_counts_thresh = 10
    # attr_val_share_thresh = 0.975

    filter_sorted_color_val = [kv for kv in sorted_color_val if kv[1] >= attr_val_counts_thresh]

    # qualify_sorted_color_val = []
    # color_counter = 0.
    # color_count = zip(*filter_sorted_color_val)
    # sum_color_counts = sum(color_count[1])
    # total_color_counts = attr_val_share_thresh * sum_color_counts
    # for color_count in filter_sorted_color_val:
    #     if color_counter < total_color_counts:
    #         qualify_sorted_color_val.append(color_count)
    #         color_counter += color_count[1]
    #     else:
    #         break

    with codecs.open(os.path.join(save_dir, 'mogu_color_val.txt'), 'w', 'utf-8') as f:
        for color, count in filter_sorted_color_val:
            f.write('{}:{}\n'.format(color, count))


def create_color_val():
    color_val = []
    with codecs.open(os.path.join(save_dir, 'mogu_color_val_byhand.txt'), 'r', 'utf-8') as f:
        for line in f:
            color_val.append(line.strip().split(':')[0])

    color_val_label_dict = {each[1]: each[0] for each in enumerate(color_val)}
    with codecs.open(os.path.join(save_dir, 'mogu_color_val.json'), 'w', 'utf-8') as f:
        f.write(json.dumps(color_val_label_dict, ensure_ascii=False))


def create_train_eval_set():
    with open(os.path.join(save_dir, 'mogu_cls_label_v1.json'), 'r') as f:
        cls_label_dict = json.loads(f.readline().strip())
    with open(os.path.join(save_dir, 'mogu_color_val.json'), 'r') as f:
        color_label_dict = json.loads(f.readline().strip())
    with open(os.path.join(save_dir, 'mogu_selected_attr.json'), 'r') as f:
        attr_label_dict = json.loads(f.readline().strip())

    selected_cid = set(cls_label_dict.keys())
    color_attr_in_chinese = u'\u989c\u8272'

    with open(os.path.join(save_dir, mogu_valid_json), 'r') as f, \
         open(os.path.join(save_dir, 'mogu_train.csv'), 'wb') as g, \
         open(os.path.join(save_dir, 'mogu_eval.csv'), 'wb') as h:

        train_set_writer = csv.writer(g)
        eval_set_writer = csv.writer(h)
        train_save_info = []
        eval_save_info = []

        for line in f:
            data = json.loads(line)
            cid_str = data['cid']
            iid_str = data['iid']
            im_list = data['im_list']
            attr_dict = data['attributes']

            if cid_str in selected_cid:
                folder_path = os.path.join(src_dir, cid_str)
                num_files = len(im_list)
                file_paths = [os.path.join(folder_path, '{}_{}.jpg'.format(iid_str, idx)) for idx in range(num_files)]

                cls_label = str(cls_label_dict[cid_str])

                if attr_dict.has_key(color_attr_in_chinese):
                    color_list = _clean_text(attr_dict[color_attr_in_chinese])
                    color_labels = [str(color_label_dict[color]) for color in color_list
                                    if color_label_dict.has_key(color)]
                    if color_labels:
                        color_labels = ' '.join(color_labels)
                    else:
                        color_labels = '-1'
                else:
                    color_labels = '-1'

                attr_labels = []
                for attr_key, attr_text in attr_dict.items():
                    attr_val_list = _clean_text(attr_text)
                    labels = [str(attr_label_dict[attr_key][attr_val]) for attr_val in attr_val_list
                              if attr_label_dict.has_key(attr_key) and attr_label_dict[attr_key].has_key(attr_val)]
                    attr_labels.extend(labels)

                if attr_labels:
                    attr_labels = ' '.join(attr_labels)
                else:
                    attr_labels = '-1'

                if num_files > avg_images_per_id:
                    for path in file_paths:
                        if random.random() < train_set_keep_prob:
                            train_save_info.append((path, cls_label, color_labels, attr_labels))
                        else:
                            eval_save_info.append((path, cls_label, color_labels, attr_labels))
                else:
                    for path in file_paths:
                        eval_save_info.append((path, cls_label, color_labels, attr_labels))

        if train_save_info:
            train_set_writer.writerows(train_save_info)
        if eval_save_info:
            eval_set_writer.writerows(eval_save_info)


def create_train_eval_set_full_info():
    with open(os.path.join(save_dir, 'mogu_cls_label_v1.json'), 'r') as f:
        cls_label_dict = json.loads(f.readline().strip())
    with open(os.path.join(save_dir, 'mogu_color_val.json'), 'r') as f:
        color_label_dict = json.loads(f.readline().strip())
    with open(os.path.join(save_dir, 'mogu_selected_attr.json'), 'r') as f:
        attr_label_dict = json.loads(f.readline().strip())

    selected_cid = set(cls_label_dict.keys())
    color_attr_in_chinese = u'\u989c\u8272'

    with open(os.path.join(save_dir, mogu_valid_json), 'r') as f, \
         open(os.path.join(save_dir, 'mogu_train_full_info.csv'), 'wb') as g, \
         open(os.path.join(save_dir, 'mogu_eval_rest.csv'), 'wb') as h:

        train_set_writer = csv.writer(g)
        eval_set_writer = csv.writer(h)
        train_save_info = []
        eval_save_info = []

        keep_prob = 0.8
        for line in f:
            data = json.loads(line)
            cid_str = data['cid']
            iid_str = data['iid']
            im_list = data['im_list']
            attr_dict = data['attributes']

            if cid_str in selected_cid:
                folder_path = os.path.join(src_dir, cid_str)
                num_files = len(im_list)
                file_paths = [os.path.join(folder_path, '{}_{}.jpg'.format(iid_str, idx)) for idx in range(num_files)]

                cls_label = str(cls_label_dict[cid_str])

                if attr_dict.has_key(color_attr_in_chinese):
                    color_list = _clean_text(attr_dict[color_attr_in_chinese])
                    color_labels = [str(color_label_dict[color]) for color in color_list
                                    if color_label_dict.has_key(color)]
                    if color_labels:
                        color_labels = ' '.join(color_labels)
                    else:
                        color_labels = '-1'
                else:
                    color_labels = '-1'

                attr_labels = []
                for attr_key, attr_text in attr_dict.items():
                    attr_val_list = _clean_text(attr_text)
                    labels = [str(attr_label_dict[attr_key][attr_val]) for attr_val in attr_val_list
                              if attr_label_dict.has_key(attr_key) and attr_label_dict[attr_key].has_key(attr_val)]
                    attr_labels.extend(labels)

                if attr_labels:
                    attr_labels = ' '.join(attr_labels)
                else:
                    attr_labels = '-1'

                if _qualified_data_record(color_labels, attr_labels):
                    for path in file_paths:
                        if random.random() < keep_prob:
                            train_save_info.append((path, cls_label, color_labels, attr_labels))
                        else:
                            eval_save_info.append((path, cls_label, color_labels, attr_labels))
                else:
                    for path in file_paths:
                        eval_save_info.append((path, cls_label, color_labels, attr_labels))

        if train_save_info:
            train_set_writer.writerows(train_save_info)
        if eval_save_info:
            eval_set_writer.writerows(eval_save_info)


def create_train_eval_set_ahocorasick():
    with open(os.path.join(save_dir, 'mogu_cls_label_v1.json'), 'r') as f:
        cls_label_dict = json.loads(f.readline().strip())
    with open(os.path.join(save_dir, 'mogu_color_val.json'), 'r') as f:
        color_label_dict = json.loads(f.readline().strip())
    with open(os.path.join(save_dir, 'mogu_selected_attr.json'), 'r') as f:
        attr_label_dict = json.loads(f.readline().strip())

    selected_cid = set(cls_label_dict.keys())
    color_attr_in_chinese = u'\u989c\u8272'

    color_aho = ahocorasick.Automaton()
    # attr_aho = ahocorasick.Automaton()
    for color, label in color_label_dict.items():
        color_aho.add_word(color.encode('utf-8'), (label, color))
    color_aho.make_automaton()

    with open(os.path.join(save_dir, mogu_valid_json), 'r') as f, \
         open(os.path.join(save_dir, 'mogu_train_full_info_aho.csv'), 'wb') as g, \
         open(os.path.join(save_dir, 'mogu_eval_rest_aho.csv'), 'wb') as h:

        train_set_writer = csv.writer(g)
        eval_set_writer = csv.writer(h)
        train_save_info = []
        eval_save_info = []

        keep_prob = 0.8
        for line in f:
            data = json.loads(line)
            cid_str = data['cid']
            iid_str = data['iid']
            im_list = data['im_list']
            attr_dict = data['attributes']

            if cid_str in selected_cid:
                folder_path = os.path.join(src_dir, cid_str)
                num_files = len(im_list)
                file_paths = [os.path.join(folder_path, '{}_{}.jpg'.format(iid_str, idx)) for idx in range(num_files)]

                cls_label = str(cls_label_dict[cid_str])

                if attr_dict.has_key(color_attr_in_chinese):
                    color_list = _clean_text(attr_dict[color_attr_in_chinese])

                    color_labels = []
                    for clr in color_list:
                        max_len_clr_label = None
                        max_len = 0
                        for _, label_wd in color_aho.iter(clr.encode('utf-8')):
                            if len(label_wd[1]) > max_len:
                                max_len_clr_label = label_wd[0]

                        if max_len_clr_label:
                            if str(max_len_clr_label) not in color_labels:
                                color_labels.append(str(max_len_clr_label))

                    color_labels = [str(color_label_dict[color]) for color in color_list
                                    if color_label_dict.has_key(color)]
                    if color_labels:
                        color_labels = ' '.join(color_labels)
                    else:
                        color_labels = '-1'
                else:
                    color_labels = '-1'

                attr_labels = []
                for attr_key, attr_text in attr_dict.items():
                    attr_val_list = _clean_text(attr_text)
                    labels = [str(attr_label_dict[attr_key][attr_val]) for attr_val in attr_val_list
                              if attr_label_dict.has_key(attr_key) and attr_label_dict[attr_key].has_key(attr_val)]
                    attr_labels.extend(labels)

                if attr_labels:
                    attr_labels = ' '.join(attr_labels)
                else:
                    attr_labels = '-1'

                if _qualified_data_record(color_labels, attr_labels):
                    for path in file_paths:
                        if random.random() < keep_prob:
                            train_save_info.append((path, cls_label, color_labels, attr_labels))
                        else:
                            eval_save_info.append((path, cls_label, color_labels, attr_labels))
                else:
                    for path in file_paths:
                        eval_save_info.append((path, cls_label, color_labels, attr_labels))

        if train_save_info:
            train_set_writer.writerows(train_save_info)
        if eval_save_info:
            eval_set_writer.writerows(eval_save_info)


def create_train_eval_set_ahocorasick_with_image_id_color_bytze():
    with open(os.path.join(save_dir, 'mogu_cls_label_v1.json'), 'r') as f:
        cls_label_dict = json.loads(f.readline().strip())
    with open(os.path.join(save_dir, 'mogu_color_val.json'), 'r') as f:
        color_label_dict = json.loads(f.readline().strip())
    with open(os.path.join(save_dir, 'mogu_selected_attr.json'), 'r') as f:
        attr_label_dict = json.loads(f.readline().strip())
    with open(os.path.join(save_dir, 'mogu_color_val_tze.json'), 'r') as f:
        color_label_dict_fin = json.loads(f.readline().strip())

    selected_cid = set(cls_label_dict.keys())
    color_attr_in_chinese = u'\u989c\u8272'

    color_aho = ahocorasick.Automaton()
    # attr_aho = ahocorasick.Automaton()
    for color, label in color_label_dict.items():
        color_aho.add_word(color.encode('utf-8'), (label, color))
    color_aho.make_automaton()

    with open(os.path.join(save_dir, mogu_valid_json), 'r') as f, \
         open(os.path.join(save_dir, 'mogu_train_full_info_color_tze.csv'), 'wb') as g, \
         open(os.path.join(save_dir, 'mogu_eval_rest_color_tze.csv'), 'wb') as h:

        train_set_writer = csv.writer(g)
        eval_set_writer = csv.writer(h)
        train_save_info = []
        eval_save_info = []

        keep_prob = 0.8
        #image_id = 0
        qualified_image_id = 0
        #eval_image_id = 0
        num_imgs_thresh = 2
        for line in f:
            data = json.loads(line)
            cid_str = data['cid']
            iid_str = data['iid']
            im_list = data['im_list']
            attr_dict = data['attributes']

            if cid_str in selected_cid:
                folder_path = os.path.join(src_dir, cid_str)
                num_files = len(im_list)
                file_paths = [os.path.join(folder_path, '{}_{}.jpg'.format(iid_str, idx)) for idx in range(num_files)]

                #image_id_label = str(image_id)
                #image_id += 1

                cls_label = str(cls_label_dict[cid_str])

                if attr_dict.has_key(color_attr_in_chinese):
                    color_list = _clean_text(attr_dict[color_attr_in_chinese])

                    color_labels = []
                    for clr in color_list:
                        max_len_clr_str = None
                        max_len = 0
                        for _, label_wd in color_aho.iter(clr.encode('utf-8')):
                            if len(label_wd[1]) > max_len:
                                max_len_clr_str = label_wd[1]

                        if max_len_clr_str:
                            for color_vals_str, label in color_label_dict_fin.items():
                                color_vals = color_vals_str.split()
                                if max_len_clr_str in color_vals:
                                    if str(label) not in color_labels:
                                        color_labels.append(str(label))
                                        break

                    if color_labels:
                        color_labels = ' '.join(color_labels)
                    else:
                        color_labels = '-1'
                else:
                    color_labels = '-1'

                attr_labels = []
                for attr_key, attr_text in attr_dict.items():
                    attr_val_list = _clean_text(attr_text)
                    labels = [str(attr_label_dict[attr_key][attr_val]) for attr_val in attr_val_list
                              if attr_label_dict.has_key(attr_key) and attr_label_dict[attr_key].has_key(attr_val)]
                    attr_labels.extend(labels)

                if attr_labels:
                    attr_labels = ' '.join(attr_labels)
                else:
                    attr_labels = '-1'

                if _qualified_data_record(color_labels, attr_labels) and num_files >= num_imgs_thresh:
                    qualified_image_id_label = str(qualified_image_id)
                    qualified_image_id += 1

                    if num_files == num_imgs_thresh:
                        for path in file_paths:
                            train_save_info.append(
                                (path, qualified_image_id_label, cls_label, color_labels, attr_labels))
                    else:
                        num_samples_for_eval = int((1.0 - keep_prob) * num_files + 1.0)
                        eval_idx = random.sample(range(num_files), num_samples_for_eval)
                        for idx, path in enumerate(file_paths):
                            if idx in eval_idx:
                                eval_save_info.append(
                                    (path, qualified_image_id_label, cls_label, color_labels, attr_labels))
                            else:
                                train_save_info.append(
                                    (path, qualified_image_id_label, cls_label, color_labels, attr_labels))
                # else:
                #     for path in file_paths:
                #         eval_save_info.append((path, image_id_label, cls_label, color_labels, attr_labels))

        if train_save_info:
            train_set_writer.writerows(train_save_info)
        if eval_save_info:
            eval_set_writer.writerows(eval_save_info)


def test_color_attr_clustering():
    file_path = []
    file_label = []
    file_color = []
    file_attr = []
    with open(os.path.join(save_dir, 'mogu_whole.csv'), 'rb') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            file_path.append(line[0])

            file_attr_label = [int(label) for label in line[3].split() if int(label) != -1]
            file_attr.append(file_attr_label)

    path_attr_pair = zip(file_path, file_attr)
    anchor_sample = random.choice(path_attr_pair)

    anchor_attr_np = np.zeros(num_attr_val)
    anchor_attr_np[anchor_sample[1]] = 1.

    l2_attr_dist = []
    cosine_attr_dist = []
    for path, attr_label in path_attr_pair:
        if attr_label:
            comp_attr_np = np.zeros(num_attr_val)
            comp_attr_np[attr_label] = 1.

        attr_l2_diff = _l2_dist(anchor_attr_np, comp_attr_np)
        attr_cosine_diff = _cosine_dist(anchor_attr_np, comp_attr_np)
        l2_attr_dist.append((path, attr_l2_diff))
        cosine_attr_dist.append((path, attr_cosine_diff))

    sorted_l2_dist = heapq.nsmallest(30, l2_attr_dist, key=lambda path_dist: path_dist[1])
    sorted_cosine_dist = heapq.nlargest(30, cosine_attr_dist, key=lambda path_dist: path_dist[1])

    print('anchor sample path: {}'.format(anchor_sample[0]))
    print('anchor sample attr: {}'.format(anchor_sample[1]))

    print('\n')

    print('l2 dist:')
    for idx in range(20):
        print(sorted_l2_dist[idx])

    print('\n')

    print('cosine dist:')
    for idx in range(20):
        print(sorted_cosine_dist[idx])


def _l2_dist(anchor, comp):
    return np.linalg.norm(anchor-comp)


def _cosine_dist(anchor, comp):
    norm1 = np.linalg.norm(anchor)
    norm2 = np.linalg.norm(comp)
    return np.sum(anchor * comp) / (norm1 * norm2)


def _make_whole_data_set():
    with open(os.path.join(save_dir, 'mogu_cls_label_v1.json'), 'r') as f:
        cls_label_dict = json.loads(f.readline().strip())
    with open(os.path.join(save_dir, 'mogu_color_val.json'), 'r') as f:
        color_label_dict = json.loads(f.readline().strip())
    with open(os.path.join(save_dir, 'mogu_selected_attr.json'), 'r') as f:
        attr_label_dict = json.loads(f.readline().strip())

    selected_cid = set(cls_label_dict.keys())
    color_attr_in_chinese = u'\u989c\u8272'

    with open(os.path.join(save_dir, mogu_valid_json), 'r') as f, \
         open(os.path.join(save_dir, 'mogu_whole.csv'), 'wb') as g:

        set_writer = csv.writer(g)
        save_info = []

        for line in f:
            data = json.loads(line)
            cid_str = data['cid']
            iid_str = data['iid']
            im_list = data['im_list']
            attr_dict = data['attributes']

            if cid_str in selected_cid:
                folder_path = os.path.join(src_dir, cid_str)
                num_files = len(im_list)
                file_paths = [os.path.join(folder_path, '{}_{}.jpg'.format(iid_str, idx)) for idx in range(num_files)]

                cls_label = str(cls_label_dict[cid_str])

                if attr_dict.has_key(color_attr_in_chinese):
                    color_list = _clean_text(attr_dict[color_attr_in_chinese])
                    color_labels = [str(color_label_dict[color]) for color in color_list
                                    if color_label_dict.has_key(color)]
                    if color_labels:
                        color_labels = ' '.join(color_labels)
                    else:
                        color_labels = '-1'
                else:
                    color_labels = '-1'

                attr_labels = []
                for attr_key, attr_text in attr_dict.items():
                    attr_val_list = _clean_text(attr_text)
                    labels = [str(attr_label_dict[attr_key][attr_val]) for attr_val in attr_val_list
                              if attr_label_dict.has_key(attr_key) and attr_label_dict[attr_key].has_key(attr_val)]
                    attr_labels.extend(labels)

                if attr_labels:
                    attr_labels = ' '.join(attr_labels)
                else:
                    attr_labels = '-1'

                for path in file_paths:
                    save_info.append((path, cls_label, color_labels, attr_labels))

        set_writer.writerows(save_info)


def _create_full_info_data_set():
    with open(os.path.join(save_dir, 'mogu_cls_label_v1.json'), 'r') as f:
        cls_label_dict = json.loads(f.readline().strip())
    with open(os.path.join(save_dir, 'mogu_color_val.json'), 'r') as f:
        color_label_dict = json.loads(f.readline().strip())
    with open(os.path.join(save_dir, 'mogu_selected_attr.json'), 'r') as f:
        attr_label_dict = json.loads(f.readline().strip())

    selected_cid = set(cls_label_dict.keys())
    color_attr_in_chinese = u'\u989c\u8272'

    with open(os.path.join(save_dir, mogu_valid_json), 'r') as f, \
         open(os.path.join(save_dir, 'mogu_whole_full_info.csv'), 'wb') as g:

        set_writer = csv.writer(g)
        save_info = []

        for line in f:
            data = json.loads(line)
            cid_str = data['cid']
            iid_str = data['iid']
            im_list = data['im_list']
            attr_dict = data['attributes']

            if cid_str in selected_cid:
                folder_path = os.path.join(src_dir, cid_str)
                num_files = len(im_list)
                file_paths = [os.path.join(folder_path, '{}_{}.jpg'.format(iid_str, idx)) for idx in range(num_files)]

                cls_label = str(cls_label_dict[cid_str])

                if attr_dict.has_key(color_attr_in_chinese):
                    color_list = _clean_text(attr_dict[color_attr_in_chinese])
                    color_labels = [str(color_label_dict[color]) for color in color_list
                                    if color_label_dict.has_key(color)]
                    if color_labels:
                        color_labels = ' '.join(color_labels)
                    else:
                        color_labels = '-1'
                else:
                    color_labels = '-1'

                attr_labels = []
                for attr_key, attr_text in attr_dict.items():
                    attr_val_list = _clean_text(attr_text)
                    labels = [str(attr_label_dict[attr_key][attr_val]) for attr_val in attr_val_list
                              if attr_label_dict.has_key(attr_key) and attr_label_dict[attr_key].has_key(attr_val)]
                    attr_labels.extend(labels)

                if attr_labels:
                    attr_labels = ' '.join(attr_labels)
                else:
                    attr_labels = '-1'

                for path in file_paths:
                    if _qualified_data_record(color_labels, attr_labels):
                        save_info.append((path, cls_label, color_labels, attr_labels))

        set_writer.writerows(save_info)


def _is_valid_character(nv):
    ulen = 0
    for ch in nv:
        if u'\u4e00' <= ch <= u'\u9fff':
            ulen += 1
        else:
            return False
    return ulen > 1


def _clean_text(line):
    line = line.lower()
    # p_0 = re.compile(ur'\((.*?)\)')
    # line = p_0.sub(u'', line)

    # seek alpha, num, chinese character, dash, brackets (both chinese and english)
    p_1 = re.compile(u'[\w\u4e00-\u9fa5\-\uff08\uff09\(\)]+')
    words = p_1.findall(line)
    return words


def _qualified_data_record(color_labels, attr_labels):
    num_color_attrs = len(color_labels.split(' '))
    return color_labels != '-1' and num_color_attrs <= 1 and attr_labels != '-1'


# def disp_json(cid_str, iid_str, im_list, title_str, prop_list, attr_dict):
#     print('{')
#
#     print('     cid: {}'.format(cid_str))
#
#     print('     iid: {}'.format(iid_str))
#
#     print('     im_list: ')
#     for im in im_list:
#         print('              {}'.format(im))
#
#     print('     title: {}'.format(title_str))
#
#     print('     props:', end='')
#     for prop in prop_list:
#         print(' {}'.format(prop), end='')
#     print()
#
#     print('     attributes:', end='')
#     idx = 0
#     for k, v in attr_dict.items():
#         if idx % 5 == 0:
#             print()
#             print('                ', end='')
#         print(' ({}: {})'.format(k, v), end='')
#         idx += 1
#     print()
#
#     print('}')


def main():
    # with open(os.path.join(data_dir, file)) as f:
    #     for _ in range(10):
    #         cid, iid, im_li, title, prop_li, attr_di = parse_json(f.readline())
    #         disp_json(cid, iid, im_li, title, prop_li, attr_di)

    # _json_filter()

    # _preprocess_attrs_props()

    # get_image_path()

    # _get_global_attr_key()

    # process_color_attr()

    # create_selected_attr_key_val()

    # create_color_val()

    # create_train_eval_set()

    # create_train_eval_set_full_info()

    create_train_eval_set_ahocorasick()

    # _make_whole_data_set()

    # _create_full_info_data_set()

    # test_color_attr_clustering()


if __name__ == '__main__':
    main()

