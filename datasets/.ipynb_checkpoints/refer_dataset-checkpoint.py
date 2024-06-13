import functools
import re

import numpy as np

import torch
from torch.utils.data import Dataset

from ._volta_refer_image_features_reader import RefImageFeaturesH5Reader
from tools.refer.refer import REFER


class ReferExpressionDataset(Dataset):
    def __init__(self,
                 dataroot,
                 name,
                 split,
                 image_feature_reader,
                 tokenizer,
                 max_region_num: int = 40,
                 max_seq_length: int = 40,
                 ):
        super().__init__()
        self.dataroot = dataroot
        self.name = name
        self.split = split
        self.image_feature_reader = image_feature_reader
        self.tokenizer = tokenizer
        self.max_region_num = max_region_num
        self.max_seq_length = max_seq_length
        self.last_word_pat = re.compile("[ ,]")
        #
        self.refer = REFER(dataroot, name)
        self.ref_ids = self.refer.getRefIds(split=split)
        self.entries = self.load_annotations()

    def load_annotations(self):
        entries = []
        for ref_id in self.ref_ids:
            ref = self.refer.loadRefs(ref_id)[0]
            unformatted_box = self.refer.getRefBox(ref_id)
            box = [unformatted_box[0],
                   unformatted_box[1],
                   unformatted_box[0] + unformatted_box[2],
                   unformatted_box[1] + unformatted_box[3]]
            image_id = ref['image_id']
            for sentence in ref['sentences']:
                sent = sentence['sent']
                tokens = sentence['tokens']
                sent_id = sentence['sent_id']
                entries.append({
                    "ref_id": ref_id,
                    "sent_id": sent_id,
                    "image_id": image_id,
                    "sent": sent,
                    "tokens": tokens,
                    "box": box,
                })
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]

        # pad image features to max_region_num
        image_id = entry['image_id']
        features, num_boxes, _, boxes_ori, objs_words, _ = self.image_features_reader[image_id]

        mix_num_boxes = min(int(num_boxes), self.max_region_num)
        # obj feature
        mix_features_pad = torch.zeros((self.max_region_num, self.image_features_reader.feature_size))
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]
        # boxes_ori
        boxes_ori_tensor = torch.zeros(self.max_region_num, 4)
        boxes_ori_tensor[:mix_num_boxes] = torch.tensor(boxes_ori[:mix_num_boxes])

        # obj attr words
        objs_id = [self.tokenize_obj_or_attr(w) for w in objs_words]
        mix_objs_id = torch.zeros(self.max_region_num, dtype=torch.long)
        mix_objs_id[:mix_num_boxes] = torch.tensor(objs_id[:mix_num_boxes], dtype=torch.long)

        # tokenize
        encoding = self._tokenize(entry)
        token_ids = torch.tensor(encoding.ids)
        length = torch.tensor(sum(encoding.attention_mask), dtype=torch.long)

        # ref_box
        ref_boxes = torch.tensor(entry["box"])
        return (torch.tensor(index, dtype=torch.long),
                image_id, mix_features_pad, mix_objs_id, torch.tensor(mix_num_boxes, dtype=torch.long),
                token_ids, length,
                entry, boxes_ori, ref_boxes, boxes_ori_tensor,
                )

    def collate_fn(self, batch):
        (index,
         image_id, mix_features_pad, mix_objs_id, mix_num_boxes,
         token_ids, length,
         entry, boxes_ori, ref_boxes, boxes_ori_tensor,
         ) = zip(*batch)

        should_stack = (
            index,
            mix_features_pad, mix_objs_id, mix_num_boxes,
            token_ids, length,
            ref_boxes, boxes_ori_tensor,
        )
        stacked = map(lambda x: torch.stack(x), should_stack)
        (
            index,
            mix_features_pad, mix_objs_id, mix_num_boxes,
            token_ids, length,
            ref_boxes, boxes_ori_tensor,
        ) = stacked

        return (index,
                image_id, mix_features_pad, mix_objs_id, mix_num_boxes,
                token_ids, length,
                entry, boxes_ori, ref_boxes, boxes_ori_tensor,
                )

    @functools.lru_cache(maxsize=4000)
    def tokenize_obj_or_attr(self, words):
        wd = self.last_word_pat.split(words)[-1]
        return self.tokenizer.get_vocab().get(wd, self.tokenizer.unk_token_id)

    def _tokenize(self, entry):
        caption_encoding = self.tokenizer(entry['tokens'],
                                          is_split_into_words=True,
                                          max_length=self.max_seq_length,
                                          padding='max_length',
                                          truncation=True,
                                          return_tensors='pt')
        return caption_encoding
