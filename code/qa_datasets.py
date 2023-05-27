import json
import numpy as np
import torch
import utils
from transformers import BertTokenizer
from torch.utils.data import Dataset
import random


class QA_Dataset_Baseline(Dataset):
    def __init__(self, split):
        print('Preparing data for split %s' % split)
        filename = '../data/questions/processed_questions/{split}.json'.format(split=split)
        with open(filename, 'r', encoding='utf-8') as obj:
            questions = json.load(obj)
        print('Only use first 50000 questions')
        questions = questions[:50000]
        for q in questions:
            if split == 'test':
                q['answers'] = []
            else:
                q['answers'] = [x.replace('_', ' ') for x in q['answers']]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.all_dicts = utils.getAllDicts()
        print('Total questions = ', len(questions))
        self.data = questions
        self.num_total_entities = len(self.all_dicts['ent2id'])
        self.num_total_times = len(self.all_dicts['ts2id'])
        self.prepared_data = self.prepare_data_(self.data)

    def getEntitiesLocations(self, question):
        entities = question['entity_positions']
        ent2id = self.all_dicts['ent2id']
        loc_ent = []
        for e in entities:
            e_id = ent2id[e['entity'][0]]
            location = e['position'][0]
            loc_ent.append((location, e_id))
        return loc_ent

    def entitiesToIds(self, entities):
        output = []
        ent2id = self.all_dicts['ent2id']
        for e in entities:
            output.append(ent2id[e])
        return output

    def getIdType(self, id):
        if id < len(self.all_dicts['ent2id']):
            return 'entity'
        else:
            return 'time'

    def getEntityIdToText(self, id):
        ent = self.all_dicts['id2ent'][id]
        return ent

    def timesToIds(self, times):
        output = []
        ts2id = self.all_dicts['ts2id']
        for t in times:
            if t not in ts2id.keys():
                output.append(0)
            else:
                output.append(ts2id[t])
        return output

    def getAnswersFromScores(self, scores, largest=True, k=10):
        _, ind = torch.topk(scores, k, largest=largest)
        predict = ind
        answers = []
        for a_id in predict:
            a_id = a_id.item()
            type = self.getIdType(a_id)
            if type == 'entity':
                # answers.append(self.getEntityIdToText(a_id))
                answers.append(self.getEntityIdToText(a_id))
            else:
                time_id = a_id - len(self.all_dicts['ent2id'])
                time = self.all_dicts['id2ts'][time_id]
                answers.append(time)
        return answers

    def prepare_data_(self, data):
        question_text = []
        heads = []
        tails = []
        times = []
        answers_arr = []
        ent2id = self.all_dicts['ent2id']
        self.data_ids_filtered = []
        for i, question in enumerate(data):
            self.data_ids_filtered.append(i)
            q_text = question['question']
            entities_list_with_locations = self.getEntitiesLocations(question)
            entities_list_with_locations.sort()
            entities = [id for location, id in
                        entities_list_with_locations]
            if len(entities) == 0:
                head = 0
                tail = 0
            else:
                head = entities[0]  # take an entity
                if len(entities) > 1:
                    tail = entities[1]
                else:
                    tail = entities[0]
            times_in_question = question['time']
            if len(times_in_question) > 0:
                time = self.timesToIds(times_in_question)
            else:
                time = [0]

            time = [int(x) + self.num_total_entities for x in time]
            heads.append(int(head))
            times.append(time)
            tails.append(int(tail))
            question_text.append(q_text)

            # For test dataset
            if len(question['answers']) == 0:
                answers = [0]
            else:
                if question['answers'][0] in self.all_dicts['ent2id'].keys():
                    answers = self.entitiesToIds(question['answers'])
                else:
                    # adding num_total_entities to each time id
                    answers = [int(x) + self.num_total_entities for x in self.timesToIds(question['answers'])]
            answers_arr.append(answers)

        self.data = [self.data[idx] for idx in self.data_ids_filtered]
        return {'question_text': question_text,
                'head': heads,
                'tail': tails,
                'time': times,
                'answers_arr': answers_arr}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.prepared_data
        question_text = data['question_text'][index]
        head = data['head'][index]
        tail = data['tail'][index]
        time = data['time'][index]
        answers_arr = data['answers_arr'][index]
        answers_single = random.choice(answers_arr)
        return question_text, head, tail, time, answers_single

    def _collate_fn(self, items):
        batch_sentences = [item[0] for item in items]
        b = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        heads = torch.from_numpy(np.array([item[1] for item in items]))
        tails = torch.from_numpy(np.array([item[2] for item in items]))
        times = torch.from_numpy(np.array([item[3][0] for item in items]))
        answers_single = torch.from_numpy(np.array([item[4] for item in items]))
        return b['input_ids'], b['attention_mask'], heads, tails, times, answers_single
