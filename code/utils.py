import pickle
import torch
import numpy as np
import json
from tcomplex import TComplEx
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from collections import defaultdict


def loadTkbcModel(tkbc_model_file):
    print('Loading tkbc model from', tkbc_model_file)
    x = torch.load(tkbc_model_file, map_location=torch.device("cpu"))
    num_ent = x['embeddings.0.weight'].shape[0]
    num_rel = x['embeddings.1.weight'].shape[0]
    num_ts = x['embeddings.2.weight'].shape[0]
    print('Number ent,rel,ts from loaded model:', num_ent, num_rel, num_ts)
    sizes = (num_ent, num_rel, num_ent, num_ts)
    rank = x['embeddings.0.weight'].shape[1] // 2  # complex has 2*rank embedding size
    tkbc_model = TComplEx(sizes, rank, no_time_emb=False)
    tkbc_model.load_state_dict(x)
    tkbc_model.cuda()
    print('Loaded tkbc model')
    return tkbc_model

def getAllDicts():
    def readDict(filename):
        f = open(filename, 'r', encoding='utf-8')
        d = {}
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 1:
                line.append('')  # in case literal was blank or whitespace
            d[line[0]] = int(line[1])
        f.close()
        return d

    def getReverseDict(d):
        return {value: key for key, value in d.items()}

    base_path = '../data/kg/tkbc_processed_data/'
    ent2id = readDict(base_path + '/ent_id')
    rel2id = readDict(base_path + '/rel_id')
    ts2id = readDict(base_path + '/ts_id')

    id2rel = getReverseDict(rel2id)
    id2ent = getReverseDict(ent2id)
    id2ts = getReverseDict(ts2id)
    all_dicts = {'rel2id': rel2id,
                 'id2rel': id2rel,
                 'ent2id': ent2id,
                 'id2ent': id2ent,
                 'ts2id': ts2id,
                 'id2ts': id2ts}
    return all_dicts


def save_model(qa_model, filename):
    print('Saving model to', filename)
    torch.save(qa_model.state_dict(), filename)
    print('Saved model to ', filename)
    return


def print_info(args):
    print('#######################')
    print('Model: BERT')
    print('TKG Embeddings: ' + args.tkbc_model_file)
    print('TKG for QA (if applicable): ' + args.tkg_file)
    print('#######################')


def predict(qa_model,dataset,batch_size=100):
    qa_model.eval()
    num_workers = 1
    max_k = 10
    print('Start Predicting')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=dataset._collate_fn)
    topk_answers = []
    loader = tqdm(data_loader, total=len(data_loader), unit="batches")

    for i_batch, a in enumerate(loader):
        if i_batch * batch_size == len(dataset.data):
            break
        scores = qa_model.forward(a)
        for s in scores:
            pred = dataset.getAnswersFromScores(s, k=max_k)
            topk_answers.append(pred)
    submit = []
    for i,q in enumerate(dataset.data):
        submit.append({'quid':q['quid'],'answers':topk_answers[i]})
    with open('./submit/submit.json','w',encoding = 'utf-8') as f:
        json.dump(submit,f,indent = 4)

def eval(qa_model, dataset, batch_size=128, split='valid', k=10):
    num_workers = 1
    qa_model.eval()
    eval_log = []
    print_numbers_only = False
    k_for_reporting = k  # not change name in fn signature since named param used in places
    k_list = [1, 3, 10]
    max_k = max(k_list)
    eval_log.append("Split %s" % (split))
    print('Evaluating split', split)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=dataset._collate_fn)
    topk_answers = []
    total_loss = 0
    loader = tqdm(data_loader, total=len(data_loader), unit="batches")

    for i_batch, a in enumerate(loader):
        # if size of split is multiple of batch size, we need this
        # todo: is there a more elegant way?
        if i_batch * batch_size == len(dataset.data):
            break
        answers_khot = a[-1]  # last one assumed to be target
        scores = qa_model.forward(a)
        for s in scores:
            pred = dataset.getAnswersFromScores(s, k=max_k)
            topk_answers.append(pred)
        loss = qa_model.loss(scores, answers_khot.long().cuda())
        total_loss += loss.item()
    eval_log.append('Loss %f' % total_loss)
    eval_log.append('Eval batch size %d' % batch_size)

    eval_accuracy_for_reporting = 0
    for k in k_list:
        hits_at_k = 0
        total = 0
        for i, question in enumerate(dataset.data):
            actual_answers = question['answers']
            predicted = topk_answers[i][:k]
            if len(set(actual_answers).intersection(set(predicted))) > 0:
                hits_at_k += 1
            total += 1

        eval_accuracy = hits_at_k / total
        if k == k_for_reporting:
            eval_accuracy_for_reporting = eval_accuracy
        if not print_numbers_only:
            eval_log.append('Hits at %d: %f' % (k, round(eval_accuracy, 3)))
        else:
            eval_log.append(str(round(eval_accuracy, 3)))

    # print eval log as well as return it
    for s in eval_log:
        print(s)
    return eval_accuracy_for_reporting, eval_log


def append_log_to_file(eval_log, epoch, filename):
    f = open(filename, 'a+', encoding='utf-8')
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    f.write('Log time: %s\n' % dt_string)
    f.write('Epoch %d\n' % epoch)
    for line in eval_log:
        f.write('%s\n' % line)
    f.write('\n')
    f.close()


def train(qa_model, dataset, valid_dataset, args, result_filename=None):
    num_workers = 5
    optimizer = torch.optim.Adam(qa_model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    batch_size = args.batch_size
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             collate_fn=dataset._collate_fn)
    max_eval_score = 0
    if args.save_to == '':
        args.save_to = 'temp'
    if result_filename is None:
        result_filename = 'results/{model_file}.log'.format(
            dataset_name=args.dataset_name,
            model_file=args.save_to
        )
    checkpoint_file_name = 'models/qa_models/{model_file}.ckpt'.format(
        model_file=args.save_to
    )

    if args.load_from == '':
        print('Creating new log file')
        f = open(result_filename, 'a+', encoding='utf-8')
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('Log time: %s\n' % dt_string)
        f.write('Config: \n')
        for key, value in vars(args).items():
            key = str(key)
            value = str(value)
            f.write('%s:\t%s\n' % (key, value))
        f.write('\n')
        f.close()

    max_eval_score = 0.

    print('Starting training')
    for epoch in range(args.max_epochs):
        qa_model.train()
        epoch_loss = 0
        loader = tqdm(data_loader, total=len(data_loader), unit="batches")
        running_loss = 0
        for i_batch, a in enumerate(loader):
            qa_model.zero_grad()
            answers_khot = a[-1]  # last one assumed to be target
            scores = qa_model.forward(a)

            loss = qa_model.loss(scores, answers_khot.long().cuda())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            running_loss += loss.item()
            loader.set_postfix(Loss=running_loss / ((i_batch + 1) * batch_size), Epoch=epoch)
            loader.set_description('{}/{}'.format(epoch, args.max_epochs))
            loader.update()

        print('Epoch loss = ', epoch_loss)
        if (epoch + 1) % args.valid_freq == 0:
            print('Starting eval')
            eval_score, eval_log = eval(qa_model, valid_dataset, batch_size=args.valid_batch_size,
                                        split='valid')
            if eval_score > max_eval_score:
                print('Valid score increased')
                save_model(qa_model, checkpoint_file_name)
                max_eval_score = eval_score
            append_log_to_file(eval_log, epoch, result_filename)
