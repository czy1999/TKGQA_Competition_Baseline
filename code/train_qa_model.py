import argparse
import torch
from qa_baselines import QA_baseline
from qa_datasets import QA_Dataset_Baseline
from utils import loadTkbcModel, train, eval, print_info,predict


parser = argparse.ArgumentParser(
    description="TKGQA Baseline for bigdata competition"
)

parser.add_argument(
    '--tkbc_model_file', default='tcomplex.ckpt', type=str,
    help="Pretrained tkbc model checkpoint enhanced_icews.ckpt"
)

parser.add_argument(
    '--tkg_file', default='full.txt', type=str,
    help="TKG to use for hard-supervision"
)

parser.add_argument(
    '--load_from', default='', type=str,
    help="Pretrained qa model checkpoint"
)

parser.add_argument(
    '--save_to', default='', type=str,
    help="Where to save checkpoint."
)

parser.add_argument(
    '--max_epochs', default=5, type=int,
    help="Number of epochs."
)

parser.add_argument(
    '--valid_freq', default=1, type=int,
    help="Number of epochs between each valid."
)

parser.add_argument(
    '--batch_size', default=200, type=int,
    help="Batch size."
)

parser.add_argument(
    '--valid_batch_size', default=50, type=int,
    help="Valid batch size."
)

parser.add_argument(
    '--lr', default=1e-3, type=float,
    help="Learning rate"
)

parser.add_argument(
    '--mode', default='train', type=str,
    help="Whether train or eval."
)


if __name__ == "__main__":
    args = parser.parse_args()
    print_info(args)

    tkbc_model = loadTkbcModel('models/kg_embeddings/{tkbc_model_file}'.format(tkbc_model_file=args.tkbc_model_file))
    qa_model = QA_baseline(tkbc_model, args)
    dataset = QA_Dataset_Baseline(split='train')
    valid_dataset = QA_Dataset_Baseline(split='dev')
    test_dataset = QA_Dataset_Baseline(split='test')

    if args.load_from != '':
        filename = 'models/qa_models/{model_file}.ckpt'.format(model_file=args.load_from)
        print('Loading model from', filename)
        qa_model.load_state_dict(torch.load(filename))
        print('Loaded qa model from ', filename)
    else:
        print('Not loading from checkpoint. Starting fresh!')

    qa_model = qa_model.cuda()
    if args.mode == 'eval':
        score, log = eval(qa_model, valid_dataset, batch_size=args.valid_batch_size, split='valid')
        exit(0)
    elif args.mode == 'test':
        predict(qa_model, test_dataset, batch_size=args.valid_batch_size)
        exit(0)
    result_filename = 'results/{model_file}.log'.format(model_file=args.save_to)
    train(qa_model, dataset, valid_dataset, args, result_filename=result_filename)

    filename = 'models/qa_models/{model_file}.ckpt'.format(model_file=args.save_to)
    print('Loading best model from', filename)
    qa_model.load_state_dict(torch.load(filename))
    predict(qa_model, test_dataset, batch_size=args.valid_batch_size)
    print('Predict finished, file saved at submit/submit.json')
