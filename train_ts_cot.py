
import torch
import numpy as np
import argparse
import os
from datetime import datetime
from ts_cot_model import TS_CoT
import tasks
import datautils
from utils import init_cuda, load_config
import random



def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True,
                        help='The experimental dataset to be used: HAR, Epi, SleepEDF, Waveform.')
    parser.add_argument('--gpu', type=int, default=0, help='The experimental GPU index.')
    parser.add_argument('--max-threads', type=int, default=8, help='The maximum threads')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--repr-dims', type=int, default=32, help='Dimension of Representation')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs to be trained')
    parser.add_argument('--seed', type=int, default=1024, help='The random seed to be fixed')
    parser.add_argument('--eval', action="store_true", help='Set true for evaluation')
    parser.add_argument('--num-cluster', default='5', type=str, help='number of clusters')
    parser.add_argument('--temperature', default=0.1, type=float, help='softmax temperature of InfoNCE')
    parser.add_argument('--warmup', default=0.50, type=float, help='Warmup epoch before using co-training')
    parser.add_argument('--prototype-lambda', default=0.1, type=float, help='Prototypical loss scale adjustment')
    parser.add_argument('--eval_protocol', default='mlp', type=str, help='Classification backbone for downstreaming tasks.')
    parser.add_argument('--backbone_type', default='TS_CoT', type=str,
                        help='Which backone to use for representation learning. ')
    parser.add_argument('--dropmask', default=0.9, type=float, help='Masking ratio for augmentation')
    parser.add_argument('--model_path', default=None, type=str, help='The path of the model to be loaded')
    parser.add_argument('--ma_gamma', default=0.9999, type=float, help='The moving average parameter for prototype updating')

    args = parser.parse_args()



    device = init_cuda(args.gpu, seed=args.seed, max_threads=args.max_threads)
    print('Loading data... \n', end='')

    if args.backbone_type == 'TS_CoT':
        if args.dataset == 'HAR':
            train_data, train_labels, test_data, test_labels = datautils.load_HAR_two_view()

        if args.dataset == 'SleepEDF':
            train_data, train_labels, test_data, test_labels = datautils.load_EEG_two_view()
        if args.dataset == 'Epi':
            train_data, train_labels, test_data, test_labels = datautils.load_Epi_two_view()

        if args.dataset == 'Waveform':
            train_data, train_labels, test_data, test_labels = datautils.load_Waveform_two_view()
    else:
        print('Unknown Backbone')

    args = load_config(args.dataset, args)
    args.num_cluster = args.num_cluster.split(',')

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = 'save_dir/test' + '/' + args.dataset + '/' + args.backbone_type+ '/'+ now
    args.run_dir = run_dir
    os.makedirs(run_dir, exist_ok=True)

    if args.backbone_type == 'TS_CoT':
        if args.eval:
            args.epochs = 0
        model = TS_CoT(
            input_dims=train_data[0].shape[-1],
            output_dims=args.repr_dims,
            device=device,
            args=args
        )
        train_model = model.fit_ts_cot(
            train_data,
            n_epochs=args.epochs
        )
        model.save(f'{run_dir}/model.pkl')
    else:
        print('Unknown Backbone')

    if args.eval:
        if args.model_path:
            model.tem_encoder.load_state_dict(torch.load(args.model_path)['TemEncoder'])
            model.fre_encoder.load_state_dict(torch.load(args.model_path)['FreEncoder'])
            print('Pre-trained Model Loading...')
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels,
                                                      eval_protocol=args.eval_protocol, args = args)
            print('Evaluation result: ACC:', eval_res['acc'], '  AUROC:', eval_res['auroc'])
        else:
            print('No loaded pre-trained model.')



    print("Finished.")



