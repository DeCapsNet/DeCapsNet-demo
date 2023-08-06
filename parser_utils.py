import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                        type=int,
                        help='random seed',
                        default=1328)
    parser.add_argument('--input_dir',
                        type=str,
                        default="../../data/depression/eRisk2018/processed/phq9_temp2_train_val_test_maxsim16",
                        help='path to dataset')
    parser.add_argument('--save_path',
                        type=str,
                        default="../../data/depression/eRisk2018/result/model.pth",
                        help='path to save model')
    parser.add_argument('--save_log_path',
                        type=str,
                        default="../../data/depression/eRisk2018/result/result.txt",
                        help='path to save log')
    parser.add_argument('--model_type',
                        type=str,
                        default="bert-base-uncased",
                        help='pretrained model')
    parser.add_argument('--train_batch_size',
                        type=int,
                        help='batch size of train dataset',
                        default=16)
    parser.add_argument('--test_batch_size',
                        type=int,
                        help='batch size of test dataset and dev dataset',
                        default=16)
    parser.add_argument('--max_len',
                        type=int,
                        help='posts',
                        default=128)
    parser.add_argument('--max_len_templates',
                        type=int,
                        help='posts',
                        default=32)
    parser.add_argument('--learning_rate',
                        type=float,
                        help='learning rate for the model',
                        default=2e-5)
    parser.add_argument('--alpha',
                        type=float,
                        help='coefficient of attention loss',
                        default=0.2)
    parser.add_argument('--beta',
                        type=float,
                        help='coefficient of user contrastive loss',
                        default=0.5)
    parser.add_argument('--gamma',
                        type=float,
                        help='coefficient of post contrastive loss',
                        default=0.2)
    parser.add_argument('--temperature1',
                        type=float,
                        help='temperature of user contrastive loss',
                        default=0.5)
    parser.add_argument('--temperature2',
                        type=float,
                        help='temperature of post contrastive loss',
                        default=0.5)
    parser.add_argument('--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=15)
    parser.add_argument('--num_routing',
                        type=int,
                        help='number of epochs to route',
                        default=3)
    parser.add_argument('--dropout',
                        type=float,
                        help='possibility of dropout',
                        default=0.3)
    parser.add_argument('--bert_dim',
                        type=int,
                        help='bert size',
                        default=768)
    parser.add_argument('--num_layer',
                        type=int,
                        help='bert layer',
                        default=12)
    parser.add_argument('--num_head',
                        type=int,
                        help='bert layer',
                        default=12)
    parser.add_argument('--num_symptoms',
                        type=int,
                        help='number of symptoms',
                        default=9)
    parser.add_argument('--output_dim',
                        type=int,
                        help='number of classes',
                        default=2)
    parser.add_argument('--output_atoms',
                        type=int,
                        help='number of output dims',
                        default=50)
    parser.add_argument('--save_loss',
                        type=bool,
                        help='save loss',
                        default=False)
    parser.add_argument('--save_embeddings',
                        type=bool,
                        help='save embeddings',
                        default=False)
    parser.add_argument('--save_ls_path',
                        type=str,
                        help='path to save loss')
    parser.add_argument('--save_embedding_path',
                        type=str,
                        help='path to save embedding pkl')
    parser.add_argument('--only_test',
                        type=bool,
                        help='test mode',
                        default=False)
    parser.add_argument('--local_rank',
                        type=int,
                        default=0)
    return parser
