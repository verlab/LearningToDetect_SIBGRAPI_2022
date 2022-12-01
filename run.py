# TESTES INICIAIS

import utils.train as t
from models.our_detector import Our
import losses
import torch
import argparse

from NRDataset import NRDataset

def parseArg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--h5path", help="dataset .h5 five path"
    , required=True) 
    parser.add_argument("-o", "--output", help="Output path and name to save .pth final file."
    , required=True) 
    parser.add_argument("--epochs", help="Number of epochs.", type=int
    , required=False, default = 10) 
    parser.add_argument("--batch", help="Batch size.", type=int
    , required=False, default = 8) 
    parser.add_argument("--lr", help="Learning rate.", type=float
    , required=False, default = 1e-4) 
    parser.add_argument("--chunk", help="Chunk size.", type=int
    , required=False, default = 10)
    parser.add_argument("--passes", help="Passes.", type=int
    , required=False, default = 1)
    
    parser.add_argument("--is_testing", help="Run with 2 folders to test the code"
    , action = 'store_true')

    args = parser.parse_args()
    return args

# python3 run.py -i test_dataset.h5 -o ./test --chunk 2 --passes 2 --batch 8 --epochs 10 --lr 1e-3
def main():
    global args
    args = parseArg()
    inputFile = args.h5path
    output = args.output

    dataset = NRDataset(inputFile, batch_size = args.batch, chunk_size = args.chunk, passes=args.passes)
    net = Our()
    t.train(net, dataset, nepochs=args.epochs, lr=args.lr, resume = None)

    torch.save(net.state_dict(), output+'_model.pth' )    

main()