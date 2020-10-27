"""Console script for GOlite."""
import argparse
import sys
from GOlite.CNNmodel import CNNmodel

def main():
    """Console script for GOlite."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filters", default="512",
                        help="filters count, default is 512")
    parser.add_argument("-s", "--filterSize", default="8,128,8",
                        help="start fiter size, end filter size, step size. \
                        Defualt is 8,128,8")
    parser.add_argument("-d", "--dPrefix", required=True,
                        help="files name pattern that only includes the data")
    parser.add_argument("-l", "--lPrefix", required=True,
                        help="files name pattern that only includes the labels")
    parser.add_argument("-i", "--dDim", default="1000,58000", required=True,
                        help="data dimension; default is 1000,58000")
    parser.add_argument("-I", "--lDim", default="1000,500", required=True,
                        help="label dimension; Defualt is 1000,500")
    parser.add_argument("-b", "--batchSize", default="32",
                        help="batch size of the generator")
    parser.add_argument("-v", "--validation", default="0.2",
                        help="validation percentage of the data")
    parser.add_argument("-e", "--epochs", default="13",
                        help="number of ebochs")
    parser.add_argument("-t", "--trainingSize", default="1",
                        help="training size of each batch")
    parser.add_argument("-g", "--generator", default="1",
                        help="use generator for loading data;\
                        0: False, 1: (default) True")
    args = parser.parse_args()

    filters = int(args.filters)
    filterSize = args.filterSize
    dPrefix = args.dPrefix
    lPrefix = args.lPrefix
    dDim = args.dDim
    lDim = args.lDim
    batchSize = int(args.batchSize)
    validation = float(args.validation)
    epochs = int(args.epochs)
    trainSize = float(args.trainingSize)
    generator = bool(int(args.generator))

    CNN = CNNmodel(dPrefix, lPrefix, dDim, lDim, batchSize, validation,
                   filters, filterSize)
    if generator:
        CNN.fit_model_generator(epochs, trainSize)
    else:
        CNN.fit_model_bitByBit(epochs, trainSize)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
