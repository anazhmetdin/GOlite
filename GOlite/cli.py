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
    parser.add_argument("-i", "--dDim", default="50,29000", required=True,
                        help="data dimension; default is 50,29000")
    parser.add_argument("-I", "--lDim", default="50,1000", required=True,
                        help="label dimension; Defualt is 50,1000")
    parser.add_argument("-b", "--batchSize", default="32",
                        help="batch size of the generator")
    parser.add_argument("-v", "--validation", default="0.2",
                        help="validation percentage of the data")
    parser.add_argument("-e", "--epochs", default="13",
                        help="number of ebochs")
    parser.add_argument("-t", "--trainingSize", default="1",
                        help="training size of each batch")
    parser.add_argument("-g", "--generator", default="0",
                        help="use generator for loading data;\
                        0: (default) False, 1: True")
    parser.add_argument("-m", "--model", default="CN",
                        help="what model structure to be used; CN, DN")
    parser.add_argument("-p", "--DNdepth", default="121",
                        help="The depth of DenseNet")
    parser.add_argument("-o", "--Oprefix", default="./",
                        help="number of parameters in case of DN model")
    parser.add_argument("-M", "--modelPrefix", default="",
                        help="file path prefix of a model to be loaded")
    parser.add_argument("-P", "--predict", default="",
                        help="file path of encoded proteins")
    parser.add_argument("-k", "--topK", default="50",
                        help="top k predictions")
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
    model = args.model
    params = args.DNdepth
    Oprefix = args.Oprefix
    modelPrefix = args.modelPrefix
    predict = args.predict
    k = int(args.topK)

    CNN = CNNmodel(dPrefix, lPrefix, dDim, lDim, validation,
                   filters, filterSize, model, params, modelPrefix)
    if generator:
        CNN.fit_model_generator(batchSize, epochs, trainSize)
    elif predict == "":
        CNN.fit_model_bitByBit(Oprefix, batchSize, epochs, trainSize)
    elif predict != "":
        CNN.predict(predict, Oprefix, k)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
