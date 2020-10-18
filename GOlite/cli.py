"""Console script for GOlite."""
import argparse
import sys


def main():
    """Console script for GOlite."""
    parser = argparse.ArgumentParser()
    parser.add_argument('_', nargs='*')
    parser.add_argument("-f", "--filters", default="512",
                        help="filters count, default is 512")
    parser.add_argument("-s", "--filterSize", default="8,128,8",
                        help="start fiter size, end filter size, step size. \
                        Defualt is 8,128,8")
    args = parser.parse_args()

    print("Arguments: " + str(args._))
    print("Replace this message by putting your code into "
          "GOlite.cli.main")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
