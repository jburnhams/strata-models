import sys

from stratamodels.cli import main

if __name__ == "__main__":  # not when a worker process (fontfit's) imports this as its parent's main module
    sys.exit(main())
