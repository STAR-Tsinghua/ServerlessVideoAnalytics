from parser import parse_args
from run import run



def main():
    args = parse_args()
    run(args)

if __name__ == '__main__':
    main()