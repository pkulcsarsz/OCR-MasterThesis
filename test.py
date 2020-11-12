import sys, getopt

opts, args = getopt.getopt(sys.argv[1:], 'm:')
for opt, arg in opts:
    if opt == '-m':
        print("model", arg)
        print(arg)

