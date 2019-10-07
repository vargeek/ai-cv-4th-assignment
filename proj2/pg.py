# %%
import argparse
parser = argparse.ArgumentParser(description='xx')
# for param in params:
#     args, kwargs = param
parser.add_argument('--foo', type=str)


# print(parser.parse_args(args=[]))
# print(parser.parse_known_args(args=[]))
# print(parser.parse_known_intermixed_args())
# print(parser.parse_known_args())

# hasattr()
# getattr()
# delattr()


# def _get_args_parser():
#     import argparse
#     parser = argparse.ArgumentParser(description='logloader')
#     # for param in params:
#     #     args, kwargs = param
#     #     parser.add_argument(*args, **kwargs)

#     # args = parser.parse_args()
#     # pre_process_args(args)

#     return parser


# _parser = _get_args_parser()
# _parser.parse_args([])

def foo(args=parser.parse_args([])):
    print(args)
    args.foo = 100
    print(args)


print(foo())


# %%
