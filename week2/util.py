import os
import sys


def getAssetDir():
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(os.path.dirname(proj_dir), 'assets')
    return assets_dir


def getLennaFilepath():
    return os.path.join(getAssetDir(), 'lenna.jpg')


def getLenna2Filepath():
    return os.path.join(getAssetDir(), 'lenna2.jpg')


def getImagePath():
    filepath = len(sys.argv) > 1 and sys.argv[1] or None
    if not filepath:
        filepath = os.path.join(getAssetDir(), 'lenna.jpg')

    return filepath
