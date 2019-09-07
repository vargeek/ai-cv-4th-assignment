import time
import math
from uuid import uuid4

# int(math.ceil(math.log(2 ** 128, len_alphabet)))
# int(math.ceil(math.log(2 ** 128, 56))) -> 23
# int(math.ceil(math.log(2 ** 128, 57))) -> 22
# 所以选择57个字符
_alphabet = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "abcdefghijklmnopqrstuvwxyz" "01234")


class ShortUUID():
    def __init__(self, alphabet=_alphabet):
        self._alphabet = _alphabet
        len_alphabet = len(_alphabet)
        assert(len_alphabet > 1)
        self._uuid_len = int(math.ceil(math.log(2 ** 128, len_alphabet)))

    def uuid(self):
        numerator = uuid4().int
        output = ''
        len_alphabet = len(self._alphabet)
        while numerator:
            numerator, remainder = divmod(numerator, len_alphabet)
            output += self._alphabet[remainder]

        padding = max(self._uuid_len - len(output), 0)
        output = output + self._alphabet[0] * padding

        return output


_default_uuid = None


def uuid():
    global _default_uuid
    if _default_uuid is None:
        _default_uuid = ShortUUID()
    return _default_uuid.uuid()
