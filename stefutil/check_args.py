"""
An easy, readable interface for checking string arguments as effectively enums

Intended for high-level arguments instead of actual data processing as not as efficient
"""


from typing import List

from stefutil.prettier import logi


__all__ = ['CheckArg', 'ca']


class CheckArg:
    """
    Raise errors when common arguments don't match the expected values
    """

    @staticmethod
    def check_mismatch(arg_type: str, arg_value: str, expected_values: List[str]):
        if arg_value not in expected_values:
            raise ValueError(f'Unexpected {logi(arg_type)}: '
                             f'expect one of {logi(expected_values)}, got {logi(arg_value)}')

    def __init__(self):
        self.d_name2func = dict()

    def __call__(self, **kwargs):
        for k in kwargs:
            self.d_name2func[k](kwargs[k])

    def cache_mismatch(self, display_name: str, attr_name: str, accepted_values: List[str]):
        self.d_name2func[attr_name] = lambda x: CheckArg.check_mismatch(display_name, x, accepted_values)


ca = CheckArg()


if __name__ == '__main__':
    ca.cache_mismatch(
        'Bar Plot Orientation', attr_name='bar_orient', accepted_values=['v', 'h', 'vertical', 'horizontal']
    )

    ori = 'v'
    ca(bar_orient=ori)
