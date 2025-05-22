from unittest import TestCase
import polars as pl
from thesis.model.enums import Relative_Position

class TestEnums(TestCase):
    
    def test_relative_position_order(self):
        """
        Ensure that ``Back`` has a higher value than ``Front``.
        """
        expected = ["Front", "Front", "Front", "Back", "Back"]
        unsorted = ["Back", "Front", "Front", "Back", "Front"]
        unsorted_series = pl.Series(unsorted, dtype=Relative_Position)
        sorted_series = unsorted_series.sort()
        self.assertEqual(expected, sorted_series.to_list())