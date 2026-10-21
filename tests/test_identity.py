"""Small regressions for identity merges that change which evidence can be recommended."""

import unittest

import pandas as pd

from landscape.identity import work_aliases


class WorkIdentityTest(unittest.TestCase):
    def aliases(self, records, authors):
        return work_aliases(pd.DataFrame(records, columns=["paper_id", "title", "year", "doi"]),
                            pd.DataFrame(authors, columns=["paper_id", "author_id"]))

    def test_preprint_and_article_share_work(self):
        a = self.aliases([
            ("journal", "A robust method for decisions under uncertainty", 2020, None),
            ("preprint", "A ROBUST METHOD for decisions under uncertainty!", 2018, None)],
            [("journal", "A1"), ("preprint", "A1")])
        self.assertEqual(a, {"journal": "preprint", "preprint": "preprint"})

    def test_equal_titles_with_disjoint_authors_stay_separate(self):
        a = self.aliases([
            ("one", "A robust method for decisions under uncertainty", 2018, None),
            ("two", "A robust method for decisions under uncertainty", 2020, None)],
            [("one", "A1"), ("two", "A2")])
        self.assertNotEqual(a["one"], a["two"])

    def test_equal_doi_does_not_require_equal_title(self):
        a = self.aliases([("one", "Old title", 2018, "https://doi.org/10.1234/TEST"),
                          ("two", "Revised title", 2020, "10.1234/test")], [])
        self.assertEqual(a["one"], a["two"])

    def test_generic_titles_are_not_merged(self):
        a = self.aliases([("one", "Introduction", 2018, None), ("two", "Introduction", 2020, None)],
                         [("one", "A1"), ("two", "A1")])
        self.assertNotEqual(a["one"], a["two"])

    def test_representative_uses_only_supplied_cutoff_records(self):
        a = self.aliases([("known", "A robust method for decisions under uncertainty", 2018, None)],
                         [("known", "A1"), ("future", "A1")])
        self.assertEqual(a, {"known": "known"})


if __name__ == "__main__":
    unittest.main()
