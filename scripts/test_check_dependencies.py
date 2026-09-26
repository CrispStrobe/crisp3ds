import unittest

from check_dependencies import license_allowed, selected_license


class LicensePolicyTests(unittest.TestCase):
    def test_allowed_terms_and_combinations(self):
        for expression in ("MIT", "MPL-2.0", "Apache-2.0 OR MIT", "MIT/Apache-2.0",
                           "(BSD-3-Clause OR MIT) AND MPL-2.0", "Apache-2.0 WITH LLVM-exception",
                           "MIT OR LGPL-2.1-or-later"):
            with self.subTest(expression=expression):
                self.assertTrue(license_allowed(expression))

    def test_unknown_rejected_and_malformed_terms(self):
        for expression in (None, "", "GPL-3.0", "AGPL-3.0-only", "LGPL-2.1-only", "custom",
                           "MIT AND GPL-3.0", "MIT OR", "MIT Apache-2.0", "MIT WITH LLVM-exception"):
            with self.subTest(expression=expression):
                self.assertFalse(license_allowed(expression))

    def test_selected_license_branch_is_explicit(self):
        self.assertEqual(selected_license("MIT OR LGPL-2.1-or-later"), "MIT")
        self.assertEqual(selected_license("GPL-3.0 OR Apache-2.0"), "Apache-2.0")


if __name__ == "__main__":
    unittest.main()
