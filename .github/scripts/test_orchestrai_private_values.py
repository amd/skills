"""Private provisioning inputs must be masked at the test-log boundary."""

import unittest

from orchestrai_logs import private_log_values
from orchestrai_stdout import sanitize_stream


class DriverSourceRedactionTests(unittest.TestCase):
    def test_windows_source_is_private_from_environment_or_run_payload(self):
        source = r"\\fixture-host\fixture-share\driver-package"
        inputs = (
            ({}, {"ORCHESTRAI_WINDOWS_DRIVER_SOURCE": source}),
            ({"builds_json": {"vars": {"driver_source": source}}}, {}),
        )
        for plan, environ in inputs:
            with self.subTest(plan=bool(plan)):
                private = private_log_values(plan, {}, environ)
                self.assertIn(source, private)
                rendered = sanitize_stream("Copying " + source, private)
                self.assertNotIn("fixture-host", rendered)
                self.assertNotIn("driver-package", rendered)


if __name__ == "__main__":
    unittest.main()
