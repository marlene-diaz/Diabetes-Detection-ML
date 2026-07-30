import unittest

from streamlit.testing.v1 import AppTest


class StreamlitAppTests(unittest.TestCase):
    def test_app_loads_and_generates_a_result(self):
        app = AppTest.from_file("app.py")
        app.run(timeout=30)
        self.assertEqual(list(app.exception), [])

        app.button[0].click()
        app.run(timeout=30)

        self.assertEqual(list(app.exception), [])
        self.assertTrue(
            any(metric.label == "Model-estimated likelihood" for metric in app.metric)
        )


if __name__ == "__main__":
    unittest.main()
