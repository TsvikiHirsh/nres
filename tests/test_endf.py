import unittest
import os
import pandas as pd
from your_script_name import get_xs, _check_endf_file_exists

class TestENDFFetcher(unittest.TestCase):
    def setUp(self):
        """
        Set up the directory and file paths for testing.
        """
        self.isotope = "Mn-55"
        self.energies = [1, 10, 100]  # Energies in eV
        self.endf_directory = "endf_database"
        self.endf_file = os.path.join(self.endf_directory, f"{self.isotope}.dat")

    def tearDown(self):
        """
        Clean up by removing downloaded and extracted files.
        """
        if os.path.exists(self.endf_file):
            os.remove(self.endf_file)
        if os.path.exists(self.endf_directory) and not os.listdir(self.endf_directory):
            os.rmdir(self.endf_directory)

    def test_fetch_new_endf_file(self):
        """
        Test fetching a new ENDF file for a specific isotope (e.g., "Mn-55").
        """
        if os.path.exists(self.endf_file):
            os.remove(self.endf_file)
        
        xs = get_xs(self.isotope)
        
        # Verify that the ENDF file was downloaded and extracted correctly
        self.assertTrue(os.path.exists(self.endf_file))
        self.assertIsInstance(xs, pd.Series)
        self.assertFalse(xs.empty)

    def test_fetch_existing_endf_file(self):
        """
        Test fetching an existing ENDF file for a specific isotope.
        """
        # Ensure the file is already downloaded
        get_xs(self.isotope)
        
        xs = get_xs(self.isotope)
        
        # Verify that the file is reused and no additional download occurs
        self.assertTrue(os.path.exists(self.endf_file))
        self.assertIsInstance(xs, pd.Series)
        self.assertFalse(xs.empty)

    def test_interpolated_energies(self):
        """s
        Test fetching cross sections with specific energy values for interpolation.
        """
        xs = get_xs(self.isotope, energies=self.energies)
        
        # Verify the result is a pandas.Series and matches the expected energies
        self.assertIsInstance(xs, pd.Series)
        self.assertEqual(xs.index.tolist(), self.energies)
        self.assertFalse(xs.empty)
        self.assertEqual(len(xs), len(self.energies))

if __name__ == "__main__":
    unittest.main()
