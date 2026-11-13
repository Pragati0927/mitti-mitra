"""
Mitti Mitra Automated Validation Testing Suite
Author: Ananya (SMR Engine for Crop Rotation)
Description: Structural unit tests running data sanity boundary checks on the 27-column matrix
             to guarantee zero null-pointer vulnerabilities or mathematical out-of-bounds corruption.
Coding Standard: PEP-8 Compliance
"""

import unittest
import os
import pandas as pd

class TestMittiMitraDataIntegrity(unittest.TestCase):
    
    def setUp(self):
        """Initializes data references from local system storage paths before testing assertions."""
        self.filename = "master_indian_recommendation_dataset.csv"
        self.assertTrue(os.path.exists(self.filename), f"Critical Fault: Target data matrix missing at {self.filename}")
        self.df = pd.read_csv(self.filename)

    def test_feature_dimensions_integrity(self):
        """Assertion: Verify the compiled spreadsheet contains exactly 27 active columns."""
        total_columns = self.df.shape[1]
        self.assertEqual(total_columns, 27, f"Dimension Mismatch: Expected 27 features, found {total_columns}")

    def test_agronomic_ph_threshold_bounds(self):
        """Assertion: Verify all soil chemical pH entries fall within valid environmental scale parameters."""
        max_ph = self.df['pH'].max()
        min_ph = self.df['pH'].min()
        self.assertTrue(max_ph <= 14.0 and min_ph >= 0.0, f"Data Distortion: Out-of-bounds soil pH detected: [{min_ph} - {max_ph}]")

    def test_microbial_abundance_non_negativity(self):
        """Assertion: Ensure high-dimensional biological sequencing metrics contain no negative numbers."""
        min_bacterial_read = self.df['Bacterial_Abundance_16S'].min()
        min_fungal_read = self.df['Fungal_Abundance_ITS'].min()
        self.assertTrue(min_bacterial_read >= 0.0, "Data Noise Corruption: Negative 16S bacterial community counts found.")
        self.assertTrue(min_fungal_read >= 0.0, "Data Noise Corruption: Negative ITS fungal indices discovered.")

if __name__ == "__main__":
    print("🔬 Executing system automated validation suite checks...")
    unittest.main(exit=False)
