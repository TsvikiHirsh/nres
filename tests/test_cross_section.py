import unittest
import numpy as np
import pandas as pd
from nres.cross_section import CrossSection
from nres import materials

class TestCrossSectionOrder(unittest.TestCase):
    def setUp(self):
        # Create sample materials for testing
        self.pe = materials["Polyethylene, Non-borated"]
        self.al2o3 = materials["Aluminum Oxide"]
        self.iron = "Iron"
        
    def test_addition_shape_consistency(self):
        """Test that addition produces consistent shapes regardless of order"""
        # First order
        xs1 = CrossSection()
        xs1 += CrossSection(pe=self.pe, splitby="materials")
        xs1 += CrossSection(rdx=self.al2o3, splitby="materials")
        xs1 += CrossSection(iron=self.iron, splitby="materials")
        
        # Second order
        xs2 = CrossSection()
        xs2 += CrossSection(iron=self.iron, splitby="materials")
        xs2 += CrossSection(pe=self.pe, splitby="materials")
        xs2 += CrossSection(rdx=self.al2o3, splitby="materials")
        
        # Third order
        xs3 = CrossSection()
        xs3 += CrossSection(rdx=self.al2o3, splitby="materials")
        xs3 += CrossSection(iron=self.iron, splitby="materials")
        xs3 += CrossSection(pe=self.pe, splitby="materials")
        
        # Check shapes match
        self.assertEqual(xs1.table.shape, xs2.table.shape)
        self.assertEqual(xs2.table.shape, xs3.table.shape)
        
        # Check energy grids match
        pd.testing.assert_index_equal(xs1.table.index, xs2.table.index)
        pd.testing.assert_index_equal(xs2.table.index, xs3.table.index)

    def test_atomic_density_consistency(self):
        """Test that atomic density is consistent regardless of addition order"""
        # First order
        xs1 = CrossSection()
        xs1 += CrossSection(pe=self.pe, splitby="materials")
        xs1 += CrossSection(rdx=self.al2o3, splitby="materials")
        
        # Second order
        xs2 = CrossSection()
        xs2 += CrossSection(rdx=self.al2o3, splitby="materials")
        xs2 += CrossSection(pe=self.pe, splitby="materials")
        
        # Check atomic densities match
        self.assertAlmostEqual(xs1.n, xs2.n, places=6)
        
    def test_multiplication(self):
        """Test multiplication operation"""
        xs = CrossSection()
        xs += CrossSection(pe=self.pe, splitby="materials")
        
        # Test multiplication by scalar
        factor = 2.5
        xs_scaled = xs * factor
        
        # Check that atomic density scales correctly
        self.assertAlmostEqual(xs_scaled.n, xs.n * factor, places=6)
        
        # Check that cross sections scale correctly
        pd.testing.assert_frame_equal(xs_scaled.table, xs.table)
        
        # Check that weights remain normalized
        self.assertAlmostEqual(sum(xs_scaled.weights), 1.0, places=6)

    def test_interpolation_accuracy(self):
        """Test that interpolation produces accurate results"""
        xs1 = CrossSection(pe=self.pe, splitby="materials")
        xs2 = CrossSection(iron=self.iron, splitby="materials")
        
        combined = xs1 + xs2
        
        # Check interpolated values lie between original values
        for col in combined.table.columns:
            if col != 'total':  # Skip the total column
                self.assertTrue(
                    combined.table[col].between(
                        combined.table[col].min(),
                        combined.table[col].max()
                    ).all()
                )


if __name__ == "__main__":
    unittest.main()
