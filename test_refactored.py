#!/usr/bin/env python3
"""
Test script to verify that the refactored potts_shunt_test.py still works correctly.

This script tests the helper classes and verifies that the UI components can be instantiated
without errors.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

# Set up Qt environment before importing anything else
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Use offscreen platform for testing


class TestSliderMapper(unittest.TestCase):
    """Test the SliderMapper helper class"""

    def setUp(self):
        """Import SliderMapper for testing"""
        from main import SliderMapper

        self.SliderMapper = SliderMapper

    def test_force_scale_mapping(self):
        """Test force scale slider value mapping"""
        # Test zero value
        self.assertEqual(self.SliderMapper.force_scale_slider_to_value(0), 0.0)

        # Test positive value
        self.assertEqual(self.SliderMapper.force_scale_slider_to_value(500), 0.5)

        # Test negative value
        self.assertEqual(self.SliderMapper.force_scale_slider_to_value(-500), -0.5)

        # Test round trip conversion
        original_value = 0.75
        slider_val = self.SliderMapper.force_scale_value_to_slider(original_value)
        converted_back = self.SliderMapper.force_scale_slider_to_value(slider_val)
        self.assertAlmostEqual(original_value, converted_back, places=3)

    def test_stent_diameter_mapping(self):
        """Test stent diameter value mapping"""
        # Test minimum value
        min_slider = self.SliderMapper.stent_diameter_value_to_slider(0.1)
        self.assertEqual(min_slider, 0)

        # Test maximum value
        max_slider = self.SliderMapper.stent_diameter_value_to_slider(1.0)
        self.assertEqual(max_slider, 900)

        # Test clamping - values below minimum
        clamped_slider = self.SliderMapper.stent_diameter_value_to_slider(0.05)
        self.assertEqual(clamped_slider, 0)

        # Test clamping - values above maximum
        clamped_slider = self.SliderMapper.stent_diameter_value_to_slider(2.0)
        self.assertEqual(clamped_slider, 900)

        # Test round trip conversion
        original_value = 0.5
        slider_val = self.SliderMapper.stent_diameter_value_to_slider(original_value)
        converted_back = self.SliderMapper.stent_diameter_slider_to_value(slider_val)
        self.assertAlmostEqual(original_value, converted_back, places=3)

    def test_stent_length_mapping(self):
        """Test stent length value mapping"""
        # Test minimum value
        min_slider = self.SliderMapper.stent_length_value_to_slider(2.0)
        self.assertEqual(min_slider, 0)

        # Test maximum value
        max_slider = self.SliderMapper.stent_length_value_to_slider(8.0)
        self.assertEqual(max_slider, 24)

        # Test round trip conversion
        original_value = 5.0
        slider_val = self.SliderMapper.stent_length_value_to_slider(original_value)
        converted_back = self.SliderMapper.stent_length_slider_to_value(slider_val)
        self.assertAlmostEqual(original_value, converted_back, places=3)


class TestUIStyleManager(unittest.TestCase):
    """Test the UIStyleManager helper class"""

    def setUp(self):
        """Import UIStyleManager for testing"""
        from main import UIStyleManager

        self.UIStyleManager = UIStyleManager

    def test_set_button_active(self):
        """Test setting button active state"""
        # Mock button
        mock_button = MagicMock()

        # Test active state
        self.UIStyleManager.set_button_active(mock_button, True)
        mock_button.setStyleSheet.assert_called_with("background-color: #d84005")

        # Test inactive state
        self.UIStyleManager.set_button_active(mock_button, False)
        mock_button.setStyleSheet.assert_called_with("background-color: white")

    def test_toggle_button_style(self):
        """Test toggling button style"""
        # Mock button with active style
        mock_button = MagicMock()
        mock_button.styleSheet.return_value = "background-color: #d84005"

        result = self.UIStyleManager.toggle_button_style(mock_button)
        self.assertFalse(result)  # Should return False (inactive) after toggle
        mock_button.setStyleSheet.assert_called_with("background-color: white")

        # Mock button with inactive style
        mock_button.styleSheet.return_value = "background-color: white"
        result = self.UIStyleManager.toggle_button_style(mock_button)
        self.assertTrue(result)  # Should return True (active) after toggle
        mock_button.setStyleSheet.assert_called_with("background-color: #d84005")


class TestMainWindowStructure(unittest.TestCase):
    """Test that the MainWindow class structure is correct"""

    def setUp(self):
        """Set up QApplication and mocks for testing"""
        # Create QApplication if it doesn't exist
        from PyQt6.QtWidgets import QApplication, QWidget

        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)
        self.QWidget = QWidget

    @patch("potts_shunt_test.VTKHandler")
    def test_main_window_instantiation(self, mock_vtk_handler):
        """Test that MainWindow can be instantiated without errors"""
        try:
            # Mock the VTK handler to avoid VTK dependencies
            mock_vtk_handler.return_value = MagicMock()

            # Create a real QWidget for the VTK widget mock instead of MagicMock
            with patch(
                "potts_shunt_test.QVTKRenderWindowInteractor"
            ) as mock_vtk_widget:
                mock_widget_instance = self.QWidget()  # Use real QWidget

                # Create proper VTK mock chain with specific return values
                mock_render_window = MagicMock()
                mock_renderers = MagicMock()
                mock_renderers.GetNumberOfItems.return_value = 0  # Return integer 0
                mock_render_window.GetRenderers.return_value = mock_renderers
                mock_render_window.GetInteractor.return_value = MagicMock()
                mock_widget_instance.GetRenderWindow = MagicMock(
                    return_value=mock_render_window
                )
                mock_vtk_widget.return_value = mock_widget_instance

                # This should not raise any exceptions
                from main import MainWindow

                window = MainWindow()

                # Verify that the helper methods exist
                self.assertTrue(hasattr(window, "_setup_window_geometry"))
                self.assertTrue(hasattr(window, "_setup_main_layout"))
                self.assertTrue(hasattr(window, "_setup_timers"))
                self.assertTrue(hasattr(window, "_setup_ui_controls"))
                self.assertTrue(hasattr(window, "_setup_vtk_components"))
                self.assertTrue(hasattr(window, "_load_default_files"))

                # Verify that required instance variables exist
                self.assertTrue(hasattr(window, "mesh_file"))
                self.assertTrue(hasattr(window, "centerline_file"))
                self.assertTrue(hasattr(window, "vtk_handler"))

                print("✓ MainWindow instantiation test passed")

        except Exception as e:
            self.fail(f"MainWindow instantiation failed: {e}")


def run_tests():
    """Run all tests"""
    print("Running refactoring verification tests...")
    print("=" * 50)

    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(unittest.makeSuite(TestSliderMapper))
    suite.addTest(unittest.makeSuite(TestUIStyleManager))
    suite.addTest(unittest.makeSuite(TestMainWindowStructure))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✓ ALL TESTS PASSED' if success else '✗ SOME TESTS FAILED'}")

    return success


if __name__ == "__main__":
    # Set up minimal environment for testing
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Run tests
    success = run_tests()
    sys.exit(0 if success else 1)
