import unittest

from engine.material import Material


class MaterialTests(unittest.TestCase):
    def test_default_material_is_dielectric(self):
        material = Material()
        self.assertEqual(material.base_color, (0.8, 0.8, 0.8))
        self.assertEqual(material.metallic, 0.0)
        self.assertEqual(material.roughness, 0.5)

    def test_custom_material_parameters(self):
        material = Material(
            name="BrushedMetal",
            base_color=(0.7, 0.72, 0.75),
            metallic=1.0,
            roughness=0.24,
            emissive=(0.0, 0.01, 0.02),
        )
        self.assertEqual(material.shader_parameters()["metallic"], 1.0)
        self.assertEqual(material.roughness, 0.24)

    def test_base_color_must_have_three_components_in_unit_range(self):
        with self.assertRaises(ValueError):
            Material(base_color=(1.1, 0.0, 0.0))
        with self.assertRaises(ValueError):
            Material(base_color=(1.0, 0.0))

    def test_metallic_and_roughness_are_validated(self):
        with self.assertRaises(ValueError):
            Material(metallic=-0.1)
        with self.assertRaises(ValueError):
            Material(roughness=1.1)

    def test_emissive_can_exceed_one(self):
        material = Material(emissive=(3.0, 1.5, 0.25))
        self.assertEqual(material.emissive, (3.0, 1.5, 0.25))


if __name__ == "__main__":
    unittest.main()
