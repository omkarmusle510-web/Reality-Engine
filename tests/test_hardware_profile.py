"""Focused Block 6 tests for hardware_profile.py."""

import ast
import importlib
from pathlib import Path
import unittest

MODULE = "apps.reality_painter.optimization.hardware_profile"


def mod():
    return importlib.import_module(MODULE)


class TestCPU(unittest.TestCase):
    def test_success(self):
        self.assertEqual(mod().detect_cpu(lambda: 8).logical_cores, 8)

    def test_invalid_and_exception_safe(self):
        m = mod()
        for value in (None, 0, -1, 1.5, "8"):
            self.assertIsNone(m.detect_cpu(lambda value=value: value).logical_cores)

        def broken():
            raise RuntimeError
        self.assertIsNone(m.detect_cpu(broken).logical_cores)


class TestMemory(unittest.TestCase):
    def test_success(self):
        self.assertEqual(mod().detect_memory(detector_override=lambda: 16384).total_ram_mb, 16384)

    def test_invalid_and_exception_safe(self):
        m = mod()
        for value in (None, 0, -1, 1.5, "16384"):
            self.assertIsNone(m.detect_memory(detector_override=lambda value=value: value).total_ram_mb)

        def broken():
            raise RuntimeError
        self.assertIsNone(m.detect_memory(detector_override=broken).total_ram_mb)


class TestPlatform(unittest.TestCase):
    def test_detection(self):
        m = mod()

        class Fake:
            @staticmethod
            def system(): return "Windows"
            @staticmethod
            def release(): return "11"
            @staticmethod
            def machine(): return "AMD64"

        r = m.detect_platform(Fake)
        self.assertEqual((r.system, r.release, r.architecture), ("Windows", "11", "AMD64"))

    def test_failure_is_per_field_safe(self):
        m = mod()

        class Broken:
            @staticmethod
            def system(): raise RuntimeError
            @staticmethod
            def release(): return ""
            @staticmethod
            def machine(): raise RuntimeError

        r = m.detect_platform(Broken)
        self.assertEqual(r.system, "Unknown")
        self.assertIsNone(r.release)
        self.assertIsNone(r.architecture)


class TestGPU(unittest.TestCase):
    def test_windows_gpu_success(self):
        m = mod()

        def runner(args):
            self.assertEqual(tuple(args), m._WINDOWS_GPU_COMMAND)
            return "Name\nNVIDIA GeForce RTX Test 4060\n"

        r = m.detect_gpu(system="Windows", runner=runner)
        self.assertEqual(r.name, "NVIDIA GeForce RTX Test 4060")
        self.assertEqual(r.vendor, "NVIDIA")
        self.assertEqual(r.memory_kind, m.GPUMemoryKind.UNKNOWN)
        self.assertIsNone(r.memory_mb)

    def test_linux_gpu_success(self):
        m = mod()
        r = m.detect_gpu(
            system="Linux",
            runner=lambda args: "01:00.0 VGA compatible controller: NVIDIA Corporation Test GPU\n",
        )
        self.assertEqual(r.name, "NVIDIA Corporation Test GPU")
        self.assertEqual(r.vendor, "NVIDIA")

    def test_mac_gpu_success(self):
        m = mod()
        r = m.detect_gpu(
            system="Darwin",
            runner=lambda args: "Chipset Model: Apple M Test\n",
        )
        self.assertEqual(r.name, "Apple M Test")
        self.assertEqual(r.vendor, "Apple")

    def test_unavailable_and_unknown_system_safe(self):
        m = mod()

        def broken(_):
            raise FileNotFoundError

        r = m.detect_gpu(system="Windows", runner=broken)
        self.assertIsNone(r.name)
        self.assertIsNone(r.vendor)
        self.assertEqual(r.memory_kind, m.GPUMemoryKind.UNKNOWN)

        called = []
        r = m.detect_gpu(system="UnknownOS", runner=lambda a: called.append(a))
        self.assertFalse(called)
        self.assertIsNone(r.name)

    def test_memory_never_fabricated(self):
        m = mod()
        r = m.detect_gpu(system="Windows", runner=lambda _: "Name\nIntel Test Graphics\n")
        self.assertEqual(r.memory_kind, m.GPUMemoryKind.UNKNOWN)
        self.assertIsNone(r.memory_mb)


class TestTiers(unittest.TestCase):
    def test_low_medium_high_unknown(self):
        m = mod()
        self.assertEqual(m.classify_tier(m.CPUInfo(2), m.MemoryInfo(4096)), m.PerformanceTier.LOW)
        self.assertEqual(m.classify_tier(m.CPUInfo(4), m.MemoryInfo(8192)), m.PerformanceTier.MEDIUM)
        self.assertEqual(m.classify_tier(m.CPUInfo(8), m.MemoryInfo(16384)), m.PerformanceTier.HIGH)
        self.assertEqual(m.classify_tier(m.CPUInfo(None), m.MemoryInfo(16384)), m.PerformanceTier.UNKNOWN)
        self.assertEqual(m.classify_tier(m.CPUInfo(8), m.MemoryInfo(None)), m.PerformanceTier.UNKNOWN)

    def test_deterministic(self):
        m = mod()
        values = [m.classify_tier(m.CPUInfo(8), m.MemoryInfo(16384)) for _ in range(20)]
        self.assertEqual(values, [m.PerformanceTier.HIGH] * 20)

    def test_custom_thresholds(self):
        m = mod()
        t = m.TierThresholds(2, 4, 4096, 8192)
        self.assertEqual(m.classify_tier(m.CPUInfo(4), m.MemoryInfo(8192), t), m.PerformanceTier.HIGH)


class TestPolicy(unittest.TestCase):
    def test_explicit_policies(self):
        m = mod()
        expected = {
            m.PerformanceTier.HIGH: (90.0, 45.0),
            m.PerformanceTier.MEDIUM: (60.0, 30.0),
            m.PerformanceTier.LOW: (30.0, 15.0),
            m.PerformanceTier.UNKNOWN: (30.0, 15.0),
        }
        for tier, pair in expected.items():
            p = m.recommended_policy_for_tier(tier)
            self.assertEqual((p.target_fps, p.minimum_fps), pair)

    def test_deterministic_and_valid(self):
        m = mod()
        for tier in m.PerformanceTier:
            a = m.recommended_policy_for_tier(tier)
            b = m.recommended_policy_for_tier(tier)
            self.assertEqual(a, b)
            self.assertGreaterEqual(a.target_fps, a.minimum_fps)


class TestComposition(unittest.TestCase):
    def test_build_profile(self):
        m = mod()

        class FakePlatform:
            @staticmethod
            def system(): return "Windows"
            @staticmethod
            def release(): return "11"
            @staticmethod
            def machine(): return "AMD64"

        r = m.build_hardware_profile(
            cpu_count_fn=lambda: 8,
            memory_detector=lambda: 16384,
            platform_module=FakePlatform,
            gpu_runner=lambda _: "Name\nAMD Radeon Test GPU\n",
        )
        self.assertEqual(r.cpu.logical_cores, 8)
        self.assertEqual(r.memory.total_ram_mb, 16384)
        self.assertEqual(r.platform.system, "Windows")
        self.assertEqual(r.gpu.vendor, "AMD")
        self.assertEqual(r.tier, m.PerformanceTier.HIGH)

    def test_all_detectors_fail_safely(self):
        m = mod()

        class BrokenPlatform:
            @staticmethod
            def system(): raise RuntimeError
            @staticmethod
            def release(): raise RuntimeError
            @staticmethod
            def machine(): raise RuntimeError

        def broken(): raise RuntimeError
        r = m.build_hardware_profile(
            cpu_count_fn=broken,
            memory_detector=broken,
            platform_module=BrokenPlatform,
            gpu_runner=lambda _: (_ for _ in ()).throw(RuntimeError()),
        )
        self.assertIsNone(r.cpu.logical_cores)
        self.assertIsNone(r.memory.total_ram_mb)
        self.assertEqual(r.platform.system, "Unknown")
        self.assertIsNone(r.gpu.name)
        self.assertEqual(r.tier, m.PerformanceTier.UNKNOWN)


class TestScope(unittest.TestCase):
    def source(self):
        return Path(mod().__file__).read_text(encoding="utf-8")

    def test_no_forbidden_optimizer_imports(self):
        tree = ast.parse(self.source())
        forbidden = {
            "apps.reality_painter.optimization.analyzer",
            "apps.reality_painter.optimization.candidate_generator",
            "apps.reality_painter.optimization.benchmark",
            "apps.reality_painter.optimization.selector",
            "apps.reality_painter.optimization.cache",
        }
        found = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                found.update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom) and n.module:
                found.add(n.module)
        self.assertTrue(forbidden.isdisjoint(found))

    def test_no_network_renderer_or_glb_behavior(self):
        source = self.source().lower()
        tree = ast.parse(source)
        imports = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imports.update(a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.ImportFrom) and n.module:
                imports.add(n.module.split(".")[0])

        self.assertTrue({"requests", "httpx", "aiohttp", "socket", "urllib"}.isdisjoint(imports))
        self.assertNotIn("renderer3d", source)
        self.assertNotIn("load_glb", source)

    def test_no_glb_load_call(self):
        tree = ast.parse(self.source())
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name):
                    self.assertNotEqual(n.func.id, "load_glb")
                elif isinstance(n.func, ast.Attribute):
                    self.assertNotEqual(n.func.attr, "load_glb")


if __name__ == "__main__":
    unittest.main(verbosity=2)
