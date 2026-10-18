#!/usr/bin/env python3
"""Package routing/layout tests with fake executables, no SDK or DSP required."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
MOCK = '#!/bin/sh\nprintf "%s\\n" "$@"\nprintf "LIB=%s\\n" "$LD_LIBRARY_PATH"\nprintf "DSP=%s\\n" "$ADSP_LIBRARY_PATH"\n'


class PackageTests(unittest.TestCase):
    def build_fixture(self, base, sensevoice=True):
        build = base / 'build'
        files = {'CMakeCache.txt': 'CMAKE_STRIP:FILEPATH=/usr/bin/strip\n',
                 'hexagon-build-id.txt': 'test-build-id\n',
                 'bin/llama-cli': MOCK, 'bin/libggml-base.so': 'main ggml',
                 'bin/libggml-hexagon.so': 'fake host module',
                 'ggml/src/ggml-hexagon/libggml-htp-v73.so': 'fake skel'}
        if sensevoice:
            files.update({'sensevoice/sensevoice-build/bin/llama-funasr-sensevoice': MOCK,
                          'sensevoice/sensevoice-build/bin/test-sensevoice-backend': MOCK,
                          'sensevoice/sensevoice-build/bin/libggml-base.so': 'private SenseVoice ggml',
                          'sensevoice/sensevoice-build/bin/libggml-cpu.so': 'CPU linked'})
        for name, text in files.items():
            path = build / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            if text.startswith('#!'): path.chmod(0o755)
        return build

    def test_flat_split_and_sensevoice_only(self):
        with tempfile.TemporaryDirectory(prefix='sensevoice-package-') as directory:
            base = Path(directory)
            build = self.build_fixture(base)
            for mode in ('flat', 'split', 'sensevoice'):
                with self.subTest(mode=mode):
                    out = base / mode
                    command = [str(ROOT/'scripts/package.sh')] + ([] if mode == 'flat' else ['--'+mode]) + [str(build), str(out)]
                    result = subprocess.run(command, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertTrue((out/'licenses/SenseVoice-LICENSE').exists())
                    self.assertTrue((out/'licenses/miniaudio-LICENSE').exists())
                    self.assertIn('SenseVoice:', (out/'BUILD_INFO.txt').read_text())
                    for script in ('run.sh', 'run-sensevoice.sh'):
                        self.assertEqual(subprocess.run(['sh','-n',str(out/script)]).returncode, 0)
                    reply = subprocess.run([str(out/'run-sensevoice.sh'),'model file.gguf','audio file.wav','--threads','2'],
                        env=dict(os.environ,SENSEVOICE_DEVICE='HTP1'), capture_output=True,text=True)
                    self.assertEqual(reply.returncode, 0, reply.stderr)
                    lines = reply.stdout.splitlines()
                    self.assertEqual(lines[:7], ['-m','model file.gguf','-a','audio file.wav','--backend','hexagon','--backend-lib'])
                    self.assertIn('HTP1', lines)
                    self.assertIn('--threads', lines)
                    library = out/'lib/sensevoice' if mode=='split' else out/'sensevoice' if mode=='flat' else out
                    self.assertIn('LIB='+str(library)+':', reply.stdout)
                    self.assertEqual(subprocess.run([str(out/'run.sh'),'llama-funasr-sensevoice','--help'],capture_output=True).returncode,0)
                    if mode=='sensevoice': self.assertFalse((out/'llama-cli').exists())
                    else: self.assertTrue((out/('bin/llama-cli' if mode=='split' else 'llama-cli')).exists())

    def test_optional_off_and_missing_runtime(self):
        with tempfile.TemporaryDirectory(prefix='sensevoice-package-') as directory:
            base = Path(directory)
            build = self.build_fixture(base, sensevoice=False)
            out = base/'flat'
            self.assertEqual(subprocess.run([str(ROOT/'scripts/package.sh'),str(build),str(out)],capture_output=True).returncode,0)
            self.assertFalse((out/'run-sensevoice.sh').exists())
            self.assertEqual(subprocess.run([str(ROOT/'scripts/package.sh'),'--sensevoice',str(build),str(base/'missing')],capture_output=True).returncode,2)


if __name__=='__main__':
    unittest.main()
