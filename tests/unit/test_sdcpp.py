"""The stable-diffusion.cpp runtime (stratamodels.runtimes.sdcpp) without the binary: files, the command line, a run with a fake sd-cli, refusals."""
import os, sys, types

import pytest

from stratamodels.registry import Model
from stratamodels.runtimes import RunError, sdcpp


@pytest.fixture
def handle(tmp_path, monkeypatch):
    exe = tmp_path / 'sd-cli'; exe.write_text('#!/bin/sh\n'); d = tmp_path / 'm.gguf'; d.write_text('x')
    monkeypatch.setenv('STRATAMODELS_SD_BIN', str(exe))
    hub = types.ModuleType('huggingface_hub'); hub.hf_hub_download = lambda repo, path, local_dir: os.path.join(local_dir, path)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    m = Model(key='klein', runtime='sd-cpp', name='k', roles=('image-generate',), vram_gb=7, options=dict(diffusion=str(d), llm='org/te-GGUF:te.gguf', steps=16, args=['--diffusion-fa']))
    return sdcpp.load(m, str(tmp_path))


def test_load_resolves_files_and_refuses_what_is_missing(handle, tmp_path, monkeypatch):
    assert handle['files']['llm'].endswith(os.path.join('sd', 'org--te-GGUF', 'te.gguf')) and handle['defaults']['steps'] == 16
    m = Model(key='x', runtime='sd-cpp', name='x', roles=(), vram_gb=1, options={})
    with pytest.raises(RunError, match='names no diffusion'): sdcpp.load(m, str(tmp_path))
    with pytest.raises(RunError, match='no such model file'): sdcpp.fetch('/nope.gguf', str(tmp_path))
    with pytest.raises(RunError, match='repo:path'): sdcpp.fetch('just-a-name', str(tmp_path))
    def gated(*a, **k): raise PermissionError('401')
    sys.modules['huggingface_hub'].hf_hub_download = gated
    with pytest.raises(RunError, match='licence accepted'): sdcpp.fetch('bfl/klein:ae.safetensors', str(tmp_path))
    monkeypatch.setenv('STRATAMODELS_SD_BIN', str(tmp_path / 'none'))
    with pytest.raises(RunError, match='not installed'): sdcpp.load(m, str(tmp_path))


def test_the_command_line(handle, tmp_path):
    ref = tmp_path / 'ref.png'; ref.write_bytes(b'png')
    cmd, seed = sdcpp.command(handle, dict(prompt='a fox', width=768, height=512, seed=7, negative_prompt='blur', ref_images=[str(ref)], cfg_scale=2), '/out.png')
    assert seed == 7 and cmd[cmd.index('-W') + 1] == '768' and cmd[cmd.index('--steps') + 1] == '16' and cmd[cmd.index('--cfg-scale') + 1] == '2.0' and '-r' in cmd and '--llm' in cmd and cmd[-1] == '--diffusion-fa'
    assert 0 <= sdcpp.command(handle, dict(prompt='x'), '/o.png')[1] < 2 ** 31
    for p, msg in ((dict(), 'give a prompt'), (dict(prompt='x', width=1000), 'multiple of 16'), (dict(prompt='x', height='big'), 'whole number'), (dict(prompt='x', colour=1), 'unknown params'),
                   (dict(prompt='x', ref_images=['/nope.png']), 'no such reference'), (dict(prompt='x', ref_images='a'), 'at most 10')):
        with pytest.raises(RunError, match=msg): sdcpp.command(handle, p, '/o.png')


class FakeSd:
    def __init__(self, cmd, **kw): self.cmd, self.pid, self.returncode = cmd, 4242, 0; out = cmd[cmd.index('-o') + 1]; open(out, 'wb').write(b'png') if 'fail' not in cmd[cmd.index('-p') + 1] else None; self.returncode = 0 if 'fail' not in cmd[cmd.index('-p') + 1] else 1
    def communicate(self, timeout=None): return ('generated', None)
    def kill(self): pass


def test_a_run_writes_the_image_and_reports_time_and_peak_vram(handle, tmp_path):
    probe = lambda: dict(processes=[dict(pid=4242, used_mb=6100.0)])
    r = sdcpp.run(handle, dict(prompt='a fox', seed=3, width=512, height=512), popen=FakeSd, probe=probe, out_dir=str(tmp_path / 'g'))
    assert os.path.isfile(r['path']) and r['seed'] == 3 and r['width'] == 512 and r['model'] == 'klein' and r['steps'] == 16
    with pytest.raises(RunError, match='sd-cli failed'): sdcpp.run(handle, dict(prompt='fail please'), popen=FakeSd, probe=probe, out_dir=str(tmp_path / 'g'))
    sdcpp.unload(handle); assert handle == {}


def test_a_run_that_takes_too_long_is_stopped(handle, tmp_path):
    import subprocess
    class Slow(FakeSd):
        calls = 0
        def communicate(self, timeout=None):
            Slow.calls += 1
            if Slow.calls == 1: raise subprocess.TimeoutExpired('sd-cli', timeout)
            return ('', None)
    with pytest.raises(RunError, match='30 minutes'): sdcpp.run(handle, dict(prompt='a'), popen=Slow, probe=lambda: None, out_dir=str(tmp_path))
