"""Small PyTorch 2.8.0 MPS scalar-view/scheduler reproduction; no pretrained weights.

The second loop keeps scheduler arithmetic on CPU and transfers only its result.
This is a reference validation tool, not a DeepJ runtime dependency.
"""
from pathlib import Path
import torch
from diffusers import DPMSolverMultistepScheduler

ROOT = Path(__file__).resolve().parents[1] / 'src/test/resources/sana-video/pipeline/scheduler'
if not torch.backends.mps.is_available():
    raise RuntimeError('Physical Apple MPS required')


def scheduler(device):
    result = DPMSolverMultistepScheduler.from_pretrained(ROOT, local_files_only=True)
    result.set_timesteps(3, device=device)
    return result


def compare(cpu_scheduler):
    reference = scheduler('cpu')
    candidate = scheduler('cpu' if cpu_scheduler else 'mps')
    expected = torch.linspace(-2, 2, 64).reshape(1, 4, 2, 2, 4)
    actual = expected.to('mps')
    for index in range(3):
        flow = torch.sin(expected * 0.1 + index)
        expected = reference.step(flow, reference.timesteps[index], expected).prev_sample
        value = actual.cpu() if cpu_scheduler else actual
        prediction = flow if cpu_scheduler else flow.to('mps')
        actual = candidate.step(prediction, candidate.timesteps[index], value).prev_sample.to('mps')
        print('CPU scheduler' if cpu_scheduler else 'MPS scheduler', index,
              (expected - actual.cpu()).abs().max().item(), flush=True)


scalar = torch.tensor([0.1, 0.2, 0.3])[1]
print('CPU view', scalar.item(), 'MPS multiplication', (torch.ones(1, device='mps') * scalar).cpu().item())
compare(False)
compare(True)
