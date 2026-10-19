"""Build an E5-only Kaggle commit with observable, supervised subprocesses."""
from pathlib import Path
import ast
import base64
import json

ROOT = Path(__file__).resolve().parents[1]


def code(source):
    ast.parse(source)
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(True)}


setup = '''import os, sys, json, shutil, subprocess, time, signal
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
assert torch.cuda.is_available(), "Enable GPU T4 x2 in Kaggle Settings"
print("GPU:", torch.cuda.get_device_name(0), flush=True)
roots = sorted(p for p in Path("/kaggle/input").rglob("labrepo")
               if (p/"student/e5.py").is_file() and (p/"data/splits/train.jsonl").is_file())
assert roots, "Attach the completed E0-E4 notebook output as Input"
INPUT_ROOT = roots[0]
ROOT = Path("/kaggle/working/labrepo")
ROOT.mkdir(parents=True, exist_ok=True)
for name in ("scripts", "student", "student_tests", "configs", "data", "work", "report"):
    if (INPUT_ROOT/name).is_dir():
        shutil.copytree(INPUT_ROOT/name, ROOT/name, dirs_exist_ok=True)
for name in ("requirements-colab.txt", "pytest.ini", "REPORT.md"):
    if (INPUT_ROOT/name).is_file(): shutil.copy2(INPUT_ROOT/name, ROOT/name)
CHECKPOINT_ROOT = Path("/kaggle/working/day10_e5_observable_v1")
candidates = sorted(p for p in Path("/kaggle/input").rglob("step-0300")
                    if p.parent.name == "e3" and (p/"adapter/adapter_model.safetensors").is_file()
                    and (p/"projector.pt").is_file())
assert candidates, "Attach E3 step-0300 adapter and projector from the completed run"
target = CHECKPOINT_ROOT/"e3/step-0300"
shutil.copytree(candidates[0]/"adapter", target/"adapter", dirs_exist_ok=True)
shutil.copy2(candidates[0]/"projector.pt", target/"projector.pt")
if (candidates[0]/"run.json").is_file(): shutil.copy2(candidates[0]/"run.json", target/"run.json")
(CHECKPOINT_ROOT/"e3/latest.txt").write_text(str(target), encoding="utf-8")
os.environ["PYTHONPATH"] = str(ROOT)
os.chdir(ROOT)
meta = json.loads((ROOT/"data/splits/meta.json").read_text())
assert meta["dataset_version"] == "d10-21e4070301", meta
for split, expected in (("train",1422),("val",276),("test",311)):
    rows = [json.loads(x) for x in (ROOT/f"data/splits/{split}.jsonl").read_text().splitlines() if x.strip()]
    assert len(rows) == expected, (split,len(rows))
    assert all((ROOT/"data"/r["image_path"]).is_file() for r in rows), split
assert (ROOT/"work/e3_test.jsonl").is_file(), "Missing SFT baseline test predictions"
print("Dataset and E3 step 300 restored", flush=True)
subprocess.run([sys.executable,"-m","pip","install","-q","-r",str(ROOT/"requirements-colab.txt")], check=True)
subprocess.run([sys.executable,"-m","pip","uninstall","-y","torchao"], capture_output=True)
'''

patch = 'import base64, importlib.metadata\n'
for name in ("e5.py", "inference.py", "model.py"):
    source = (ROOT/"student"/name).read_text(encoding="utf-8")
    ast.parse(source)
    encoded = base64.b64encode(source.encode()).decode()
    patch += f'source = base64.b64decode({encoded!r}).decode("utf-8")\n'
    patch += f'compile(source, "student/{name}", "exec")\n'
    patch += f'(ROOT/"student/{name}").write_text(source, encoding="utf-8")\n'
patch += '''versions = {p: importlib.metadata.version(p) for p in
            ("torch","transformers","peft","trl","bitsandbytes","accelerate")}
print("PACKAGES", json.dumps(versions), flush=True)
print("E5 source replaced: plain logs, per-step progress, incremental predictions", flush=True)
'''

supervisor = '''LOG_ROOT = Path("/kaggle/working/e5_job_logs")
LOG_ROOT.mkdir(exist_ok=True)
def run_job(name, args, *, inactivity=900, total_timeout=7200):
    logfile = LOG_ROOT/f"{name}.log"
    # The child writes to a regular file, avoiding notebook stdout pipe backpressure.
    launcher = "import faulthandler,runpy; faulthandler.enable(); faulthandler.dump_traceback_later(600,repeat=True); runpy.run_module('student.e5',run_name='__main__')"
    command = [sys.executable,"-u","-c",launcher,*args]
    start = last_progress = last_heartbeat = time.monotonic()
    print("START", name, args, flush=True)
    with logfile.open("wb", buffering=0) as sink:
        child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                                 stdout=sink, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            with logfile.open("r", encoding="utf-8", errors="replace") as reader:
                while True:
                    while True:
                        line = reader.readline()
                        if not line: break
                        if "E5_PROGRESS " in line or "INFERENCE_PROGRESS " in line:
                            last_progress = time.monotonic()
                        if line.strip(): print(f"[{name}] {line.rstrip()[:1800]}", flush=True)
                    now = time.monotonic()
                    if now-last_heartbeat >= 60:
                        gpu = subprocess.run(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used,memory.total",
                                              "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
                        print("HEARTBEAT",name,"elapsed",round(now-start),"seconds; last progress",
                              round(now-last_progress),"seconds ago; GPU",gpu.stdout.strip(),flush=True)
                        last_heartbeat = now
                    if child.poll() is not None: break
                    if now-start > total_timeout or (inactivity and now-last_progress > inactivity):
                        raise TimeoutError(f"{name}: progress timeout; diagnostic traceback in {logfile}")
                    time.sleep(2)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
    if child.returncode:
        raise RuntimeError(f"{name} failed ({child.returncode}); see {logfile}")
    print("DONE",name,"seconds",round(time.monotonic()-start),flush=True)
'''

smoke = '''subprocess.run([sys.executable,"-u","-m","pytest","-q","student_tests/test_student_e5.py"],cwd=ROOT,check=True)
for steps, limit in ((1,4),(5,20)):
    output = Path(f"/kaggle/working/e5_smoke_observable_{steps}")
    run_job(f"smoke_{steps}",["--mode","train","--base-exp","e3","--smoke",
            "--max-steps",str(steps),"--train-limit",str(limit),
            "--checkpoint-root",str(CHECKPOINT_ROOT),"--output-root",str(output),"--no-resume"],
            total_timeout=1800)
    run = json.loads((output/"run.json").read_text())
    assert run["steps"] == steps and run["max_completion_length"] == 32, run
    import math
    assert run["log_history"] and all(math.isfinite(float(row["loss"]))
           for row in run["log_history"] if "loss" in row), "Non-finite smoke loss"
print("SMOKE GATES PASSED",flush=True)
'''

full = '''run_job("full_60",["--mode","train","--base-exp","e3","--max-steps","60",
        "--checkpoint-root",str(CHECKPOINT_ROOT)], total_timeout=7200)
run = json.loads((ROOT/"report/evidence/e5_train_log.json").read_text())
assert run["steps"] == 60 and not run["smoke"] and run["max_completion_length"] == 128, run
assert run["train_images"] == 1422, run
print("FULL 60 TRAINING STEPS SAVED",flush=True)
'''

evaluate = '''run_job("test_inference",["--mode","infer","--split","test","--checkpoint-root",str(CHECKPOINT_ROOT)],total_timeout=7200)
predictions = [json.loads(x) for x in (ROOT/"work/e5_test.jsonl").read_text().splitlines() if x.strip()]
assert len(predictions) == len({x["image_id"] for x in predictions}) == 311
run_job("analysis",["--mode","analyze","--baseline-exp","e3"],inactivity=None,total_timeout=7200)
for name in ("e5_train_log.json","e5_completion_samples.jsonl","e5.json","e5_analysis.json"):
    p = ROOT/"report/evidence"/name
    assert p.is_file() and p.stat().st_size > 0, p
    print("EVIDENCE",name,p.stat().st_size,"bytes",flush=True)
shutil.make_archive("/kaggle/working/e5_bonus_evidence","zip",ROOT/"report/evidence")
print("E5 COMPLETE: /kaggle/working/e5_bonus_evidence.zip",flush=True)
'''

cells = [{"cell_type": "markdown", "metadata": {}, "source": [
    "# E5 GRPO from E3 step 300\n",
    "Reward tests, 1/5-step smoke gates, full 60-step training, test inference and bonus evidence.\n",
    "Only E5 runs; E0-E4 results are restored from the attached completed notebook output.\n"]}]
cells += [code(s) for s in (setup, patch, supervisor, smoke, full, evaluate)]
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"}},
            "nbformat":4,"nbformat_minor":5}
output = ROOT/"lab_day10_e5_observable.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(output)
