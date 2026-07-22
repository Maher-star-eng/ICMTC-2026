# Full-Repo Type-Diagnostic Sweep — ICMTC 2026 Mission 2

**Scope:** every `.py` file under `core/`, `geo/`, `ioutils/`, `vision/`, `training/` (excluding `tests/`, `__pycache__/`, `.venv/`, and the paths `pyrightconfig.json` itself excludes: `training/runs`, `training/flag detection.v1i.yolov11`, `untitled folder`).
**Tool:** `.venv/bin/pyright 1.1.411`, invoked as `pyright core geo ioutils vision training` (existing `pyrightconfig.json` respected — `.venv` venv, Python 3.12, `basic` mode).
**Result:** 0 unresolved-import diagnostics. Every finding below is a real type/attribute finding on a codebase that now type-checks cleanly at the import level.

Files analyzed: 21. Total diagnostics: **50** (all severity `error`; 0 warnings, 0 information).

---

## Executive Summary

| Bucket | Root causes | Diagnostics |
|---|---:|---:|
| **REAL BUG** | 4 | 5 |
| **UNCERTAIN — needs a human decision** | 1 | 1 |
| **COSMETIC / TYPE-LOOSENESS** | 11 | 44 |
| **Total** | **16** | **50** |

The 50 raw diagnostics collapse to **16 independent root causes**. Two of the REAL BUG root causes are the already-known `MedianGeolocatorAdapter`/`Tracker` Protocol-narrowness bugs (mission.py:599-600) — confirmed and detailed below with their concrete failure scenarios. Sweeping the rest of the repo for the *same shape* of bug (declared Protocol wider than what the concrete class actually accepts/returns) turned up two more real findings: a `pymavlink` connection-object narrowness bug in `geo/telemetry.py` (this is also the answer to the CSVReader question — see its own section), and an unguarded `None` from `cv2.imread` reaching a shared preprocessing helper in `training/masked_validation_experiment.py`. Everything else is either a well-understood type-checker false positive (third-party stub gaps, closure-narrowing limitations, a documented try/except relying on a `TypeError`) or one genuinely unverifiable-without-real-weights case, called out under UNCERTAIN.

---

## Part 1 — REAL BUG (4 root causes, 5 diagnostics)

### RB-1. `Tracker.update()` is narrower than the `TrackerLike` Protocol it's wired in as

**Affected:** `core/mission.py:599:17`

```
Argument of type "Tracker" cannot be assigned to parameter "tracker" of type "TrackerLike" in function "__init__"
  "Tracker" is incompatible with protocol "TrackerLike"
    "update" is an incompatible type
      Type "(detections: list[Detection], frame_idx: int) -> list[Track]" is not assignable to
      type "(detections: Sequence[object], frame_idx: int) -> Sequence[TrackLike]"
        Parameter 1: type "Sequence[object]" is incompatible with type "list[Detection]"
        Function return type "list[Track]" is incompatible with type "Sequence[TrackLike]"
```

**Classification: REAL BUG.** `ioutils/output.py`'s `TrackerLike.update(self, detections: Sequence[object], frame_idx: int) -> Sequence[TrackLike]` promises that *any* sequence of opaque objects can be handed to the tracker — that's the whole point of `DetectorLike.detect() -> Sequence[object]` being duck-typed (ASSUMPTION O15 in output.py). But the concrete `vision/tracker.py:Tracker.update()` does:
```python
for detection in detections:
    if not isinstance(detection, Detection):
        raise TypeError(...)
```
It demands real `tracker.Detection` dataclass instances, which is strictly narrower than what it promised to accept as a `TrackerLike`.

**Concrete failure scenario:** `VisionSystem.process_frame` (output.py) is written against the Protocol, so it's entirely valid — and exactly the kind of thing this project's own "hardware-free tests" philosophy invites — for a test or a future detector implementation to satisfy `DetectorLike` with a lightweight fake that returns simple namedtuples/dicts as "opaque detections" instead of real `Detection` objects. Wire that fake detector together with the real `Tracker` (both individually valid per their declared Protocols) and `Tracker.update()`'s `isinstance` check raises `TypeError` on the very first detection, every frame. Because `VisionSystem.process_frame` wraps everything in `try/except Exception` (ASSUMPTION O1), this doesn't crash the mission — it silently turns every frame into `MissionStatus.DEGRADED` forever, with no target ever found and no visible error beyond a log line. That's arguably worse than a crash: a silent total pipeline failure that looks like "no flags in view" rather than "wiring bug."

### RB-2. `MedianGeolocatorAdapter.locate()` is narrower than the `GeolocatorLike` Protocol it's wired in as

**Affected:** `core/mission.py:600:20`

```
Argument of type "MedianGeolocatorAdapter" cannot be assigned to parameter "geolocator" of type "GeolocatorLike" in function "__init__"
  "MedianGeolocatorAdapter" is incompatible with protocol "GeolocatorLike"
    "locate" is an incompatible type
      Type "(track: _TrackWithBox, pose: Pose | None) -> (GPSCoord | None)" is not assignable to
      type "(track: TrackLike, pose: object) -> (tuple[float, float] | None)"
        Parameter 1: "TrackLike" is incompatible with protocol "_TrackWithBox"
        Parameter 2: "object" is incompatible with type "Pose | None"
```

**Classification: REAL BUG** (already identified prior to this sweep; confirmed here with the exact mechanism). `output.py`'s `GeolocatorLike.locate(self, track: TrackLike, pose: object) -> ...` only guarantees a track with `.id`, `.hits`, `.class_votes`, `.committed_class()` (no `.box`!) and a `pose` that is completely opaque (`object`). `mission.py`'s own `_TrackWithBox` Protocol demands `.box` in addition, and `MedianGeolocatorAdapter.locate()`'s body reads `track.box` (line 254) and `pose.gimbal_frame` / `pose.gimbal_quat` / `pose.alt_agl` / `pose.lat` / `pose.lon` (lines 258-282) — i.e. it assumes the real `geo.telemetry.Pose` shape, not an opaque `object`.

**Concrete failure scenario:** exactly as the task's own framing puts it — if a `TrackLike`-shaped fake track from `output.py`'s own test suite (satisfying only `id`/`hits`/`class_votes`/`committed_class()`, deliberately *not* including `.box` since the Protocol doesn't require it) were ever substituted for a real `tracker.Track`, `MedianGeolocatorAdapter.locate()` would raise `AttributeError: 'FakeTrack' object has no attribute 'box'` the first time it runs. Symmetrically, if a `TelemetryLike`-conforming fake telemetry object returned anything other than a real `geo.telemetry.Pose` (which `TelemetryLike.pose_at() -> object | None` fully permits), `locate()` would raise `AttributeError` on `.gimbal_frame`/`.gimbal_quat`/etc. Today this never triggers because `main()` always wires the real `Tracker`/`TelemetryBuffer`, but any test or future component swap that trusts the declared Protocols (not the accidental concrete shape) breaks it immediately.

### RB-3. `pymavlink.mavutil.mavlink_connection()`'s return type is a union of connection classes with an inconsistent surface — `conn.close()`/`conn.recv_match(timeout=...)` assume every branch supports it, but `CSVReader` doesn't

**Affected:** `geo/telemetry.py:514:30`, `geo/telemetry.py:543:24` (this is also the direct answer to the CSVReader question — see its dedicated section below)

```
514: Cannot access attribute "close" for class "CSVReader" | Attribute "close" is unknown
543: Cannot access attribute "recv_match" for class "object" | Attribute "recv_match" is unknown
```

**Classification: REAL BUG.** See the CSVReader section below for the full trace. Short version: `mavutil.mavlink_connection(device)` (pymavlink) is an untyped function with one `return` statement per connection-string prefix — `mavtcp`, `mavudp`, `DFReader_binary`, `pymavlink.CSVReader.CSVReader`, etc. — so pyright infers its return type as the union of everything it can return, one member of which is `pymavlink.CSVReader.CSVReader`. `TelemetryBuffer._mavlink_loop` types its `conn` parameter as bare `object` (line 520) specifically to paper over this union, then calls `conn.recv_match(type=_WANTED, blocking=True, timeout=1.0)` and, in the caller's `finally`, `conn.close()`.

### RB-4. `cv2.imread()`'s `None` return can reach `forward_raw()`/`letterbox()` unguarded

**Affected:** `training/masked_validation_experiment.py:415:38`

```
Argument of type "MatLike | None" cannot be assigned to parameter "frame" of type "ndarray[_AnyShape, dtype[Any]]" in function "forward_raw"
  "None" is not assignable to "ndarray[_AnyShape, dtype[Any]]"
```

**Classification: REAL BUG.** In `main()`:
```python
sample_img, _ = images_by_target[sample_target][0]
sample_frame = cv2.imread(str(sample_img))          # cv2.imread returns MatLike | None
sample_raw, _ = forward_raw(net, sample_frame, INPUT_SIZE)   # no None check
```
`forward_raw()` immediately calls `letterbox(frame, input_size)` (`vision/detector.py:244`), whose very first line is `orig_h, orig_w = int(frame.shape[0]), int(frame.shape[1])` — no guard at all. Note that *elsewhere* in this same file, `verify_before_argmax_invariant()` (line ~305) does check `if frame is None: continue` before using a `cv2.imread()` result — the author clearly knows `cv2.imread` can fail — but this second, later call site (picking the sample image for the `verify_masking` sanity check) skips that check and re-reads the file fresh.

**Concrete failure scenario:** `images_by_target[sample_target][0]` is just "the first image path found on disk containing this target class" (from `find_images_with_class`), with no guarantee it is one of the frames that successfully decoded earlier in the script. Any single corrupt, truncated, or unsupported-format image file in the real validation dataset directory (`training/flag detection.v1i.yolov11/valid/images/`) that happens to be first in sorted order for a target class will make `cv2.imread` return `None` here, and the script dies with an opaque `AttributeError: 'NoneType' object has no attribute 'shape'` several stack frames away from the actual cause, instead of a clear "corrupt/unreadable image: `<path>`" message. This is a real dataset dependent on file downloads/exports (Roboflow), where a single bad file is entirely plausible.

---

## Part 2 — UNCERTAIN, needs a human decision (1 root cause, 1 diagnostic)

### UN-1. `OnnxBackend.infer()`'s output type is broader than what `_decode_yolov8` accepts — genuinely unverifiable without real ONNX weights

**Affected:** `vision/detector.py:440:31`

```
Argument of type "ndarray[_AnyShape, dtype[Any]] | SparseTensor | list[Unknown] | dict[Unknown, Unknown]"
cannot be assigned to parameter "output" of type "ndarray[_AnyShape, dtype[Any]]" in function "_decode_yolov8"
  "SparseTensor" is not assignable to "ndarray[_AnyShape, dtype[Any]]"
```

`onnxruntime.InferenceSession.run()`'s stub reflects the full ONNX spec — a graph output can be a dense tensor, a sparse tensor, a sequence, or a map — so its return type is a broad union. `_decode_yolov8` only accepts a plain `np.ndarray`.

**Why this is UNCERTAIN rather than confidently COSMETIC:** `vision/detector.py`'s own class docstring flags `OnnxBackend` as *"UNVERIFIED SCAFFOLDING — requires trained model... makes no accuracy claim and cannot until real weights exist."* A standard Ultralytics YOLO→ONNX export for a detection model produces exactly one dense output tensor, which would make this diagnostic a pure stub-breadth artifact (COSMETIC) — but that is an inference from how ONNX exports *typically* behave, not something this repo can currently verify by running the actual exported `.onnx` file end to end, since no ONNX weights exist yet in this repo. Whoever first exercises `OnnxBackend` against a real `.onnx` export should confirm `outputs[0]` is always a dense `ndarray` for this project's specific export configuration (single output node, no NMS-in-graph, no dynamic sequence outputs) before dismissing this diagnostic — if a future export ever bakes in NMS or multiple output heads (some ONNX export options do), the broader union becomes real and `_decode_yolov8(outputs[0], ...)` would need an explicit `isinstance(outputs[0], np.ndarray)` guard.

---

## Part 3 — COSMETIC / TYPE-LOOSENESS (11 root causes, 44 diagnostics)

### C-1. `_quat_to_euler(quat)` is called with an `Optional` quaternion by design (ASSUMPTION MS6)

**Affected:** `core/mission.py:269:68`

`pose.gimbal_quat: tuple[...] | None`, but `_quat_to_euler(quat: tuple[float,float,float,float])` declares a non-Optional parameter. Tuple-unpacking `None` (`w, x, y, z = None`) raises `TypeError`. **This is intentional**: mission.py's own module docstring, ASSUMPTION MS6, states explicitly — *"a TypeError from an unusable gimbal quaternion (e.g. None) [is] treated identically [to a ValueError]... that frame contributes nothing to history."* The call sits inside `try: ... except (ValueError, TypeError): ...` (lines 268-292), which is exactly this documented behavior. **Classification: COSMETIC** — correct, deliberate runtime behavior; the annotation is simply narrower than what's actually passed, relying on an intentionally-caught exception rather than an explicit `if quat is None` branch. No fix needed beyond (optionally) widening the signature to `tuple[...] | None` for documentation clarity.

### C-2. `dist_coeffs` is typed `np.ndarray` but `AppConfig.camera.dist_coeffs` is a `tuple[float, ...]`

**Affected:** `core/mission.py:271:59`, `core/mission.py:282:30`

`geo/geolocate.py`'s `pixel_to_ray`/`pixel_to_gps` declare `dist_coeffs: np.ndarray | None`. `core/config.py`'s `CameraConfig.dist_coeffs: tuple[float, ...] = ()` (ASSUMPTION C2) is a validated tuple, never an ndarray. **Classification: COSMETIC.** Both functions immediately call `_validate_dist_coeffs()` → `_as_finite_float_array()` → `np.asarray(values, dtype=np.float64)`, which coerces a tuple perfectly well. The declared parameter type is stricter than what the function body actually requires (any array-like), so this never breaks at runtime. Widening the signature to `Sequence[float] | np.ndarray | None` (or just `ArrayLike | None`) would silence this with zero behavior change.

### C-3. Locally-declared `YoloModel`/`YoloFactory` Protocol methods use a bare docstring as their body, so pyright doesn't treat them as stubs

**Affected:** `training/export.py:167:40`, `training/export.py:177:41`, `training/train.py:81:41`, `vision/infer.py:110:41`

All four are `Protocol` method definitions whose body is only a docstring (no `...`), e.g.:
```python
class YoloFactory(Protocol):
    def __call__(self, weights: str) -> YoloModel:
        """Return a YOLO-like model loaded from weights."""
```
Because the body isn't literally `...`, pyright analyzes it as a real implementation that implicitly returns `None`, which doesn't match the declared return type. **Classification: COSMETIC.** These `Protocol` classes are never instantiated — they exist purely as structural type-checking contracts for dependency injection (mirrored from output.py's own `*Like` Protocol style). The diagnostic is about the Protocol's own placeholder body, not about any real caller or implementation; nothing at runtime ever executes this code path. Purely a style nit (swap the docstring for `...`, or add both).

### C-4. Real `ultralytics.YOLO` doesn't structurally satisfy the local `YoloModel` Protocol, due to ultralytics' own inconsistent stub for `predict()` plus a read-only `names` property

**Affected:** `training/export.py:189:12`, `vision/infer.py:123:16`

```
"predict" is an incompatible type
  Type "(source: str | Path | int | Image | ... = None, ...) -> list[Results]" is not assignable to "(**kwargs: Any) -> Any"
    "None" is not assignable to type "str | Path | int | Image | ..."
```
Verified directly in `.venv/.../ultralytics/engine/model.py:484`: `source: str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor = None` — the default value `None` is not itself a member of the declared type union. This is a pre-existing inconsistency in ultralytics 8.4.90's own bundled type hints, external to this codebase. `vision/infer.py:123` additionally flags `names` as "invariant because it is mutable": `YoloModel.names: dict[int, str]` is a plain (implicitly read/write) attribute, but the real `YOLO.names` (`ultralytics/engine/model.py:905`) is a read-only `@property`. **Classification: COSMETIC.** Nothing in this repo ever assigns to `.names` (verified: no `.names = ` write anywhere in `vision/` or `training/`), and every real call to `.predict()`/`.export()` in this codebase uses ordinary keyword arguments (`source=`, `conf=`, `format=`, etc.) that work fine against the real ultralytics API — the incompatibility is purely a byproduct of a third-party stub quirk plus Python's Protocol-attribute invariance rule, not a bug in how this project calls Ultralytics.

### C-5. `ncnn` is an untyped compiled C-extension — pyright can't see `Net`/`Mat`

**Affected:** `training/export.py:333:16`, `training/export.py:338:19`, `training/export.py:340:14`, `vision/detector.py:505:26`, `vision/detector.py:522:23`, `vision/detector.py:524:18`

`ncnn/__init__.py` does `from .ncnn import *`, where `ncnn/ncnn.cpython-312-darwin.so` is a compiled binary extension with no accompanying `.pyi` stub. Pyright has literally no way to know `ncnn.Net`/`ncnn.Mat` exist. **Classification: COSMETIC.** `ncnn` is installed in `.venv` (confirmed: `ncnn-1.0.20260526`) and `Net()`/`Mat.from_pixels(...)` work fine at runtime — this is purely a missing-stub limitation for a third-party compiled package, identical in nature to a `# type: ignore[import-untyped]` situation. No project-code fix is possible short of hand-writing a `.pyi` stub for `ncnn` (optional future nicety, not a correctness issue).

### C-6. Ultralytics `Boxes.xyxy`/`.conf`/`.cls` are typed `Tensor | ndarray`; `.cpu()` only exists on the `Tensor` half

**Affected:** `training/masked_validation_experiment.py:215:27`, `:216:25`, `:217:28`

Verified directly in `ultralytics/engine/results.py`: `Boxes` extends `BaseTensor`, whose `.data` can be either a `torch.Tensor` or an `np.ndarray` depending on whether `.cpu()`/`.numpy()` was already called on the wrapping `Results`/`Boxes` object — hence the union return type on `.xyxy`/`.conf`/`.cls`. **Classification: COSMETIC.** `decode_after_argmax_filter()` calls `.xyxy.cpu().numpy()` immediately after a fresh `model.predict(...)` call, before anything has converted the underlying data to numpy — at that point `.data` is always a live `torch.Tensor` (CPU tensors also support `.cpu()` as a no-op), so `.cpu()` never fails for this call pattern. The union only becomes "real" if this code ever called `results.numpy()` first, which it doesn't.

### C-7. `model.model` (ultralytics `Model.model`) is inferred as `None | str | Path | nn.Module | ...` since it's assigned differently across ultralytics' own `_load()` branches — but always resolves to a real `nn.Module` for this project's usage

**Affected:** (same pattern, 3 near-duplicate scripts)
- `training/masked_validation_experiment.py:363:23` (×2), `:365:29`, `:365:41` (×2)
- `training/resolution_and_remap_report.py:168:23` (×2), `:170:29`, `:170:41` (×2)
- `training/threshold_sweep_report.py:250:23` (×2), `:252:29`, `:252:41` (×2)

```python
model = YOLO(str(WEIGHTS))
net = model.model.to(DEVICE)          # "to" not known on None / str
net.eval()
names: dict[int, str] = model.model.names   # "names" not known on None / str; assignment-type mismatch
```
`ultralytics/engine/model.py`'s untyped `self.model` attribute is assigned `None` at `__init__` (line 110), then in `_load()` either `self.model, self.ckpt = load_checkpoint(weights)` (a real `nn.Module`, when the suffix is `.pt` — line 283) or `self.model, self.ckpt = weights, None` (a bare string, for non-`.pt` weight formats — line 289). Pyright infers the attribute's type as the union of every assignment site in the whole ultralytics class. **Classification: COSMETIC**, and *provably* so: all three scripts check `WEIGHTS.exists()` before constructing `YOLO(str(WEIGHTS))`, and `WEIGHTS` is a hard-coded path ending in `.pt`. That guarantees `str(weights).rpartition(".")[-1] == "pt"` is true, which forces the `load_checkpoint(weights)` branch every single time — `self.model` is never `None` or `str` for any of these three scripts. `resolution_and_remap_report.py` and `threshold_sweep_report.py` import `masked_validation_experiment` as `mv` and copy this exact block verbatim (`mv.WEIGHTS`, `mv.YOLO`), which is why the same 5-diagnostic fingerprint appears three times.

### C-8. `max(dict, key=dict.get)` idiom — `dict.get(key)` (1-arg) types as `V | None`, but iterating the dict's own keys guarantees a hit

**Affected:** `training/masked_validation_experiment.py:437:25`, `:437:49`; `training/resolution_and_remap_report.py:142:23`, `:142:47`; `training/threshold_sweep_report.py:505:14`, `:505:37`

```python
best_sibling_id = max(sibling_scores, key=sibling_scores.get)
```
`typeshed`'s `dict.get` has a 1-arg overload returning `V | None`; `max`'s `key` callback must return `SupportsRichComparison`, and `None` doesn't support `<`. **Classification: COSMETIC.** `max(some_dict, ...)` iterates over `some_dict`'s own keys, so every call to `.get(key)` is guaranteed to hit an existing entry and never actually returns `None` — this is a well-known, extremely common pyright/mypy false positive for exactly this idiom. Same pattern, three copies (int/float/str-keyed variants), same root cause.

### C-9. `r` (`RecallResult` / `dict[str, RecallResult]`) reused across two loops in the same function body; captured inside a `lambda`, pyright falls back to the whole-function union type

**Affected:** `training/masked_validation_experiment.py:673:70`

`main()` reuses the bare name `r` for two different purposes: `r = results[target_name]["C"]` (a single `RecallResult`, line 596, in one `for` loop) and `r = results[n]` (a `dict[str, RecallResult]`, lines 630/662, in later loops). At line 673, `best_t = max(SUPPLEMENTARY_THRESHOLDS, key=lambda t: r[f"C@{t}"].recall)` captures `r` inside a `lambda`. Pyright does not apply flow-narrowing for variables captured by a nested function/lambda (correctly so in general — the closure could run after the variable is reassigned), so it falls back to `r`'s declared type across the *entire* function: `RecallResult | dict[str, RecallResult]`. `RecallResult` (a plain `@dataclass`) has no `__getitem__`, hence the error. **Classification: COSMETIC.** At the point this specific lambda is actually invoked — synchronously, as `max()`'s `key=` callback, within the same `for n in weak:` loop iteration where `r = results[n]` was just assigned two lines earlier (line 662) and never reassigned before line 673 — `r` is always the `dict[str, RecallResult]`. This is a textbook closure-narrowing limitation, not a bug. (Worth a cheap readability fix regardless: reusing `r` for two unrelated types in one function is a minor code smell independent of the type-checker complaint.)

### C-10. `getattr(value, "mean", None)` narrowed by `callable()` loses its real return type

**Affected:** `training/train.py:857:32`

```python
mean = getattr(value, "mean", None)
if callable(mean):
    parsed = float(mean())   # mean() inferred as returning `object`
```
Modern typeshed types `callable()` as `TypeGuard[Callable[..., object]]`. Since `mean`'s original type from the 3-arg `getattr` overload is essentially `Any`/unknown, narrowing through `callable()` collapses it to `Callable[..., object]` — so `mean()`'s return type becomes bare `object`, which `float()` rejects (`object` doesn't implement `__float__`/`__index__`). **Classification: COSMETIC.** At runtime, `value` here is expected to be a numpy array or similar (this is `_as_optional_float`'s fallback path for coercing metrics/confusion-matrix values), and `numpy.ndarray.mean()` returns a real numpy scalar that supports `__float__` fine. This is a well-known typeshed/pyright precision loss for the `getattr(...) `+`callable()` pattern, not a real runtime risk.

### C-11. Hand-rolled `cv2.VideoCapture` Protocols name the `get()` parameter `prop_id`; cv2's real stub calls it `propId`

**Affected:** `vision/capture.py:300:25`, `vision/webcam_infer.py:233:41`

Both `vision/capture.py`'s `_CvCapture` Protocol and `vision/webcam_infer.py`'s `VideoCapture` Protocol declare `def get(self, prop_id: int) -> float: ...` (snake_case), while cv2's real stub declares `def get(self, propId: int) -> float`. Python's structural typing checks keyword-parameter names for non-positional-only parameters, so `cv2.VideoCapture` doesn't structurally satisfy either local Protocol, which cascades into `self._cap = cap` (capture.py:300) and `cap_factory = capture_factory or cv2.VideoCapture` → `open_webcam(..., cap_factory)` (webcam_infer.py:233) both failing. **Classification: COSMETIC.** Every call site in this codebase calls `.get(cv2.CAP_PROP_FPS)` / `.get(cv2.CAP_PROP_FRAME_WIDTH)` **positionally** — never as `.get(prop_id=...)` — so the naming mismatch has zero runtime consequence today. It would only bite if some future caller (or a test fake written strictly against the local Protocol's spelling) called `.get()` by keyword against a real `cv2.VideoCapture`, or vice versa; currently dormant. Trivial one-line fix (rename the Protocol parameter to `propId`) whenever convenient, no urgency.

### C-12. `Detector.target_class_ids` is a `frozenset[int]`, but `DetectorBackend.infer()` declares `Sequence[int]`

**Affected:** `vision/detector.py:890:62`

`Detector.set_target_classes()` stores `self.target_class_ids: frozenset[int] | None`, then `detect()` passes it to `self.backend.infer(model_input, transform, self.target_class_ids)`, where every backend's `infer()` declares `target_class_ids: Sequence[int] | None`. `frozenset` is a `Collection`/`AbstractSet`, not a `Sequence` (no `__getitem__`/ordering). **Classification: COSMETIC.** Traced every actual use of `target_class_ids` inside `_decode_yolov8`/`_masked_class_argmax` (`vision/detector.py:573-631`): the parameter is *only ever iterated* (`{int(class_id) for class_id in target_class_ids}`, then `sorted(...)` into a fresh array) — never indexed, sliced, or reversed. A `frozenset[int]` supports iteration perfectly well, so nothing in the actual code path relies on the `Sequence`-specific operations the declared type promises. The declared parameter type is simply narrower than what the function body needs (`Iterable[int]` would be sufficient and accurate).

---

## The CSVReader Question — Direct Answer

**Question:** does `CSVReader` hold a genuinely unreleased resource, or is `.close()` cosmetic?

**Answer: it is a genuine, real resource leak — not cosmetic — and it is one head of the RB-3 bug above.**

This `CSVReader` is not a project-local class; it's `pymavlink.CSVReader.CSVReader` (`.venv/lib/python3.12/site-packages/pymavlink/CSVReader.py`), one of the many possible concrete return types of `pymavlink.mavutil.mavlink_connection()`, which `geo/telemetry.py:TelemetryBuffer._reader_loop()` calls with whatever connection string is passed to `TelemetryBuffer.start()`.

Tracing `CSVReader.__init__`:
```python
def __init__(self, filename, ...):
    ...
    self.f = None
    self._rewind()      # opens files etc etc
    self.recv_msg()      # populate self.messages
    self._rewind()       # opens files etc etc

def _rewind(self):
    if self.f is not None:
        self.f.close()
    self.f = open(self.filename, mode='r')   # <-- a real, live file handle
```
`CSVReader` opens a real OS file handle (`self.f`) in its constructor and keeps it open for the object's lifetime. **It has no public `close()` method at all** — only the private `_rewind()` closes-then-reopens it internally, and there is no `__del__`/context-manager finalizer either.

`geo/telemetry.py`'s `_reader_loop()`:
```python
conn = mavutil.mavlink_connection(self._connection_str)
...
self._mavlink_loop(conn)
...
finally:
    if conn is not None:
        with contextlib.suppress(Exception):
            conn.close()
```
If `self._connection_str` is ever a `"csv:..."` string — a real, documented pymavlink connection string (see `CSVReader.py`'s own module docstring, `MAVExplorer.py "csv:$LOG:..."`), plausible for offline log-replay/debugging given this project's own "hardware-free" testing ethos — `conn` is a `CSVReader`, and:

1. `_mavlink_loop(conn)`'s very first message-receive call is `conn.recv_match(type=_WANTED, blocking=True, timeout=1.0)`. `CSVReader.recv_match(self, condition=None, type=None, blocking=False)` has **no `timeout` parameter and no `**kwargs`** — this raises `TypeError: recv_match() got an unexpected keyword argument 'timeout'` immediately, before a single pose sample is ever read.
2. That `TypeError` propagates up and is caught by `_reader_loop`'s own `except Exception:` (logged), then the `finally` block calls `conn.close()` — which raises `AttributeError` (no such method), silently swallowed by `contextlib.suppress(Exception)`.
3. The outer reconnect loop backs off and retries, calling `mavlink_connection(...)` again, which constructs a **brand-new** `CSVReader` — opening **another** real file handle via `_rewind()` — and immediately hits the same `TypeError` again.

Net effect if a CSV connection string is ever used: telemetry ingestion never produces a single pose sample (permanently silent failure, no crash reaches the mission loop), **and** every reconnect attempt (every 1-30s, capped by the existing exponential backoff) leaks one more open file descriptor that is never released, for as long as `TelemetryBuffer` keeps running. This is a genuine, unbounded file-descriptor leak, not a cosmetic type-checker complaint — it just happens to currently be dormant because production wiring (`core/mission.py:main()`) always uses `"udpin:0.0.0.0:14550"`, never a CSV string.

---

## Prioritized Fix Order

### Tier 1 — REAL BUGS, ranked by ease of triggering

1. **RB-1 / RB-2 (tie) — `Tracker`/`MedianGeolocatorAdapter` vs `TrackerLike`/`GeolocatorLike`** (`core/mission.py:599-600`). Highest priority despite requiring a "swap the collaborator" trigger: this codebase's own stated design philosophy is dependency injection against minimal Protocols with hardware-free fakes (see `test_mission.py`, output.py ASSUMPTION O15) — writing a new Protocol-conforming fake is not an edge case here, it's the normal way this codebase is tested and extended. Any next test author or hardware-integration swap that trusts the documented Protocol (not the accidental extra requirements of the current concrete classes) hits an `AttributeError`/`TypeError` immediately. Fix before any new test fixture or alternate tracker/geolocator is introduced.
2. **RB-4 — `cv2.imread()` → `forward_raw` None crash** (`training/masked_validation_experiment.py:415`). Requires no deliberate substitution at all — just one bad/corrupt file already present in a real image dataset, entirely plausible for data pulled from an external export. Low blast radius (kills an offline analysis script, not the flight pipeline), but trivially easy to hit by accident. Fix with a simple `if sample_frame is None: ...` guard and a clear error message.
3. **RB-3 — `mavlink_connection()`/`CSVReader` narrowness (file-descriptor leak + silent telemetry failure)** (`geo/telemetry.py:514,543`). Lowest priority of the four REAL BUGs to *trigger* today: it requires deliberately passing a non-default (`"csv:..."`) connection string, which nothing in the current production wiring does. But it should be fixed before this project ever grows log-replay/offline-debugging tooling around `TelemetryBuffer`, since the failure mode (silent, permanent, leaking) is the worst kind to discover late.

### Tier 2 — UNCERTAIN, needs a human decision

4. **UN-1 — `OnnxBackend` output-type breadth** (`vision/detector.py:440`). Not urgent (the backend is explicitly unverified scaffolding with no real weights yet), but flag it as a checklist item for whoever first exercises the ONNX export path: confirm `outputs[0]` is always a dense `ndarray` for this project's specific export config, or add an explicit `isinstance` guard if not.

### Tier 3 — COSMETIC, batch-fixable whenever convenient, no urgency

All 11 remaining root causes (C-1 through C-12, 44 diagnostics) are safe to leave as-is indefinitely or batch-fix in one pass; none of them reflect incorrect runtime behavior under this codebase's current or reasonably foreseeable usage:
- Third-party stub/annotation gaps external to this project: **C-4** (ultralytics `predict` default-value stub bug + `names` property invariance), **C-5** (`ncnn` compiled extension, no stubs), **C-6** (ultralytics `Boxes.xyxy` Tensor|ndarray union), **C-7** (ultralytics `Model.model` broad union, provably always populated here), **C-10** (`callable()` TypeGuard precision loss).
- Type-checker idiom false positives: **C-8** (`max(dict, key=dict.get)`), **C-9** (lambda-closure narrowing).
- Project-local annotations narrower/wider than necessary but functionally harmless: **C-1** (documented `TypeError`-as-control-flow), **C-2** (`tuple` vs `ndarray`, coerced internally), **C-3** (Protocol stub bodies), **C-11** (`prop_id`/`propId` naming), **C-12** (`frozenset` vs `Sequence`, only ever iterated).

Suggested batch-fix order if/when addressed: C-3 and C-11 first (one-line, zero-risk stylistic fixes); then C-7/C-8/C-9 together (they're all in the same three `training/*_report.py` files); then the remaining third-party-stub items only if a `# type: ignore` policy or hand-written stub package is desired.
