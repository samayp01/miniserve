import base64
import io

from PIL import Image

from benchmarks.workload import Workload


def test_image_spec_attaches_a_unique_512px_image_to_image_requests():
    reqs = Workload("image", model="smolvlm").schedule(4, 40, "t")
    images = [r["media"][0]["data"] for r in reqs if r["bucket"] == "image"]
    assert images and all("media" not in r for r in reqs if r["bucket"] != "image")
    assert len(set(images)) == len(images)
    assert Image.open(io.BytesIO(base64.b64decode(images[0]))).size == (512, 512)


def test_schedule_is_reproducible():
    a = Workload("image", model="smolvlm").schedule(4, 20, "t")
    b = Workload("image", model="smolvlm").schedule(4, 20, "t")
    assert a == b


def test_images_differ_between_levels():
    w = Workload("image", model="smolvlm")
    first = {r["media"][0]["data"] for r in w.schedule(4, 40, "a") if "media" in r}
    second = {r["media"][0]["data"] for r in w.schedule(4, 40, "b") if "media" in r}
    assert first and not first & second


def test_repeat_resends_images_from_the_same_level():
    reqs = [r for r in Workload("image", model="smolvlm", repeat=0.9).schedule(4, 64, "a") if "media" in r]
    assert len({r["media"][0]["data"] for r in reqs}) < len(reqs)
