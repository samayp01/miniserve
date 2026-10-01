import base64
import io

from PIL import Image

from benchmarks.workload import Workload


def test_image_spec_attaches_a_unique_512px_image_to_image_requests():
    reqs = Workload("image", model="smolvlm").schedule(4, 40, "t")
    images = [r["image"] for r in reqs if r["bucket"] == "image"]
    assert images and all("image" not in r for r in reqs if r["bucket"] != "image")
    assert len(set(images)) == len(images)
    assert Image.open(io.BytesIO(base64.b64decode(images[0]))).size == (512, 512)


def test_schedule_is_reproducible():
    a = Workload("image", model="smolvlm").schedule(4, 20, "t")
    b = Workload("image", model="smolvlm").schedule(4, 20, "t")
    assert a == b
